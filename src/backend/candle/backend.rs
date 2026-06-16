use crate::backend::candle::quantized_llama_fork::ModelWeights;
use crate::backend::traits::*;
use crate::error::{Result, ZllmError};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor as CandleTensor};
use std::path::Path;

pub struct CandleCpuBackend {
    model: Option<ModelWeights>,
    device: Device,
    n_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    hidden_states: Vec<Vec<f32>>,
    next_block_id: BlockId,
    index_pos: usize,
}

/// Pick the best Candle device available at runtime given the compiled
/// feature set. Metal takes precedence on macOS, CUDA on Linux/Windows,
/// CPU as a safe fallback if either init fails (e.g., compiled with
/// `--features cuda` but no NVIDIA driver present).
///
/// The struct name `CandleCpuBackend` is historical from before v0.6 —
/// it now backs all three devices. Renaming is held back to avoid a
/// large mechanical diff across callers.
fn select_best_device() -> Device {
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(d) => {
                tracing::info!("Candle backend: selected Metal device 0");
                return d;
            }
            Err(e) => tracing::debug!("Metal init failed, falling back: {e}"),
        }
    }
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(d) => {
                tracing::info!("Candle backend: selected CUDA device 0");
                return d;
            }
            Err(e) => tracing::debug!("CUDA init failed, falling back: {e}"),
        }
    }
    tracing::info!("Candle backend: selected CPU");
    Device::Cpu
}

impl CandleCpuBackend {
    pub fn new() -> Self {
        Self {
            model: None,
            device: select_best_device(),
            n_layers: 0,
            hidden_size: 0,
            vocab_size: 0,
            hidden_states: Vec::new(),
            next_block_id: 0,
            index_pos: 0,
        }
    }

    /// Build a backend with an explicit device. Useful for tests, for
    /// pinning to a specific GPU ordinal, or for forcing CPU on a box
    /// that has a flaky GPU.
    pub fn with_device(device: Device) -> Self {
        Self {
            model: None,
            device,
            n_layers: 0,
            hidden_size: 0,
            vocab_size: 0,
            hidden_states: Vec::new(),
            next_block_id: 0,
            index_pos: 0,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Number of token positions currently in the KV cache. Equals
    /// `index_pos`. Used by speculative decoding to know how far to
    /// truncate after a rejected draft.
    pub fn position(&self) -> usize {
        self.index_pos
    }

    /// Reset both the position counter and the underlying KV cache so
    /// the next forward pass starts from a clean slate. Call between
    /// independent chat requests — without this, position + cache
    /// accumulate across requests until the model's effective range is
    /// exceeded and generation collapses to immediate EOS.
    pub fn reset_position(&mut self) {
        self.index_pos = 0;
        if let Some(model) = self.model.as_mut() {
            model.clear_kv_cache();
        }
    }

    /// Keep the first `n` token positions in the KV cache and set
    /// `index_pos = n`. Used by prompt-prefix caching to reuse a
    /// previous request's prefill K/V instead of recomputing it.
    /// Returns `Err` if the KV-truncation tensor op fails.
    pub fn truncate_to(&mut self, n: usize) -> Result<()> {
        self.index_pos = n;
        if let Some(model) = self.model.as_mut() {
            model
                .truncate_kv_cache(n)
                .map_err(|e| ZllmError::Backend(format!("truncate_kv_cache: {e}")))?;
        }
        Ok(())
    }

    /// Number of transformer blocks in the loaded model. Returns 0 if no
    /// model is loaded yet.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Run the forward pass and return the next-token logits as a
    /// `Vec<f32>` of length `vocab_size`. Increments `index_pos` for
    /// KV-cache state. Use this when you want to apply a custom sampler
    /// (the chat path in `src/server/rest.rs` does — it calls
    /// `engine::sampler::sample`). The greedy path stays available via
    /// `generate_token`, which delegates here and applies argmax.
    pub fn forward_logits(&mut self, prompt_tokens: &[u32]) -> Result<Vec<f32>> {
        self.forward_logits_with_observer(prompt_tokens, |_, _| {})
    }

    /// Same as `forward_logits`, but invokes `on_layer(layer_idx, &hidden)`
    /// after every transformer block. `hidden` is the live residual
    /// stream as a borrow on the underlying candle tensor (shape
    /// `(1, seq_len, n_embd)`). The hook can read it cheaply (no copy)
    /// for memory capture; mutation requires building a replacement
    /// tensor and is out of v0.7 scope.
    ///
    /// This is the surface the chat handler / inference runner uses to
    /// fire its `HookRegistry` mid-inference — see
    /// `crate::backend::candle::quantized_llama_fork::ModelWeights::forward_with_callback`
    /// for why the fork exists.
    pub fn forward_logits_with_observer<F: FnMut(usize, &CandleTensor)>(
        &mut self,
        prompt_tokens: &[u32],
        on_layer: F,
    ) -> Result<Vec<f32>> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| ZllmError::Model("model not loaded".into()))?;

        let input = CandleTensor::new(prompt_tokens, &self.device)
            .map_err(|e| ZllmError::Backend(format!("tensor creation: {e}")))?
            .unsqueeze(0)
            .map_err(|e| ZllmError::Backend(format!("unsqueeze: {e}")))?;

        let logits = model
            .forward_with_callback(&input, self.index_pos, on_layer)
            .map_err(|e| ZllmError::Backend(format!("forward pass: {e}")))?;

        self.index_pos += prompt_tokens.len();

        let logits_vec: Vec<f32> = logits
            .squeeze(0)
            .map_err(|e| ZllmError::Backend(format!("squeeze: {e}")))?
            .to_vec1()
            .map_err(|e| ZllmError::Backend(format!("to_vec1: {e}")))?;

        Ok(logits_vec)
    }

    /// Diagnostic: project every layer's hidden state through the
    /// final norm + LM head, returning the top-1 token per layer.
    /// Used to investigate "is early exit viable on this model?" by
    /// measuring layer-vs-final agreement. Expensive — runs n_layers
    /// extra LM head matmuls — so for analysis only.
    pub fn forward_per_layer_argmax(
        &mut self,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, Vec<u32>)> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| ZllmError::Model("model not loaded".into()))?;
        let input = CandleTensor::new(tokens, &self.device)
            .map_err(|e| ZllmError::Backend(format!("tensor creation: {e}")))?
            .unsqueeze(0)
            .map_err(|e| ZllmError::Backend(format!("unsqueeze: {e}")))?;
        let (logits, top1_per_layer) = model
            .forward_per_layer_argmax(&input, self.index_pos)
            .map_err(|e| ZllmError::Backend(format!("forward pass: {e}")))?;
        self.index_pos += tokens.len();
        let logits_vec: Vec<f32> = logits
            .squeeze(0)
            .map_err(|e| ZllmError::Backend(format!("squeeze: {e}")))?
            .to_vec1()
            .map_err(|e| ZllmError::Backend(format!("to_vec1: {e}")))?;
        Ok((logits_vec, top1_per_layer))
    }

    /// Forward with a per-layer "should we exit now?" callback. Wraps
    /// `ModelWeights::forward_with_early_exit`. Returns
    /// `(logits, exit_layer_idx)` — `exit_layer_idx == n_layers - 1`
    /// means no early exit fired.
    pub fn forward_logits_early_exit<F>(
        &mut self,
        prompt_tokens: &[u32],
        should_exit: F,
    ) -> Result<(Vec<f32>, usize)>
    where
        F: FnMut(usize, &CandleTensor) -> bool,
    {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| ZllmError::Model("model not loaded".into()))?;
        let input = CandleTensor::new(prompt_tokens, &self.device)
            .map_err(|e| ZllmError::Backend(format!("tensor creation: {e}")))?
            .unsqueeze(0)
            .map_err(|e| ZllmError::Backend(format!("unsqueeze: {e}")))?;
        let (logits, exit_at) = model
            .forward_with_early_exit(&input, self.index_pos, should_exit)
            .map_err(|e| ZllmError::Backend(format!("forward pass: {e}")))?;
        self.index_pos += prompt_tokens.len();
        let logits_vec: Vec<f32> = logits
            .squeeze(0)
            .map_err(|e| ZllmError::Backend(format!("squeeze: {e}")))?
            .to_vec1()
            .map_err(|e| ZllmError::Backend(format!("to_vec1: {e}")))?;
        Ok((logits_vec, exit_at))
    }

    /// Multi-position forward used by speculative decoding. Returns
    /// `(seq_len, vocab)` — one logit vector per input token. Caller
    /// is responsible for truncating the KV cache afterwards if any
    /// of the input positions are rejected.
    pub fn forward_all_logits(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| ZllmError::Model("model not loaded".into()))?;
        let input = CandleTensor::new(tokens, &self.device)
            .map_err(|e| ZllmError::Backend(format!("tensor creation: {e}")))?
            .unsqueeze(0)
            .map_err(|e| ZllmError::Backend(format!("unsqueeze: {e}")))?;
        let logits = model
            .forward_all_positions(&input, self.index_pos)
            .map_err(|e| ZllmError::Backend(format!("forward pass: {e}")))?;
        self.index_pos += tokens.len();
        // logits shape: (1, seq_len, vocab). Squeeze batch, return vec-of-vec.
        let squeezed = logits
            .squeeze(0)
            .map_err(|e| ZllmError::Backend(format!("squeeze: {e}")))?;
        let rows: Vec<Vec<f32>> = squeezed
            .to_vec2()
            .map_err(|e| ZllmError::Backend(format!("to_vec2: {e}")))?;
        Ok(rows)
    }

    pub fn generate_token(&mut self, prompt_tokens: &[u32]) -> Result<(u32, Vec<Vec<f32>>)> {
        let logits_vec = self.forward_logits(prompt_tokens)?;
        let token_id = argmax_token(&logits_vec);
        // Hidden states not extractable from quantized forward without
        // modification — second return is reserved for a future forked
        // forward pass.
        Ok((token_id, vec![]))
    }
}

fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

impl Backend for CandleCpuBackend {
    fn load_model(&mut self, path: &Path, _config: &QuantConfig) -> Result<()> {
        tracing::info!("Loading GGUF model from {:?}", path);

        let mut file = std::fs::File::open(path)
            .map_err(|e| ZllmError::Model(format!("cannot open model file: {e}")))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| ZllmError::Model(format!("invalid GGUF file: {e}")))?;

        // Extract model metadata
        let n_layers = content
            .metadata
            .get("llama.block_count")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(32) as usize;

        let hidden_size = content
            .metadata
            .get("llama.embedding_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4096) as usize;

        let vocab_size = content
            .metadata
            .get("llama.vocab_size")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(128256) as usize;

        tracing::info!(
            "Model metadata: layers={n_layers}, hidden={hidden_size}, vocab={vocab_size}"
        );

        let model = ModelWeights::from_gguf(content, &mut file, &self.device)
            .map_err(|e| ZllmError::Model(format!("failed to load model weights: {e}")))?;

        self.model = Some(model);
        self.n_layers = n_layers;
        self.hidden_size = hidden_size;
        self.vocab_size = vocab_size;
        self.hidden_states = vec![vec![0.0; hidden_size]; n_layers];
        self.index_pos = 0;

        tracing::info!("Model loaded successfully");
        Ok(())
    }

    fn unload_model(&mut self) -> Result<()> {
        self.model = None;
        self.hidden_states.clear();
        self.index_pos = 0;
        tracing::info!("Model unloaded");
        Ok(())
    }

    fn forward_layer(
        &self,
        _layer_idx: usize,
        hidden_state: &Tensor,
        _seq_len: usize,
    ) -> Result<Tensor> {
        // Per-layer forward not directly supported by quantized model
        // Return input unchanged (hooks still work on the hidden state)
        Ok(hidden_state.clone())
    }

    fn read_hidden_state(&self, layer_idx: usize) -> Result<Tensor> {
        if layer_idx < self.hidden_states.len() {
            Ok(self.hidden_states[layer_idx].clone())
        } else {
            Err(ZllmError::Backend(format!(
                "layer {layer_idx} out of range (max {})",
                self.hidden_states.len()
            )))
        }
    }

    fn write_hidden_state(&mut self, layer_idx: usize, state: &Tensor) -> Result<()> {
        if layer_idx < self.hidden_states.len() {
            self.hidden_states[layer_idx] = state.clone();
            Ok(())
        } else {
            Err(ZllmError::Backend(format!(
                "layer {layer_idx} out of range"
            )))
        }
    }

    fn compute_logits(&self, _hidden_state: &Tensor) -> Result<Tensor> {
        // This is handled by generate_token() for the candle backend
        // Return zeros as placeholder — real logits come from forward pass
        Ok(vec![0.0f32; self.vocab_size])
    }

    fn alloc_kv_block(&mut self, _n_tokens: usize) -> Result<BlockId> {
        self.next_block_id += 1;
        Ok(self.next_block_id)
    }

    fn free_kv_block(&mut self, _id: BlockId) -> Result<()> {
        Ok(())
    }

    fn device_info(&self) -> DeviceInfo {
        let (name, backend) = if self.device.is_cuda() {
            ("CUDA (candle)".to_string(), "candle-cuda".to_string())
        } else if self.device.is_metal() {
            ("Metal (candle)".to_string(), "candle-metal".to_string())
        } else {
            ("CPU (candle)".to_string(), "candle-cpu".to_string())
        };
        DeviceInfo {
            name,
            backend,
            memory_total_mb: 0, // TODO: detect via candle once exposed
            memory_used_mb: 0,
            supports_fp8: false,
            supports_fp4: false,
        }
    }
}

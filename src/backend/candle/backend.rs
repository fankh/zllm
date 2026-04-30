use crate::backend::traits::*;
use crate::error::{Result, ZllmError};
use candle_core::{DType, Device, Tensor as CandleTensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama::ModelWeights;
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

impl CandleCpuBackend {
    pub fn new() -> Self {
        Self {
            model: None,
            device: Device::Cpu,
            n_layers: 0,
            hidden_size: 0,
            vocab_size: 0,
            hidden_states: Vec::new(),
            next_block_id: 0,
            index_pos: 0,
        }
    }

    pub fn reset_position(&mut self) {
        self.index_pos = 0;
    }

    pub fn generate_token(&mut self, prompt_tokens: &[u32]) -> Result<(u32, Vec<Vec<f32>>)> {
        let model = self.model.as_mut()
            .ok_or_else(|| ZllmError::Model("model not loaded".into()))?;

        let input = CandleTensor::new(prompt_tokens, &self.device)
            .map_err(|e| ZllmError::Backend(format!("tensor creation: {e}")))?
            .unsqueeze(0)
            .map_err(|e| ZllmError::Backend(format!("unsqueeze: {e}")))?;

        // Forward pass — gets logits
        let logits = model.forward(&input, self.index_pos)
            .map_err(|e| ZllmError::Backend(format!("forward pass: {e}")))?;

        self.index_pos += prompt_tokens.len();

        // Extract logits as Vec<f32>
        let logits_vec: Vec<f32> = logits
            .squeeze(0)
            .map_err(|e| ZllmError::Backend(format!("squeeze: {e}")))?
            .to_vec1()
            .map_err(|e| ZllmError::Backend(format!("to_vec1: {e}")))?;

        // Get the argmax token for now (greedy)
        let token_id = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        // Hidden states not extractable from quantized forward without modification
        // Return empty vec for now — will implement with forked forward pass
        Ok((token_id, vec![]))
    }
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
        DeviceInfo {
            name: "CPU (candle)".to_string(),
            backend: "candle-cpu".to_string(),
            memory_total_mb: 0, // TODO: detect system RAM
            memory_used_mb: 0,
            supports_fp8: false,
            supports_fp4: false,
        }
    }
}

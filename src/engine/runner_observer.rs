use crate::engine::confidence::ConfidenceHead;
use crate::engine::hooks::registry::HookRegistry;
use crate::engine::hooks::traits::{HookAction, HookContext};
use crate::engine::memory_store::{InspectionTrace, LayerSnapshot, TokenSnapshot};
use candle_core::Tensor as CandleTensor;
use std::cell::Cell;
use std::sync::{Arc, RwLock};

/// Observer-driven bridge from the backend's push-shaped
/// `CandleCpuBackend::forward_logits_with_observer(|layer_idx, &CandleTensor|)`
/// callback into the runner's pull-shaped HookRegistry. Lets the chat
/// path get per-layer hook firings, confidence updates, memory captures,
/// and inspection traces — the same features `InferenceRunner::generate`
/// provides for the (still-stubbed) `forward_layer` path.
///
/// Read-only in v0.8: hook mutations to `&mut Tensor` are computed but
/// discarded — the backend tensor is borrowed immutably and there is no
/// path back into the running forward pass. Capture-only hook branches
/// (e.g. `MemoryInjectHook` at `capture_layer`) work correctly because
/// they don't depend on the mutation propagating. True mid-forward
/// injection requires extending the observer signature in
/// `quantized_llama_fork.rs` and lands in v0.9.
pub struct RunnerObserver {
    hooks: Arc<HookRegistry>,
    pub context: HookContext,
    enable_inspection: bool,
    /// Filled progressively as layers fire. Behind a `RwLock` so the
    /// observer can be used through `&self` from inside the `FnMut`
    /// closure passed to the backend.
    layer_snapshots: RwLock<Vec<LayerSnapshot>>,
    /// One per sampled token in the autoregressive decode. Populated
    /// from the chat loop via `record_token`. Empty if the request
    /// generated nothing (e.g. early-exit before any token).
    token_snapshots: RwLock<Vec<TokenSnapshot>>,
    /// Mean-pooled hidden state from the most recent layer, kept so
    /// the chat path can score importance / confidence after the
    /// forward pass returns.
    last_hidden: RwLock<Vec<f32>>,
    pub last_confidence: Cell<f32>,
    pub early_exit_signal: Cell<bool>,
    pub early_exit_reason: RwLock<Option<String>>,
}

impl RunnerObserver {
    pub fn new(hooks: Arc<HookRegistry>, request_id: impl Into<String>) -> Self {
        Self {
            hooks,
            context: HookContext::new(request_id),
            enable_inspection: false,
            layer_snapshots: RwLock::new(Vec::new()),
            token_snapshots: RwLock::new(Vec::new()),
            last_hidden: RwLock::new(Vec::new()),
            last_confidence: Cell::new(0.0),
            early_exit_signal: Cell::new(false),
            early_exit_reason: RwLock::new(None),
        }
    }

    pub fn with_inspection(mut self, enabled: bool) -> Self {
        self.enable_inspection = enabled;
        self
    }

    pub fn last_hidden(&self) -> Vec<f32> {
        self.last_hidden.read().unwrap().clone()
    }

    /// Called once per layer by `forward_logits_with_observer`. Mean-pools
    /// the `(1, seq_len, d_model)` tensor down to a `d_model` vector so
    /// hooks see a per-token-equivalent signal without paying for the
    /// full `seq_len * d_model` flatten on every layer.
    /// Returns `Some(new_hidden)` when a registered hook contributes a write-back
    /// to the full `(1, seq_len, d_model)` residual stream (steering / inject);
    /// `None` = observe-only. The forward continues from the returned tensor.
    pub fn on_layer(&self, layer_idx: usize, hidden: &CandleTensor) -> Option<CandleTensor> {
        let pooled = match hidden
            .mean(1)
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_vec1::<f32>())
        {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("RunnerObserver mean-pool failed at layer {layer_idx}: {e}");
                return None;
            }
        };

        let confidence = ConfidenceHead::estimate(&pooled);
        self.last_confidence.set(confidence);
        self.context.current_confidence.set(confidence);

        // Observe path: hooks see the pooled `&mut Vec<f32>` (confidence/capture/
        // early-exit signal). Write-back (full-tensor edits) is the separate
        // `residual_delta` path below.
        let mut staging = pooled.clone();
        let action = self.hooks.fire(layer_idx, 0, &mut staging, &self.context);
        if let HookAction::EarlyExit { reason } = action {
            self.early_exit_signal.set(true);
            *self.early_exit_reason.write().unwrap() = Some(reason);
        }

        if self.enable_inspection {
            self.layer_snapshots
                .write()
                .unwrap()
                .push(LayerSnapshot::from_hidden_state(layer_idx, 0, &pooled));
        }
        *self.last_hidden.write().unwrap() = pooled;

        // Write-back channel (wired in Phase 2): a hook may add a d_model delta,
        // broadcast over tokens, to the full residual stream.
        self.apply_residual_delta(layer_idx, hidden)
    }

    /// If any hook contributes a `d_model` residual delta for this layer, add it
    /// (broadcast over the `seq_len` tokens) to the full hidden state and return
    /// the modified tensor; otherwise `None`.
    fn apply_residual_delta(&self, layer_idx: usize, hidden: &CandleTensor) -> Option<CandleTensor> {
        let delta = self.hooks.residual_delta(layer_idx, &self.context)?;
        let d = delta.len();
        let dt = CandleTensor::from_vec(delta, (1, 1, d), hidden.device())
            .and_then(|t| hidden.broadcast_add(&t));
        match dt {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!("RunnerObserver residual write-back failed at layer {layer_idx}: {e}");
                None
            }
        }
    }

    pub fn into_inspection_trace(self) -> Option<InspectionTrace> {
        if !self.enable_inspection {
            return None;
        }
        Some(InspectionTrace {
            request_id: self.context.request_id,
            layers: self.layer_snapshots.into_inner().unwrap(),
            tokens: self.token_snapshots.into_inner().unwrap(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    /// Drain accumulated layer + token snapshots into an
    /// `InspectionTrace` through a shared `&self` borrow — needed
    /// because the observer lives behind an `Arc` after being moved
    /// into the per-layer closure. Returns `None` only when inspection
    /// was never enabled. Safe to call multiple times; subsequent calls
    /// return an empty-or-shorter trace because the inner vectors are
    /// drained on each call.
    pub fn take_inspection_trace(&self) -> Option<InspectionTrace> {
        if !self.enable_inspection {
            return None;
        }
        let layers = std::mem::take(&mut *self.layer_snapshots.write().unwrap());
        let tokens = std::mem::take(&mut *self.token_snapshots.write().unwrap());
        if layers.is_empty() && tokens.is_empty() {
            return None;
        }
        Some(InspectionTrace {
            request_id: self.context.request_id.clone(),
            layers,
            tokens,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    /// Compute a `TokenSnapshot` from the raw logits + the sampled
    /// token. `confidence` is the softmax probability of the chosen
    /// token; `top_alternatives` is the top-K runner-up tokens by
    /// logit (the model's actual considered candidates). Pure CPU,
    /// runs once per decoded token.
    pub fn record_token(
        &self,
        step: usize,
        token_id: u32,
        token_text: String,
        logits: &[f32],
        top_k: usize,
    ) {
        if !self.enable_inspection {
            return;
        }
        // Numerically-stable softmax for confidence + top alternatives.
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 {
            for e in exps.iter_mut() {
                *e /= sum;
            }
        }
        let confidence = exps.get(token_id as usize).copied().unwrap_or(0.0);
        // Top-K including the chosen token, sorted descending by prob.
        let mut indexed: Vec<(u32, f32)> = exps
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u32, *p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);
        self.token_snapshots.write().unwrap().push(TokenSnapshot {
            step,
            token_id,
            token_text,
            confidence,
            top_alternatives: indexed,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::hooks::memory_inject::MemoryInjectHook;
    use crate::engine::memory_store::{MemoryCategory, MemoryStore};
    use candle_core::{Device, Tensor as CandleTensor};

    fn fake_hidden(seq_len: usize, d_model: usize, fill: f32) -> CandleTensor {
        let data: Vec<f32> = (0..seq_len * d_model).map(|_| fill).collect();
        CandleTensor::from_vec(data, (1, seq_len, d_model), &Device::Cpu).unwrap()
    }

    #[test]
    fn observer_routes_capture_through_memory_inject_hook() {
        let memory = Arc::new(RwLock::new(MemoryStore::new(1024, 256)));
        let mut registry = HookRegistry::new();
        registry.register(Box::new(MemoryInjectHook {
            memory: memory.clone(),
            inject_layer: 0,
            capture_layer: 3,
            alpha: 0.0,
            max_memories: 0,
        }));
        let observer = RunnerObserver::new(Arc::new(registry), "test-req");

        for layer in 0..8 {
            observer.on_layer(layer, &fake_hidden(4, 16, 0.5));
        }

        let store = memory.read().unwrap();
        let captured = store.query_by_category(&MemoryCategory::Context);
        assert_eq!(captured.len(), 1, "capture_layer 3 should have written exactly one entry");
        assert_eq!(captured[0].metadata.layer_captured, 3);
    }

    #[test]
    fn observer_applies_steering_writeback_to_full_tensor() {
        use crate::engine::hooks::steering::SteeringHook;
        let mut registry = HookRegistry::new();
        // Add 0.5 * [1,2,3,4] to the full residual stream (broadcast over tokens) at layer 2.
        registry.register(Box::new(SteeringHook { vector: vec![1.0, 2.0, 3.0, 4.0], alpha: 0.5, layer: 2 }));
        let observer = RunnerObserver::new(Arc::new(registry), "test-req");

        // Non-target layer → observe-only, no write-back.
        assert!(observer.on_layer(0, &fake_hidden(3, 4, 1.0)).is_none());

        // Target layer → hidden(all 1.0) + 0.5*[1,2,3,4] = [1.5, 2.0, 2.5, 3.0] per token.
        let out = observer
            .on_layer(2, &fake_hidden(3, 4, 1.0))
            .expect("steering should return a modified tensor");
        assert_eq!(out.dims(), &[1, 3, 4]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = [1.5f32, 2.0, 2.5, 3.0];
        for (i, &x) in v.iter().enumerate() {
            assert!((x - expected[i % 4]).abs() < 1e-5, "elem {i} = {x}");
        }
    }

    #[test]
    fn observer_records_inspection_snapshots_when_enabled() {
        let registry = Arc::new(HookRegistry::new());
        let observer = RunnerObserver::new(registry, "test-req").with_inspection(true);
        for layer in 0..5 {
            observer.on_layer(layer, &fake_hidden(2, 8, 1.0));
        }
        let trace = observer.into_inspection_trace().expect("trace");
        assert_eq!(trace.layers.len(), 5);
    }
}

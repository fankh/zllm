use super::traits::{Hook, HookAction, HookContext};
use crate::backend::traits::Tensor;
use crate::engine::memory_store::MemoryStore;
use std::sync::{Arc, RwLock};

pub struct MemoryInjectHook {
    pub memory: Arc<RwLock<MemoryStore>>,
    pub inject_layer: usize,
    pub capture_layer: usize,
    pub alpha: f32,
    pub max_memories: usize,
}

impl Hook for MemoryInjectHook {
    fn on_layer(
        &self,
        layer_idx: usize,
        _loop_idx: usize,
        hidden_state: &mut Tensor,
        context: &HookContext,
    ) -> HookAction {
        // Capture: save hidden state at capture_layer for future requests.
        // Honor the per-request write quota — hooks must back off once the
        // budget is exhausted to prevent a single inference from flooding
        // the store across many layer firings.
        if layer_idx == self.capture_layer {
            let remaining = context.stores_remaining.get();
            if remaining == 0 {
                crate::metrics::memory_write_quota_refusals().inc();
                return HookAction::Continue;
            }
            context.stores_remaining.set(remaining - 1);

            let snapshot = hidden_state.clone();

            if let Ok(mut store) = self.memory.write() {
                let key = format!("{}:layer{}:auto", context.request_id, layer_idx);
                let metadata = crate::engine::memory_store::MemoryMetadata {
                    source_request_id: context.request_id.clone(),
                    layer_captured: layer_idx,
                    category: crate::engine::memory_store::MemoryCategory::Context,
                    tags: vec![],
                    text_summary: String::new(),
                };
                // Reasoning captures are not pinned and have no TTL —
                // they're plain Context entries managed by score-based
                // eviction under byte pressure.
                let _ = store.store_with_options(
                    key,
                    snapshot,
                    metadata,
                    crate::engine::memory_store::StoreOptions::default(),
                );
            }

            return HookAction::Continue;
        }

        // Inject is a full-residual-stream edit — see `residual_delta` (the
        // observe-path `hidden_state` here is a pooled, non-live copy).
        HookAction::Continue
    }

    fn target_layers(&self) -> Vec<usize> {
        vec![self.capture_layer, self.inject_layer]
    }

    fn name(&self) -> &str {
        "memory_inject"
    }

    /// Inject: at `inject_layer`, retrieve memories relevant to the current
    /// (pooled) activation and return the injection vector to add to the
    /// residual stream. Applied to the full hidden state by `RunnerObserver`
    /// (previously this edited a discarded pooled copy — the v0.9 wake-up).
    fn residual_delta(&self, layer_idx: usize, hidden: &Tensor, _context: &HookContext) -> Option<Vec<f32>> {
        // alpha <= 0 = inject disabled (capture-only). Default config; see
        // `EngineConfig::memory_inject_alpha` for the live A/B that set it.
        if layer_idx != self.inject_layer || self.alpha <= 0.0 {
            return None;
        }
        let store = self.memory.read().ok()?;
        store.build_injection_vector(hidden, self.max_memories, self.alpha)
    }
}

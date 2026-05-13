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
        loop_idx: usize,
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
                    tenant_id: context.tenant_id.clone(),
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

        // Inject: at inject_layer (first time through only — once per request,
        // preserving the v0.1 inline-inject semantic where injection happened
        // before the reasoning loop began), add relevant memories to hidden
        // state. Firing on every loop_idx would let the same memory accumulate
        // too much influence on the hidden state.
        if layer_idx == self.inject_layer && loop_idx == 0 {
            if let Ok(store) = self.memory.read() {
                if let Some(injection) = store.build_injection_vector(
                    hidden_state,
                    &context.tenant_id,
                    self.max_memories,
                    self.alpha,
                ) {
                    // h_t += injection_vector
                    for (h, v) in hidden_state.iter_mut().zip(injection.iter()) {
                        *h += v;
                    }
                    return HookAction::ModifyState;
                }
            }
        }

        HookAction::Continue
    }

    fn target_layers(&self) -> Vec<usize> {
        vec![self.capture_layer, self.inject_layer]
    }

    fn name(&self) -> &str {
        "memory_inject"
    }
}

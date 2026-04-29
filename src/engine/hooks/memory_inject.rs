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
        // Capture: save hidden state at capture_layer for future requests
        if layer_idx == self.capture_layer {
            // Read current hidden state as a memory snapshot
            // (In a real implementation, this would extract a compressed representation)
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
                store.store(key, snapshot, metadata);
            }

            return HookAction::Continue;
        }

        // Inject: at inject_layer, add relevant memories to hidden state
        if layer_idx == self.inject_layer {
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

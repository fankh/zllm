use super::traits::{Hook, HookAction, HookContext};
use crate::backend::traits::Tensor;

pub struct EarlyExitHook {
    pub threshold: f32,
    pub layer: usize,
}

impl Hook for EarlyExitHook {
    fn on_layer(
        &self,
        _layer_idx: usize,
        _loop_idx: usize,
        _hidden_state: &mut Tensor,
        context: &HookContext,
    ) -> HookAction {
        // Stub: check confidence from context
        if context.current_confidence > self.threshold {
            HookAction::EarlyExit {
                reason: format!("confidence {:.3} > threshold {:.3}", context.current_confidence, self.threshold),
            }
        } else {
            HookAction::Continue
        }
    }

    fn target_layers(&self) -> Vec<usize> {
        vec![self.layer]
    }

    fn name(&self) -> &str {
        "early_exit"
    }
}

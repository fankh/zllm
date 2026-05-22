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
        // Read the runner-updated confidence via Cell interior mutability.
        // Real signal source: `ConfidenceHead::estimate` in `engine::confidence`,
        // computed per layer inside the reasoning loop.
        let c = context.current_confidence.get();
        if c > self.threshold {
            HookAction::EarlyExit {
                reason: format!("confidence {:.3} > threshold {:.3}", c, self.threshold),
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

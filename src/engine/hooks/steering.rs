use super::traits::{Hook, HookAction, HookContext};
use crate::backend::traits::Tensor;

pub struct SteeringHook {
    pub vector: Tensor,
    pub alpha: f32,
    pub layer: usize,
}

impl Hook for SteeringHook {
    fn on_layer(
        &self,
        _layer_idx: usize,
        _loop_idx: usize,
        hidden_state: &mut Tensor,
        _context: &HookContext,
    ) -> HookAction {
        // h_t += alpha * concept_vector
        for (h, v) in hidden_state.iter_mut().zip(self.vector.iter()) {
            *h += self.alpha * v;
        }
        HookAction::ModifyState
    }

    fn target_layers(&self) -> Vec<usize> {
        vec![self.layer]
    }

    fn name(&self) -> &str {
        "steering"
    }
}

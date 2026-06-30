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
        _hidden_state: &mut Tensor,
        _context: &HookContext,
    ) -> HookAction {
        // Steering is a full-residual-stream edit, applied via `residual_delta`
        // below (the observe-path `hidden_state` is a pooled, non-live copy).
        // Nothing to do on the observe path.
        HookAction::Continue
    }

    fn target_layers(&self) -> Vec<usize> {
        vec![self.layer]
    }

    fn name(&self) -> &str {
        "steering"
    }

    /// `alpha * concept_vector`, added (broadcast over tokens) to the residual
    /// stream after `self.layer` — the actual steering write-back.
    fn residual_delta(&self, layer_idx: usize, _hidden: &Tensor, _context: &HookContext) -> Option<Vec<f32>> {
        if layer_idx != self.layer {
            return None;
        }
        Some(self.vector.iter().map(|v| self.alpha * v).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::hooks::registry::HookRegistry;
    use crate::engine::hooks::traits::HookContext;

    #[test]
    fn steering_delta_only_fires_on_target_layer() {
        let h = SteeringHook { vector: vec![1.0, 2.0, 3.0, 4.0], alpha: 0.5, layer: 3 };
        let ctx = HookContext::new("t");
        let dummy = vec![0.0f32; 4]; // steering ignores the current activation
        assert_eq!(h.residual_delta(3, &dummy, &ctx), Some(vec![0.5, 1.0, 1.5, 2.0]));
        assert_eq!(h.residual_delta(2, &dummy, &ctx), None);
    }

    #[test]
    fn registry_sums_steering_deltas() {
        let mut reg = HookRegistry::new();
        reg.register(Box::new(SteeringHook { vector: vec![1.0, 1.0], alpha: 1.0, layer: 5 }));
        reg.register(Box::new(SteeringHook { vector: vec![2.0, 3.0], alpha: 2.0, layer: 5 }));
        let ctx = HookContext::new("t");
        let dummy = vec![0.0f32; 2];
        // 1*[1,1] + 2*[2,3] = [5, 7]
        assert_eq!(reg.residual_delta(5, &dummy, &ctx), Some(vec![5.0, 7.0]));
        assert_eq!(reg.residual_delta(4, &dummy, &ctx), None);
    }
}

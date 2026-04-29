use super::traits::{Hook, HookAction, HookContext};
use crate::backend::traits::Tensor;

pub struct HookRegistry {
    hooks: Vec<Box<dyn Hook>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    pub fn register(&mut self, hook: Box<dyn Hook>) {
        tracing::info!("Registered hook: {}", hook.name());
        self.hooks.push(hook);
    }

    pub fn fire(
        &self,
        layer_idx: usize,
        loop_idx: usize,
        hidden_state: &mut Tensor,
        context: &HookContext,
    ) -> HookAction {
        for hook in &self.hooks {
            if hook.target_layers().contains(&layer_idx) {
                let action = hook.on_layer(layer_idx, loop_idx, hidden_state, context);
                match &action {
                    HookAction::EarlyExit { reason } => {
                        tracing::warn!("Early exit at layer {layer_idx}: {reason}");
                        return action;
                    }
                    HookAction::SkipRemaining => return action,
                    _ => {}
                }
            }
        }
        HookAction::Continue
    }

    pub fn clear(&mut self) {
        self.hooks.clear();
    }

    pub fn count(&self) -> usize {
        self.hooks.len()
    }
}

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

    /// Sum the `d_model` residual deltas (write-back) from every hook targeting
    /// `layer_idx`. `None` when no hook contributes one (the common case — keeps
    /// the forward observe-only). Deltas must agree on length (`d_model`); a
    /// mismatch is skipped with a warning rather than corrupting the stream.
    pub fn residual_delta(&self, layer_idx: usize, hidden: &Tensor, context: &HookContext) -> Option<Vec<f32>> {
        let mut acc: Option<Vec<f32>> = None;
        for hook in &self.hooks {
            if !hook.target_layers().contains(&layer_idx) { continue; }
            let Some(delta) = hook.residual_delta(layer_idx, hidden, context) else { continue; };
            match &mut acc {
                None => acc = Some(delta),
                Some(a) if a.len() == delta.len() => {
                    for (x, y) in a.iter_mut().zip(&delta) { *x += *y; }
                }
                Some(a) => tracing::warn!(
                    "hook {} residual_delta len {} != {} at layer {layer_idx}; skipped",
                    hook.name(), delta.len(), a.len()
                ),
            }
        }
        acc
    }

    pub fn clear(&mut self) {
        self.hooks.clear();
    }

    pub fn count(&self) -> usize {
        self.hooks.len()
    }
}

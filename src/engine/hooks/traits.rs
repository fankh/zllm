use crate::backend::traits::Tensor;

#[derive(Debug, Clone)]
pub enum HookAction {
    Continue,
    EarlyExit { reason: String },
    ModifyState,
    SkipRemaining,
}

#[derive(Debug, Clone)]
pub struct HookContext {
    pub tenant_id: String,
    pub request_id: String,
    pub tokens_generated: usize,
    pub current_confidence: f32,
}

pub trait Hook: Send + Sync {
    fn on_layer(
        &self,
        layer_idx: usize,
        loop_idx: usize,
        hidden_state: &mut Tensor,
        context: &HookContext,
    ) -> HookAction;

    fn target_layers(&self) -> Vec<usize>;
    fn name(&self) -> &str;
}

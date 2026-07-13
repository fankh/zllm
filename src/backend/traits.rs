use crate::error::Result;
use std::path::Path;

pub type Tensor = Vec<f32>;

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: String,
}

pub trait Backend: Send + Sync {
    fn load_model(&mut self, path: &Path) -> Result<()>;
    fn unload_model(&mut self) -> Result<()>;

    /// Look up token embeddings for `tokens`, returning a flat
    /// `seq_len * d_model` tensor — the real layer-0 input for a
    /// manually-driven forward pass.
    fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor>;

    /// Number of transformer blocks the loaded model actually has.
    /// Callers driving layers manually (the runner's zone loop) must
    /// clamp their layer ranges to this.
    fn n_layers(&self) -> usize;

    /// Run one transformer block over `hidden_state` (flat
    /// `seq_len * d_model`). Takes `&mut self` because a real layer
    /// forward touches backend state (KV cache, mask cache).
    fn forward_layer(
        &mut self,
        layer_idx: usize,
        hidden_state: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor>;

    fn compute_logits(&self, hidden_state: &Tensor) -> Result<Tensor>;

    fn device_info(&self) -> DeviceInfo;
}

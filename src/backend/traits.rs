use crate::error::Result;
use std::path::Path;

pub type Tensor = Vec<f32>;
pub type BlockId = u64;

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: String,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub supports_fp8: bool,
    pub supports_fp4: bool,
}

#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub method: String,
    pub bits: u8,
}

pub trait Backend: Send + Sync {
    fn load_model(&mut self, path: &Path, config: &QuantConfig) -> Result<()>;
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

    fn read_hidden_state(&self, layer_idx: usize) -> Result<Tensor>;
    fn write_hidden_state(&mut self, layer_idx: usize, state: &Tensor) -> Result<()>;

    fn compute_logits(&self, hidden_state: &Tensor) -> Result<Tensor>;

    fn alloc_kv_block(&mut self, n_tokens: usize) -> Result<BlockId>;
    fn free_kv_block(&mut self, id: BlockId) -> Result<()>;

    fn device_info(&self) -> DeviceInfo;
}

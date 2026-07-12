use crate::backend::traits::{Backend, Tensor};
use crate::error::Result;

pub struct LayerStepper;

impl LayerStepper {
    pub fn step_layer(
        backend: &mut dyn Backend,
        layer_idx: usize,
        hidden_state: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        backend.forward_layer(layer_idx, hidden_state, seq_len)
    }
}

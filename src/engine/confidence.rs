use crate::backend::traits::Tensor;

pub struct ConfidenceHead;

impl ConfidenceHead {
    pub fn should_exit(_hidden_state: &Tensor, _threshold: f32) -> bool {
        // Stub: never exit early
        false
    }
}

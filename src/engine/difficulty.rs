use crate::backend::traits::Tensor;

pub struct DifficultyEstimator;

impl DifficultyEstimator {
    pub fn estimate(_hidden_state: &Tensor) -> usize {
        // Stub: always returns 1 loop
        1
    }

    pub fn map_to_loops(difficulty: usize) -> usize {
        match difficulty {
            0 => 1,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => 16,
        }
    }
}

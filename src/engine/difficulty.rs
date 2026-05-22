use crate::backend::traits::Tensor;
use crate::engine::confidence::ConfidenceHead;

/// Coarse difficulty bucket for a prompt's hidden state.
///
/// "Difficulty" here means: how much further reasoning should the model
/// do before producing an answer? The signal is the inverse of
/// `ConfidenceHead::estimate` — a peaked hidden state (model has settled
/// on a sharp representation) is easy; a diffuse hidden state (model is
/// uncertain) is hard.
///
/// Quantized into 5 buckets (0..=4) that `map_to_loops` translates into
/// reasoning loop counts: 0→1, 1→2, 2→4, 3→8, 4→16. The non-adaptive
/// branch of `runner.rs::generate` uses this when
/// `ReasoningBudget.per_token_adaptive == false`.
pub struct DifficultyEstimator;

impl DifficultyEstimator {
    /// Difficulty level in `0..=4`. Higher = harder = more reasoning.
    pub fn estimate(hidden_state: &Tensor) -> usize {
        if hidden_state.is_empty() {
            return 0;
        }
        let confidence = ConfidenceHead::estimate(hidden_state);
        let difficulty = (1.0 - confidence).clamp(0.0, 1.0);
        // 5 buckets:
        //   [0.0, 0.2) → 0 (trivial — peaked activations)
        //   [0.2, 0.4) → 1
        //   [0.4, 0.6) → 2
        //   [0.6, 0.8) → 3
        //   [0.8, 1.0] → 4 (hardest — fully diffuse)
        ((difficulty * 5.0) as usize).min(4)
    }

    /// Map a difficulty level to a reasoning loop count.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_is_trivial() {
        assert_eq!(DifficultyEstimator::estimate(&vec![]), 0);
    }

    #[test]
    fn one_hot_state_is_trivial() {
        // Maximally peaked → confidence ≈ 1 → difficulty ≈ 0.
        let mut h = vec![0.0f32; 64];
        h[3] = 5.0;
        assert_eq!(DifficultyEstimator::estimate(&h), 0);
    }

    #[test]
    fn uniform_state_is_hardest() {
        // Maximally diffuse → confidence ≈ 0 → difficulty ≈ 1 → bucket 4.
        let h = vec![1.0f32; 128];
        assert_eq!(DifficultyEstimator::estimate(&h), 4);
    }

    #[test]
    fn map_to_loops_table() {
        assert_eq!(DifficultyEstimator::map_to_loops(0), 1);
        assert_eq!(DifficultyEstimator::map_to_loops(1), 2);
        assert_eq!(DifficultyEstimator::map_to_loops(2), 4);
        assert_eq!(DifficultyEstimator::map_to_loops(3), 8);
        assert_eq!(DifficultyEstimator::map_to_loops(4), 16);
        assert_eq!(DifficultyEstimator::map_to_loops(99), 16);
    }

    #[test]
    fn difficulty_is_inverse_of_confidence() {
        // Half zeros + half ones — moderately diffuse. Falls in the
        // middle of the difficulty range, not at either extreme.
        let h = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let d = DifficultyEstimator::estimate(&h);
        assert!(d >= 1 && d <= 3, "expected middle bucket, got {d}");
    }

    #[test]
    fn difficulty_monotonic_in_diffuseness() {
        // More spikes = more diffuse = higher difficulty.
        let mut peaked = vec![0.0f32; 32];
        peaked[0] = 5.0;

        let mut mid = vec![0.0f32; 32];
        for i in 0..8 {
            mid[i] = 1.0;
        }

        let diffuse = vec![1.0f32; 32];

        let d_peaked = DifficultyEstimator::estimate(&peaked);
        let d_mid = DifficultyEstimator::estimate(&mid);
        let d_diffuse = DifficultyEstimator::estimate(&diffuse);

        assert!(
            d_diffuse >= d_mid && d_mid >= d_peaked,
            "expected diffuse>=mid>=peaked, got {d_diffuse} >= {d_mid} >= {d_peaked}"
        );
    }
}

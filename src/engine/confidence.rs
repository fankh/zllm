use crate::backend::traits::Tensor;

/// Hidden-state-derived confidence signal.
///
/// We don't have a learned confidence head and don't have logits available
/// inside the reasoning loop (those come out at Zone 3 after reasoning is
/// done), so we use an inverse-participation-ratio (IPR) heuristic:
/// confident states concentrate energy in a few dimensions; uncertain
/// states spread energy across many. Returns a value in `[0.0, 1.0]`.
///
/// `IPR = (Σ h_i²)² / Σ h_i⁴` —
/// IPR ≈ 1 for a one-hot vector, IPR ≈ N for a uniform vector of length N.
/// We map onto `[0, 1]` as `confidence = 1 − (IPR − 1) / (N − 1)`:
/// - one-hot           → 1.0  (maximally peaked)
/// - uniform           → 0.0  (maximally diffuse)
/// - all zeros / empty → 0.0  (no signal)
///
/// This is a real signal (not a constant or a linear ramp) and is cheap to
/// compute. It's **not** a calibrated probability — don't reach for it
/// when you actually need one.
pub struct ConfidenceHead;

impl ConfidenceHead {
    pub fn estimate(hidden_state: &Tensor) -> f32 {
        let n = hidden_state.len();
        if n < 2 {
            return 0.0;
        }
        let mut sq_sum = 0.0f32;
        let mut quad_sum = 0.0f32;
        for &v in hidden_state {
            let sq = v * v;
            sq_sum += sq;
            quad_sum += sq * sq;
        }
        if quad_sum < 1e-12 || sq_sum < 1e-12 {
            return 0.0;
        }
        let ipr = (sq_sum * sq_sum) / quad_sum;
        let confidence = 1.0 - (ipr - 1.0) / (n as f32 - 1.0);
        confidence.clamp(0.0, 1.0)
    }

    pub fn should_exit(hidden_state: &Tensor, threshold: f32) -> bool {
        Self::estimate(hidden_state) >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_is_zero_confidence() {
        assert_eq!(ConfidenceHead::estimate(&vec![]), 0.0);
    }

    #[test]
    fn all_zero_state_is_zero_confidence() {
        assert_eq!(ConfidenceHead::estimate(&vec![0.0f32; 64]), 0.0);
    }

    #[test]
    fn one_hot_state_is_max_confidence() {
        let mut h = vec![0.0f32; 64];
        h[7] = 5.0;
        let c = ConfidenceHead::estimate(&h);
        assert!((c - 1.0).abs() < 1e-5, "expected 1.0, got {c}");
    }

    #[test]
    fn uniform_state_is_min_confidence() {
        // All dims equal — maximally diffuse → IPR = N → confidence ≈ 0.
        let h = vec![1.0f32; 64];
        let c = ConfidenceHead::estimate(&h);
        assert!(c < 0.05, "expected ~0, got {c}");
    }

    #[test]
    fn signs_dont_change_confidence() {
        // IPR uses h^2 / h^4 so sign doesn't matter.
        let pos = vec![0.1, 0.2, 0.3, 4.0];
        let mixed = vec![-0.1, 0.2, -0.3, 4.0];
        let c_pos = ConfidenceHead::estimate(&pos);
        let c_mixed = ConfidenceHead::estimate(&mixed);
        assert!((c_pos - c_mixed).abs() < 1e-5);
    }

    #[test]
    fn peakedness_monotonic() {
        let diffuse = vec![1.0f32; 8];
        let mid = vec![0.1f32, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0];
        let peaked = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0];
        let c_diffuse = ConfidenceHead::estimate(&diffuse);
        let c_mid = ConfidenceHead::estimate(&mid);
        let c_peaked = ConfidenceHead::estimate(&peaked);
        assert!(
            c_peaked > c_mid && c_mid > c_diffuse,
            "expected peaked>mid>diffuse, got {c_peaked} > {c_mid} > {c_diffuse}"
        );
    }

    #[test]
    fn should_exit_honors_threshold() {
        let mut h = vec![0.0f32; 16];
        h[3] = 5.0;
        assert!(ConfidenceHead::should_exit(&h, 0.5));
        assert!(!ConfidenceHead::should_exit(&h, 1.5));
    }
}

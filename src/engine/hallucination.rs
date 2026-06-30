//! Output-distribution hallucination / uncertainty detection.
//!
//! Honest framing (cf. [`crate::engine::confidence`]): this is an **uncertainty**
//! signal derived from the model's own per-token output distribution — predictive
//! entropy, top-token probability, and the top-1/top-2 margin. High uncertainty
//! correlates with confabulation/hallucination but is **not** a calibrated
//! hallucination oracle: treat the score as a risk flag, not ground truth. It needs
//! no training data and rides logits the sampling path already has on the CPU.
//!
//! A learned hidden-state probe (the stronger white-box signal, via `RunnerObserver`)
//! is the planned follow-up; this `Detector` is built so a probe score can be folded
//! in later without changing the report shape.

/// Per-generated-token uncertainty, computed from that step's logits.
#[derive(Debug, Clone, Copy)]
pub struct TokenUncertainty {
    pub token: u32,
    /// Predictive entropy of `softmax(logits)`, in nats. 0 = certain, `ln(vocab)` = uniform.
    pub entropy: f32,
    /// Probability of the most likely token. Low = the model spread its bet.
    pub max_prob: f32,
    /// `p(top1) − p(top2)`. Low = the model was torn between two options.
    pub margin: f32,
    /// `ln p(chosen)` — log-prob of the token actually emitted.
    pub chosen_logprob: f32,
}

impl TokenUncertainty {
    /// Per-token risk in `[0,1]`: the probability mass the model put on something
    /// *other* than its top choice. Bounded, monotone, interpretable.
    pub fn risk(&self) -> f32 {
        (1.0 - self.max_prob).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DetectorConfig {
    /// A token is flagged "risky" when its entropy exceeds this (nats).
    pub entropy_threshold: f32,
    /// …or when its top-token probability falls below this.
    pub max_prob_threshold: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        // ~2.0 nats ≈ the model effectively choosing among ~7+ tokens; max_prob < 0.4
        // means the top token holds under 40% of the mass. Conservative defaults.
        Self { entropy_threshold: 2.0, max_prob_threshold: 0.4 }
    }
}

/// Response-level summary plus the per-token detail.
#[derive(Debug, Clone)]
pub struct HallucinationReport {
    /// Headline risk in `[0,1]`: mean per-token `risk()` (= mean `1 − max_prob`).
    pub risk_score: f32,
    /// Mean predictive entropy (nats) over the generated tokens.
    pub mean_entropy: f32,
    /// `mean_entropy / ln(vocab)` in `[0,1]` — entropy normalized by the max possible.
    pub normalized_entropy: f32,
    /// Fraction of tokens that tripped a `DetectorConfig` threshold.
    pub risky_fraction: f32,
    /// Index (into `per_token`) of the single most uncertain token, if any.
    pub peak_token: Option<usize>,
    pub n_tokens: usize,
    pub per_token: Vec<TokenUncertainty>,
}

impl HallucinationReport {
    /// Whether the response as a whole should be flagged for review.
    pub fn flagged(&self, risk_threshold: f32) -> bool {
        self.risk_score >= risk_threshold
    }
}

/// Accumulates per-token uncertainty across a generation and produces a report.
pub struct Detector {
    cfg: DetectorConfig,
    vocab: usize,
    tokens: Vec<TokenUncertainty>,
    risky: usize,
}

impl Detector {
    pub fn new(cfg: DetectorConfig) -> Self {
        Self { cfg, vocab: 0, tokens: Vec::new(), risky: 0 }
    }

    /// Feed one decode step's full logits and the token that was emitted.
    pub fn observe(&mut self, logits: &[f32], chosen: u32) {
        let u = token_uncertainty(logits, chosen);
        self.vocab = logits.len();
        if u.entropy > self.cfg.entropy_threshold || u.max_prob < self.cfg.max_prob_threshold {
            self.risky += 1;
        }
        self.tokens.push(u);
    }

    pub fn is_empty(&self) -> bool { self.tokens.is_empty() }

    pub fn report(self) -> HallucinationReport {
        let n = self.tokens.len();
        if n == 0 {
            return HallucinationReport {
                risk_score: 0.0, mean_entropy: 0.0, normalized_entropy: 0.0,
                risky_fraction: 0.0, peak_token: None, n_tokens: 0, per_token: Vec::new(),
            };
        }
        let mean_entropy = self.tokens.iter().map(|t| t.entropy).sum::<f32>() / n as f32;
        let risk_score = self.tokens.iter().map(|t| t.risk()).sum::<f32>() / n as f32;
        let ln_vocab = (self.vocab.max(2) as f32).ln();
        let peak_token = self.tokens.iter().enumerate()
            .max_by(|a, b| a.1.entropy.partial_cmp(&b.1.entropy).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        HallucinationReport {
            risk_score,
            mean_entropy,
            normalized_entropy: (mean_entropy / ln_vocab).clamp(0.0, 1.0),
            risky_fraction: self.risky as f32 / n as f32,
            peak_token,
            n_tokens: n,
            per_token: self.tokens,
        }
    }
}

/// Compute entropy / max_prob / margin / chosen_logprob from one logit vector.
/// Numerically stable (max-subtracted softmax); `p ln p → 0` as `p → 0`.
pub fn token_uncertainty(logits: &[f32], chosen: u32) -> TokenUncertainty {
    if logits.is_empty() {
        return TokenUncertainty { token: chosen, entropy: 0.0, max_prob: 0.0, margin: 0.0, chosen_logprob: f32::NEG_INFINITY };
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in logits { sum += (l - max).exp(); }
    let inv = 1.0 / sum;
    // entropy = -Σ p ln p ; with p = e^{l-max}/sum,  ln p = (l-max) - ln(sum).
    let ln_sum = sum.ln();
    let mut entropy = 0.0f32;
    let (mut top1, mut top2) = (f32::NEG_INFINITY, f32::NEG_INFINITY); // top two probs
    for &l in logits {
        let p = (l - max).exp() * inv;
        if p > 0.0 { entropy -= p * ((l - max) - ln_sum); }
        if p > top1 { top2 = top1; top1 = p; } else if p > top2 { top2 = p; }
    }
    let chosen_p = logits.get(chosen as usize).map(|&l| (l - max).exp() * inv).unwrap_or(0.0);
    TokenUncertainty {
        token: chosen,
        entropy: entropy.max(0.0),
        max_prob: top1.max(0.0),
        margin: (top1 - top2).max(0.0),
        chosen_logprob: if chosen_p > 0.0 { chosen_p.ln() } else { f32::NEG_INFINITY },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() < eps }

    #[test]
    fn one_hot_is_certain() {
        let mut l = vec![0.0f32; 64];
        l[7] = 50.0; // dominant logit
        let u = token_uncertainty(&l, 7);
        assert!(u.entropy < 1e-3, "entropy {}", u.entropy);
        assert!(approx(u.max_prob, 1.0, 1e-3), "max_prob {}", u.max_prob);
        assert!(approx(u.margin, 1.0, 1e-3));
        assert!(u.chosen_logprob > -1e-3);
        assert!(u.risk() < 1e-3);
    }

    #[test]
    fn uniform_is_maximally_uncertain() {
        let v = 64usize;
        let l = vec![0.0f32; v];
        let u = token_uncertainty(&l, 0);
        assert!(approx(u.entropy, (v as f32).ln(), 1e-3), "entropy {} vs ln{v}", u.entropy);
        assert!(approx(u.max_prob, 1.0 / v as f32, 1e-4));
        assert!(u.margin < 1e-4);
        assert!(approx(u.risk(), 1.0 - 1.0 / v as f32, 1e-4));
    }

    #[test]
    fn sharper_distribution_is_less_risky() {
        let diffuse = vec![0.0f32, 0.1, 0.2, 0.0];
        let peaked = vec![0.0f32, 0.1, 6.0, 0.0];
        let ud = token_uncertainty(&diffuse, 2);
        let up = token_uncertainty(&peaked, 2);
        assert!(up.entropy < ud.entropy, "{} !< {}", up.entropy, ud.entropy);
        assert!(up.max_prob > ud.max_prob);
        assert!(up.risk() < ud.risk());
    }

    #[test]
    fn chosen_logprob_tracks_choice() {
        let l = vec![3.0f32, 1.0, 0.0];
        let top = token_uncertainty(&l, 0);   // chose the most likely
        let low = token_uncertainty(&l, 2);   // chose the least likely
        assert!(top.chosen_logprob > low.chosen_logprob);
    }

    #[test]
    fn report_aggregates_and_flags() {
        let mut d = Detector::new(DetectorConfig::default());
        // two confident tokens, one very uncertain (uniform) token
        let mut sharp = vec![0.0f32; 32]; sharp[1] = 40.0;
        d.observe(&sharp, 1);
        d.observe(&sharp, 1);
        d.observe(&vec![0.0f32; 32], 0); // uniform → risky
        let r = d.report();
        assert_eq!(r.n_tokens, 3);
        assert!(approx(r.risky_fraction, 1.0 / 3.0, 1e-5), "risky {}", r.risky_fraction);
        assert_eq!(r.peak_token, Some(2)); // the uniform token is the most uncertain
        assert!(r.risk_score > 0.0 && r.risk_score < 1.0);
        assert!(r.normalized_entropy > 0.0 && r.normalized_entropy <= 1.0);
        // a low bar flags it; an impossibly high bar does not
        assert!(r.flagged(0.1));
        assert!(!r.flagged(0.99));
    }

    #[test]
    fn empty_report_is_zero() {
        let d = Detector::new(DetectorConfig::default());
        assert!(d.is_empty());
        let r = d.report();
        assert_eq!(r.n_tokens, 0);
        assert_eq!(r.risk_score, 0.0);
        assert_eq!(r.peak_token, None);
    }
}

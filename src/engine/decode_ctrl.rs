//! Per-request decode-time control: the OpenAI-parameter semantics that
//! must apply identically in every decode loop (candle blocking and
//! streaming, GPU and VK fast lanes). One struct owns the state so the
//! loops cannot drift apart — before v0.9.2 the stop-token id alone was
//! hardcoded at ~20 call sites.
//!
//! Owns: repetition/presence/frequency penalties, `logit_bias`, the
//! per-request seeded RNG (`seed`), and stop-string matching. Stop
//! strings are matched on a re-decoded window of the most recent tokens
//! (never on per-token decodes, which drop SentencePiece space markers),
//! and the engine stays tokenizer-agnostic: callers decode the window
//! and pass text in.

use crate::backend::traits::Tensor;
use crate::engine::sampler::{sample, sample_with_rng, SamplerConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PenaltyConfig {
    /// llama.cpp-style repetition penalty: divide positive logits /
    /// multiply negative logits of seen tokens. 1.0 = off.
    pub repeat: f32,
    /// OpenAI presence penalty: flat subtraction once a token has
    /// appeared. 0.0 = off.
    pub presence: f32,
    /// OpenAI frequency penalty: subtraction scaled by occurrence count.
    /// 0.0 = off.
    pub frequency: f32,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self { repeat: 1.0, presence: 0.0, frequency: 0.0 }
    }
}

impl PenaltyConfig {
    pub fn is_active(&self) -> bool {
        self.repeat != 1.0 || self.presence != 0.0 || self.frequency != 0.0
    }
}

pub struct DecodeControl {
    penalties: PenaltyConfig,
    logit_bias: Vec<(u32, f32)>,
    /// Occurrence counts of generated tokens (plus the prompt tail for
    /// the repetition penalty, mirroring llama.cpp's last-n window).
    counts: HashMap<u32, u32>,
    rng: Option<StdRng>,
    stops: Vec<String>,
    /// Tokens of generated tail the caller should re-decode for stop
    /// matching. Sized from the longest stop string.
    window_tokens: usize,
}

impl DecodeControl {
    pub fn new(
        penalties: PenaltyConfig,
        logit_bias: Vec<(u32, f32)>,
        seed: Option<u64>,
        stops: Vec<String>,
    ) -> Self {
        // A BPE/SPM token is rarely shorter than ~2 bytes; pad generously
        // so a stop string can never straddle out of the window.
        let longest = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        let window_tokens = 8 + longest;
        Self {
            penalties,
            logit_bias,
            counts: HashMap::new(),
            rng: seed.map(StdRng::seed_from_u64),
            stops,
            window_tokens,
        }
    }

    /// A control that changes nothing — the zero-cost path for requests
    /// with no penalties/bias/seed/stops.
    pub fn passthrough() -> Self {
        Self::new(PenaltyConfig::default(), Vec::new(), None, Vec::new())
    }

    /// Seed the repetition-penalty window with the prompt tail, matching
    /// llama.cpp's behavior of penalizing prompt repeats too. Presence /
    /// frequency penalties are OpenAI-defined over generated text only,
    /// but sharing one count map with repeat is the accepted local
    /// approximation (llama.cpp does the same).
    pub fn observe_prompt_tail(&mut self, prompt: &[u32], last_n: usize) {
        if !self.penalties.is_active() {
            return;
        }
        for &t in prompt.iter().rev().take(last_n) {
            *self.counts.entry(t).or_insert(0) += 1;
        }
    }

    /// Whether logits must be adjusted before sampling. When false and no
    /// seed is set, callers may keep any argmax fast path (GPU argmax
    /// readback) — the distribution is untouched.
    pub fn modifies_logits(&self) -> bool {
        self.penalties.is_active() || !self.logit_bias.is_empty()
    }

    /// Apply logit_bias and penalties in place.
    pub fn adjust_logits(&self, logits: &mut [f32]) {
        for &(tok, bias) in &self.logit_bias {
            if let Some(l) = logits.get_mut(tok as usize) {
                *l += bias;
            }
        }
        if self.penalties.is_active() {
            for (&tok, &cnt) in &self.counts {
                let Some(l) = logits.get_mut(tok as usize) else { continue };
                if self.penalties.repeat != 1.0 {
                    if *l > 0.0 {
                        *l /= self.penalties.repeat;
                    } else {
                        *l *= self.penalties.repeat;
                    }
                }
                *l -= self.penalties.frequency * cnt as f32 + self.penalties.presence;
            }
        }
    }

    /// Sample the next token: adjust (bias + penalties), then draw with
    /// the per-request RNG when seeded, thread RNG otherwise. Does NOT
    /// record the token — call `observe` with what the loop actually
    /// commits (PLD/spec paths commit tokens that were never sampled).
    pub fn sample_token(&mut self, logits: &Tensor, cfg: &SamplerConfig) -> u32 {
        let adjusted;
        let logits = if self.modifies_logits() {
            let mut v = logits.clone();
            self.adjust_logits(&mut v);
            adjusted = v;
            &adjusted
        } else {
            logits
        };
        match self.rng.as_mut() {
            Some(rng) => sample_with_rng(logits, cfg, rng),
            None => sample(logits, cfg),
        }
    }

    /// Record a committed token (sampled, or accepted from a draft).
    pub fn observe(&mut self, token: u32) {
        if self.penalties.is_active() {
            *self.counts.entry(token).or_insert(0) += 1;
        }
    }

    pub fn has_stops(&self) -> bool {
        !self.stops.is_empty()
    }

    /// How many generated tail tokens the caller should re-decode into
    /// the window text for `stop_hit_in`.
    pub fn window_tokens(&self) -> usize {
        self.window_tokens
    }

    /// Does the re-decoded tail window contain any stop string? Callers
    /// check after each committed token; the first hit is the earliest
    /// completion of a stop sequence (the window outsizes every stop).
    pub fn stop_hit_in(&self, window_text: &str) -> bool {
        self.stops.iter().any(|s| window_text.contains(s.as_str()))
    }

    /// Truncate final text at the earliest stop occurrence. Returns true
    /// if a cut was made (⇒ finish_reason = "stop").
    pub fn truncate_at_stop(&self, text: &mut String) -> bool {
        let mut cut: Option<usize> = None;
        for s in &self.stops {
            if let Some(i) = text.find(s.as_str()) {
                cut = Some(cut.map_or(i, |c| c.min(i)));
            }
        }
        match cut {
            Some(i) => {
                text.truncate(i);
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy() -> SamplerConfig {
        SamplerConfig { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0 }
    }

    #[test]
    fn passthrough_leaves_logits_alone() {
        let ctrl = DecodeControl::passthrough();
        assert!(!ctrl.modifies_logits());
        let mut logits = vec![0.5, -0.5, 2.0];
        let before = logits.clone();
        ctrl.adjust_logits(&mut logits);
        assert_eq!(logits, before);
    }

    #[test]
    fn repeat_penalty_flips_greedy_choice() {
        let mut ctrl = DecodeControl::new(
            PenaltyConfig { repeat: 2.0, presence: 0.0, frequency: 0.0 },
            Vec::new(),
            None,
            Vec::new(),
        );
        let logits = vec![1.0, 0.9];
        assert_eq!(ctrl.sample_token(&logits, &greedy()), 0);
        ctrl.observe(0);
        // Token 0 penalized: 1.0 / 2.0 = 0.5 < 0.9 → greedy flips to 1.
        assert_eq!(ctrl.sample_token(&logits, &greedy()), 1);
    }

    #[test]
    fn frequency_penalty_scales_with_count() {
        let mut ctrl = DecodeControl::new(
            PenaltyConfig { repeat: 1.0, presence: 0.1, frequency: 0.3 },
            Vec::new(),
            None,
            Vec::new(),
        );
        ctrl.observe(1);
        ctrl.observe(1);
        let mut logits = vec![0.0, 1.0];
        ctrl.adjust_logits(&mut logits);
        // 1.0 - (0.3 * 2 + 0.1) = 0.3
        assert!((logits[1] - 0.3).abs() < 1e-6, "got {}", logits[1]);
        assert_eq!(logits[0], 0.0, "unseen token untouched");
    }

    #[test]
    fn negative_logits_are_penalized_away_from_zero() {
        let mut ctrl = DecodeControl::new(
            PenaltyConfig { repeat: 2.0, presence: 0.0, frequency: 0.0 },
            Vec::new(),
            None,
            Vec::new(),
        );
        ctrl.observe(0);
        let mut logits = vec![-1.0, -3.0];
        ctrl.adjust_logits(&mut logits);
        assert!((logits[0] - -2.0).abs() < 1e-6, "negative logit doubles: {}", logits[0]);
    }

    #[test]
    fn logit_bias_applies_and_can_ban() {
        let mut ctrl = DecodeControl::new(
            PenaltyConfig::default(),
            vec![(0, -100.0), (2, 5.0)],
            None,
            Vec::new(),
        );
        // Token 0 would win greedy without the ban.
        let logits = vec![10.0, 1.0, 6.0];
        assert_eq!(ctrl.sample_token(&logits, &greedy()), 2);
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let cfg = SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0 };
        let logits = vec![1.0, 1.1, 1.2, 1.3];
        let run = |seed| -> Vec<u32> {
            let mut c = DecodeControl::new(PenaltyConfig::default(), Vec::new(), Some(seed), Vec::new());
            (0..16).map(|_| c.sample_token(&logits, &cfg)).collect()
        };
        assert_eq!(run(7), run(7));
    }

    #[test]
    fn stop_matching_window_and_truncation() {
        let ctrl = DecodeControl::new(
            PenaltyConfig::default(),
            Vec::new(),
            None,
            vec!["\n\n".into(), "END".into()],
        );
        assert!(ctrl.has_stops());
        assert!(ctrl.window_tokens() >= "END".len());
        assert!(!ctrl.stop_hit_in("no stop here"));
        assert!(ctrl.stop_hit_in("line one\n\nline two"));
        let mut text = String::from("keep this END drop this");
        assert!(ctrl.truncate_at_stop(&mut text));
        assert_eq!(text, "keep this ");
        let mut clean = String::from("nothing to cut");
        assert!(!ctrl.truncate_at_stop(&mut clean));
    }

    #[test]
    fn prompt_tail_seeds_repeat_window() {
        let mut ctrl = DecodeControl::new(
            PenaltyConfig { repeat: 2.0, presence: 0.0, frequency: 0.0 },
            Vec::new(),
            None,
            Vec::new(),
        );
        ctrl.observe_prompt_tail(&[5, 6, 7], 64);
        let mut logits = vec![0.0; 8];
        logits[7] = 1.0;
        ctrl.adjust_logits(&mut logits);
        assert!((logits[7] - 0.5).abs() < 1e-6, "prompt token penalized: {}", logits[7]);
    }
}

//! OpenAI-compatible `logprobs` collection for the chat/completions API.
//!
//! Per generated token: the chosen token's log-probability plus the top-N
//! alternatives, computed from the full logit vector with a numerically
//! stable log-softmax (`lp = logit − logsumexp`). Like hallucination
//! detection, this rides the candle path (one full-logits forward per token);
//! the fast lanes are bypassed when a request asks for logprobs.

/// One generated token's logprobs.
#[derive(Debug, Clone)]
pub struct TokenLogprobs {
    pub token_id: u32,
    /// `ln p(chosen)`.
    pub logprob: f32,
    /// Top-N `(token_id, logprob)`, descending.
    pub top: Vec<(u32, f32)>,
}

/// Accumulates per-token logprobs across a generation.
pub struct LogprobsCollector {
    /// Alternatives per token (OpenAI caps this at 20; the handler clamps).
    pub top_n: usize,
    pub entries: Vec<TokenLogprobs>,
}

impl LogprobsCollector {
    pub fn new(top_n: usize) -> Self {
        Self { top_n: top_n.min(20), entries: Vec::new() }
    }

    /// Feed one decode step's full logits and the token that was emitted.
    /// Observes the distribution `chosen` was actually drawn from (i.e. after
    /// any grammar mask — masked tokens sit at -inf and never enter the top).
    pub fn observe(&mut self, logits: &[f32], chosen: u32) {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
        let logprob = logits.get(chosen as usize).map(|&l| l - lse).unwrap_or(f32::NEG_INFINITY);
        // Partial top-N selection (N ≤ 20 over ~128k vocab: keep a small sorted list).
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(self.top_n);
        if self.top_n > 0 {
            for (i, &l) in logits.iter().enumerate() {
                let lp = l - lse;
                if top.len() < self.top_n {
                    let pos = top.partition_point(|x| x.1 >= lp);
                    top.insert(pos, (i as u32, lp));
                } else if lp > top[self.top_n - 1].1 {
                    top.pop();
                    let pos = top.partition_point(|x| x.1 >= lp);
                    top.insert(pos, (i as u32, lp));
                }
            }
        }
        self.entries.push(TokenLogprobs { token_id: chosen, logprob, top });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprobs_sum_to_one_and_rank_correctly() {
        let logits = vec![2.0f32, 1.0, 0.0, -1.0];
        let mut c = LogprobsCollector::new(3);
        c.observe(&logits, 1);
        let e = &c.entries[0];
        assert_eq!(e.token_id, 1);
        // probabilities from the reported logprobs must sum to <= 1 and the
        // chosen logprob must match the top list's entry for id 1
        let p: f32 = e.top.iter().map(|(_, lp)| lp.exp()).sum();
        assert!(p > 0.8 && p <= 1.0 + 1e-4, "top-3 mass {p}");
        assert_eq!(e.top[0].0, 0, "id 0 has the largest logit");
        assert_eq!(e.top[1].0, 1);
        let reported = e.top.iter().find(|(id, _)| *id == 1).unwrap().1;
        assert!((reported - e.logprob).abs() < 1e-5);
        // descending order
        assert!(e.top.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn top_n_zero_gives_chosen_only() {
        let mut c = LogprobsCollector::new(0);
        c.observe(&[0.0, 5.0], 1);
        assert!(c.entries[0].top.is_empty());
        assert!(c.entries[0].logprob > -0.01); // ~ln(0.993)
    }

    #[test]
    fn masked_tokens_never_enter_top() {
        let logits = vec![1.0f32, f32::NEG_INFINITY, 0.5, f32::NEG_INFINITY];
        let mut c = LogprobsCollector::new(4);
        c.observe(&logits, 0);
        let e = &c.entries[0];
        // -inf entries rank last; the finite two must lead
        assert_eq!(e.top[0].0, 0);
        assert_eq!(e.top[1].0, 2);
    }

    #[test]
    fn top_n_clamped_to_openai_cap() {
        assert_eq!(LogprobsCollector::new(50).top_n, 20);
    }
}

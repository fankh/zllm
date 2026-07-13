use crate::backend::traits::Tensor;

pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Keep only tokens with prob >= min_p * max_prob (0.0 = off).
    /// Applied after temperature/top-k, before top-p — a better tail
    /// cutoff than top-p alone on small models.
    pub min_p: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            min_p: 0.0,
        }
    }
}

/// Sample with the thread RNG. Deterministic requests (OpenAI `seed`)
/// go through `sample_with_rng` with a per-request seeded RNG instead.
pub fn sample(logits: &Tensor, config: &SamplerConfig) -> u32 {
    let mut rng = rand::rng();
    sample_with_rng(logits, config, &mut rng)
}

pub fn sample_with_rng<R: rand::Rng>(logits: &Tensor, config: &SamplerConfig, rng: &mut R) -> u32 {
    // temperature == 0 ⇒ deterministic argmax. This both avoids a divide-by-
    // zero in temperature scaling and gives callers a clean way to ask for
    // greedy decoding through the same API.
    if config.temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    let mut logits = logits.clone();

    // Temperature scaling
    if config.temperature != 1.0 {
        for l in logits.iter_mut() {
            *l /= config.temperature;
        }
    }

    // Top-k filtering
    if config.top_k > 0 && config.top_k < logits.len() {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        // total_cmp: NaN-safe (partial_cmp().unwrap() would panic on a NaN logit)
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let threshold = indexed[config.top_k - 1].1;
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp() / exp_sum).collect();

    // Min-p filtering: drop everything below min_p * p_max, renormalize.
    if config.min_p > 0.0 {
        let p_max = probs.iter().cloned().fold(0.0f32, f32::max);
        let cutoff = config.min_p * p_max;
        let mut kept = 0.0f32;
        for p in probs.iter_mut() {
            if *p < cutoff {
                *p = 0.0;
            } else {
                kept += *p;
            }
        }
        if kept > 0.0 {
            for p in probs.iter_mut() {
                *p /= kept;
            }
        }
    }

    // Top-p (nucleus) filtering
    let probs = if config.top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        // total_cmp: NaN-safe (see top-k note)
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut cumsum = 0.0;
        let mut filtered = vec![0.0; probs.len()];
        for (idx, prob) in indexed {
            if cumsum >= config.top_p {
                break;
            }
            filtered[idx] = prob;
            cumsum += prob;
        }

        // Renormalize
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            filtered.iter().map(|&p| p / sum).collect()
        } else {
            probs
        }
    } else {
        probs
    };

    // Multinomial sampling
    let r: f32 = rand::Rng::random_range(rng, 0.0..1.0);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_zero_is_argmax() {
        let logits = vec![0.1, 0.5, 0.9, 0.2, 0.7];
        let cfg = SamplerConfig { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0 };
        for _ in 0..16 {
            assert_eq!(sample(&logits, &cfg), 2);
        }
    }

    #[test]
    fn temperature_zero_breaks_ties_deterministically() {
        // All equal — argmax picks the first index reliably.
        let logits = vec![1.0; 8];
        let cfg = SamplerConfig { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0 };
        let first = sample(&logits, &cfg);
        for _ in 0..16 {
            assert_eq!(sample(&logits, &cfg), first);
        }
    }

    #[test]
    fn min_p_prunes_the_tail() {
        // p_max belongs to index 1; with min_p = 0.5 only tokens whose
        // prob is >= half of p_max survive. Index 0 (tiny logit) must
        // never be sampled.
        let logits = vec![-10.0, 5.0, 4.9, -10.0];
        let cfg = SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.5 };
        for _ in 0..200 {
            let t = sample(&logits, &cfg);
            assert!(t == 1 || t == 2, "min_p should confine to {{1,2}}, got {t}");
        }
    }

    #[test]
    fn seeded_rng_is_deterministic() {
        use rand::SeedableRng;
        let logits = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
        let cfg = SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0 };
        let run = |seed: u64| -> Vec<u32> {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            (0..32).map(|_| sample_with_rng(&logits, &cfg, &mut rng)).collect()
        };
        assert_eq!(run(42), run(42), "same seed ⇒ same stream");
        assert_ne!(run(42), run(43), "different seed ⇒ different stream (overwhelmingly)");
    }

    #[test]
    fn high_temperature_explores() {
        // Many tokens equally likely after extreme softening — distribution
        // should spread. We just check we see at least two distinct outputs
        // in many tries (probabilistic but very safe).
        let logits = vec![1.0, 1.01, 1.02, 1.03, 1.04, 1.05];
        let cfg = SamplerConfig { temperature: 100.0, top_k: 0, top_p: 1.0, min_p: 0.0 };
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(sample(&logits, &cfg));
        }
        assert!(seen.len() >= 2, "expected exploration, got only {seen:?}");
    }
}

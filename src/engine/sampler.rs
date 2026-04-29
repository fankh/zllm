use crate::backend::traits::Tensor;

pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
        }
    }
}

pub fn sample(logits: &Tensor, config: &SamplerConfig) -> u32 {
    let mut logits = logits.clone();

    // Temperature scaling
    if config.temperature != 1.0 && config.temperature > 0.0 {
        for l in logits.iter_mut() {
            *l /= config.temperature;
        }
    }

    // Top-k filtering
    if config.top_k > 0 && config.top_k < logits.len() {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
    let probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp() / exp_sum).collect();

    // Top-p (nucleus) filtering
    let probs = if config.top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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
    let mut rng = rand::rng();
    let r: f32 = rand::Rng::random_range(&mut rng, 0.0..1.0);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}

use crate::backend::traits::Tensor;

#[derive(Debug, Clone)]
pub struct ReasoningBudget {
    pub max_loops: usize,
    pub max_memory_mb: usize,
    pub confidence_threshold: f32,
    pub per_token_adaptive: bool,
}

#[derive(Debug, Clone)]
pub struct ReasoningState {
    pub loops_used: usize,
    pub memory_used_mb: usize,
    pub current_confidence: f32,
    pub token_importances: Vec<f32>,
}

impl ReasoningBudget {
    pub fn should_continue(&self, state: &ReasoningState) -> bool {
        if state.loops_used >= self.max_loops {
            return false;
        }
        if state.memory_used_mb >= self.max_memory_mb {
            return false;
        }
        if state.current_confidence >= self.confidence_threshold {
            return false;
        }
        true
    }

    pub fn from_tier(tier: &str) -> Self {
        match tier {
            "free" => Self {
                max_loops: 2,
                max_memory_mb: 64,
                confidence_threshold: 0.8,
                per_token_adaptive: false,
            },
            "standard" => Self {
                max_loops: 8,
                max_memory_mb: 256,
                confidence_threshold: 0.9,
                per_token_adaptive: true,
            },
            "premium" => Self {
                max_loops: 16,
                max_memory_mb: 512,
                confidence_threshold: 0.95,
                per_token_adaptive: true,
            },
            _ => Self {
                max_loops: 4,
                max_memory_mb: 128,
                confidence_threshold: 0.9,
                per_token_adaptive: false,
            },
        }
    }

    pub fn estimate_memory_per_loop(seq_len: usize, d_model: usize, reasoning_layers: usize) -> usize {
        // bytes = reasoning_layers × seq_len × d_model × 2 (fp16)
        let bytes = reasoning_layers * seq_len * d_model * 2;
        bytes / (1024 * 1024) // convert to MB
    }
}

impl ReasoningState {
    pub fn new(seq_len: usize) -> Self {
        Self {
            loops_used: 0,
            memory_used_mb: 0,
            current_confidence: 0.0,
            token_importances: vec![0.5; seq_len],
        }
    }

    pub fn record_loop(&mut self, memory_mb: usize, confidence: f32) {
        self.loops_used += 1;
        self.memory_used_mb += memory_mb;
        self.current_confidence = confidence;
    }
}

pub struct TokenImportanceScorer;

impl TokenImportanceScorer {
    pub fn score(hidden_state: &Tensor, seq_len: usize) -> Vec<f32> {
        if seq_len == 0 || hidden_state.is_empty() {
            return vec![];
        }

        let d_model = hidden_state.len() / seq_len;
        if d_model == 0 {
            return vec![0.5; seq_len];
        }

        let mut scores = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let start = t * d_model;
            let end = (start + d_model).min(hidden_state.len());
            let token_hidden = &hidden_state[start..end];

            // Importance = L2 norm of hidden state (higher activation = more important)
            let norm: f32 = token_hidden.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Normalize to 0-1 range (rough heuristic)
            let score = (norm / (d_model as f32).sqrt()).min(1.0);
            scores.push(score);
        }

        // Boost first tokens (anchors) and last tokens (recent context)
        if scores.len() > 4 {
            for s in scores.iter_mut().take(2) {
                *s = (*s + 1.0).min(1.0);
            }
            let len = scores.len();
            for s in scores.iter_mut().skip(len - 2) {
                *s = (*s + 0.5).min(1.0);
            }
        }

        scores
    }

    pub fn tokens_needing_deep_reasoning(scores: &[f32], threshold: f32) -> Vec<usize> {
        scores
            .iter()
            .enumerate()
            .filter(|(_, s)| **s >= threshold)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn average_importance(scores: &[f32]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        scores.iter().sum::<f32>() / scores.len() as f32
    }
}

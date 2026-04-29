use crate::backend::traits::{Backend, Tensor};
use crate::engine::hooks::registry::HookRegistry;
use crate::engine::hooks::traits::{HookAction, HookContext};
use crate::engine::reasoning_budget::{ReasoningBudget, ReasoningState, TokenImportanceScorer};
use crate::engine::sampler::{SamplerConfig, sample};

pub struct InferenceRunner {
    backend: Box<dyn Backend>,
    hook_registry: HookRegistry,
    d_model: usize,
    reasoning_layers: usize,
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub reasoning_loops_used: usize,
    pub reasoning_memory_mb: usize,
    pub early_exit: bool,
    pub avg_token_importance: f32,
}

impl InferenceRunner {
    pub fn new(backend: Box<dyn Backend>, d_model: usize, reasoning_layers: usize) -> Self {
        Self {
            backend,
            hook_registry: HookRegistry::new(),
            d_model,
            reasoning_layers,
        }
    }

    pub fn hooks_mut(&mut self) -> &mut HookRegistry {
        &mut self.hook_registry
    }

    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        config: &SamplerConfig,
        budget: &ReasoningBudget,
    ) -> GenerationResult {
        let seq_len = prompt_tokens.len();
        let mut state = ReasoningState::new(seq_len);
        let memory_per_loop = ReasoningBudget::estimate_memory_per_loop(
            seq_len,
            self.d_model,
            self.reasoning_layers,
        );

        // Zone 1: Encode (always runs once)
        let mut hidden = vec![0.1f32; seq_len * self.d_model];
        for layer_idx in 0..8 {
            hidden = self.backend.forward_layer(layer_idx, &hidden, seq_len).unwrap();
        }

        // Score token importance
        let importances = TokenImportanceScorer::score(&hidden, seq_len);
        state.token_importances = importances.clone();
        let avg_importance = TokenImportanceScorer::average_importance(&importances);

        // Determine loop count from importance
        let n_loops_needed = if budget.per_token_adaptive {
            let high_importance_ratio = importances.iter().filter(|&&s| s >= 0.7).count() as f32
                / importances.len().max(1) as f32;
            // More high-importance tokens = more loops
            let adaptive = (high_importance_ratio * budget.max_loops as f32).ceil() as usize;
            adaptive.max(1).min(budget.max_loops)
        } else {
            budget.max_loops
        };

        // Zone 2: Reasoning loops (budgeted)
        let mut early_exit = false;
        let hook_context = HookContext {
            tenant_id: "default".into(),
            request_id: "req".into(),
            tokens_generated: 0,
            current_confidence: 0.0,
        };

        for loop_idx in 0..n_loops_needed {
            if !budget.should_continue(&state) {
                tracing::info!(
                    "Reasoning budget exhausted: loops={}, memory={}MB, confidence={:.3}",
                    state.loops_used, state.memory_used_mb, state.current_confidence
                );
                break;
            }

            // Forward through reasoning block
            for layer_idx in 8..8 + self.reasoning_layers {
                hidden = self.backend.forward_layer(layer_idx, &hidden, seq_len).unwrap();

                // Fire hooks at each layer
                let action = self.hook_registry.fire(layer_idx, loop_idx, &mut hidden, &hook_context);
                match action {
                    HookAction::EarlyExit { reason } => {
                        tracing::warn!("Early exit in reasoning loop {loop_idx}, layer {layer_idx}: {reason}");
                        early_exit = true;
                        break;
                    }
                    _ => {}
                }
            }

            if early_exit {
                break;
            }

            // Estimate confidence (stub: increases with each loop)
            let confidence = (loop_idx as f32 + 1.0) / n_loops_needed as f32;
            state.record_loop(memory_per_loop, confidence);
        }

        // Zone 3: Output layers (always runs once)
        for layer_idx in 8 + self.reasoning_layers..32 {
            hidden = self.backend.forward_layer(layer_idx, &hidden, seq_len).unwrap();
        }

        // Decode: generate output tokens
        let mut output_tokens = Vec::new();
        for _ in 0..max_tokens {
            let logits = self.backend.compute_logits(&hidden).unwrap();
            let token_id = sample(&logits, config);
            output_tokens.push(token_id);
            if token_id == 2 {
                break;
            }
        }

        GenerationResult {
            tokens: output_tokens,
            reasoning_loops_used: state.loops_used,
            reasoning_memory_mb: state.memory_used_mb,
            early_exit,
            avg_token_importance: avg_importance,
        }
    }
}

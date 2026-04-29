use crate::backend::traits::{Backend, Tensor};
use crate::engine::hooks::registry::HookRegistry;
use crate::engine::hooks::traits::{HookAction, HookContext};
use crate::engine::memory_store::{InspectionTrace, LayerSnapshot, MemoryStore};
use crate::engine::reasoning_budget::{ReasoningBudget, ReasoningState, TokenImportanceScorer};
use crate::engine::sampler::{SamplerConfig, sample};
use std::sync::{Arc, RwLock};

pub struct InferenceRunner {
    backend: Box<dyn Backend>,
    hook_registry: HookRegistry,
    memory: Arc<RwLock<MemoryStore>>,
    d_model: usize,
    reasoning_layers: usize,
    enable_inspection: bool,
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub reasoning_loops_used: usize,
    pub reasoning_memory_mb: usize,
    pub early_exit: bool,
    pub avg_token_importance: f32,
    pub inspection_trace: Option<InspectionTrace>,
    pub memories_injected: usize,
    pub memories_captured: usize,
}

impl InferenceRunner {
    pub fn new(backend: Box<dyn Backend>, d_model: usize, reasoning_layers: usize) -> Self {
        Self {
            backend,
            hook_registry: HookRegistry::new(),
            memory: Arc::new(RwLock::new(MemoryStore::new(1024, 256))),
            d_model,
            reasoning_layers,
            enable_inspection: false,
        }
    }

    pub fn with_memory(mut self, memory: Arc<RwLock<MemoryStore>>) -> Self {
        self.memory = memory;
        self
    }

    pub fn with_inspection(mut self, enabled: bool) -> Self {
        self.enable_inspection = enabled;
        self
    }

    pub fn hooks_mut(&mut self) -> &mut HookRegistry {
        &mut self.hook_registry
    }

    pub fn memory(&self) -> &Arc<RwLock<MemoryStore>> {
        &self.memory
    }

    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        config: &SamplerConfig,
        budget: &ReasoningBudget,
        request_id: &str,
        tenant_id: &str,
    ) -> GenerationResult {
        let seq_len = prompt_tokens.len();
        let mut state = ReasoningState::new(seq_len);
        let memory_per_loop = ReasoningBudget::estimate_memory_per_loop(
            seq_len,
            self.d_model,
            self.reasoning_layers,
        );

        let mut layer_snapshots: Vec<LayerSnapshot> = Vec::new();
        let mut memories_injected = 0usize;
        let mut memories_captured = 0usize;

        // Zone 1: Encode (always runs once)
        let mut hidden = vec![0.1f32; seq_len * self.d_model];
        for layer_idx in 0..8 {
            hidden = self.backend.forward_layer(layer_idx, &hidden, seq_len).unwrap();

            if self.enable_inspection {
                layer_snapshots.push(LayerSnapshot::from_hidden_state(layer_idx, 0, &hidden));
            }
        }

        // Inject: retrieve relevant memories and add to hidden state after encoding
        if let Ok(store) = self.memory.read() {
            if let Some(injection) = store.build_injection_vector(
                &hidden,
                tenant_id,
                5,    // max 5 memories
                0.3,  // alpha = 0.3
            ) {
                for (h, v) in hidden.iter_mut().zip(injection.iter()) {
                    *h += v;
                }
                memories_injected += 1;
                tracing::debug!("Injected memory context for tenant {tenant_id}");
            }
        }

        // Score token importance
        let importances = TokenImportanceScorer::score(&hidden, seq_len);
        state.token_importances = importances.clone();
        let avg_importance = TokenImportanceScorer::average_importance(&importances);

        // Determine loop count from importance
        let n_loops_needed = if budget.per_token_adaptive {
            let high_importance_ratio = importances.iter().filter(|&&s| s >= 0.7).count() as f32
                / importances.len().max(1) as f32;
            let adaptive = (high_importance_ratio * budget.max_loops as f32).ceil() as usize;
            adaptive.max(1).min(budget.max_loops)
        } else {
            budget.max_loops
        };

        // Zone 2: Reasoning loops (budgeted)
        let mut early_exit = false;
        let hook_context = HookContext {
            tenant_id: tenant_id.to_string(),
            request_id: request_id.to_string(),
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

            for layer_idx in 8..8 + self.reasoning_layers {
                hidden = self.backend.forward_layer(layer_idx, &hidden, seq_len).unwrap();

                // Fire hooks (including MemoryInjectHook if registered)
                let action = self.hook_registry.fire(layer_idx, loop_idx, &mut hidden, &hook_context);
                match action {
                    HookAction::EarlyExit { reason } => {
                        tracing::warn!("Early exit in reasoning loop {loop_idx}, layer {layer_idx}: {reason}");
                        early_exit = true;
                        break;
                    }
                    _ => {}
                }

                if self.enable_inspection {
                    layer_snapshots.push(LayerSnapshot::from_hidden_state(layer_idx, loop_idx, &hidden));
                }
            }

            if early_exit {
                break;
            }

            let confidence = (loop_idx as f32 + 1.0) / n_loops_needed as f32;
            state.record_loop(memory_per_loop, confidence);
        }

        // Capture: store final reasoning state as memory for future requests
        if let Ok(mut store) = self.memory.write() {
            let capture_key = format!("{request_id}:reasoning_final");
            let metadata = crate::engine::memory_store::MemoryMetadata {
                source_request_id: request_id.to_string(),
                tenant_id: tenant_id.to_string(),
                layer_captured: 8 + self.reasoning_layers - 1,
                category: crate::engine::memory_store::MemoryCategory::Context,
                tags: vec![],
                text_summary: format!("Reasoning state after {} loops", state.loops_used),
            };
            store.store(capture_key, hidden.clone(), metadata);
            memories_captured += 1;
        }

        // Zone 3: Output layers (always runs once)
        for layer_idx in 8 + self.reasoning_layers..32 {
            hidden = self.backend.forward_layer(layer_idx, &hidden, seq_len).unwrap();

            if self.enable_inspection {
                layer_snapshots.push(LayerSnapshot::from_hidden_state(layer_idx, 0, &hidden));
            }
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

        // Record inspection trace
        let inspection_trace = if self.enable_inspection {
            let trace = InspectionTrace {
                request_id: request_id.to_string(),
                layers: layer_snapshots,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };
            if let Ok(mut store) = self.memory.write() {
                store.record_trace(trace.clone());
            }
            Some(trace)
        } else {
            None
        };

        GenerationResult {
            tokens: output_tokens,
            reasoning_loops_used: state.loops_used,
            reasoning_memory_mb: state.memory_used_mb,
            early_exit,
            avg_token_importance: avg_importance,
            inspection_trace,
            memories_injected,
            memories_captured,
        }
    }
}

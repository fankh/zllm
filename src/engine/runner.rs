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
        let memory = Arc::new(RwLock::new(MemoryStore::new(1024, 256)));
        let mut hook_registry = HookRegistry::new();
        // Register the default MemoryInjectHook so cross-request memory
        // works out of the box. v0.2 collapsed the previous inline
        // inject/capture in this file into this single chokepoint —
        // configure inject at the end of Zone 1 and capture at the end
        // of Zone 2.
        let inject_layer = 8usize.saturating_sub(1);
        let capture_layer = 8 + reasoning_layers - 1;
        hook_registry.register(Box::new(
            crate::engine::hooks::memory_inject::MemoryInjectHook {
                memory: memory.clone(),
                inject_layer,
                capture_layer,
                alpha: 0.3,
                max_memories: 5,
            },
        ));
        Self {
            backend,
            hook_registry,
            memory,
            d_model,
            reasoning_layers,
            enable_inspection: false,
        }
    }

    pub fn with_memory(mut self, memory: Arc<RwLock<MemoryStore>>) -> Self {
        self.memory = memory.clone();
        // The default `MemoryInjectHook` registered in `new()` points at the
        // old (now-orphaned) internal store. Re-register it against the
        // supplied memory so captures actually land where callers expect.
        // This also clears any other hooks the caller may have registered
        // beforehand — callers wanting custom hooks should call
        // `with_memory()` first, then `hooks_mut().register(...)` for their
        // additions.
        self.hook_registry.clear();
        let inject_layer = 8usize.saturating_sub(1);
        let capture_layer = 8 + self.reasoning_layers - 1;
        self.hook_registry.register(Box::new(
            crate::engine::hooks::memory_inject::MemoryInjectHook {
                memory,
                inject_layer,
                capture_layer,
                alpha: 0.3,
                max_memories: 5,
            },
        ));
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
    ) -> GenerationResult {
        let seq_len = prompt_tokens.len();
        let mut state = ReasoningState::new(seq_len);
        let memory_per_loop = ReasoningBudget::estimate_memory_per_loop(
            seq_len,
            self.d_model,
            self.reasoning_layers,
        );

        let mut layer_snapshots: Vec<LayerSnapshot> = Vec::new();
        let memories_injected = 0usize;
        let memories_captured = 0usize;
        // Note: in v0.2 the inline inject/capture at the Zone 1/Zone 2 boundary
        // was removed. Memory inject + capture is now driven by the
        // `MemoryInjectHook` registered by default in `InferenceRunner::new()`
        // — see the hook for the inject/capture layer indices. `memories_*`
        // counters stay at zero in this code path; if you need accurate hook
        // counters they should move into HookContext (atomic counters).

        // Zone 1: Encode (always runs once)
        let mut hidden = vec![0.1f32; seq_len * self.d_model];
        for layer_idx in 0..8 {
            hidden = self.backend.forward_layer(layer_idx, &hidden, seq_len).unwrap();

            if self.enable_inspection {
                layer_snapshots.push(LayerSnapshot::from_hidden_state(layer_idx, 0, &hidden));
            }
        }

        // Score token importance
        let importances = TokenImportanceScorer::score(&hidden, seq_len);
        state.token_importances = importances.clone();
        let avg_importance = TokenImportanceScorer::average_importance(&importances);

        // Determine loop count.
        // - `per_token_adaptive` mode: weight by the share of high-
        //   importance tokens (existing logic).
        // - Otherwise: use DifficultyEstimator::estimate over the
        //   current hidden state, mapped through map_to_loops, capped
        //   at the budget. v0.1 just returned budget.max_loops here.
        let n_loops_needed = if budget.per_token_adaptive {
            let high_importance_ratio = importances.iter().filter(|&&s| s >= 0.7).count() as f32
                / importances.len().max(1) as f32;
            let adaptive = (high_importance_ratio * budget.max_loops as f32).ceil() as usize;
            adaptive.max(1).min(budget.max_loops)
        } else {
            let level = crate::engine::difficulty::DifficultyEstimator::estimate(&hidden);
            crate::engine::difficulty::DifficultyEstimator::map_to_loops(level)
                .min(budget.max_loops)
                .max(1)
        };

        // Zone 2: Reasoning loops (budgeted)
        let mut early_exit = false;
        let hook_context = HookContext::new(request_id);

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

                // Update HookContext confidence from the live hidden state
                // so hooks like EarlyExitHook see a real signal. Source:
                // engine::confidence::ConfidenceHead::estimate (IPR-based).
                hook_context
                    .current_confidence
                    .set(crate::engine::confidence::ConfidenceHead::estimate(&hidden));

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

            // Confidence at end of this reasoning loop, from the final
            // hidden state. Replaces the v0.1 placeholder linear ramp
            // `(loop_idx + 1) / n_loops_needed`.
            let confidence = crate::engine::confidence::ConfidenceHead::estimate(&hidden);
            state.record_loop(memory_per_loop, confidence);
        }

        // Capture is now handled by `MemoryInjectHook` firing at
        // `capture_layer` inside the reasoning loop. The inline
        // capture-at-end-of-Zone-2 path was removed in v0.2 to give us a
        // single chokepoint that honors HookContext.stores_remaining.

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

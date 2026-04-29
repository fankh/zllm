use crate::backend::traits::{Backend, Tensor};
use crate::engine::hooks::registry::HookRegistry;
use crate::engine::hooks::traits::HookContext;
use crate::engine::sampler::{SamplerConfig, sample};

pub struct InferenceRunner {
    backend: Box<dyn Backend>,
    hook_registry: HookRegistry,
}

impl InferenceRunner {
    pub fn new(backend: Box<dyn Backend>) -> Self {
        Self {
            backend,
            hook_registry: HookRegistry::new(),
        }
    }

    pub fn hooks_mut(&mut self) -> &mut HookRegistry {
        &mut self.hook_registry
    }

    pub fn generate(
        &self,
        _prompt_tokens: &[u32],
        max_tokens: usize,
        config: &SamplerConfig,
    ) -> Vec<u32> {
        // Stub: generate tokens using dummy backend
        let mut output_tokens = Vec::new();
        let hidden = vec![0.0f32; 4096]; // dummy hidden state

        for _ in 0..max_tokens {
            let logits = self.backend.compute_logits(&hidden).unwrap();
            let token_id = sample(&logits, config);
            output_tokens.push(token_id);

            // EOS check (token 2 = EOS for many tokenizers)
            if token_id == 2 {
                break;
            }
        }

        output_tokens
    }
}

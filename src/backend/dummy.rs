use crate::backend::traits::*;
use crate::error::Result;
use rand::Rng;
use std::path::Path;

pub struct DummyBackend {
    vocab_size: usize,
    d_model: usize,
    n_layers: usize,
}

impl DummyBackend {
    pub fn new(vocab_size: usize, d_model: usize, n_layers: usize) -> Self {
        Self {
            vocab_size,
            d_model,
            n_layers,
        }
    }
}

impl Backend for DummyBackend {
    fn load_model(&mut self, _path: &Path) -> Result<()> {
        tracing::info!("DummyBackend: model loaded (no-op)");
        Ok(())
    }

    fn unload_model(&mut self) -> Result<()> {
        Ok(())
    }

    fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        // Deterministic pseudo-embedding: unique per (token, dim, position),
        // bounded, and reproducible across runs — good enough for the
        // engine tests this backend exists for.
        let mut out = Vec::with_capacity(tokens.len() * self.d_model);
        for (pos, &tok) in tokens.iter().enumerate() {
            for dim in 0..self.d_model {
                let phase = tok as f32 * 0.618_034
                    + dim as f32 * 0.070_71
                    + pos as f32 * 0.001;
                out.push(phase.sin() * 0.02);
            }
        }
        Ok(out)
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn forward_layer(
        &mut self,
        _layer_idx: usize,
        hidden_state: &Tensor,
        _seq_len: usize,
    ) -> Result<Tensor> {
        // Return input with small random perturbation (simulates layer transform)
        let mut rng = rand::rng();
        let output: Tensor = hidden_state
            .iter()
            .map(|&x| x + rng.random_range(-0.01..0.01))
            .collect();
        Ok(output)
    }

    fn compute_logits(&self, _hidden_state: &Tensor) -> Result<Tensor> {
        let mut rng = rand::rng();
        let logits: Tensor = (0..self.vocab_size)
            .map(|_| rng.random_range(-5.0..5.0))
            .collect();
        Ok(logits)
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: "Dummy CPU".to_string(),
            backend: "dummy".to_string(),
        }
    }
}

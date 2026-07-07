use crate::backend::traits::*;
use crate::error::Result;
use rand::Rng;
use std::path::Path;

pub struct DummyBackend {
    vocab_size: usize,
    #[allow(dead_code)] d_model: usize, // part of the Backend construction contract
    n_layers: usize,
    next_block_id: BlockId,
    hidden_states: Vec<Tensor>,
}

impl DummyBackend {
    pub fn new(vocab_size: usize, d_model: usize, n_layers: usize) -> Self {
        Self {
            vocab_size,
            d_model,
            n_layers,
            next_block_id: 0,
            hidden_states: vec![vec![0.0; d_model]; n_layers],
        }
    }
}

impl Backend for DummyBackend {
    fn load_model(&mut self, _path: &Path, _config: &QuantConfig) -> Result<()> {
        tracing::info!("DummyBackend: model loaded (no-op)");
        Ok(())
    }

    fn unload_model(&mut self) -> Result<()> {
        Ok(())
    }

    fn forward_layer(
        &self,
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

    fn read_hidden_state(&self, layer_idx: usize) -> Result<Tensor> {
        Ok(self.hidden_states[layer_idx % self.n_layers].clone())
    }

    fn write_hidden_state(&mut self, layer_idx: usize, state: &Tensor) -> Result<()> {
        self.hidden_states[layer_idx % self.n_layers] = state.clone();
        Ok(())
    }

    fn compute_logits(&self, _hidden_state: &Tensor) -> Result<Tensor> {
        let mut rng = rand::rng();
        let logits: Tensor = (0..self.vocab_size)
            .map(|_| rng.random_range(-5.0..5.0))
            .collect();
        Ok(logits)
    }

    fn alloc_kv_block(&mut self, _n_tokens: usize) -> Result<BlockId> {
        self.next_block_id += 1;
        Ok(self.next_block_id)
    }

    fn free_kv_block(&mut self, _id: BlockId) -> Result<()> {
        Ok(())
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: "Dummy CPU".to_string(),
            backend: "dummy".to_string(),
            memory_total_mb: 1024,
            memory_used_mb: 0,
            supports_fp8: false,
            supports_fp4: false,
        }
    }
}

use crate::backend::traits::Tensor;

pub struct LogitFSM {
    grammar: String,
}

impl LogitFSM {
    pub fn new(grammar: &str) -> Self {
        Self {
            grammar: grammar.to_string(),
        }
    }

    pub fn apply_mask(&self, _logits: &mut Tensor) {
        // Stub: no masking applied
    }

    pub fn advance(&mut self, _token_id: u32) {
        // Stub: no state transition
    }

    pub fn grammar(&self) -> &str {
        &self.grammar
    }
}

use crate::backend::traits::BlockId;

pub const BLOCK_SIZE: usize = 16; // tokens per block

#[derive(Debug, Clone)]
pub struct KvBlock {
    pub id: BlockId,
    pub ref_count: u32,
    pub tokens_used: usize,
}

impl KvBlock {
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            ref_count: 1,
            tokens_used: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.tokens_used >= BLOCK_SIZE
    }

    pub fn remaining(&self) -> usize {
        BLOCK_SIZE - self.tokens_used
    }
}

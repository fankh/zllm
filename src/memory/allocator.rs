use crate::backend::traits::BlockId;
use crate::error::{Result, ZllmError};
use crate::memory::block::KvBlock;
use std::collections::HashMap;

pub struct PagedAllocator {
    blocks: HashMap<BlockId, KvBlock>,
    free_ids: Vec<BlockId>,
    next_id: BlockId,
    max_blocks: usize,
}

impl PagedAllocator {
    pub fn new(max_blocks: usize) -> Self {
        Self {
            blocks: HashMap::new(),
            free_ids: Vec::new(),
            next_id: 0,
            max_blocks,
        }
    }

    pub fn alloc(&mut self) -> Result<BlockId> {
        if let Some(id) = self.free_ids.pop() {
            let block = KvBlock::new(id);
            self.blocks.insert(id, block);
            Ok(id)
        } else if self.blocks.len() < self.max_blocks {
            self.next_id += 1;
            let id = self.next_id;
            let block = KvBlock::new(id);
            self.blocks.insert(id, block);
            Ok(id)
        } else {
            Err(ZllmError::Memory("no free blocks".into()))
        }
    }

    pub fn free(&mut self, id: BlockId) -> Result<()> {
        if self.blocks.remove(&id).is_some() {
            self.free_ids.push(id);
            Ok(())
        } else {
            Err(ZllmError::Memory(format!("block {id} not found")))
        }
    }

    pub fn used_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn free_count(&self) -> usize {
        self.max_blocks - self.blocks.len()
    }

    pub fn fragmentation_ratio(&self) -> f32 {
        if self.blocks.is_empty() {
            return 0.0;
        }
        let total_capacity: usize = self.blocks.len() * crate::memory::block::BLOCK_SIZE;
        let total_used: usize = self.blocks.values().map(|b| b.tokens_used).sum();
        if total_capacity == 0 {
            0.0
        } else {
            1.0 - (total_used as f32 / total_capacity as f32)
        }
    }
}

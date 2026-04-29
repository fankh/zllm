use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct InferRequest {
    pub request_id: String,
    pub tenant_id: String,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
}

pub struct BatchScheduler {
    waiting: VecDeque<InferRequest>,
    max_batch_size: usize,
}

impl BatchScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            max_batch_size,
        }
    }

    pub fn add_request(&mut self, req: InferRequest) {
        self.waiting.push_back(req);
    }

    pub fn schedule_step(&mut self) -> Vec<InferRequest> {
        let n = self.waiting.len().min(self.max_batch_size);
        self.waiting.drain(..n).collect()
    }

    pub fn pending_count(&self) -> usize {
        self.waiting.len()
    }
}

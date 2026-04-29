#[derive(Debug)]
pub enum GatewayResult {
    Pass,
    Reject { reason: String },
}

pub struct SemanticGateway;

impl SemanticGateway {
    pub fn new() -> Self {
        Self
    }

    pub fn classify(&self, _prompt: &str) -> GatewayResult {
        // Stub: always pass
        GatewayResult::Pass
    }
}

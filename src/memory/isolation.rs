use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct TenantSession {
    pub tenant_id: String,
    pub session_token: String,
    pub memory_limit_mb: u64,
    pub memory_used_mb: u64,
}

pub struct TenantMemoryPool {
    sessions: HashMap<String, TenantSession>,
}

impl TenantMemoryPool {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    pub fn create_session(&mut self, tenant_id: &str, memory_limit_mb: u64) -> String {
        let token = Uuid::new_v4().to_string();
        let session = TenantSession {
            tenant_id: tenant_id.to_string(),
            session_token: token.clone(),
            memory_limit_mb,
            memory_used_mb: 0,
        };
        self.sessions.insert(token.clone(), session);
        token
    }

    pub fn validate_access(&self, token: &str) -> bool {
        self.sessions.contains_key(token)
    }

    pub fn destroy_session(&mut self, token: &str) {
        self.sessions.remove(token);
    }

    pub fn tenant_count(&self) -> usize {
        self.sessions.len()
    }
}

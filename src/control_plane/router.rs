#[derive(Debug, Clone)]
pub struct TenantContext {
    pub tenant_id: String,
    pub hooks: Vec<String>,
    pub grammar: Option<String>,
}

pub struct RequestRouter;

impl RequestRouter {
    pub fn new() -> Self {
        Self
    }

    pub fn route(&self, tenant_id: &str) -> TenantContext {
        // Stub: return default context for any tenant
        TenantContext {
            tenant_id: tenant_id.to_string(),
            hooks: vec![],
            grammar: None,
        }
    }
}

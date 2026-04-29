use thiserror::Error;

#[derive(Error, Debug)]
pub enum ZllmError {
    #[error("config error: {0}")]
    Config(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("memory error: {0}")]
    Memory(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("hook error: {0}")]
    Hook(String),

    #[error("tenant not found: {0}")]
    TenantNotFound(String),

    #[error("early exit: {0}")]
    EarlyExit(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ZllmError>;

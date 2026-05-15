use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct ZllmConfig {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub engine: EngineConfig,
    pub memory: MemoryConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub rest_port: u16,
    pub max_concurrent: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub path: String,
    pub quantization: String,
    pub max_seq_len: usize,
    /// Optional path to tokenizer.json. If empty / absent, the server
    /// looks for `tokenizer.json` next to `path`.
    #[serde(default)]
    pub tokenizer_path: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EngineConfig {
    pub encoder_layers: usize,
    pub reasoning_layers: usize,
    pub max_loops: usize,
    pub confidence_threshold: f32,
    pub default_temperature: f32,
    pub default_top_k: usize,
    pub default_top_p: f32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MemoryConfig {
    pub block_size: usize,
    pub max_blocks: usize,
}

impl ZllmConfig {
    pub fn load(path: &Path) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::error::ZllmError::Config(format!("failed to read config: {e}")))?;
        let config: ZllmConfig = toml::from_str(&content)
            .map_err(|e| crate::error::ZllmError::Config(format!("invalid config: {e}")))?;
        Ok(config)
    }
}

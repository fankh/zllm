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
    pub max_seq_len: usize,
    /// Optional path to tokenizer.json. If empty / absent, the server
    /// looks for `tokenizer.json` next to `path`.
    #[serde(default)]
    pub tokenizer_path: String,
    /// Optional directory the server scans for additional `.gguf` files
    /// that the chat UI's model-picker dropdown will offer. Each file
    /// must have a sibling `tokenizer.json`. Empty / absent disables
    /// the picker.
    #[serde(default)]
    pub dir: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EngineConfig {
    /// Width of the encoder zone (layers `0..encoder_layers`) in the
    /// 3-zone reasoning program. Its last layer is where the
    /// MemoryInjectHook injects retrieved memories; capture happens at
    /// `encoder_layers + reasoning_layers - 1`. The canonical split for
    /// a 32-layer model is 8 / 8+N / 32.
    pub encoder_layers: usize,
    pub reasoning_layers: usize,
    pub max_loops: usize,
    pub confidence_threshold: f32,
    pub default_temperature: f32,
    pub default_top_k: usize,
    pub default_top_p: f32,
    /// How many backend slots to spin up. Each slot loads its own copy
    /// of the model weights (~N × model_size RAM) but lets that many
    /// chat requests run in parallel without contending on a single
    /// write lock. Default 2 — set to 1 for memory-constrained boxes,
    /// 4+ if you have RAM to spare and want more concurrency headroom.
    #[serde(default)]
    pub backend_pool_size: Option<usize>,
    /// Path to a smaller "draft" GGUF used by speculative decoding.
    /// Must share the main model's tokenizer (e.g. main = Llama 3.2
    /// 3B, draft = Llama 3.2 1B). Loaded once per pool slot
    /// alongside the main backend. Empty/absent disables spec-decode.
    #[serde(default)]
    pub draft_model_path: Option<String>,
    /// Strength of the MemoryInjectHook write-back (`h += alpha * retrieved`).
    /// **Default 0.0 = inject OFF** (capture still runs). Live A/B on the 1B
    /// showed alpha=0.3 injecting pooled prefill states from unrelated requests
    /// derails generation ("Paris" → quiz-format gibberish on the identical
    /// prompt once the store had entries). Opt in deliberately, with a curated
    /// store and a small alpha.
    #[serde(default)]
    pub memory_inject_alpha: f32,
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

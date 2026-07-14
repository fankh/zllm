use axum::{
    Json, Router,
    extract::{Path, State},
    http::{StatusCode, header},
    response::{IntoResponse, Sse, sse::Event},
    routing::{get, patch, post},
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::backend::candle::backend::CandleCpuBackend;
use crate::backend::candle::tokenizer::LlamaTokenizer;
use crate::backend::traits::Backend;
use crate::config::EngineConfig;
use crate::control_plane::goal_manager::{GoalManager, TaskStatus};
use crate::engine::decode_ctrl::{DecodeControl, PenaltyConfig};
use crate::engine::hooks::registry::HookRegistry;
use crate::engine::logit_fsm::LogitFSM;
use crate::engine::memory_store::{MemoryCategory, MemoryMetadata, MemoryStore};
use crate::engine::runner_observer::RunnerObserver;
use crate::engine::sampler::SamplerConfig;

const CHAT_UI_HTML: &str = include_str!("chat_ui.html");

/// Optional bearer-token auth: when ZLLM_API_KEY is set, every route
/// except /health requires `Authorization: Bearer <key>`. Read once —
/// the key is process-lifetime configuration, not hot-reloadable.
async fn require_api_key(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    static KEY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let key = KEY.get_or_init(|| std::env::var("ZLLM_API_KEY").ok().filter(|k| !k.is_empty()));
    if let Some(key) = key {
        if req.uri().path() != "/health" {
            let ok = req
                .headers()
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(|k| k == key)
                .unwrap_or(false);
            if !ok {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({"error": "missing or invalid API key"})),
                )
                    .into_response();
            }
        }
    }
    next.run(req).await
}

/// RAII in-flight counter: decrements on drop so streams and early
/// returns can't leak a slot in the saturation accounting.
struct ActiveGuard(Arc<AtomicUsize>);
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// 503 when the server is already running `max_concurrent` generations
/// — bounded queueing beats every caller blocking on slot mutexes.
fn admit(s: &AppState) -> Result<ActiveGuard, axum::response::Response> {
    let now = s.active_requests.fetch_add(1, Ordering::Relaxed) + 1;
    let guard = ActiveGuard(s.active_requests.clone());
    if now > s.max_concurrent {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": format!("server saturated ({now} in flight, limit {})", s.max_concurrent)})),
        )
            .into_response());
    }
    Ok(guard)
}

/// One slot in the backend pool: a fully-loaded backend plus the
/// tokens currently materialized in its KV cache. The pool size is
/// controlled by `cfg.engine.backend_pool_size` (default 2). Each
/// chat request acquires one slot exclusively for the duration of
/// generation; the prefix cache is per-slot so reuse only happens
/// when a follow-up request lands on the same slot.
pub struct BackendSlot {
    pub backend: CandleCpuBackend,
    pub prompt_cache: Vec<u32>,
    /// Optional draft model for speculative decoding. Same tokenizer
    /// as `backend`; loaded once at startup. `None` means no
    /// spec-decode available for this slot.
    pub draft: Option<CandleCpuBackend>,
    /// KV state of `draft` — synced to `prompt_cache` whenever the
    /// chat handler runs the speculative path. Stays empty when the
    /// draft is unused.
    pub draft_prompt_cache: Vec<u32>,
}

#[derive(Clone)]
pub struct AppState {
    pub pool: Arc<Vec<Mutex<BackendSlot>>>,
    /// Tokenizer is `RwLock`-wrapped so model swap can replace it with
    /// the new model's tokenizer (sibling `tokenizer.json` to the GGUF).
    pub tokenizer: Arc<RwLock<LlamaTokenizer>>,
    pub goals: Arc<GoalManager>,
    pub memory: Arc<RwLock<MemoryStore>>,
    pub engine: Arc<EngineConfig>,
    /// Directory scanned for selectable `.gguf` files. `None` disables
    /// the picker; `/v1/models` returns only the current model.
    pub models_dir: Option<std::path::PathBuf>,
    /// Filename (no extension) of the currently-loaded model, e.g.
    /// `Llama-3.2-1B-Instruct-Q4_K_M`. Falls back to `"zllm"` if no
    /// model is loaded.
    pub current_model: Arc<RwLock<String>>,
    /// In-memory cache of `(path, mtime_secs) -> architecture` strings
    /// from `general.architecture` in each GGUF header. The probe
    /// opens the file and reads metadata — fast (50-300 ms even on
    /// 19 GB files) but enough to make `/v1/models` perceptibly slow
    /// when there are several large files. Cached for the process
    /// lifetime; invalidated by mtime change.
    pub arch_cache: Arc<RwLock<std::collections::HashMap<std::path::PathBuf, (u64, String)>>>,
    /// Lazily-built vocab→surface-bytes table for grammar-constrained decoding
    /// (`regex:` mode). One decode per vocab entry (~100ms for 128k) on first
    /// grammar request; invalidated on model swap (tokenizer changes).
    pub token_table: Arc<RwLock<Option<Arc<crate::engine::logit_fsm::TokenByteTable>>>>,
    /// Chat-relevant GGUF metadata (embedded chat template + DECLARED
    /// stop ids + BOS), read at load and refreshed on model swap. The
    /// template renders via minijinja; the ChatFamily vocab heuristics
    /// are the fallback for template-less GGUFs. Declared stop ids are
    /// unioned with the vocab probe in `stop_set`.
    pub chat_meta: Arc<RwLock<crate::server::chat_template::GgufChatMeta>>,
    /// Effective context window of the loaded model (min of its declared
    /// context_length and the configured cap); 0 = no model. Requests
    /// whose prompt exceeds it get an OpenAI-style 400 instead of a
    /// forward-pass failure.
    pub model_ctx: Arc<AtomicUsize>,
    /// In-flight generation count vs `server.max_concurrent`: excess
    /// requests get 503 instead of queueing unboundedly on slot locks.
    pub active_requests: Arc<AtomicUsize>,
    pub max_concurrent: usize,
    /// Hook registry consulted on every chat prefill via `RunnerObserver`.
    /// Built once at startup with the default `MemoryInjectHook` plus
    /// anything callers add before serving — see `main.rs`.
    pub hooks: Arc<HookRegistry>,
    /// Runtime flag controlling whether the inspection pipeline runs
    /// (per-layer mean-pool, hook firing, per-token softmax for
    /// confidence, inspection trace recording). When `false` the chat
    /// path skips all observer work and runs as a thin forward + sample
    /// loop — the "fast lane" you want for raw throughput benchmarks.
    /// Toggled at runtime via `GET/POST /v1/inspect/enabled`.
    pub inspection_enabled: Arc<AtomicBool>,
    /// Whether prompt-lookup decoding is active for greedy
    /// (temperature=0) chat requests. Trades wasted compute on
    /// rejected drafts for batched speedup on accepted ones — big win
    /// on summarize/quote workloads, neutral on open-ended chat.
    /// Toggled via `GET/POST /v1/pld/enabled`.
    pub pld_enabled: Arc<AtomicBool>,
    /// Whether classic speculative decoding (small draft model
    /// proposing tokens, main model verifying) is active for greedy
    /// requests. Requires the slot's draft model to be loaded.
    /// Toggled via `GET/POST /v1/spec_decode/enabled`.
    pub spec_decode_enabled: Arc<AtomicBool>,
    /// Whether confidence-driven early-exit is active for greedy
    /// decode. When on, per-token forwards check confidence (IPR) at
    /// the layer index stored in `early_exit_min_layer`; if it's
    /// above the threshold in `early_exit_threshold_bits`, the
    /// remaining layers are skipped (final norm + LM head still run).
    /// Skipped when inspection / PLD / spec-decode are active or
    /// temperature != 0. Toggled via `/v1/early_exit/enabled`.
    pub early_exit_enabled: Arc<AtomicBool>,
    /// Minimum layer at which early exit is allowed to fire (inclusive).
    /// Default 12 of 16 (75% of depth). Caller adjusts via
    /// `/v1/early_exit/config`.
    pub early_exit_min_layer: Arc<AtomicUsize>,
    /// Confidence threshold (IPR) above which early exit fires.
    /// Stored as bits-of-f32 in an AtomicU32 (Rust doesn't have
    /// AtomicF32). Default 0.30 — calibrated against typical
    /// Llama 3.2 1B per-layer IPR distributions.
    pub early_exit_threshold_bits: Arc<AtomicU32>,
    /// Optional resident iGPU inference engine (cargo feature `gpu`,
    /// enabled at startup via `ZLLM_GPU=1`). When `Some` and a request is
    /// on the "fast lane" (inspection off + no spec-decode / PLD / early-exit
    /// / grammar) with a prompt of 1..=512 tokens, the whole generation runs
    /// on the iGPU — batched prefill (fills the resident KV cache) + decode —
    /// bypassing the candle pool. Serialized through the Mutex because the
    /// GpuModel has a single resident KV cache. Reloaded on model swap.
    #[cfg(feature = "gpu")]
    pub gpu: Arc<Mutex<Option<crate::backend::gpu::GpuModel>>>,
    /// Optional GPU continuous-batching server (cargo feature `gpu`, enabled at
    /// startup via `ZLLM_CB=1`). Owns its own GpuModel on a dedicated thread and
    /// decodes all in-flight `/v1/cb/completions` requests together (vLLM-style
    /// in-flight batching) — high aggregate throughput under concurrency.
    /// Greedy (argmax) decode only; does not hot-swap with the model selector.
    #[cfg(feature = "gpu")]
    pub cb: Option<Arc<crate::backend::gpu::GpuBatchServer>>,
    /// Optional resident raw-Vulkan (ash) decode engine (cargo feature
    /// `vulkan`, enabled via `ZLLM_VK=1`). Same fast-lane contract as `gpu`
    /// but uses the VkModel (validated bit-exact vs candle). Prompts ≤ 128
    /// (sequential prefill). Serialized through the Mutex (single KV cache).
    #[cfg(feature = "vulkan")]
    pub vk: Arc<Mutex<Option<crate::backend::vulkan::VkModel>>>,
}

/// Round-robin tie-breaker when every pool slot is busy. Atomic so
/// concurrent acquire calls don't all park on the same slot.
static POOL_FALLBACK_RR: AtomicUsize = AtomicUsize::new(0);

/// Try every slot's `try_lock` in order; if all are busy, block on a
/// round-robin-selected slot for fairness. Single-slot pools degrade
/// to the previous "always lock slot 0" behavior with zero overhead.
pub fn acquire_slot<'a>(pool: &'a [Mutex<BackendSlot>]) -> std::sync::MutexGuard<'a, BackendSlot> {
    for slot in pool {
        if let Ok(g) = slot.try_lock() {
            return g;
        }
    }
    let i = POOL_FALLBACK_RR.fetch_add(1, Ordering::Relaxed) % pool.len();
    pool[i].lock().unwrap_or_else(|e| e.into_inner())
}

pub fn router(state: AppState) -> Router {
    Router::new()
        // UI + ops
        .route("/", get(chat_ui))
        .route("/health", get(health))
        .route("/v1/info", get(info))
        .route("/metrics", get(metrics))
        // OpenAI-compatible
        .route("/v1/models", get(list_models))
        .route("/v1/models/select", post(select_model))
        .route("/v1/models/download", post(download_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(text_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/cb/completions", post(cb_completions))
        // Goal CRUD
        .route("/v1/goal/state", get(get_state))
        .route("/v1/goal/set", post(set_goal))
        .route("/v1/goal/list", get(list_goals))
        .route("/v1/goal/current", post(set_current_goal))
        .route("/v1/goal/task", post(add_task))
        .route("/v1/goal/tasks", get(list_tasks))
        .route("/v1/goal/task/{id}", patch(update_task))
        .route("/v1/goal/status", post(set_status))
        // Inspection
        .route("/v1/inspect", get(list_traces))
        .route("/v1/inspect/enabled", get(get_inspect_enabled).post(set_inspect_enabled))
        .route("/v1/pld/enabled", get(get_pld_enabled).post(set_pld_enabled))
        .route("/v1/spec_decode/enabled", get(get_spec_decode_enabled).post(set_spec_decode_enabled))
        .route("/v1/early_exit/enabled", get(get_early_exit_enabled).post(set_early_exit_enabled))
        .route("/v1/early_exit/config", get(get_early_exit_config).post(set_early_exit_config))
        .route("/v1/settings", get(get_settings))
        // Profiling (only available when built with --features profile).
        .route("/v1/debug/pprof/flamegraph", get(pprof_flamegraph))
        .route("/v1/debug/layer_agreement", post(layer_agreement))
        .route("/v1/debug/matmul_bench", get(matmul_bench))
        .route("/v1/inspect/{request_id}", get(get_trace))
        .layer(axum::middleware::from_fn(require_api_key))
        // Outermost: any unexpected panic in a handler (or an inner layer)
        // is caught and turned into a clean 500 so one bad request can't
        // drop the connection abruptly or wedge the worker. Explicit
        // panics are already gone from production paths; this backstops
        // the rest (indexing, arithmetic, third-party code).
        .layer(tower_http::catch_panic::CatchPanicLayer::custom(handle_panic))
        .with_state(state)
}

/// Turn a caught handler panic into a 500 JSON error and log it, instead
/// of the default abrupt connection drop.
fn handle_panic(err: Box<dyn std::any::Any + Send + 'static>) -> axum::response::Response {
    let msg = err
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| err.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string());
    tracing::error!("caught handler panic: {msg}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": "internal server error"})),
    )
        .into_response()
}

// --- UI + ops ---

/// Serve the embedded chat UI HTML.
///
/// Built as an explicit response (rather than `axum::response::Html`) to be
/// defensive against browsers that download the page instead of rendering it:
/// - `Content-Disposition: inline` — explicitly tells the browser to render,
///   not download. Fixes the bug where Chrome was saving the page as a
///   UUID-named file with no extension.
/// - `X-Content-Type-Options: nosniff` — prevents MIME sniffing from
///   second-guessing the declared `text/html`.
/// - `Cache-Control: no-cache` — ensures chat UI updates take effect on
///   reload during development; for an installed app the binary IS the cache
///   buster, but explicit no-cache avoids stale-UI confusion.
async fn chat_ui() -> impl IntoResponse {
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CONTENT_DISPOSITION, "inline"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        CHAT_UI_HTML,
    )
}

async fn health() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "engine": "zllm",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// A UUID minted once at process start. The chat UI uses it to gate the
/// inspect pill: only show it for messages produced in the current
/// server session, since `MemoryStore` (and the traces in it) live in
/// process memory and are wiped on restart.
static SERVER_SESSION_ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
fn server_session_id() -> &'static str {
    SERVER_SESSION_ID.get_or_init(|| Uuid::new_v4().to_string())
}

async fn info() -> Json<Value> {
    Json(json!({
        "name": "ZLLM",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "White-box LLM inference engine — installed-app, REST + chat UI",
        "session_id": server_session_id(),
        "features": [
            "openai_compat_chat",
            "goal_manager",
            "memory_store",
            "hallucination_detection",
            "early_exit",
            "hook_writeback",
            "logit_fsm_ban_regex"
        ]
    }))
}

async fn metrics() -> impl IntoResponse {
    use prometheus::{Encoder, TextEncoder};
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    if encoder.encode(&metric_families, &mut buffer).is_err() {
        return (StatusCode::INTERNAL_SERVER_ERROR, "encode error").into_response();
    }
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, encoder.format_type())],
        buffer,
    )
        .into_response()
}

// --- OpenAI compat ---

/// Probe of a GGUF file's architecture metadata, with mtime-keyed
/// caching in AppState. The uncached probe reads only the GGUF header
/// (tensor data is not touched) but still hits the disk; the cache
/// makes subsequent `/v1/models` requests effectively free.
fn gguf_architecture(
    cache: &Arc<RwLock<std::collections::HashMap<std::path::PathBuf, (u64, String)>>>,
    path: &std::path::Path,
) -> Option<String> {
    let mtime = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Fast path: cache hit on the same mtime.
    if let Ok(c) = cache.read() {
        if let Some((cached_mtime, arch)) = c.get(path) {
            if *cached_mtime == mtime {
                return Some(arch.clone());
            }
        }
    }
    // Cold path: open + parse + store.
    let mut f = std::fs::File::open(path).ok()?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut f).ok()?;
    let v = content.metadata.get("general.architecture")?;
    let arch = v.to_string().ok().map(|s| s.to_string())?;
    if let Ok(mut c) = cache.write() {
        c.insert(path.to_path_buf(), (mtime, arch.clone()));
    }
    Some(arch)
}

/// Our forked quantized_llama only handles GGUFs that declare
/// `general.architecture = "llama"`. Other arches (qwen2, mistral,
/// gemma, phi, …) need their own forks.
fn arch_is_supported(arch: &str) -> bool {
    // Supported ⇔ the arch registry has a spec for it. Widening support
    // (qwen2, …) happens in backend::arch, not here.
    crate::backend::arch::spec_for(arch).is_some()
}

async fn list_models(State(s): State<AppState>) -> Json<Value> {
    let now = unix_secs();
    let current = s.current_model.read().unwrap_or_else(|e| e.into_inner()).clone();
    let mut entries: Vec<Value> = Vec::new();

    // Scan models_dir for additional .gguf files. Each becomes a
    // selectable model. The currently-loaded one (matched by stem) is
    // marked with current=true.
    let mut seen_current = false;
    if let Some(dir) = &s.models_dir {
        let mut walk: Vec<std::path::PathBuf> = vec![dir.clone()];
        while let Some(d) = walk.pop() {
            if let Ok(rd) = std::fs::read_dir(&d) {
                for ent in rd.flatten() {
                    let p = ent.path();
                    if p.is_dir() {
                        walk.push(p);
                    } else if p.extension().and_then(|e| e.to_str()) == Some("gguf") {
                        let id = p
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_string();
                        if id.is_empty() {
                            continue;
                        }
                        let size = ent.metadata().map(|m| m.len()).unwrap_or(0);
                        let is_current = id == current;
                        if is_current {
                            seen_current = true;
                        }
                        let tok_present = p.with_file_name("tokenizer.json").exists();
                        let arch = gguf_architecture(&s.arch_cache, &p);
                        let arch_ok = arch.as_deref().map(arch_is_supported).unwrap_or(false);
                        entries.push(json!({
                            "id": id,
                            "object": "model",
                            "created": now,
                            "owned_by": "local",
                            "size_bytes": size,
                            "current": is_current,
                            "loadable": tok_present && arch_ok,
                            "architecture": arch.unwrap_or_else(|| "unknown".into()),
                            "path": p.to_string_lossy(),
                        }));
                    }
                }
            }
        }
    }
    // If the currently-loaded model lives outside the scan dir, include
    // it anyway so the picker always shows what's actually running.
    if !seen_current && !current.is_empty() && current != "zllm" {
        entries.push(json!({
            "id": current,
            "object": "model",
            "created": now,
            "owned_by": "local",
            "current": true,
            "loadable": true,
        }));
    }
    if entries.is_empty() {
        entries.push(json!({
            "id": current,
            "object": "model",
            "created": now,
            "owned_by": "local",
            "current": true,
            "loadable": true,
        }));
    }
    // Sort: current first, then alphabetically
    entries.sort_by(|a, b| {
        let ac = a.get("current").and_then(|v| v.as_bool()).unwrap_or(false);
        let bc = b.get("current").and_then(|v| v.as_bool()).unwrap_or(false);
        bc.cmp(&ac).then_with(|| {
            let ai = a.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let bi = b.get("id").and_then(|v| v.as_str()).unwrap_or("");
            ai.cmp(bi)
        })
    });
    Json(json!({ "object": "list", "data": entries }))
}

#[derive(Deserialize)]
struct SelectModelReq {
    id: String,
}

#[derive(Deserialize)]
struct DownloadModelReq {
    /// HuggingFace repo id, e.g. `bartowski/Llama-3.2-3B-Instruct-GGUF`.
    repo: String,
    /// GGUF filename in that repo, e.g. `Llama-3.2-3B-Instruct-Q4_K_M.gguf`.
    filename: String,
    /// Optional separate tokenizer repo. Defaults to `repo`.
    #[serde(default)]
    tokenizer_repo: Option<String>,
}

/// POST /v1/models/download — fetch a GGUF and its sibling `tokenizer.json`
/// from a HuggingFace repo into the configured `models_dir`. Blocks the
/// response for the full download (no progress reporting in v0.7).
/// Subsequent `/v1/models` calls will list the new file.
async fn download_model(
    State(s): State<AppState>,
    Json(req): Json<DownloadModelReq>,
) -> impl IntoResponse {
    let Some(dir) = s.models_dir.clone() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "models_dir not configured"})),
        )
            .into_response();
    };
    if req.repo.trim().is_empty() || req.filename.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "repo and filename required"})),
        )
            .into_response();
    }
    if !req.filename.ends_with(".gguf") {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "filename must end in .gguf"})),
        )
            .into_response();
    }
    // Defense-in-depth: reject path traversal in either field
    if req.repo.contains("..") || req.filename.contains("..") || req.filename.contains('/') || req.filename.contains('\\') {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid characters in repo or filename"})),
        )
            .into_response();
    }
    let tok_repo = req.tokenizer_repo.clone().unwrap_or_else(|| req.repo.clone());
    let id = req.filename.trim_end_matches(".gguf").to_string();
    let target_subdir = dir.join(&id);
    let target_gguf = target_subdir.join(&req.filename);
    let target_tok = target_subdir.join("tokenizer.json");
    let repo_id = req.repo.clone();
    let filename = req.filename.clone();

    let res = tokio::task::spawn_blocking(move || -> anyhow::Result<u64> {
        use hf_hub::api::sync::ApiBuilder;
        std::fs::create_dir_all(&target_subdir)?;
        let api = ApiBuilder::new().build()?;

        // GGUF first (the big one)
        let gguf_cached = api.model(repo_id).get(&filename)?;
        if !target_gguf.exists() {
            std::fs::copy(&gguf_cached, &target_gguf)?;
        }
        // Tokenizer next to it
        let tok_cached = api.model(tok_repo).get("tokenizer.json")?;
        if !target_tok.exists() {
            std::fs::copy(&tok_cached, &target_tok)?;
        }
        let size = std::fs::metadata(&target_gguf)?.len();
        Ok(size)
    })
    .await;

    match res {
        Ok(Ok(size)) => {
            tracing::info!("downloaded {} ({} bytes)", id, size);
            Json(json!({"success": true, "id": id, "size_bytes": size})).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("download failed: {e}")})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("task join: {e}")})),
        )
            .into_response(),
    }
}

/// POST /v1/models/select — unload the current model and load a new one
/// from the models_dir scan. Blocks the response while the swap happens
/// (~2-30s depending on GGUF size). Clears MemoryStore captures (the
/// new model's n_embd may differ, which would dilute injection vectors).
async fn select_model(
    State(s): State<AppState>,
    Json(req): Json<SelectModelReq>,
) -> impl IntoResponse {
    let Some(dir) = s.models_dir.clone() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "models_dir not configured"})),
        )
            .into_response();
    };
    if req.id.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "id required"}))).into_response();
    }

    // Resolve the requested id to a file path by walking the scan dir.
    let mut found: Option<std::path::PathBuf> = None;
    let mut walk: Vec<std::path::PathBuf> = vec![dir];
    while let Some(d) = walk.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for ent in rd.flatten() {
                let p = ent.path();
                if p.is_dir() {
                    walk.push(p);
                } else if p.extension().and_then(|e| e.to_str()) == Some("gguf")
                    && p.file_stem().and_then(|s| s.to_str()) == Some(req.id.as_str())
                {
                    found = Some(p);
                    break;
                }
            }
        }
        if found.is_some() {
            break;
        }
    }
    let Some(gguf_path) = found else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("model {:?} not found in scan dir", req.id)})),
        )
            .into_response();
    };
    let tok_path = gguf_path.with_file_name("tokenizer.json");

    // Architecture pre-check. We refuse non-Llama BEFORE unloading the
    // current model so a failed swap doesn't leave the backend empty.
    let arch = gguf_architecture(&s.arch_cache, &gguf_path).unwrap_or_else(|| "unknown".into());
    if !arch_is_supported(&arch) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!(
                    "architecture {:?} is not supported by zllm v0.7 — only \"llama\" GGUFs work today. Other arches need their own backend fork (parallel to src/backend/candle/quantized_llama_fork.rs).",
                    arch
                )
            })),
        )
            .into_response();
    }

    // Load the new tokenizer first so a tokenizer error doesn't leave us
    // with an unloaded backend. Sibling tokenizer.json preferred; the
    // GGUF-embedded vocab (BPE) makes bare .gguf files swappable.
    let new_tok = if tok_path.exists() {
        match LlamaTokenizer::from_file(tok_path.to_str().unwrap_or("")) {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": format!("tokenizer load failed: {e}")})),
                )
                    .into_response();
            }
        }
    } else {
        match LlamaTokenizer::from_gguf_file(&gguf_path) {
            Ok(t) => {
                tracing::info!("no sibling tokenizer.json — using the GGUF-embedded vocab");
                t
            }
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": format!(
                        "no tokenizer.json next to {} and the embedded vocab is unusable: {e}",
                        gguf_path.display()
                    )})),
                )
                    .into_response();
            }
        }
    };

    // Acquire every slot up front so a chat-in-flight doesn't see a
    // half-swapped pool. Holds them simultaneously for the duration
    // of the reload — could take a few seconds per slot.
    let mut guards: Vec<_> = s
        .pool
        .iter()
        .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
        .collect();
    for g in guards.iter_mut() {
        let _ = g.backend.unload_model();
        if let Err(e) = g.backend.load_model(&gguf_path) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("model load failed in pool slot: {e}")})),
            )
                .into_response();
        }
        // Different model → different tokenizer → prior cached tokens
        // are no longer meaningful for this slot.
        g.prompt_cache.clear();
    }
    drop(guards);
    // Refresh the effective context window for the new model.
    let new_ctx = s
        .pool
        .first()
        .and_then(|m| m.lock().ok())
        .map(|g| g.backend.max_seq())
        .unwrap_or(0);
    s.model_ctx.store(new_ctx, Ordering::Relaxed);

    // Reload the GPU engine for the new model (else the fast-lane would keep
    // serving the OLD weights). Same gate as startup; on any failure (e.g. a
    // non-Llama arch the GPU loader doesn't support) we disable the GPU
    // fast-lane and fall back to the candle pool for this model.
    #[cfg(feature = "gpu")]
    if std::env::var("ZLLM_GPU").is_ok() {
        let path_str = gguf_path.to_str().unwrap_or("").to_string();
        let loaded = crate::backend::gpu::GpuContext::new()
            .and_then(|ctx| crate::backend::gpu::GpuModel::load(&path_str, ctx));
        match loaded {
            Ok(m) => {
                *s.gpu.lock().unwrap_or_else(|e| e.into_inner()) = Some(m);
                tracing::info!("GPU engine reloaded for swapped model");
            }
            Err(e) => {
                *s.gpu.lock().unwrap_or_else(|e| e.into_inner()) = None;
                tracing::warn!("GPU reload failed ({e}); GPU fast-lane disabled for this model");
            }
        }
    }
    // Hot-swap the continuous-batching server's model too (it owns its own copy
    // on a dedicated thread), so the default chat backend follows model selection.
    #[cfg(feature = "gpu")]
    if let Some(cb) = &s.cb {
        let path_str = gguf_path.to_str().unwrap_or("").to_string();
        if cb.swap_model(path_str) {
            tracing::info!("continuous-batching server reloaded for swapped model");
        } else {
            tracing::warn!("continuous-batching server model swap failed");
        }
    }
    #[cfg(feature = "vulkan")]
    if std::env::var("ZLLM_VK").is_ok() {
        let path_str = gguf_path.to_str().unwrap_or("").to_string();
        let loaded = crate::backend::vulkan::VkContext::new()
            .and_then(|ctx| crate::backend::vulkan::VkModel::load(&path_str, ctx));
        match loaded {
            Ok(m) => { *s.vk.lock().unwrap_or_else(|e| e.into_inner()) = Some(m); tracing::info!("Vulkan engine reloaded for swapped model"); }
            Err(e) => { *s.vk.lock().unwrap_or_else(|e| e.into_inner()) = None; tracing::warn!("Vulkan reload failed ({e}); fast-lane disabled for this model"); }
        }
    }

    *s.tokenizer.write().unwrap_or_else(|e| e.into_inner()) = new_tok;
    *s.current_model.write().unwrap_or_else(|e| e.into_inner()) = req.id.clone();
    // Grammar byte table is tokenizer-specific — rebuild lazily on next use.
    *s.token_table.write().unwrap_or_else(|e| e.into_inner()) = None;
    // Chat template + declared stop ids follow the new GGUF.
    *s.chat_meta.write().unwrap_or_else(|e| e.into_inner()) =
        crate::server::chat_template::read_gguf_chat_meta(&gguf_path);

    // Selectively clear Context captures — they have an n_embd from
    // the previous model and would dilute injections done on the new
    // one. Goal / Task / Status entries survive the swap (their text is
    // model-agnostic) but their vectors are real embeddings in the OLD
    // model's space and width, so they must be re-encoded below.
    if let Ok(mut store) = s.memory.write() {
        let to_remove: Vec<String> = store
            .query_by_category(&MemoryCategory::Context)
            .into_iter()
            .map(|e| e.key.clone())
            .collect();
        for key in to_remove {
            store.remove(&key);
        }
    }
    // Re-embed goals/tasks/status against the swapped-in model. The
    // encoder reads the tokenizer + pool we just updated; slot locks are
    // released at this point, so the try_lock inside the encoder succeeds.
    s.goals.reencode_all();

    tracing::info!("model swapped to {} ({})", req.id, gguf_path.display());
    Json(json!({"success": true, "current": req.id})).into_response()
}

/// OpenAI `stop`: a single string or an array of strings.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum StopParam {
    One(String),
    Many(Vec<String>),
}

impl StopParam {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopParam::One(s) => vec![s],
            StopParam::Many(v) => v,
        }
    }
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    /// Optional RNG seed for reproducible sampling (all lanes).
    #[serde(default)]
    seed: Option<u32>,
    /// OpenAI stop strings: generation halts when the decoded tail
    /// contains any of them; the match is trimmed from the output.
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    /// llama.cpp-style repetition penalty (divide/multiply), 1.0 = off.
    /// `repetition_penalty` accepted as an alias.
    #[serde(default, alias = "repetition_penalty")]
    repeat_penalty: Option<f32>,
    /// OpenAI logit_bias: stringified token id → additive bias (±100
    /// effectively bans/forces).
    #[serde(default)]
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    // --- Recognized-but-unsupported OpenAI params: rejected with 400
    // instead of silently ignored (V1_PLAN M1 "no silent lies") ---
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    /// Optional logit-constraint string. v0.5 supports `"ban:<id>,<id>,…"`;
    /// see `engine::logit_fsm::LogitFSM` for the full list of modes.
    /// Non-OpenAI-standard but cheap to add and useful for the
    /// installed-app case.
    #[serde(default)]
    grammar: Option<String>,
    /// Attach an output-distribution hallucination/uncertainty report to the
    /// response. Forces the candle path (full per-token logits) like inspection.
    #[serde(default)]
    detect_hallucination: Option<bool>,
    /// OpenAI-compatible logprobs: `logprobs: true` returns each generated
    /// token's log-probability; `top_logprobs: N` (0..=20) adds the top-N
    /// alternatives. Forces the candle path like detection.
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<u32>,
}

fn sampler_from_request(
    engine: &EngineConfig,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
) -> SamplerConfig {
    SamplerConfig {
        temperature: temperature.unwrap_or(engine.default_temperature),
        top_k: top_k.unwrap_or(engine.default_top_k),
        top_p: top_p.unwrap_or(engine.default_top_p),
        min_p: 0.0,
    }
}

/// Build the per-request DecodeControl from OpenAI-style params. Stop
/// strings are capped at 8 (OpenAI allows 4); logit_bias values are
/// clamped to ±100 per the OpenAI contract.
fn build_decode_ctrl(
    stop: Option<StopParam>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repeat_penalty: Option<f32>,
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    seed: Option<u32>,
) -> DecodeControl {
    let stops: Vec<String> = stop
        .map(|s| s.into_vec())
        .unwrap_or_default()
        .into_iter()
        .filter(|s| !s.is_empty())
        .take(8)
        .collect();
    let bias: Vec<(u32, f32)> = logit_bias
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(k, v)| k.trim().parse::<u32>().ok().map(|id| (id, v.clamp(-100.0, 100.0))))
        .collect();
    DecodeControl::new(
        PenaltyConfig {
            repeat: repeat_penalty.unwrap_or(1.0),
            presence: presence_penalty.unwrap_or(0.0),
            frequency: frequency_penalty.unwrap_or(0.0),
        },
        bias,
        seed.map(u64::from),
        stops,
    )
}

/// Per-request wall-clock budget (ZLLM_REQ_TIMEOUT_SECS, default 600).
fn request_timeout() -> std::time::Duration {
    std::time::Duration::from_secs(
        std::env::var("ZLLM_REQ_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(600),
    )
}

/// The stop-TOKEN set for the loaded model: ids the GGUF declares
/// (ground truth) unioned with the vocab probe (covers GGUFs that
/// predate the declared fields).
fn stop_set(s: &AppState) -> Vec<u32> {
    let mut ids = s.chat_meta.read().unwrap_or_else(|e| e.into_inner()).stop_ids.clone();
    for id in s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).stop_token_ids() {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    ids
}

/// Decode the generated tail into text for stop-string matching. Always
/// a window re-decode — per-token decodes drop SentencePiece space
/// markers and would miss stops spanning token boundaries.
fn stop_window_text(s: &AppState, generated: &[u32], window: usize) -> String {
    let start = generated.len().saturating_sub(window);
    s.tokenizer
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .decode(&generated[start..])
        .unwrap_or_default()
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

fn default_max_tokens() -> usize {
    256
}

async fn chat_completions(
    State(s): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    // Build the rendered prompt with goal-prefix injected as the system
    // message.
    let prompt = render_chat_prompt(&s, &req.messages);
    let tokens = match s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("tokenize: {e}")})),
            )
                .into_response();
        }
    };

    // Context-window guard: an over-long prompt gets an OpenAI-style 400
    // instead of a mid-forward failure.
    let ctx = s.model_ctx.load(Ordering::Relaxed);
    if ctx > 0 && tokens.len() >= ctx {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": format!(
                "this model's maximum context length is {ctx} tokens; the rendered prompt has {}",
                tokens.len()
            ),
            "code": "context_length_exceeded"
        }))).into_response();
    }
    let active = match admit(&s) {
        Ok(g) => g,
        Err(resp) => return resp,
    };
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let model_id = s.current_model.read().unwrap_or_else(|e| e.into_inner()).clone();
    // Clamp generation to the remaining window so the KV cache can never
    // be asked to grow past what it preallocated.
    let max_tokens = if ctx > 0 {
        req.max_tokens.min(ctx.saturating_sub(tokens.len()).max(1))
    } else {
        req.max_tokens
    };
    // Recognized-but-unsupported OpenAI params fail loudly (V1_PLAN M1:
    // accepting a parameter and ignoring it is a silent lie).
    if req.tools.is_some() || req.tool_choice.is_some() {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": "tools / tool_choice are not supported"
        }))).into_response();
    }
    if req.n.unwrap_or(1) > 1 {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": "n > 1 is not supported"
        }))).into_response();
    }
    let mut sampler_cfg = sampler_from_request(&s.engine, req.temperature, req.top_p, req.top_k);
    sampler_cfg.min_p = req.min_p.unwrap_or(0.0);
    let mut ctrl = build_decode_ctrl(
        req.stop.clone(),
        req.presence_penalty,
        req.frequency_penalty,
        req.repeat_penalty,
        req.logit_bias.clone(),
        req.seed,
    );
    // A single decoding constraint may come from the OpenAI `response_format`
    // (json_object / json_schema) or the `grammar` extension. Resolve to one
    // grammar string, then compile. Compile errors (bad regex/schema,
    // unimplemented mode) fail loudly — silently returning unconstrained output
    // to a caller who asked for a constraint is worse than an error.
    let grammar = match resolve_constraint(&req.grammar, &req.response_format) {
        Ok(g) => g,
        Err(msg) => return (StatusCode::BAD_REQUEST, Json(json!({"error": msg}))).into_response(),
    };
    let fsm = match grammar.as_deref() {
        Some(g) => match fsm_from_grammar(&s, g) {
            Ok(f) => Some(f),
            Err(msg) => {
                return (StatusCode::BAD_REQUEST, Json(json!({"error": msg}))).into_response();
            }
        },
        None => None,
    };
    // Hallucination detection and logprobs are only wired into the
    // non-streaming path.
    if req.detect_hallucination.unwrap_or(false) && req.stream {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": "detect_hallucination is not supported with stream=true yet; use stream=false"
        }))).into_response();
    }
    if req.logprobs.unwrap_or(false) && req.stream {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": "logprobs is not supported with stream=true yet; use stream=false"
        }))).into_response();
    }

    // Continuous-batching fast lane (ZLLM_CB=1): route eligible chat requests
    // through the shared in-flight batcher (vLLM-style) instead of the candle
    // pool / single-stream GPU fast lanes. Eligible = inspection off and none of
    // the candle-only features (grammar / spec-decode / PLD / early-exit) are on,
    // since the CB engine doesn't implement those. Greedy or temp/top-k/top-p.
    #[cfg(feature = "gpu")]
    if let Some(server) = cb_chat_server(
        &s,
        fsm.is_none()
            && !req.detect_hallucination.unwrap_or(false)
            && !req.logprobs.unwrap_or(false)
            // The CB engine samples on its own thread and implements
            // none of penalties / logit_bias / stop strings / min_p —
            // such requests fall through to the candle/fast-lane paths.
            && !ctrl.modifies_logits()
            && !ctrl.has_stops()
            && req.min_p.is_none(),
    ) {
        let prompt_tokens = tokens.len();
        let temp = sampler_cfg.temperature;
        let params = if temp <= 0.0 {
            crate::backend::gpu::SamplingParams::greedy()
        } else {
            crate::backend::gpu::SamplingParams { temp, top_k: sampler_cfg.top_k as u32, top_p: sampler_cfg.top_p }
        };
        let seed = req.seed.unwrap_or(0);
        let (eos, stop_eot) = {
            let tok = s.tokenizer.read().unwrap_or_else(|e| e.into_inner());
            let eos = tok.eos_token_id().unwrap_or(128001);
            // End-of-turn stop derived from the vocab (Llama-3 <|eot_id|>,
            // ChatML <|im_end|>) instead of a hardcoded Llama-3 id; falls
            // back to EOS for vocabs with no separate turn token.
            let eot = tok
                .token_to_id("<|eot_id|>")
                .or_else(|| tok.token_to_id("<|im_end|>"))
                .unwrap_or(eos);
            (eos, eot)
        };
        match server.submit(tokens, max_tokens, stop_eot, params, seed) {
            Ok(rx) => {
                return if req.stream {
                    Sse::new(cb_chat_stream(rx, eos, stop_eot, s.clone(), id, model_id)).into_response()
                } else {
                    cb_chat_blocking(rx, eos, stop_eot, &s, id, model_id, prompt_tokens).await
                };
            }
            Err(_) => return (StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "continuous-batching server unavailable"}))).into_response(),
        }
    }

    if req.stream {
        let stream = chat_stream(s.clone(), tokens, max_tokens, sampler_cfg, fsm, id.clone(), model_id, ctrl, active);
        Sse::new(stream).into_response()
    } else {
        let _active = active;
        let mut detector = req.detect_hallucination.unwrap_or(false)
            .then(|| crate::engine::hallucination::Detector::new(Default::default()));
        let mut lp = req.logprobs.unwrap_or(false).then(|| {
            crate::engine::logprobs::LogprobsCollector::new(req.top_logprobs.unwrap_or(0) as usize)
        });
        let (text, prompt_tokens, completion_tokens, finish_reason) =
            generate_blocking(&s, tokens, max_tokens, &sampler_cfg, fsm.as_ref(), &id, detector.as_mut(), lp.as_mut(), &mut ctrl);
        let hallu = detector.map(|d| hallucination_json(&d.report()));
        let logprobs_json = lp.map(|c| chat_logprobs_json(&s, &c));
        let now = unix_secs();
        let mut resp = json!({
            "id": id,
            "object": "chat.completion",
            "created": now,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        });
        // Only present when detection was requested — a null field on every
        // response breaks byte-untouched middleware contracts (found by the
        // llm-probe proxy's T2 battery test).
        if let Some(h) = hallu {
            resp["hallucination"] = h;
        }
        if let Some(l) = logprobs_json {
            resp["choices"][0]["logprobs"] = l;
        }
        Json(resp).into_response()
    }
}

#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default, alias = "repetition_penalty")]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    // Recognized-but-unsupported → 400, not silence.
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    best_of: Option<u32>,
    #[serde(default)]
    grammar: Option<String>,
    /// Attach an output-distribution hallucination/uncertainty report to the
    /// response. Forces the candle path (full per-token logits) like inspection.
    #[serde(default)]
    detect_hallucination: Option<bool>,
    /// Legacy OpenAI completions logprobs: an INTEGER N returns each token's
    /// logprob plus the top-N alternatives (`{tokens, token_logprobs,
    /// top_logprobs}` response shape). Forces the candle path.
    #[serde(default)]
    logprobs: Option<u32>,
}

/// Build a LogitFSM from a request's `grammar` string, supplying the cached
/// vocab byte table for the DFA modes (`regex:` / `json_schema:` / `json:`),
/// built on first use (~one decode per vocab entry; invalidated on model
/// swap). `Err` = user-facing 400 message (bad pattern / bad schema /
/// unimplemented mode / no EOS).
fn fsm_from_grammar(s: &AppState, grammar: &str) -> Result<LogitFSM, String> {
    let g = grammar.trim_start();
    let needs_table =
        g.starts_with("regex:") || g.starts_with("json_schema:") || g.starts_with("json:");
    let table = if needs_table {
        if let Some(t) = s.token_table.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
            Some(t.clone())
        } else {
            let tok = s.tokenizer.read().unwrap_or_else(|e| e.into_inner());
            let eos = tok.eos_token_id().ok_or("tokenizer has no EOS token; regex grammar unavailable")?;
            let t0 = std::time::Instant::now();
            let table = Arc::new(crate::engine::logit_fsm::TokenByteTable {
                bytes: tok.token_bytes_table(),
                eos,
            });
            tracing::info!("built grammar byte table ({} tokens) in {:.0} ms",
                table.bytes.len(), t0.elapsed().as_secs_f64() * 1e3);
            *s.token_table.write().unwrap_or_else(|e| e.into_inner()) = Some(table.clone());
            Some(table)
        }
    } else {
        None
    };
    LogitFSM::compile(grammar, table)
}

/// Map an OpenAI `response_format` object to a grammar string, or `None` for
/// the unconstrained (`text`) case. `json_object` → any well-formed JSON;
/// `json_schema` → the embedded schema compiled to a constraint. `Err` is a
/// user-facing 400 message.
fn grammar_from_response_format(rf: &serde_json::Value) -> Result<Option<String>, String> {
    let ty = rf.get("type").and_then(|t| t.as_str()).unwrap_or("text");
    match ty {
        "" | "text" => Ok(None),
        // OpenAI json_object means a JSON *object* specifically (not any JSON
        // value). An empty-properties object schema compiles to exactly that:
        // a top-level `{...}` with arbitrary keys/values. (The shapeless
        // any-value `json:` grammar remains available as an extension.)
        "json_object" => Ok(Some("json_schema:{\"type\":\"object\"}".to_string())),
        "json_schema" => {
            // OpenAI nests the schema at response_format.json_schema.schema.
            let schema = rf
                .get("json_schema")
                .and_then(|j| j.get("schema"))
                .ok_or("response_format \"json_schema\" requires a json_schema.schema object")?;
            let schema_str = serde_json::to_string(schema)
                .map_err(|e| format!("response_format schema: {e}"))?;
            Ok(Some(format!("json_schema:{schema_str}")))
        }
        other => Err(format!(
            "response_format {other:?} is not supported (supported: text, json_object, json_schema)"
        )),
    }
}

/// Resolve the single decoding constraint for a request from either the
/// `grammar` extension or the OpenAI `response_format` — at most one may
/// constrain output. `Err` is a user-facing 400 message.
fn resolve_constraint(
    grammar: &Option<String>,
    response_format: &Option<serde_json::Value>,
) -> Result<Option<String>, String> {
    let rf_grammar = match response_format {
        Some(rf) => grammar_from_response_format(rf)?,
        None => None,
    };
    match (grammar, rf_grammar) {
        (Some(g), Some(_)) if !g.trim().is_empty() => Err(
            "cannot combine the `grammar` extension with a constraining `response_format`; use one"
                .to_string(),
        ),
        (Some(g), _) if !g.trim().is_empty() => Ok(Some(g.clone())),
        (_, rf) => Ok(rf),
    }
}

/// Compact JSON view of a hallucination report (the per-token detail is omitted —
/// the summary is what callers act on). `flagged` uses a 0.5 risk bar.
fn hallucination_json(r: &crate::engine::hallucination::HallucinationReport) -> serde_json::Value {
    json!({
        "risk_score": r.risk_score,
        "mean_entropy": r.mean_entropy,
        "normalized_entropy": r.normalized_entropy,
        "risky_fraction": r.risky_fraction,
        "n_tokens": r.n_tokens,
        "peak_token_index": r.peak_token,
        "flagged": r.flagged(0.5),
    })
}

async fn text_completions(
    State(s): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let prefix = s.goals.build_prompt_prefix();
    let prompt = format!("{prefix}{}", req.prompt);
    let tokens = match s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("tokenize: {e}")})),
            )
                .into_response();
        }
    };
    let ctx = s.model_ctx.load(Ordering::Relaxed);
    if ctx > 0 && tokens.len() >= ctx {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": format!(
                "this model's maximum context length is {ctx} tokens; the prompt has {}",
                tokens.len()
            ),
            "code": "context_length_exceeded"
        }))).into_response();
    }
    let _active = match admit(&s) {
        Ok(g) => g,
        Err(resp) => return resp,
    };
    let max_tokens = if ctx > 0 {
        req.max_tokens.min(ctx.saturating_sub(tokens.len()).max(1))
    } else {
        req.max_tokens
    };
    let id = format!("cmpl-{}", Uuid::new_v4());
    if req.n.unwrap_or(1) > 1 || req.best_of.unwrap_or(1) > 1 {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": "n > 1 / best_of > 1 are not supported"
        }))).into_response();
    }
    let mut sampler_cfg = sampler_from_request(&s.engine, req.temperature, req.top_p, req.top_k);
    sampler_cfg.min_p = req.min_p.unwrap_or(0.0);
    let mut ctrl = build_decode_ctrl(
        req.stop.clone(),
        req.presence_penalty,
        req.frequency_penalty,
        req.repeat_penalty,
        req.logit_bias.clone(),
        req.seed,
    );
    // Grammar compile errors fail loudly (silent unconstrained output is worse
    // than an error) — same contract as the chat endpoint.
    let fsm = match req.grammar.as_deref() {
        Some(g) => match fsm_from_grammar(&s, g) {
            Ok(f) => Some(f),
            Err(msg) => {
                return (StatusCode::BAD_REQUEST, Json(json!({"error": msg}))).into_response();
            }
        },
        None => None,
    };
    let mut detector = req.detect_hallucination.unwrap_or(false)
        .then(|| crate::engine::hallucination::Detector::new(Default::default()));
    let mut lp = req.logprobs
        .map(|n| crate::engine::logprobs::LogprobsCollector::new(n as usize));
    let (text, p, c, finish_reason) = generate_blocking(&s, tokens, max_tokens, &sampler_cfg, fsm.as_ref(), &id, detector.as_mut(), lp.as_mut(), &mut ctrl);
    let hallu = detector.map(|d| hallucination_json(&d.report()));
    let logprobs_json = lp.map(|col| legacy_logprobs_json(&s, &col));
    let now = unix_secs();
    let mut resp = json!({
        "id": id,
        "object": "text_completion",
        "created": now,
        "model": s.current_model.read().unwrap_or_else(|e| e.into_inner()).clone(),
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": p + c
        }
    });
    // Only present when detection was requested (see the chat handler note).
    if let Some(h) = hallu {
        resp["hallucination"] = h;
    }
    if let Some(l) = logprobs_json {
        resp["choices"][0]["logprobs"] = l;
    }
    Json(resp).into_response()
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct EmbeddingsRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    input: EmbeddingsInput,
}

/// OpenAI-compatible `/v1/embeddings`: mean-pooled, L2-normalized token
/// embeddings from the loaded model (embedding lookup only — no
/// transformer layers). The same construction the GoalManager encoder
/// uses, so goal-similarity and API embeddings live in one space.
async fn embeddings(
    State(s): State<AppState>,
    Json(req): Json<EmbeddingsRequest>,
) -> impl IntoResponse {
    let inputs = match req.input {
        EmbeddingsInput::One(t) => vec![t],
        EmbeddingsInput::Many(v) => v,
    };
    if inputs.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "input is empty"}))).into_response();
    }
    let mut data: Vec<Value> = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0usize;
    let slot = acquire_slot(&s.pool);
    for (i, text) in inputs.iter().enumerate() {
        let ids = match s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).encode(text) {
            Ok(v) if !v.is_empty() => v,
            Ok(_) => {
                return (StatusCode::BAD_REQUEST,
                    Json(json!({"error": format!("input {i} tokenized to nothing")}))).into_response();
            }
            Err(e) => {
                return (StatusCode::BAD_REQUEST,
                    Json(json!({"error": format!("tokenize input {i}: {e}")}))).into_response();
            }
        };
        total_tokens += ids.len();
        let flat = match slot.backend.embed_tokens(&ids) {
            Ok(f) => f,
            Err(e) => {
                return (StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({"error": format!("embedding lookup: {e}")}))).into_response();
            }
        };
        let d = flat.len() / ids.len();
        let mut mean = vec![0f32; d];
        for chunk in flat.chunks_exact(d) {
            for (m, v) in mean.iter_mut().zip(chunk) {
                *m += v;
            }
        }
        let inv_n = 1.0 / ids.len() as f32;
        for m in &mut mean {
            *m *= inv_n;
        }
        let norm = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for m in &mut mean {
                *m /= norm;
            }
        }
        data.push(json!({"object": "embedding", "index": i, "embedding": mean}));
    }
    drop(slot);
    Json(json!({
        "object": "list",
        "data": data,
        "model": s.current_model.read().unwrap_or_else(|e| e.into_inner()).clone(),
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens}
    }))
    .into_response()
}

#[derive(Deserialize)]
struct TokenizeRequest {
    content: String,
}

/// llama.cpp-compatible `/tokenize`.
async fn tokenize(State(s): State<AppState>, Json(req): Json<TokenizeRequest>) -> impl IntoResponse {
    match s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).encode(&req.content) {
        Ok(tokens) => Json(json!({"tokens": tokens})).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": format!("tokenize: {e}")}))).into_response(),
    }
}

#[derive(Deserialize)]
struct DetokenizeRequest {
    tokens: Vec<u32>,
}

/// llama.cpp-compatible `/detokenize`.
async fn detokenize(State(s): State<AppState>, Json(req): Json<DetokenizeRequest>) -> impl IntoResponse {
    match s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&req.tokens) {
        Ok(content) => Json(json!({"content": content})).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": format!("detokenize: {e}")}))).into_response(),
    }
}

/// OpenAI CHAT logprobs shape: `{"content": [{token, logprob, bytes,
/// top_logprobs: [{token, logprob, bytes}]}]}`.
fn chat_logprobs_json(s: &AppState, c: &crate::engine::logprobs::LogprobsCollector) -> serde_json::Value {
    let tok = s.tokenizer.read().unwrap_or_else(|e| e.into_inner());
    let dec = |id: u32| tok.decode(&[id]).unwrap_or_default();
    let content: Vec<serde_json::Value> = c.entries.iter().map(|e| {
        let t = dec(e.token_id);
        json!({
            "token": t,
            "logprob": e.logprob,
            "bytes": t.as_bytes(),
            "top_logprobs": e.top.iter().map(|(id, lp)| {
                let tt = dec(*id);
                json!({"token": tt, "logprob": lp, "bytes": tt.as_bytes()})
            }).collect::<Vec<_>>()
        })
    }).collect();
    json!({ "content": content })
}

/// Legacy COMPLETIONS logprobs shape: `{tokens, token_logprobs,
/// top_logprobs: [{token: logprob}]}`.
fn legacy_logprobs_json(s: &AppState, c: &crate::engine::logprobs::LogprobsCollector) -> serde_json::Value {
    let tok = s.tokenizer.read().unwrap_or_else(|e| e.into_inner());
    let dec = |id: u32| tok.decode(&[id]).unwrap_or_default();
    let tokens: Vec<String> = c.entries.iter().map(|e| dec(e.token_id)).collect();
    let token_logprobs: Vec<f32> = c.entries.iter().map(|e| e.logprob).collect();
    let top: Vec<serde_json::Value> = c.entries.iter().map(|e| {
        let m: serde_json::Map<String, serde_json::Value> = e.top.iter()
            .map(|(id, lp)| (dec(*id), json!(lp)))
            .collect();
        serde_json::Value::Object(m)
    }).collect();
    json!({ "tokens": tokens, "token_logprobs": token_logprobs, "top_logprobs": top })
}

#[derive(Deserialize)]
#[cfg_attr(not(feature = "gpu"), allow(dead_code))] // consumed only by the CB fast lane
struct CbRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    /// Sampling temperature (0 / omitted = greedy).
    #[serde(default)]
    temperature: Option<f32>,
    /// Top-k cap (0 / omitted = off). Capped at the GPU candidate pool (64).
    #[serde(default)]
    top_k: Option<u32>,
    /// Top-p (nucleus) threshold (omitted / ≥1 = off).
    #[serde(default)]
    top_p: Option<f32>,
    /// Optional RNG seed for reproducible sampling.
    #[serde(default)]
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] seed: Option<u32>, // consumed by the CB fast lane (feature = "gpu")
}

/// Continuous-batching completion (`/v1/cb/completions`). Routes to the
/// `GpuBatchServer` (started with `ZLLM_CB=1`): the prompt is admitted into a
/// free KV slot and decoded together with every other in-flight request —
/// vLLM-style in-flight batching, high aggregate throughput under concurrency.
/// Temperature sampling (`temperature`>0) or greedy; `seed` for reproducibility.
/// Streams SSE text chunks when `stream`, else returns the full text. 503 when
/// the server is not enabled / built.
async fn cb_completions(
    State(s): State<AppState>,
    Json(req): Json<CbRequest>,
) -> axum::response::Response {
    #[cfg(feature = "gpu")]
    {
        let server = match &s.cb {
            Some(srv) => srv.clone(),
            None => return (StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "continuous-batching server not enabled (start with ZLLM_CB=1)"}))).into_response(),
        };
        let tokens = match s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).encode(&req.prompt) {
            Ok(t) => t,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("tokenize: {e}")}))).into_response(),
        };
        let (eos, stop_eot) = {
            let tok = s.tokenizer.read().unwrap_or_else(|e| e.into_inner());
            let eos = tok.eos_token_id().unwrap_or(128001);
            // Chat-turn stop token from the vocab (<|eot_id|> / <|im_end|>),
            // not the hardcoded Llama-3 128009; falls back to EOS.
            let eot = tok
                .token_to_id("<|eot_id|>")
                .or_else(|| tok.token_to_id("<|im_end|>"))
                .unwrap_or(eos);
            (eos, eot)
        };
        let params = crate::backend::gpu::SamplingParams {
            temp: req.temperature.unwrap_or(0.0),
            top_k: req.top_k.unwrap_or(0),
            top_p: req.top_p.unwrap_or(1.0),
        };
        let seed = req.seed.unwrap_or(0);
        // Use eot as the server's stop so the KV slot frees on the chat stop.
        let mut tok_rx = match server.submit(tokens, req.max_tokens, stop_eot, params, seed) {
            Ok(r) => r,
            Err(_) => return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "server unavailable"}))).into_response(),
        };
        let tok = s.tokenizer.clone();
        let model_id = s.current_model.read().unwrap_or_else(|e| e.into_inner()).clone();

        if req.stream {
            use futures::channel::mpsc;
            let id = format!("cmpl-{}", Uuid::new_v4());
            let (tx, rx) = mpsc::unbounded::<Result<Event, Infallible>>();
            tokio::spawn(async move {
                let now = unix_secs();
                while let Some(item) = tok_rx.recv().await {
                    let t = match item { Some(t) => t, None => break }; // None = done sentinel
                    if t == eos || t == stop_eot { break; }
                    let text = tok.read().unwrap_or_else(|e| e.into_inner()).decode(&[t]).unwrap_or_default();
                    let chunk = json!({"id": id, "object": "text_completion.chunk", "created": now,
                        "model": model_id, "choices": [{"text": text, "index": 0, "finish_reason": null}]});
                    if tx.unbounded_send(Ok(Event::default().data(chunk.to_string()))).is_err() { break; }
                }
                let _ = tx.unbounded_send(Ok(Event::default().data("[DONE]")));
            });
            Sse::new(rx).into_response()
        } else {
            let mut out_ids: Vec<u32> = Vec::new();
            while let Some(item) = tok_rx.recv().await {
                let t = match item { Some(t) => t, None => break };
                if t == eos || t == stop_eot { break; }
                out_ids.push(t);
            }
            let text = tok.read().unwrap_or_else(|e| e.into_inner()).decode(&out_ids).unwrap_or_default();
            Json(json!({
                "id": format!("cmpl-{}", Uuid::new_v4()),
                "object": "text_completion", "created": unix_secs(), "model": model_id,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {"completion_tokens": out_ids.len()}
            })).into_response()
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (&s, &req);
        (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "built without the gpu feature"}))).into_response()
    }
}

/// Eligibility gate for routing a chat request through the continuous-batching
/// server: present (ZLLM_CB=1), inspection off, no grammar, and none of the
/// candle-only decode features (PLD / spec-decode / early-exit) enabled.
#[cfg(feature = "gpu")]
fn cb_chat_server(s: &AppState, no_fsm: bool) -> Option<Arc<crate::backend::gpu::GpuBatchServer>> {
    let server = s.cb.clone()?;
    let on = |a: &std::sync::atomic::AtomicBool| a.load(Ordering::Relaxed);
    if on(&s.inspection_enabled) || !no_fsm || on(&s.pld_enabled)
        || on(&s.spec_decode_enabled) || on(&s.early_exit_enabled) {
        return None;
    }
    Some(server)
}

/// Stream a continuous-batching chat completion as OpenAI `chat.completion.chunk`
/// SSE events: a role chunk, one content chunk per decoded token (stopping at
/// eos/eot without emitting it), then a finish chunk and `[DONE]`.
#[cfg(feature = "gpu")]
fn cb_chat_stream(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Option<u32>>,
    eos: u32, stop: u32, s: AppState, id: String, model_id: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    use futures::channel::mpsc;
    let (tx, out_rx) = mpsc::unbounded::<Result<Event, Infallible>>();
    let now = unix_secs();
    tokio::spawn(async move {
        let role = json!({"id": id, "object": "chat.completion.chunk", "created": now, "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]});
        let _ = tx.unbounded_send(Ok(Event::default().data(role.to_string())));
        while let Some(item) = rx.recv().await {
            let t = match item { Some(t) => t, None => break };
            if t == eos || t == stop { break; }
            let text = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&[t]).unwrap_or_default();
            let chunk = json!({"id": id, "object": "chat.completion.chunk", "created": now, "model": model_id,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]});
            if tx.unbounded_send(Ok(Event::default().data(chunk.to_string()))).is_err() { break; }
        }
        let fin = json!({"id": id, "object": "chat.completion.chunk", "created": now, "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]});
        let _ = tx.unbounded_send(Ok(Event::default().data(fin.to_string())));
        let _ = tx.unbounded_send(Ok(Event::default().data("[DONE]")));
    });
    out_rx
}

/// Collect a continuous-batching chat completion and return the full
/// `chat.completion` JSON (non-streaming).
#[cfg(feature = "gpu")]
async fn cb_chat_blocking(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Option<u32>>,
    eos: u32, stop: u32, s: &AppState, id: String, model_id: String, prompt_tokens: usize,
) -> axum::response::Response {
    let mut ids: Vec<u32> = Vec::new();
    while let Some(item) = rx.recv().await {
        let t = match item { Some(t) => t, None => break };
        if t == eos || t == stop { break; }
        ids.push(t);
    }
    let text = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&ids).unwrap_or_default();
    let completion = ids.len();
    Json(json!({
        "id": id, "object": "chat.completion", "created": unix_secs(), "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion, "total_tokens": prompt_tokens + completion}
    })).into_response()
}

// --- Generation ---

/// Length of the longest common prefix of two token slices.
fn lcp_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Prepare the backend KV cache for a chat prefill against
/// `prompt`. Compares against `cached` (tokens currently in KV):
///   - LCP == 0           → full reset, prefill from 0.
///   - 0 < LCP < prompt   → truncate KV to LCP, prefill from LCP.
///   - LCP >= prompt      → truncate KV to `prompt.len() - 1`, prefill
///                          the last token (needed to get fresh logits).
/// Returns the index at which prefill should start.
fn prepare_prompt_cache(
    backend: &mut CandleCpuBackend,
    cached: &mut Vec<u32>,
    prompt: &[u32],
) -> usize {
    let lcp = lcp_len(prompt, cached);
    let reuse = lcp.min(prompt.len().saturating_sub(1));
    if reuse == 0 {
        backend.reset_position();
        cached.clear();
        crate::metrics::prefix_cache_misses().inc();
        return 0;
    }
    let _ = backend.truncate_to(reuse);
    cached.truncate(reuse);
    crate::metrics::prefix_cache_hits().inc();
    crate::metrics::prefix_cache_tokens_saved().inc_by(reuse as u64);
    reuse
}

/// Mean-pool a `(1, seq_len, n_embd)` candle tensor and write it into
/// `MemoryStore` as a chat-prefill capture. Fires once per chat request
/// from inside `forward_logits_with_observer` (v0.7 Phase 2). Lets the
/// memory store actually populate from chat conversations — before this,
/// the only way to write to the store was via the `GoalManager` API.
fn capture_prefill_to_memory(
    memory: &Arc<RwLock<MemoryStore>>,
    request_id: &str,
    layer_idx: usize,
    hidden: &candle_core::Tensor,
) {
    let pooled = match hidden
        .mean(1)
        .and_then(|t| t.squeeze(0))
        .and_then(|t| t.to_vec1::<f32>())
    {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("chat capture mean-pool failed at layer {layer_idx}: {e}");
            return;
        }
    };
    if let Ok(mut store) = memory.write() {
        let metadata = MemoryMetadata {
            source_request_id: request_id.to_string(),
            layer_captured: layer_idx,
            category: MemoryCategory::Context,
            tags: vec!["chat".into(), "prefill".into()],
            text_summary: format!("chat prefill at layer {layer_idx}"),
        };
        store.store(format!("{request_id}:prefill"), pooled, metadata);
    }
}

/// Chat-template family, detected from the loaded tokenizer's vocab
/// instead of assuming Llama 3. The arch gate limits swaps to llama-arch
/// GGUFs, but that family spans three prompt formats in the wild:
/// Llama-3 header style, ChatML (many llama-arch finetunes: Hermes,
/// TinyLlama-Chat, …), and Llama-2 `[INST]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatFamily {
    Llama3,
    ChatMl,
    /// Mistral v0.3+/Tekken: `[INST]` is a control token in the vocab;
    /// no `<<SYS>>` — the system prompt folds into the first user turn.
    Mistral,
    Llama2,
}

impl ChatFamily {
    fn detect(tok: &LlamaTokenizer) -> Self {
        if tok.token_to_id("<|start_header_id|>").is_some() {
            ChatFamily::Llama3
        } else if tok.token_to_id("<|im_start|>").is_some() {
            ChatFamily::ChatMl
        } else if tok.token_to_id("[INST]").is_some() {
            ChatFamily::Mistral
        } else if tok.token_to_id("</s>").is_some() {
            // Llama-2 and pre-v0.3 Mistral (no [INST] control token) both
            // land here; the [INST]+<<SYS>> string format is the best fit.
            ChatFamily::Llama2
        } else {
            // Unknown vocab — Llama 3 headers were the previous
            // unconditional behavior, keep them as the default.
            ChatFamily::Llama3
        }
    }
}

/// Render the chat prompt with the goal prefix folded into the effective
/// system message. The GGUF-embedded chat template (minijinja) is
/// authoritative when present; the hand-built family templates below are
/// the fallback for template-less GGUFs and render failures.
fn render_chat_prompt(s: &AppState, messages: &[ChatMessage]) -> String {
    let prefix = s.goals.build_prompt_prefix();
    let mut sys = prefix.trim_end().to_string();
    let mut other_messages: Vec<&ChatMessage> = Vec::new();
    for m in messages {
        if m.role == "system" {
            if !sys.is_empty() {
                sys.push_str("\n\n");
            }
            sys.push_str(&m.content);
        } else {
            other_messages.push(m);
        }
    }

    // GGUF-embedded template first.
    let meta = s.chat_meta.read().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some(tpl) = &meta.template {
        let tok = s.tokenizer.read().unwrap_or_else(|e| e.into_inner());
        let bos = meta.bos_id.and_then(|id| tok.id_to_token(id)).unwrap_or_default();
        let eos = meta
            .stop_ids
            .first()
            .and_then(|&id| tok.id_to_token(id))
            .unwrap_or_default();
        drop(tok);
        let mut pairs: Vec<(String, String)> = Vec::new();
        if !sys.is_empty() {
            pairs.push(("system".to_string(), sys.clone()));
        }
        for m in &other_messages {
            pairs.push((m.role.clone(), m.content.clone()));
        }
        match crate::server::chat_template::render(tpl, &pairs, &bos, &eos) {
            Ok(p) => return p,
            Err(e) => tracing::warn!("{e}; falling back to family heuristics"),
        }
    }

    let family = ChatFamily::detect(&s.tokenizer.read().unwrap_or_else(|e| e.into_inner()));
    let mut out = String::new();
    match family {
        ChatFamily::Llama3 => {
            out.push_str("<|begin_of_text|>");
            if !sys.is_empty() {
                out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                out.push_str(&sys);
                out.push_str("<|eot_id|>");
            }
            for m in other_messages {
                out.push_str("<|start_header_id|>");
                out.push_str(&m.role);
                out.push_str("<|end_header_id|>\n\n");
                out.push_str(&m.content);
                out.push_str("<|eot_id|>");
            }
            out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        ChatFamily::ChatMl => {
            if !sys.is_empty() {
                out.push_str("<|im_start|>system\n");
                out.push_str(&sys);
                out.push_str("<|im_end|>\n");
            }
            for m in other_messages {
                out.push_str("<|im_start|>");
                out.push_str(&m.role);
                out.push('\n');
                out.push_str(&m.content);
                out.push_str("<|im_end|>\n");
            }
            out.push_str("<|im_start|>assistant\n");
        }
        ChatFamily::Mistral => {
            // <s> comes from tokenizer.encode(add_special_tokens=true).
            // System prompt: Mistral defines no system slot — prepend it
            // to the first user message, separated by a blank line.
            let mut pending_sys = if sys.is_empty() { None } else { Some(sys.clone()) };
            for m in other_messages {
                match m.role.as_str() {
                    "assistant" => {
                        out.push(' ');
                        out.push_str(&m.content);
                        out.push_str("</s>");
                    }
                    _ => {
                        out.push_str("[INST] ");
                        if let Some(sys) = pending_sys.take() {
                            out.push_str(&sys);
                            out.push_str("\n\n");
                        }
                        out.push_str(&m.content);
                        out.push_str(" [/INST]");
                    }
                }
            }
            if let Some(sys) = pending_sys {
                // System prompt but no user turn — emit it alone.
                out.push_str("[INST] ");
                out.push_str(&sys);
                out.push_str(" [/INST]");
            }
        }
        ChatFamily::Llama2 => {
            // BOS comes from tokenizer.encode(add_special_tokens=true);
            // assistant turns close with </s> per the Llama-2 format.
            let mut first_user = true;
            for m in other_messages {
                match m.role.as_str() {
                    "assistant" => {
                        out.push(' ');
                        out.push_str(&m.content);
                        out.push_str(" </s>");
                    }
                    _ => {
                        out.push_str("[INST] ");
                        if first_user && !sys.is_empty() {
                            out.push_str("<<SYS>>\n");
                            out.push_str(&sys);
                            out.push_str("\n<</SYS>>\n\n");
                        }
                        first_user = false;
                        out.push_str(&m.content);
                        out.push_str(" [/INST]");
                    }
                }
            }
            if first_user && !sys.is_empty() {
                // System prompt but no user turn — still emit it.
                out.push_str("[INST] <<SYS>>\n");
                out.push_str(&sys);
                out.push_str("\n<</SYS>>\n\n [/INST]");
            }
        }
    }
    out
}

/// Synchronous bypass-path generation. Returns
/// `(decoded_text, prompt_tok_count, completion_tok_count, finish_reason)`.
/// Holds the backend write lock for the full duration — fine for
/// single-user installed app, documented limitation.
///
/// Prefill runs through a `RunnerObserver` so every registered hook
/// (memory inject/capture, confidence updates, future early-exit /
/// hallucination hooks) fires per layer. Single-token continuations
/// skip the observer — running 32 hook firings per generated token
/// would be expensive and most hooks (capture, early-exit on
/// confidence) only need the prefill signal.
/// Self-contained spec-decode generation path. Called by
/// `generate_blocking` only when all conditions hold (draft model
/// loaded, spec-decode flag on, temperature 0, inspection off).
/// Same return contract as `generate_blocking`.
fn generate_spec_decode(
    s: &AppState,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    _sampler_cfg: &SamplerConfig,
    ctrl: &mut DecodeControl,
) -> (String, usize, usize, &'static str) {
    let prompt_len = prompt_tokens.len();
    let stops = stop_set(s);
    let mut all_tokens = prompt_tokens;
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut slot = acquire_slot(&s.pool);
    let BackendSlot { backend, prompt_cache, draft, draft_prompt_cache } = &mut *slot;
    let draft = draft.as_mut().expect("generate_spec_decode requires loaded draft");

    // Prefill MAIN (cache-aware).
    let main_prefill_start = prepare_prompt_cache(backend, prompt_cache, &all_tokens);
    let main_logits_res = backend.forward_logits(&all_tokens[main_prefill_start..]);
    let mut last_main_logit = match main_logits_res {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("spec-decode main prefill failed: {e}");
            return (String::new(), prompt_len, 0, "error");
        }
    };
    // Prefill DRAFT — its own cache; we need draft's KV synced to the
    // same token positions as main's.
    let draft_prefill_start = prepare_prompt_cache(draft, draft_prompt_cache, &all_tokens);
    let mut last_draft_logit = match draft.forward_logits(&all_tokens[draft_prefill_start..]) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("spec-decode draft prefill failed: {e}");
            return (String::new(), prompt_len, 0, "error");
        }
    };

    const SPEC_K: usize = 5;
    let finish_reason: &'static str = "stop";
    while generated_ids.len() < max_tokens {
        let iter = crate::engine::spec_decode::spec_iter(
            backend, draft, &last_main_logit, &last_draft_logit, SPEC_K,
        );
        let iter = match iter {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("spec-decode iter failed: {e}");
                break;
            }
        };
        crate::metrics::spec_decode_iters().inc();
        crate::metrics::spec_decode_accepted().inc_by(iter.accepted as u64);
        crate::metrics::spec_decode_rejected()
            .inc_by((iter.draft_proposed - iter.accepted) as u64);
        last_main_logit = iter.next_main_logit;
        last_draft_logit = iter.next_draft_logit;
        let mut hit_eos = false;
        for t in iter.committed {
            if stops.contains(&t) { hit_eos = true; break; }
            all_tokens.push(t);
            generated_ids.push(t);
            if generated_ids.len() >= max_tokens { break; }
        }
        // Stop-string check over the committed batch (spec runs greedy on
        // unadjusted logits; ctrl only contributes stops here).
        if !hit_eos && ctrl.has_stops() {
            let w = stop_window_text(s, &generated_ids, ctrl.window_tokens());
            if ctrl.stop_hit_in(&w) { hit_eos = true; }
        }
        if hit_eos { break; }
    }

    // Sync both caches.
    prompt_cache.clear();
    prompt_cache.extend_from_slice(&all_tokens);
    draft_prompt_cache.clear();
    draft_prompt_cache.extend_from_slice(&all_tokens);
    drop(slot);

    let mut text = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&generated_ids).unwrap_or_default();
    ctrl.truncate_at_stop(&mut text);
    (text, prompt_len, generated_ids.len(), finish_reason)
}

/// Whole-request generation on the resident iGPU engine: batched prefill
/// over the prompt (fills the GPU KV cache for positions 0..M) then decode
/// off that cache. Sampling/stop/detokenize match `generate_blocking`. The
/// caller holds the GpuModel lock for the duration (one resident KV cache,
/// so GPU requests serialize). Does NOT use the candle prefix cache or fire
/// inspection hooks — that's the explicit fast-lane trade. Same 4-tuple.
#[cfg(feature = "gpu")]
fn generate_gpu(
    s: &AppState,
    model: &crate::backend::gpu::GpuModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    sampler_cfg: &SamplerConfig,
    ctrl: &mut DecodeControl,
) -> (String, usize, usize, &'static str) {
    let prompt_len = prompt_tokens.len();
    let stops = stop_set(s);
    // Greedy decode (temperature 0) can use the GPU argmax path, which reads
    // back 4 bytes instead of the 128k-wide logit vector each token (~40% more
    // decode tok/s). Sampling — or any logit adjustment (penalties /
    // logit_bias) — needs the full logits on the CPU.
    let greedy = sampler_cfg.temperature == 0.0 && !ctrl.modifies_logits();
    ctrl.observe_prompt_tail(prompt_tokens, 64);
    // Prefill the whole prompt in one batched pass; returns the last token's
    // logits (the first sample) and leaves the KV cache filled for 0..M.
    let t_prefill = std::time::Instant::now();
    let first_logits = model.prefill_forward(prompt_tokens);
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    let mut generated: Vec<u32> = Vec::new();
    let mut finish_reason: &'static str = "length";
    let mut pos = prompt_len;
    let t_decode = std::time::Instant::now();
    let mut next = ctrl.sample_token(&first_logits, sampler_cfg);
    loop {
        if stops.contains(&next) {
            finish_reason = "stop";
            break;
        }
        generated.push(next);
        ctrl.observe(next);
        if ctrl.has_stops() {
            let w = stop_window_text(s, &generated, ctrl.window_tokens());
            if ctrl.stop_hit_in(&w) {
                finish_reason = "stop";
                break;
            }
        }
        if generated.len() >= max_tokens {
            break;
        }
        next = if greedy {
            model.forward_argmax(next, pos) // GPU argmax, 4-byte readback
        } else {
            ctrl.sample_token(&model.forward(next, pos), sampler_cfg)
        };
        pos += 1;
    }
    let dec_s = t_decode.elapsed().as_secs_f64();
    tracing::info!(
        "GPU fast-lane: prefill {prompt_len} tok in {prefill_ms:.0} ms ({:.0} tok/s), decoded {} tok at {:.0} tok/s",
        prompt_len as f64 / (prefill_ms / 1e3),
        generated.len(),
        generated.len() as f64 / dec_s.max(1e-6),
    );
    let mut text = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&generated).unwrap_or_default();
    if ctrl.truncate_at_stop(&mut text) {
        finish_reason = "stop";
    }
    (text, prompt_len, generated.len(), finish_reason)
}

/// Raw-Vulkan (ash) decode fast-lane. Mirrors `generate_gpu` but uses the
/// `VkModel` engine (validated bit-exact vs candle). Prefill is sequential
/// (one forward per prompt token filling the KV cache); decode uses the GPU
/// argmax path for greedy (4-byte readback) or full logits for sampling.
#[cfg(feature = "vulkan")]
fn generate_vk(
    s: &AppState,
    model: &crate::backend::vulkan::VkModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    sampler_cfg: &SamplerConfig,
    ctrl: &mut DecodeControl,
) -> (String, usize, usize, &'static str) {
    let prompt_len = prompt_tokens.len();
    let stops = stop_set(s);
    // GPU argmax readback only when nothing adjusts the distribution.
    let greedy = sampler_cfg.temperature == 0.0 && !ctrl.modifies_logits();
    ctrl.observe_prompt_tail(prompt_tokens, 64);
    let t_prefill = std::time::Instant::now();
    // Prefill with cross-request prefix reuse: K/V for the longest common prefix
    // with the previous request (system prompt, prior chat turns) is already
    // resident; only the suffix is computed (sequential when short, chunked
    // batched otherwise). Cold prompts take the plain batched path inside.
    let mut next = ctrl.sample_token(&model.prefill_cached(prompt_tokens), sampler_cfg);
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    let mut generated: Vec<u32> = Vec::new();
    let mut finish_reason: &'static str = "length";
    let mut pos = prompt_len;
    let t_decode = std::time::Instant::now();
    loop {
        if stops.contains(&next) { finish_reason = "stop"; break; }
        generated.push(next);
        ctrl.observe(next);
        if ctrl.has_stops() {
            let w = stop_window_text(s, &generated, ctrl.window_tokens());
            if ctrl.stop_hit_in(&w) { finish_reason = "stop"; break; }
        }
        if generated.len() >= max_tokens { break; }
        next = if greedy { model.forward_argmax(next, pos) } else { ctrl.sample_token(&model.forward(next, pos), sampler_cfg) };
        pos += 1;
    }
    // Extend the reusable prefix with the decoded tokens whose KV is resident:
    // every generated token was forwarded except the final one (max-tokens break
    // pushes without forwarding; eos stops before pushing).
    if !generated.is_empty() {
        model.note_decoded(&generated[..generated.len() - 1]);
    }
    let dec_s = t_decode.elapsed().as_secs_f64();
    tracing::info!(
        "Vulkan fast-lane: prefill {prompt_len} tok in {prefill_ms:.0} ms, decoded {} tok at {:.0} tok/s",
        generated.len(), generated.len() as f64 / dec_s.max(1e-6),
    );
    let mut text = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&generated).unwrap_or_default();
    if ctrl.truncate_at_stop(&mut text) {
        finish_reason = "stop";
    }
    (text, prompt_len, generated.len(), finish_reason)
}

fn generate_blocking(
    s: &AppState,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampler_cfg: &SamplerConfig,
    fsm: Option<&LogitFSM>,
    request_id: &str,
    mut detect: Option<&mut crate::engine::hallucination::Detector>,
    mut lp: Option<&mut crate::engine::logprobs::LogprobsCollector>,
    ctrl: &mut DecodeControl,
) -> (String, usize, usize, &'static str) {
    // Spec-decode fast path: redirect to the dedicated handler if every
    // precondition holds. Keeps the main generate_blocking unchanged.
    // Hallucination detection and logprobs collection force the candle path
    // (they need full per-token logits, one token per forward) — like
    // inspection, they disable the fast lanes. Penalties/logit_bias also
    // disable spec: drafts are verified against UNADJUSTED greedy rows.
    let inspect_on = s.inspection_enabled.load(Ordering::Relaxed);
    let spec_on = s.spec_decode_enabled.load(Ordering::Relaxed)
        && !inspect_on
        && detect.is_none()
        && lp.is_none()
        && sampler_cfg.temperature == 0.0
        && !ctrl.modifies_logits()
        && fsm.is_none();
    if spec_on {
        // Peek at slot 0 for a draft — if any slot is missing the draft
        // we conservatively fall back to the normal path.
        let has_draft = s.pool.iter().all(|m| {
            m.try_lock().map(|g| g.draft.is_some()).unwrap_or(true)
        });
        if has_draft {
            return generate_spec_decode(s, prompt_tokens, max_tokens, sampler_cfg, ctrl);
        }
    }
    // iGPU fast-lane: run the whole request on the resident GPU engine
    // (batched prefill + decode) when nothing CPU-only is active and the
    // prompt fits the batched-prefill cap. Returns the same 4-tuple. Mirrors
    // the spec-decode early return above; leaves the candle path untouched.
    #[cfg(feature = "gpu")]
    {
        let gpu_eligible = !inspect_on
            && detect.is_none()
            && lp.is_none()
            && !s.pld_enabled.load(Ordering::Relaxed)
            && !s.early_exit_enabled.load(Ordering::Relaxed)
            && fsm.is_none()
            && (1..=crate::backend::gpu::MAX_PREFILL_M).contains(&prompt_tokens.len());
        if gpu_eligible {
            if let Ok(guard) = s.gpu.lock() {
                if let Some(model) = guard.as_ref() {
                    return generate_gpu(s, model, &prompt_tokens, max_tokens, sampler_cfg, ctrl);
                }
            }
        }
    }
    // Raw-Vulkan decode fast-lane (same gate). Prompt bound is the resident
    // KV-cache capacity, not the prefill tile — longer prompts run as chunked
    // batched prefill inside prefill_forward. Two fit checks: the padded
    // prefill tiles (128-multiples) and the decode positions must both fit.
    #[cfg(feature = "vulkan")]
    {
        let plen = prompt_tokens.len();
        let vk_eligible = !inspect_on
            && detect.is_none()
            && lp.is_none()
            && !s.pld_enabled.load(Ordering::Relaxed)
            && !s.early_exit_enabled.load(Ordering::Relaxed)
            && fsm.is_none()
            && plen >= 1
            && plen <= crate::backend::vulkan::MAX_SEQ - 128 // = prefill_cap(): padded-tile headroom
            && plen + max_tokens < crate::backend::vulkan::MAX_SEQ;
        if vk_eligible {
            if let Ok(guard) = s.vk.lock() {
                if let Some(model) = guard.as_ref() {
                    return generate_vk(s, model, &prompt_tokens, max_tokens, sampler_cfg, ctrl);
                }
            }
        }
    }
    let prompt_len = prompt_tokens.len();
    let stops = stop_set(s);
    ctrl.observe_prompt_tail(&prompt_tokens, 64);
    let mut all_tokens = prompt_tokens;
    let mut generated_ids: Vec<u32> = Vec::new();
    // Acquire any free backend slot — falls back to round-robin block
    // if everything is busy. Each slot has its own KV cache + prompt
    // cache, so cache hits only happen when we land on the same slot
    // as the previous matching request (try-lock policy makes this the
    // common case for sequential traffic).
    let mut slot = acquire_slot(&s.pool);
    let BackendSlot { backend, prompt_cache, .. } = &mut *slot;
    // Hallucination detection needs REPRODUCIBLE logits: prefix-cache reuse
    // (truncate + re-append KV) shifts logits by numerical epsilons, which
    // measurably blurs the entropy/risk signal (live A/B: fresh server
    // discriminated 0.63-flagged vs 0.44; warm slots read 0.49 vs 0.48).
    // Force a cold full prefill when detecting; normal requests keep the cache.
    let prefill_start = if detect.is_some() {
        backend.reset_position();
        prompt_cache.clear();
        0
    } else {
        prepare_prompt_cache(backend, prompt_cache, &all_tokens)
    };
    let last_layer = backend.n_layers().saturating_sub(1);
    let inspect_on = s.inspection_enabled.load(Ordering::Relaxed);
    let observer = Arc::new(
        RunnerObserver::new(s.hooks.clone(), request_id.to_string())
            .with_inspection(inspect_on),
    );
    let mut finish_reason: &'static str = "stop";
    // Early-exit applies only to per-token (post-prefill) decode and
    // only when nothing fancy is active. Prefill always runs full.
    let early_exit_on = s.early_exit_enabled.load(Ordering::Relaxed)
        && !inspect_on
        && !s.pld_enabled.load(Ordering::Relaxed)
        && sampler_cfg.temperature == 0.0
        && fsm.is_none();
    let ee_min_layer = s.early_exit_min_layer.load(Ordering::Relaxed);
    let ee_threshold = f32::from_bits(s.early_exit_threshold_bits.load(Ordering::Relaxed));
    // Wall-clock cap: a wedged or absurdly slow generation frees its slot
    // instead of holding it forever (ZLLM_REQ_TIMEOUT_SECS, default 600).
    let req_deadline = std::time::Instant::now() + request_timeout();

    for _ in 0..max_tokens {
        if std::time::Instant::now() >= req_deadline {
            tracing::warn!("request wall-clock timeout — returning partial output");
            finish_reason = "length";
            break;
        }
        let is_prefill = generated_ids.is_empty();
        let input = if is_prefill {
            &all_tokens[prefill_start..]
        } else {
            &all_tokens[all_tokens.len() - 1..]
        };
        let logits_result = if is_prefill {
            let memory = s.memory.clone();
            let req_id = request_id.to_string();
            if inspect_on {
                let obs = observer.clone();
                backend.forward_logits_with_observer(input, move |layer_idx, hidden| {
                    let writeback = obs.on_layer(layer_idx, hidden);
                    if layer_idx == last_layer {
                        capture_prefill_to_memory(&memory, &req_id, layer_idx, hidden);
                    }
                    writeback
                })
            } else {
                backend.forward_logits_with_observer(input, move |layer_idx, hidden| {
                    if layer_idx == last_layer {
                        capture_prefill_to_memory(&memory, &req_id, layer_idx, hidden);
                    }
                    None
                })
            }
        } else if early_exit_on {
            // Per-token decode with confidence-driven early exit.
            // IPR is computed on the mean-pooled hidden vector — cheap
            // (no matmul) signal of "how concentrated is this state".
            // Returns (logits, exit_layer_idx).
            let ee_min = ee_min_layer;
            let ee_thr = ee_threshold;
            let res = backend.forward_logits_early_exit(input, move |layer_idx, hidden| {
                if layer_idx < ee_min { return false; }
                let pooled = match hidden.mean(1).and_then(|t| t.squeeze(0)).and_then(|t| t.to_vec1::<f32>()) {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                let conf = crate::engine::confidence::ConfidenceHead::estimate(&pooled);
                conf > ee_thr
            });
            match res {
                Ok((logits, exit_at)) => {
                    if exit_at < last_layer {
                        crate::metrics::early_exit_fires().inc();
                        crate::metrics::early_exit_layer_sum().inc_by(exit_at as u64);
                    } else {
                        crate::metrics::early_exit_full_forwards().inc();
                    }
                    Ok(logits)
                }
                Err(e) => Err(e),
            }
        } else {
            backend.forward_logits(input)
        };
        let mut logits = match logits_result {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("forward_logits failed: {e}");
                break;
            }
        };
        if is_prefill && inspect_on && observer.early_exit_signal.get() {
            crate::metrics::runner_early_exits().inc();
            finish_reason = "early_exit";
            break;
        }
        if let Some(fsm) = fsm {
            if fsm.is_active() {
                fsm.apply_mask(&mut logits);
            }
        }
        let next = ctrl.sample_token(&logits, sampler_cfg);
        // Stateful grammars (regex): feed the sampled token so the next step
        // masks from the advanced DFA state.
        if let Some(fsm) = fsm {
            if fsm.is_active() {
                fsm.advance(next);
            }
        }
        // Hallucination/uncertainty: observe the distribution `next` was drawn from
        // (post-grammar-mask, i.e. exactly what the model chose from).
        if let Some(d) = detect.as_deref_mut() {
            d.observe(&logits, next);
        }
        if let Some(c) = lp.as_deref_mut() {
            c.observe(&logits, next);
        }
        if inspect_on {
            let tok_text = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&[next]).unwrap_or_default();
            observer.record_token(generated_ids.len(), next, tok_text, &logits, 5);
        }
        if stops.contains(&next) {
            break;
        }
        all_tokens.push(next);
        generated_ids.push(next);
        ctrl.observe(next);
        if ctrl.has_stops() {
            let w = stop_window_text(s, &generated_ids, ctrl.window_tokens());
            if ctrl.stop_hit_in(&w) {
                break;
            }
        }

        // ── Prompt-lookup decoding (PLD) ──
        // Greedy-only fast path: when temperature=0 + PLD enabled, look
        // up an n-gram from the prompt and verify a draft against the
        // main model in one batched forward. Skipped when inspection is
        // on (observer pipeline assumes one-token-at-a-time semantics),
        // when a grammar is active (PLD commits draft tokens WITHOUT
        // masking — it would violate the constraint), and when sampling
        // is non-greedy (we'd need rejection sampling, out of scope).
        let pld_on = s.pld_enabled.load(Ordering::Relaxed)
            && !inspect_on
            && detect.is_none()
            && lp.is_none()
            && fsm.is_none()
            && !ctrl.modifies_logits() // drafts verify against unadjusted rows
            && sampler_cfg.temperature == 0.0;
        if pld_on && generated_ids.len() < max_tokens {
            const LOOKUP_LEN: usize = 2;
            const DRAFT_K: usize = 5;
            let draft = crate::engine::spec_decode::lookup_draft(
                &all_tokens, &all_tokens, LOOKUP_LEN, DRAFT_K,
            );
            if let Some(draft) = draft {
                crate::metrics::pld_draft_attempts().inc();
                let draft_len = draft.len();
                // Forward [next, draft_0, draft_1, ..., draft_K-1]
                // batched — we need per-position logits.
                let mut spec_input: Vec<u32> = Vec::with_capacity(1 + draft_len);
                spec_input.push(next);
                spec_input.extend_from_slice(&draft);
                // Roll back the just-pushed `next` from KV so the
                // single-token forward at top of next iter doesn't
                // double-commit it.
                // (Strategy: spec_input STARTS with `next`, replacing
                // the normal next-iteration forward of just `[next]`.)
                let pos_before_spec = backend.position();
                let multi = backend.forward_all_logits(&spec_input);
                let rows = match multi {
                    Ok(r) => r,
                    Err(e) => { tracing::warn!("PLD forward failed: {e}"); continue; }
                };
                // rows[0] predicts after `next` → first draft token check.
                let verify = crate::engine::spec_decode::verify_drafts(&draft, &rows);
                crate::metrics::pld_tokens_accepted().inc_by(verify.accepted as u64);
                crate::metrics::pld_tokens_rejected().inc_by((draft_len - verify.accepted) as u64);

                let mut early_eos = false;
                for d in &draft[..verify.accepted] {
                    if stops.contains(d) { early_eos = true; break; }
                    all_tokens.push(*d);
                    generated_ids.push(*d);
                    ctrl.observe(*d);
                    if generated_ids.len() >= max_tokens { break; }
                }
                // Stop strings over the batch-committed tokens.
                if !early_eos && ctrl.has_stops() {
                    let w = stop_window_text(s, &generated_ids, ctrl.window_tokens());
                    if ctrl.stop_hit_in(&w) { early_eos = true; }
                }
                // Truncate the KV cache to drop any rejected draft
                // tokens (they're in KV but not in our output). The
                // accepted ones stay.
                let keep = pos_before_spec + 1 + verify.accepted;
                if backend.truncate_to(keep).is_err() {
                    tracing::warn!("PLD KV truncate failed; resetting");
                    backend.reset_position();
                    prompt_cache.clear();
                }
                // Also emit the bonus/corrected token (always present).
                if !early_eos
                    && !stops.contains(&verify.bonus)
                    && generated_ids.len() < max_tokens
                {
                    all_tokens.push(verify.bonus);
                    generated_ids.push(verify.bonus);
                    ctrl.observe(verify.bonus);
                    // bonus is NOT in KV yet — next iteration's top-of-loop
                    // forward will commit it. So we *don't* update KV here;
                    // we rely on the next iteration to forward [bonus].
                    // Note: prepare_prompt_cache at next request will see
                    // KV-vs-all_tokens mismatch (KV missing the bonus), so
                    // we need to reflect this in prompt_cache too — we
                    // sync at the end of generate_blocking anyway.
                }
            }
        }
    }
    if inspect_on {
        if let Some(trace) = observer.take_inspection_trace() {
            if let Ok(mut store) = s.memory.write() {
                store.record_trace(trace);
            }
        }
    }
    // Sync this slot's prefix cache to whatever is now in its KV.
    prompt_cache.clear();
    prompt_cache.extend_from_slice(&all_tokens);
    drop(slot);
    let mut text = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&generated_ids).unwrap_or_default();
    ctrl.truncate_at_stop(&mut text);
    (text, prompt_len, generated_ids.len(), finish_reason)
}

/// Streaming generation via SSE. Spawns a blocking task that runs the
/// generation loop and pushes per-token chunks into a futures::channel,
/// which axum's Sse type consumes.
#[allow(clippy::too_many_arguments)]
fn chat_stream(
    s: AppState,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampler_cfg: SamplerConfig,
    fsm: Option<LogitFSM>,
    id: String,
    model_id: String,
    mut ctrl: DecodeControl,
    active: ActiveGuard,
) -> impl Stream<Item = Result<Event, Infallible>> {
    use futures::channel::mpsc;
    use futures::StreamExt;

    let (mut tx, rx) = mpsc::unbounded::<Result<Event, Infallible>>();
    let now = unix_secs();

    // Send the initial role chunk.
    let role_chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": null
        }]
    });
    let _ = tx.unbounded_send(Ok(Event::default().data(role_chunk.to_string())));

    tokio::task::spawn_blocking(move || {
        // Held for the stream's lifetime; drops (and frees the admission
        // slot) when generation ends OR the client disconnects.
        let _active = active;
        let stops = stop_set(&s);

        // iGPU fast-lane (streaming): same eligibility as generate_blocking's
        // GPU path. Streams tokens from the resident GPU engine over SSE, then
        // returns — bypassing the candle slot entirely. Greedy uses the 4-byte
        // argmax readback; sampling reads full logits.
        #[cfg(feature = "gpu")]
        {
            let inspect_on = s.inspection_enabled.load(Ordering::Relaxed);
            let gpu_eligible = !inspect_on
                && !s.pld_enabled.load(Ordering::Relaxed)
                && !s.early_exit_enabled.load(Ordering::Relaxed)
                && fsm.is_none()
                && (1..=crate::backend::gpu::MAX_PREFILL_M).contains(&prompt_tokens.len());
            if gpu_eligible {
                if let Ok(guard) = s.gpu.lock() {
                    if let Some(model) = guard.as_ref() {
                        let greedy = sampler_cfg.temperature == 0.0 && !ctrl.modifies_logits();
                        ctrl.observe_prompt_tail(&prompt_tokens, 64);
                        let prompt_len = prompt_tokens.len();
                        let t_prefill = std::time::Instant::now();
                        let first_logits = model.prefill_forward(&prompt_tokens);
                        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
                        let mut finish_reason: &'static str = "length";
                        let mut pos = prompt_len;
                        let mut gen_ids: Vec<u32> = Vec::new();
                        let mut next = ctrl.sample_token(&first_logits, &sampler_cfg);
                        let t_decode = std::time::Instant::now();
                        loop {
                            if stops.contains(&next) {
                                finish_reason = "stop";
                                break;
                            }
                            gen_ids.push(next);
                            ctrl.observe(next);
                            // Stop-string check BEFORE emitting the chunk so a
                            // completed stop sequence is never streamed.
                            if ctrl.has_stops() {
                                let w = stop_window_text(&s, &gen_ids, ctrl.window_tokens());
                                if ctrl.stop_hit_in(&w) {
                                    finish_reason = "stop";
                                    break;
                                }
                            }
                            let text = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&[next]).unwrap_or_default();
                            let chunk = json!({
                                "id": id, "object": "chat.completion.chunk",
                                "created": now, "model": model_id,
                                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                            });
                            if tx.unbounded_send(Ok(Event::default().data(chunk.to_string()))).is_err() {
                                break;
                            }
                            if gen_ids.len() >= max_tokens {
                                break;
                            }
                            next = if greedy {
                                model.forward_argmax(next, pos)
                            } else {
                                ctrl.sample_token(&model.forward(next, pos), &sampler_cfg)
                            };
                            pos += 1;
                        }
                        let generated = gen_ids.len();
                        let dec_s = t_decode.elapsed().as_secs_f64();
                        tracing::info!(
                            "GPU fast-lane (stream): prefill {prompt_len} tok in {prefill_ms:.0} ms ({:.0} tok/s), streamed {generated} tok at {:.0} tok/s",
                            prompt_len as f64 / (prefill_ms / 1e3),
                            generated as f64 / dec_s.max(1e-6),
                        );
                        let final_chunk = json!({
                            "id": id, "object": "chat.completion.chunk",
                            "created": now, "model": model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
                        });
                        let _ = tx.unbounded_send(Ok(Event::default().data(final_chunk.to_string())));
                        let _ = tx.unbounded_send(Ok(Event::default().data("[DONE]")));
                        tx.disconnect();
                        return;
                    }
                }
            }
        }

        // Raw-Vulkan fast-lane (streaming) — same gate, sequential prefill.
        #[cfg(feature = "vulkan")]
        {
            let inspect_on = s.inspection_enabled.load(Ordering::Relaxed);
            let vk_eligible = !inspect_on
                && !s.pld_enabled.load(Ordering::Relaxed)
                && !s.early_exit_enabled.load(Ordering::Relaxed)
                && fsm.is_none()
                && (1..=crate::backend::vulkan::MAX_PREFILL_M).contains(&prompt_tokens.len());
            if vk_eligible {
                if let Ok(guard) = s.vk.lock() {
                    if let Some(model) = guard.as_ref() {
                        let greedy = sampler_cfg.temperature == 0.0 && !ctrl.modifies_logits();
                        ctrl.observe_prompt_tail(&prompt_tokens, 64);
                        let prompt_len = prompt_tokens.len();
                        let t_decode = std::time::Instant::now();
                        let mut next = if prompt_len > 32 {
                            ctrl.sample_token(&model.prefill_forward(&prompt_tokens), &sampler_cfg)
                        } else {
                            for (i, &tk) in prompt_tokens[..prompt_len - 1].iter().enumerate() { model.prefill_step(tk, i); }
                            let last = prompt_tokens[prompt_len - 1];
                            if greedy { model.forward_argmax(last, prompt_len - 1) } else { ctrl.sample_token(&model.forward(last, prompt_len - 1), &sampler_cfg) }
                        };
                        let mut finish_reason: &'static str = "length";
                        let mut pos = prompt_len;
                        let mut gen_ids: Vec<u32> = Vec::new();
                        loop {
                            if stops.contains(&next) { finish_reason = "stop"; break; }
                            gen_ids.push(next);
                            ctrl.observe(next);
                            if ctrl.has_stops() {
                                let w = stop_window_text(&s, &gen_ids, ctrl.window_tokens());
                                if ctrl.stop_hit_in(&w) { finish_reason = "stop"; break; }
                            }
                            let text = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&[next]).unwrap_or_default();
                            let chunk = json!({
                                "id": id, "object": "chat.completion.chunk",
                                "created": now, "model": model_id,
                                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                            });
                            if tx.unbounded_send(Ok(Event::default().data(chunk.to_string()))).is_err() { break; }
                            if gen_ids.len() >= max_tokens { break; }
                            next = if greedy { model.forward_argmax(next, pos) } else { ctrl.sample_token(&model.forward(next, pos), &sampler_cfg) };
                            pos += 1;
                        }
                        let generated = gen_ids.len();
                        let dec_s = t_decode.elapsed().as_secs_f64();
                        tracing::info!("Vulkan fast-lane (stream): streamed {generated} tok at {:.0} tok/s", generated as f64 / dec_s.max(1e-6));
                        let final_chunk = json!({
                            "id": id, "object": "chat.completion.chunk",
                            "created": now, "model": model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
                        });
                        let _ = tx.unbounded_send(Ok(Event::default().data(final_chunk.to_string())));
                        let _ = tx.unbounded_send(Ok(Event::default().data("[DONE]")));
                        tx.disconnect();
                        return;
                    }
                }
            }
        }

        ctrl.observe_prompt_tail(&prompt_tokens, 64);
        let stream_prompt_len = prompt_tokens.len();
        let mut all_tokens = prompt_tokens;
        let mut slot = acquire_slot(&s.pool);
        let BackendSlot { backend, prompt_cache, .. } = &mut *slot;
        let prefill_start = prepare_prompt_cache(backend, prompt_cache, &all_tokens);
        let mut generated = 0usize;
        let last_layer = backend.n_layers().saturating_sub(1);
        let memory = s.memory.clone();
        let req_id = id.clone();
        let inspect_on = s.inspection_enabled.load(Ordering::Relaxed);
        let observer = Arc::new(
            RunnerObserver::new(s.hooks.clone(), req_id.clone())
                .with_inspection(inspect_on),
        );
        let mut finish_reason: &'static str = "stop";
        let req_deadline = std::time::Instant::now() + request_timeout();
        loop {
            if generated >= max_tokens {
                break;
            }
            if std::time::Instant::now() >= req_deadline {
                tracing::warn!("request wall-clock timeout — ending stream");
                finish_reason = "length";
                break;
            }
            let is_prefill = generated == 0;
            let input = if is_prefill {
                &all_tokens[prefill_start..]
            } else {
                &all_tokens[all_tokens.len() - 1..]
            };
            let logits_result = if is_prefill {
                let memory = memory.clone();
                let req_id = req_id.clone();
                if inspect_on {
                    let obs = observer.clone();
                    backend.forward_logits_with_observer(input, move |layer_idx, hidden| {
                        let writeback = obs.on_layer(layer_idx, hidden);
                        if layer_idx == last_layer {
                            capture_prefill_to_memory(&memory, &req_id, layer_idx, hidden);
                        }
                        writeback
                    })
                } else {
                    backend.forward_logits_with_observer(input, move |layer_idx, hidden| {
                        if layer_idx == last_layer {
                            capture_prefill_to_memory(&memory, &req_id, layer_idx, hidden);
                        }
                        None
                    })
                }
            } else {
                backend.forward_logits(input)
            };
            let mut logits = match logits_result {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("forward_logits failed: {e}");
                    break;
                }
            };
            if is_prefill && inspect_on && observer.early_exit_signal.get() {
                crate::metrics::runner_early_exits().inc();
                finish_reason = "early_exit";
                break;
            }
            if let Some(fsm) = &fsm {
                if fsm.is_active() {
                    fsm.apply_mask(&mut logits);
                }
            }
            let next = ctrl.sample_token(&logits, &sampler_cfg);
            // Stateful grammars: advance the DFA on the sampled token.
            if let Some(fsm) = &fsm {
                if fsm.is_active() {
                    fsm.advance(next);
                }
            }
            if inspect_on {
                let tok_text = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&[next]).unwrap_or_default();
                observer.record_token(generated, next, tok_text, &logits, 5);
            }
            if stops.contains(&next) {
                break;
            }
            all_tokens.push(next);
            generated += 1;
            ctrl.observe(next);
            // Stop-string check BEFORE emitting so a completed stop
            // sequence is never streamed to the client.
            if ctrl.has_stops() {
                let w = stop_window_text(&s, &all_tokens[stream_prompt_len..], ctrl.window_tokens());
                if ctrl.stop_hit_in(&w) {
                    break;
                }
            }
            // Helper: emit one token as an SSE delta chunk.
            let send_tok = |tok: u32, tx: &mut futures::channel::mpsc::UnboundedSender<Result<Event, Infallible>>| -> bool {
                if let Ok(text) = s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&[tok]) {
                    let chunk = json!({
                        "id": id, "object": "chat.completion.chunk",
                        "created": now, "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                    });
                    return tx.unbounded_send(Ok(Event::default().data(chunk.to_string()))).is_ok();
                }
                true
            };
            if !send_tok(next, &mut tx) { break; }

            // PLD fast-path (same logic as generate_blocking). fsm gate: PLD
            // commits draft tokens without masking — would violate a grammar.
            let pld_on = s.pld_enabled.load(Ordering::Relaxed)
                && !inspect_on
                && fsm.is_none()
                && !ctrl.modifies_logits() // drafts verify against unadjusted rows
                && sampler_cfg.temperature == 0.0;
            if pld_on && generated < max_tokens {
                const LOOKUP_LEN: usize = 2;
                const DRAFT_K: usize = 5;
                if let Some(draft) = crate::engine::spec_decode::lookup_draft(
                    &all_tokens, &all_tokens, LOOKUP_LEN, DRAFT_K,
                ) {
                    crate::metrics::pld_draft_attempts().inc();
                    let draft_len = draft.len();
                    let mut spec_input: Vec<u32> = Vec::with_capacity(1 + draft_len);
                    spec_input.push(next);
                    spec_input.extend_from_slice(&draft);
                    let pos_before = backend.position();
                    let rows = match backend.forward_all_logits(&spec_input) {
                        Ok(r) => r,
                        Err(e) => { tracing::warn!("PLD forward failed: {e}"); continue; }
                    };
                    let verify = crate::engine::spec_decode::verify_drafts(&draft, &rows);
                    crate::metrics::pld_tokens_accepted().inc_by(verify.accepted as u64);
                    crate::metrics::pld_tokens_rejected().inc_by((draft_len - verify.accepted) as u64);
                    let mut early_eos = false;
                    for d in &draft[..verify.accepted] {
                        if stops.contains(d) { early_eos = true; break; }
                        all_tokens.push(*d); generated += 1;
                        ctrl.observe(*d);
                        if ctrl.has_stops() {
                            let w = stop_window_text(&s, &all_tokens[stream_prompt_len..], ctrl.window_tokens());
                            if ctrl.stop_hit_in(&w) { early_eos = true; break; }
                        }
                        if !send_tok(*d, &mut tx) { early_eos = true; break; }
                        if generated >= max_tokens { break; }
                    }
                    let keep = pos_before + 1 + verify.accepted;
                    if backend.truncate_to(keep).is_err() {
                        tracing::warn!("PLD truncate failed; resetting");
                        backend.reset_position();
                        prompt_cache.clear();
                    }
                    if !early_eos
                        && !stops.contains(&verify.bonus)
                        && generated < max_tokens
                    {
                        all_tokens.push(verify.bonus); generated += 1;
                        ctrl.observe(verify.bonus);
                        let _ = send_tok(verify.bonus, &mut tx);
                    }
                }
            }
        }
        // Persist the per-request inspection trace (layers + tokens)
        // once decode is done. Skipped entirely when inspection is off.
        if inspect_on {
            if let Some(trace) = observer.take_inspection_trace() {
                if let Ok(mut store) = s.memory.write() {
                    store.record_trace(trace);
                }
            }
        }
        // Sync this slot's prefix cache (see generate_blocking).
        prompt_cache.clear();
        prompt_cache.extend_from_slice(&all_tokens);
        drop(slot);
        let final_chunk = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        });
        let _ = tx.unbounded_send(Ok(Event::default().data(final_chunk.to_string())));
        let _ = tx.unbounded_send(Ok(Event::default().data("[DONE]")));
        tx.disconnect();
    });

    rx.boxed()
}

// --- Goal CRUD ---

async fn get_state(State(s): State<AppState>) -> Json<Value> {
    let st = s.goals.get_state();
    let prefix = s.goals.build_prompt_prefix();
    Json(json!({
        "current_goal": st.current_goal.map(|g| json!({
            "goal_id": g.goal_id, "text": g.text, "is_current": g.is_current
        })),
        "active_tasks": st.active_tasks.into_iter().map(|t| json!({
            "task_id": t.task_id, "goal_id": t.goal_id, "text": t.text,
            "status": format!("{:?}", t.status).to_lowercase()
        })).collect::<Vec<_>>(),
        "latest_status": st.latest_status.map(|x| json!({
            "text": x.text, "goal_id": x.goal_id
        })),
        "prompt_prefix": prefix,
    }))
}

#[derive(Deserialize)]
struct SetGoalReq { text: String }

async fn set_goal(State(s): State<AppState>, Json(req): Json<SetGoalReq>) -> impl IntoResponse {
    if req.text.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error":"text required"}))).into_response();
    }
    let id = s.goals.set_goal(&req.text);
    Json(json!({"goal_id": id})).into_response()
}

async fn list_goals(State(s): State<AppState>) -> Json<Value> {
    let goals: Vec<Value> = s.goals.list_goals().into_iter().map(|g| json!({
        "goal_id": g.goal_id, "text": g.text, "is_current": g.is_current
    })).collect();
    Json(json!({"goals": goals}))
}

#[derive(Deserialize)]
struct SetCurrentReq { goal_id: String }

async fn set_current_goal(State(s): State<AppState>, Json(req): Json<SetCurrentReq>) -> Json<Value> {
    let success = s.goals.set_current(&req.goal_id);
    Json(json!({"success": success}))
}

#[derive(Deserialize)]
struct AddTaskReq { goal_id: String, text: String }

async fn add_task(State(s): State<AppState>, Json(req): Json<AddTaskReq>) -> impl IntoResponse {
    if req.goal_id.trim().is_empty() || req.text.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error":"goal_id and text required"}))).into_response();
    }
    let id = s.goals.add_task(&req.goal_id, &req.text);
    Json(json!({"task_id": id})).into_response()
}

#[derive(Deserialize)]
struct ListTasksQuery { goal_id: String }

async fn list_tasks(
    State(s): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<ListTasksQuery>,
) -> Json<Value> {
    let tasks: Vec<Value> = s.goals.list_tasks(&q.goal_id).into_iter().map(|t| json!({
        "task_id": t.task_id, "goal_id": t.goal_id, "text": t.text,
        "status": format!("{:?}", t.status).to_lowercase()
    })).collect();
    Json(json!({"tasks": tasks}))
}

#[derive(Deserialize)]
struct UpdateTaskReq { status: String }

async fn update_task(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateTaskReq>,
) -> Json<Value> {
    let status = match req.status.to_lowercase().as_str() {
        "active" => TaskStatus::Active,
        "done" => TaskStatus::Done,
        "blocked" => TaskStatus::Blocked,
        _ => return Json(json!({"success": false, "error": "invalid status"})),
    };
    let success = s.goals.update_task(&id, status);
    Json(json!({"success": success}))
}

fn trace_to_json(t: &crate::engine::memory_store::InspectionTrace) -> Value {
    let layers: Vec<Value> = t.layers.iter().map(|l| json!({
        "layer_idx": l.layer_idx,
        "loop_idx": l.loop_idx,
        "hidden_state_norm": l.hidden_state_norm,
        "hidden_state_hash": l.hidden_state_hash,
        "top_activations": l.top_activations.iter().map(|(i, v)| json!([i, v])).collect::<Vec<_>>(),
        "interpretation": l.interpretation,
    })).collect();
    let tokens: Vec<Value> = t.tokens.iter().map(|tk| json!({
        "step": tk.step,
        "token_id": tk.token_id,
        "token_text": tk.token_text,
        "confidence": tk.confidence,
        "top_alternatives": tk.top_alternatives.iter().map(|(i, p)| json!([i, p])).collect::<Vec<_>>(),
    })).collect();
    json!({
        "request_id": t.request_id,
        "timestamp": t.timestamp,
        "layers": layers,
        "tokens": tokens,
    })
}

async fn get_inspect_enabled(State(s): State<AppState>) -> Json<Value> {
    Json(json!({"enabled": s.inspection_enabled.load(Ordering::Relaxed)}))
}

#[derive(Deserialize)]
struct SetInspectReq { enabled: bool }

async fn set_inspect_enabled(
    State(s): State<AppState>,
    Json(req): Json<SetInspectReq>,
) -> Json<Value> {
    s.inspection_enabled.store(req.enabled, Ordering::Relaxed);
    Json(json!({"enabled": req.enabled}))
}

async fn get_pld_enabled(State(s): State<AppState>) -> Json<Value> {
    Json(json!({"enabled": s.pld_enabled.load(Ordering::Relaxed)}))
}

async fn set_pld_enabled(
    State(s): State<AppState>,
    Json(req): Json<SetInspectReq>,
) -> Json<Value> {
    s.pld_enabled.store(req.enabled, Ordering::Relaxed);
    Json(json!({"enabled": req.enabled}))
}

async fn get_spec_decode_enabled(State(s): State<AppState>) -> Json<Value> {
    Json(json!({"enabled": s.spec_decode_enabled.load(Ordering::Relaxed)}))
}

async fn set_spec_decode_enabled(
    State(s): State<AppState>,
    Json(req): Json<SetInspectReq>,
) -> Json<Value> {
    s.spec_decode_enabled.store(req.enabled, Ordering::Relaxed);
    Json(json!({"enabled": req.enabled}))
}

async fn get_early_exit_enabled(State(s): State<AppState>) -> Json<Value> {
    Json(json!({"enabled": s.early_exit_enabled.load(Ordering::Relaxed)}))
}
async fn set_early_exit_enabled(
    State(s): State<AppState>,
    Json(req): Json<SetInspectReq>,
) -> Json<Value> {
    s.early_exit_enabled.store(req.enabled, Ordering::Relaxed);
    Json(json!({"enabled": req.enabled}))
}

#[derive(Deserialize)]
struct EarlyExitConfigReq {
    min_layer: Option<usize>,
    threshold: Option<f32>,
}
async fn get_early_exit_config(State(s): State<AppState>) -> Json<Value> {
    Json(json!({
        "min_layer": s.early_exit_min_layer.load(Ordering::Relaxed),
        "threshold": f32::from_bits(s.early_exit_threshold_bits.load(Ordering::Relaxed)),
    }))
}
async fn set_early_exit_config(
    State(s): State<AppState>,
    Json(req): Json<EarlyExitConfigReq>,
) -> Json<Value> {
    if let Some(ml) = req.min_layer {
        s.early_exit_min_layer.store(ml, Ordering::Relaxed);
    }
    if let Some(t) = req.threshold {
        s.early_exit_threshold_bits.store(t.to_bits(), Ordering::Relaxed);
    }
    get_early_exit_config(State(s)).await
}

/// Return a JSON timing breakdown of the most recent forward pass.
/// Populated when zllm is built with `--features profile` and the
/// instrumentation in quantized_llama_fork.rs runs. Gives bucket
/// totals (attention, FFN, norm, LM head) instead of a flamegraph —
/// the deps for pure-Rust on-Windows sampling profilers are broken.
async fn pprof_flamegraph() -> impl IntoResponse {
    #[cfg(feature = "profile")]
    {
        let snap = crate::backend::candle::quantized_llama_fork::TIMING.snapshot();
        Json(json!({
            "n_forwards": snap.n_forwards,
            "total_ms": snap.total_ms,
            "attention_ms": snap.attention_ms,
            "ffn_ms": snap.ffn_ms,
            "norm_ms": snap.norm_ms,
            "lm_head_ms": snap.lm_head_ms,
            "qmm_attn_ms": snap.qmm_attn_ms,
            "qmm_ffn_ms": snap.qmm_ffn_ms,
            "qmm_lm_ms": snap.qmm_lm_ms,
            "qmm_calls": snap.qmm_calls,
            "per_forward_ms": if snap.n_forwards > 0 {
                snap.total_ms as f64 / snap.n_forwards as f64
            } else { 0.0 },
        })).into_response()
    }
    #[cfg(not(feature = "profile"))]
    {
        (StatusCode::NOT_FOUND, "rebuild zllm with --features profile to enable").into_response()
    }
}

/// Spike benchmark: compare two ways of computing the same matmul.
///
/// Path A — Q4_K_M aware: keep weights quantized, use Candle's
///   `QMatMul::forward` (the existing inference path).
/// Path B — dequant-and-go: dequantize the Q4_K_M weights to FP32
///   once, then do a standard FP matmul via `Tensor::matmul`.
///
/// Tests at typical Llama 3.2 1B FFN shape (hidden=2048,
/// intermediate=8192) for varying seq_len. If Path B is faster on
/// large seq_len, validates the dequant-and-BLAS plan for prefill.
///
/// **Note**: real integration would amortize the dequant cost across
/// multiple matmuls within a layer or cache the dequantized weights.
/// This spike reports both raw and amortized timings.
async fn matmul_bench() -> Json<Value> {
    use candle_core::{Device, Tensor, DType, Module};
    use candle_core::quantized::{QTensor, GgmlDType, QMatMul};
    use std::time::Instant;
    let device = Device::Cpu;
    let hidden = 2048usize;
    let intermediate = 8192usize;
    // Random weight matrix (intermediate, hidden), F32.
    let weights = match Tensor::randn(0.0f32, 1.0, (intermediate, hidden), &device) {
        Ok(t) => t,
        Err(e) => return Json(json!({"error": format!("randn: {e}")})),
    };
    // Quantize to Q4_K_M.
    let qtensor = match QTensor::quantize(&weights, GgmlDType::Q4K) {
        Ok(q) => q,
        Err(e) => return Json(json!({"error": format!("quantize: {e}")})),
    };
    let qtensor = std::sync::Arc::new(qtensor);
    let qmm = match QMatMul::from_arc(qtensor.clone()) {
        Ok(q) => q,
        Err(e) => return Json(json!({"error": format!("qmm: {e}")})),
    };

    // Dequant once (path B preamble — amortizable if cached).
    let dequant_start = Instant::now();
    let dequantized = match qtensor.dequantize(&device) {
        Ok(t) => match t.to_dtype(DType::F32) {
            Ok(t) => t,
            Err(e) => return Json(json!({"error": format!("to_dtype: {e}")})),
        },
        Err(e) => return Json(json!({"error": format!("dequant: {e}")})),
    };
    let dequant_ms = dequant_start.elapsed().as_secs_f64() * 1000.0;
    // For matmul x @ W.T we need (intermediate, hidden) → matmul against
    // (1, seq_len, hidden) producing (1, seq_len, intermediate).
    // Tensor::matmul does (M, K) @ (K, N). So transpose weights:
    let weights_t = match dequantized.t().and_then(|t| t.contiguous()) {
        Ok(t) => t,
        Err(e) => return Json(json!({"error": format!("transpose: {e}")})),
    };

    let mut results = Vec::new();
    for &seq_len in &[1usize, 16, 64, 256, 1024] {
        // Input: (1, seq_len, hidden) F32
        let x = match Tensor::randn(0.0f32, 1.0, (1, seq_len, hidden), &device) {
            Ok(t) => t,
            Err(e) => return Json(json!({"error": format!("x randn: {e}")})),
        };

        // Probe: actually run once and surface any error.
        let probe_a = qmm.forward(&x).and_then(|r| r.sum_all()).and_then(|r| r.to_scalar::<f32>());
        let probe_b = x.broadcast_matmul(&weights_t).and_then(|r| r.sum_all()).and_then(|r| r.to_scalar::<f32>());
        if let Err(e) = &probe_a {
            results.push(json!({"seq_len": seq_len, "qmatmul_error": format!("{e}")}));
            continue;
        }
        if let Err(e) = &probe_b {
            results.push(json!({"seq_len": seq_len, "fp_matmul_error": format!("{e}")}));
            continue;
        }

        let n_iters = if seq_len <= 64 { 20 } else if seq_len <= 256 { 5 } else { 3 };
        let t = Instant::now();
        for _ in 0..n_iters {
            let r = qmm.forward(&x).unwrap();
            let _ = r.sum_all().unwrap().to_scalar::<f32>().unwrap();
        }
        let a_ms = t.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;

        let t = Instant::now();
        for _ in 0..n_iters {
            let r = x.broadcast_matmul(&weights_t).unwrap();
            let _ = r.sum_all().unwrap().to_scalar::<f32>().unwrap();
        }
        let b_ms = t.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;

        results.push(json!({
            "seq_len": seq_len,
            "iters": n_iters,
            "qmatmul_ms": a_ms,
            "fp_matmul_ms": b_ms,
            "speedup": a_ms / b_ms,
        }));
    }

    Json(json!({
        "shape": {"hidden": hidden, "intermediate": intermediate, "weight_dtype": "Q4_K_M"},
        "dequant_once_ms": dequant_ms,
        "results": results,
        "interpretation": "qmatmul = current path; fp_matmul = dequantize-then-matmul. speedup > 1 means dequant-and-FP would be faster (if dequant cost is amortized).",
    }))
}

#[derive(Deserialize)]
struct LayerAgreementReq {
    prompt: String,
    #[serde(default = "default_la_tokens")]
    n_tokens: usize,
}
fn default_la_tokens() -> usize { 30 }

/// For a given prompt, generate N tokens with full forward, AND at
/// each token record the top-1 prediction at every layer (via
/// per-layer projection through final norm + LM head). Report:
///   agreement[k] = fraction of tokens where layer k's top-1 == layer
///   (n_layers - 1)'s top-1.
///
/// This is the data needed to answer "is zero-shot early exit viable
/// on this model?". If agreement[k] is high for some k < n-1, exit at
/// k saves compute with low quality risk. If low, early exit needs
/// trained per-layer heads.
async fn layer_agreement(
    State(s): State<AppState>,
    Json(req): Json<LayerAgreementReq>,
) -> impl IntoResponse {
    let tokens = match s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).encode(&req.prompt) {
        Ok(t) => t,
        Err(e) => return (StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("tokenize: {e}")}))).into_response(),
    };
    let n_tokens = req.n_tokens.clamp(1, 100);
    let mut slot = acquire_slot(&s.pool);
    let BackendSlot { backend, prompt_cache, .. } = &mut *slot;
    // Fresh KV for clean measurement.
    backend.reset_position();
    prompt_cache.clear();
    // Prefill — full forward, top1s discarded.
    if let Err(e) = backend.forward_logits(&tokens) {
        return (StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("prefill: {e}")}))).into_response();
    }
    let n_layers = backend.n_layers();
    let mut agreement = vec![0u32; n_layers];
    let mut last_token = *tokens.last().unwrap_or(&0);
    let sampler_cfg = crate::engine::sampler::SamplerConfig {
        temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0,
    };
    let mut generated: Vec<u32> = Vec::new();
    for _ in 0..n_tokens {
        let (logits, per_layer) = match backend.forward_per_layer_argmax(&[last_token]) {
            Ok(x) => x,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("decode: {e}")}))).into_response(),
        };
        let final_token = crate::engine::sampler::sample(&logits, &sampler_cfg);
        for (k, t) in per_layer.iter().enumerate() {
            if *t == final_token { agreement[k] += 1; }
        }
        generated.push(final_token);
        last_token = final_token;
        if stop_set(&s).contains(&final_token) { break; }
    }
    let n = generated.len() as f32;
    let pct: Vec<f64> = agreement.iter().map(|&c| 100.0 * c as f64 / n.max(1.0) as f64).collect();
    Json(json!({
        "n_tokens_measured": generated.len(),
        "n_layers": n_layers,
        "agreement_pct_per_layer": pct,
        "generated_preview": s.tokenizer.read().unwrap_or_else(|e| e.into_inner()).decode(&generated).unwrap_or_default().chars().take(120).collect::<String>(),
    })).into_response()
}

/// Snapshot of every runtime feature toggle + read-only metadata
/// (pool size, draft availability). Lets the settings UI render the
/// current state in one round-trip.
async fn get_settings(State(s): State<AppState>) -> Json<Value> {
    let draft_loaded = s.pool.iter().all(|m| {
        m.try_lock().map(|g| g.draft.is_some()).unwrap_or(true)
    });
    Json(json!({
        "inspection_enabled": s.inspection_enabled.load(Ordering::Relaxed),
        "pld_enabled": s.pld_enabled.load(Ordering::Relaxed),
        "spec_decode_enabled": s.spec_decode_enabled.load(Ordering::Relaxed),
        "pool_size": s.pool.len(),
        "draft_loaded": draft_loaded,
    }))
}

async fn list_traces(State(s): State<AppState>) -> Json<Value> {
    let store = s.memory.read().unwrap_or_else(|e| e.into_inner());
    let traces = store.get_traces(20);
    let summaries: Vec<Value> = traces.iter().map(|t| json!({
        "request_id": t.request_id,
        "timestamp": t.timestamp,
        "n_layers": t.layers.len(),
    })).collect();
    Json(json!({"traces": summaries}))
}

async fn get_trace(
    State(s): State<AppState>,
    Path(request_id): Path<String>,
) -> impl IntoResponse {
    let store = s.memory.read().unwrap_or_else(|e| e.into_inner());
    match store.get_trace_by_request(&request_id) {
        Some(t) => Json(trace_to_json(t)).into_response(),
        None => (StatusCode::NOT_FOUND, Json(json!({"error": "trace not found"}))).into_response(),
    }
}

#[derive(Deserialize)]
struct SetStatusReq { text: String }

async fn set_status(State(s): State<AppState>, Json(req): Json<SetStatusReq>) -> Json<Value> {
    s.goals.set_status(&req.text);
    Json(json!({"success": true}))
}

// --- Helpers ---

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt; // for `oneshot`

    // handle_panic maps every panic payload kind to a 500.
    #[test]
    fn handle_panic_maps_any_panic_to_500() {
        assert_eq!(handle_panic(Box::new("boom")).status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(handle_panic(Box::new(String::from("boom"))).status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(handle_panic(Box::new(42u8)).status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn response_format_maps_to_grammar() {
        // text / absent -> no constraint
        assert_eq!(grammar_from_response_format(&json!({"type": "text"})).unwrap(), None);
        assert_eq!(grammar_from_response_format(&json!({})).unwrap(), None);
        // json_object -> a JSON object specifically
        assert_eq!(
            grammar_from_response_format(&json!({"type": "json_object"})).unwrap(),
            Some("json_schema:{\"type\":\"object\"}".to_string())
        );
        // json_schema -> json_schema:<schema>
        let rf = json!({
            "type": "json_schema",
            "json_schema": {"name": "s", "schema": {"type": "boolean"}}
        });
        assert_eq!(
            grammar_from_response_format(&rf).unwrap(),
            Some("json_schema:{\"type\":\"boolean\"}".to_string())
        );
        // json_schema without a schema -> error
        assert!(grammar_from_response_format(&json!({"type": "json_schema"})).is_err());
        // unknown type -> error
        assert!(grammar_from_response_format(&json!({"type": "yaml"})).is_err());
    }

    #[test]
    fn resolve_constraint_precedence_and_conflict() {
        // neither
        assert_eq!(resolve_constraint(&None, &None).unwrap(), None);
        // grammar only
        assert_eq!(
            resolve_constraint(&Some("regex:a+".into()), &None).unwrap(),
            Some("regex:a+".into())
        );
        // response_format only
        assert_eq!(
            resolve_constraint(&None, &Some(json!({"type": "json_object"}))).unwrap(),
            Some("json_schema:{\"type\":\"object\"}".into())
        );
        // an empty grammar string is not a constraint — response_format wins
        assert_eq!(
            resolve_constraint(&Some("".into()), &Some(json!({"type": "json_object"}))).unwrap(),
            Some("json_schema:{\"type\":\"object\"}".into())
        );
        // both constraining -> conflict error
        assert!(resolve_constraint(
            &Some("regex:a+".into()),
            &Some(json!({"type": "json_object"}))
        )
        .is_err());
        // grammar + non-constraining response_format (text) -> grammar wins
        assert_eq!(
            resolve_constraint(&Some("regex:a+".into()), &Some(json!({"type": "text"}))).unwrap(),
            Some("regex:a+".into())
        );
    }

    // The catch-panic layer turns a panicking handler into a clean 500
    // instead of dropping the connection — the production resilience claim.
    #[tokio::test]
    async fn catch_panic_layer_turns_handler_panic_into_500() {
        async fn boom() -> axum::response::Response {
            panic!("kaboom")
        }
        let app = axum::Router::new()
            .route("/boom", axum::routing::get(boom))
            .layer(tower_http::catch_panic::CatchPanicLayer::custom(handle_panic));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/boom")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}

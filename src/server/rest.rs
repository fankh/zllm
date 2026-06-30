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
use crate::engine::hooks::registry::HookRegistry;
use crate::engine::logit_fsm::LogitFSM;
use crate::engine::memory_store::{MemoryCategory, MemoryMetadata, MemoryStore};
use crate::engine::runner_observer::RunnerObserver;
use crate::engine::sampler::{SamplerConfig, sample};

const CHAT_UI_HTML: &str = include_str!("chat_ui.html");

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
    pool[i].lock().expect("backend slot poisoned")
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
        .with_state(state)
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
            "latent_reasoning_runner",
            "logit_fsm_ban_only"
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
    arch.eq_ignore_ascii_case("llama")
}

async fn list_models(State(s): State<AppState>) -> Json<Value> {
    let now = unix_secs();
    let current = s.current_model.read().unwrap().clone();
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
    if !tok_path.exists() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("tokenizer.json not found next to {}", gguf_path.display())
            })),
        )
            .into_response();
    }

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
    // with an unloaded backend.
    let new_tok = match LlamaTokenizer::from_file(tok_path.to_str().unwrap_or("")) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("tokenizer load failed: {e}")})),
            )
                .into_response();
        }
    };

    // Acquire every slot up front so a chat-in-flight doesn't see a
    // half-swapped pool. Holds them simultaneously for the duration
    // of the reload — could take a few seconds per slot.
    let mut guards: Vec<_> = s
        .pool
        .iter()
        .map(|m| m.lock().expect("backend slot poisoned"))
        .collect();
    for g in guards.iter_mut() {
        let _ = g.backend.unload_model();
        if let Err(e) = g.backend.load_model(
            &gguf_path,
            &crate::backend::traits::QuantConfig {
                method: "gguf".into(),
                bits: 4,
            },
        ) {
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
                *s.gpu.lock().expect("gpu poisoned") = Some(m);
                tracing::info!("GPU engine reloaded for swapped model");
            }
            Err(e) => {
                *s.gpu.lock().expect("gpu poisoned") = None;
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
            Ok(m) => { *s.vk.lock().expect("vk poisoned") = Some(m); tracing::info!("Vulkan engine reloaded for swapped model"); }
            Err(e) => { *s.vk.lock().expect("vk poisoned") = None; tracing::warn!("Vulkan reload failed ({e}); fast-lane disabled for this model"); }
        }
    }

    *s.tokenizer.write().expect("tokenizer poisoned") = new_tok;
    *s.current_model.write().expect("current_model poisoned") = req.id.clone();

    // Selectively clear Context captures — they have an n_embd from
    // the previous model and would dilute injections done on the new
    // one. Goal / Task / Status entries are model-agnostic text-only
    // records (their vectors are zero-padded placeholders, not real
    // hidden states) so they survive the swap.
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

    tracing::info!("model swapped to {} ({})", req.id, gguf_path.display());
    Json(json!({"success": true, "current": req.id})).into_response()
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
    /// Optional RNG seed for reproducible sampling (continuous-batching lane).
    #[serde(default)]
    seed: Option<u32>,
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
    }
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
    let prompt = render_chat_prompt(&s.goals, &req.messages);
    let tokens = match s.tokenizer.read().unwrap().encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("tokenize: {e}")})),
            )
                .into_response();
        }
    };

    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let model_id = s.current_model.read().unwrap().clone();
    let max_tokens = req.max_tokens;
    let sampler_cfg = sampler_from_request(&s.engine, req.temperature, req.top_p, req.top_k);
    let fsm = req.grammar.as_deref().map(LogitFSM::new);

    // Continuous-batching fast lane (ZLLM_CB=1): route eligible chat requests
    // through the shared in-flight batcher (vLLM-style) instead of the candle
    // pool / single-stream GPU fast lanes. Eligible = inspection off and none of
    // the candle-only features (grammar / spec-decode / PLD / early-exit) are on,
    // since the CB engine doesn't implement those. Greedy or temp/top-k/top-p.
    #[cfg(feature = "gpu")]
    if let Some(server) = cb_chat_server(&s, fsm.is_none()) {
        let prompt_tokens = tokens.len();
        let temp = sampler_cfg.temperature;
        let params = if temp <= 0.0 {
            crate::backend::gpu::SamplingParams::greedy()
        } else {
            crate::backend::gpu::SamplingParams { temp, top_k: sampler_cfg.top_k as u32, top_p: sampler_cfg.top_p }
        };
        let seed = req.seed.unwrap_or(0);
        let eos = s.tokenizer.read().unwrap().eos_token_id().unwrap_or(128001);
        let stop_eot = 128009u32;
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
        let stream = chat_stream(s.clone(), tokens, max_tokens, sampler_cfg, fsm, id.clone(), model_id);
        Sse::new(stream).into_response()
    } else {
        let mut detector = req.detect_hallucination.unwrap_or(false)
            .then(|| crate::engine::hallucination::Detector::new(Default::default()));
        let (text, prompt_tokens, completion_tokens, finish_reason) =
            generate_blocking(&s, tokens, max_tokens, &sampler_cfg, fsm.as_ref(), &id, detector.as_mut());
        let hallu = detector.map(|d| hallucination_json(&d.report()));
        let now = unix_secs();
        Json(json!({
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
            },
            "hallucination": hallu
        }))
        .into_response()
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
    grammar: Option<String>,
    /// Attach an output-distribution hallucination/uncertainty report to the
    /// response. Forces the candle path (full per-token logits) like inspection.
    #[serde(default)]
    detect_hallucination: Option<bool>,
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
    let tokens = match s.tokenizer.read().unwrap().encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("tokenize: {e}")})),
            )
                .into_response();
        }
    };
    let id = format!("cmpl-{}", Uuid::new_v4());
    let sampler_cfg = sampler_from_request(&s.engine, req.temperature, req.top_p, req.top_k);
    let fsm = req.grammar.as_deref().map(LogitFSM::new);
    let mut detector = req.detect_hallucination.unwrap_or(false)
        .then(|| crate::engine::hallucination::Detector::new(Default::default()));
    let (text, p, c, finish_reason) = generate_blocking(&s, tokens, req.max_tokens, &sampler_cfg, fsm.as_ref(), &id, detector.as_mut());
    let hallu = detector.map(|d| hallucination_json(&d.report()));
    let now = unix_secs();
    Json(json!({
        "id": id,
        "object": "text_completion",
        "created": now,
        "model": s.current_model.read().unwrap().clone(),
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": p + c
        },
        "hallucination": hallu
    }))
    .into_response()
}

#[derive(Deserialize)]
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
    seed: Option<u32>,
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
        let tokens = match s.tokenizer.read().unwrap().encode(&req.prompt) {
            Ok(t) => t,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("tokenize: {e}")}))).into_response(),
        };
        let eos = s.tokenizer.read().unwrap().eos_token_id().unwrap_or(128001);
        let stop_eot = 128009u32; // Llama 3.2 <|eot_id|> — the chat-turn stop token
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
        let model_id = s.current_model.read().unwrap().clone();

        if req.stream {
            use futures::channel::mpsc;
            let id = format!("cmpl-{}", Uuid::new_v4());
            let (tx, rx) = mpsc::unbounded::<Result<Event, Infallible>>();
            tokio::spawn(async move {
                let now = unix_secs();
                while let Some(item) = tok_rx.recv().await {
                    let t = match item { Some(t) => t, None => break }; // None = done sentinel
                    if t == eos || t == stop_eot { break; }
                    let text = tok.read().unwrap().decode(&[t]).unwrap_or_default();
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
            let text = tok.read().unwrap().decode(&out_ids).unwrap_or_default();
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
            let text = s.tokenizer.read().unwrap().decode(&[t]).unwrap_or_default();
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
    let text = s.tokenizer.read().unwrap().decode(&ids).unwrap_or_default();
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

/// Llama 3 instruct chat template with the goal prefix folded into the
/// effective system message. Hand-built — getting the special tokens right
/// matters for output quality, but tokenizer.encode handles them.
fn render_chat_prompt(goals: &GoalManager, messages: &[ChatMessage]) -> String {
    let prefix = goals.build_prompt_prefix();
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

    let mut out = String::from("<|begin_of_text|>");
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
    sampler_cfg: &SamplerConfig,
) -> (String, usize, usize, &'static str) {
    let prompt_len = prompt_tokens.len();
    let eos = s.tokenizer.read().unwrap().eos_token_id().unwrap_or(128001);
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
    let mut finish_reason: &'static str = "stop";
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
            if t == eos || t == 128009 { hit_eos = true; break; }
            all_tokens.push(t);
            generated_ids.push(t);
            if generated_ids.len() >= max_tokens { break; }
        }
        if hit_eos { break; }
    }

    // Sync both caches.
    prompt_cache.clear();
    prompt_cache.extend_from_slice(&all_tokens);
    draft_prompt_cache.clear();
    draft_prompt_cache.extend_from_slice(&all_tokens);
    drop(slot);

    let text = s.tokenizer.read().unwrap().decode(&generated_ids).unwrap_or_default();
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
) -> (String, usize, usize, &'static str) {
    let prompt_len = prompt_tokens.len();
    let eos = s.tokenizer.read().unwrap().eos_token_id().unwrap_or(128001);
    // Greedy decode (temperature 0) can use the GPU argmax path, which reads
    // back 4 bytes instead of the 128k-wide logit vector each token (~40% more
    // decode tok/s). Sampling needs the full logits on the CPU.
    let greedy = sampler_cfg.temperature == 0.0;
    // Prefill the whole prompt in one batched pass; returns the last token's
    // logits (the first sample) and leaves the KV cache filled for 0..M.
    let t_prefill = std::time::Instant::now();
    let first_logits = model.prefill_forward(prompt_tokens);
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    let mut generated: Vec<u32> = Vec::new();
    let mut finish_reason: &'static str = "length";
    let mut pos = prompt_len;
    let t_decode = std::time::Instant::now();
    let mut next = sample(&first_logits, sampler_cfg);
    loop {
        if next == eos || next == 128009 {
            finish_reason = "stop";
            break;
        }
        generated.push(next);
        if generated.len() >= max_tokens {
            break;
        }
        next = if greedy {
            model.forward_argmax(next, pos) // GPU argmax, 4-byte readback
        } else {
            sample(&model.forward(next, pos), sampler_cfg)
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
    let text = s.tokenizer.read().unwrap().decode(&generated).unwrap_or_default();
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
) -> (String, usize, usize, &'static str) {
    let prompt_len = prompt_tokens.len();
    let eos = s.tokenizer.read().unwrap().eos_token_id().unwrap_or(128001);
    let greedy = sampler_cfg.temperature == 0.0;
    let t_prefill = std::time::Instant::now();
    // Batched prefill (one coopmat-GEMM pass) wins for longer prompts; sequential
    // is cheaper for short ones (batched always processes a padded 128 rows).
    let mut next = if prompt_len > 32 {
        sample(&model.prefill_forward(prompt_tokens), sampler_cfg)
    } else {
        for (i, &tk) in prompt_tokens[..prompt_len - 1].iter().enumerate() { model.prefill_step(tk, i); }
        let last = prompt_tokens[prompt_len - 1];
        if greedy { model.forward_argmax(last, prompt_len - 1) } else { sample(&model.forward(last, prompt_len - 1), sampler_cfg) }
    };
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    let mut generated: Vec<u32> = Vec::new();
    let mut finish_reason: &'static str = "length";
    let mut pos = prompt_len;
    let t_decode = std::time::Instant::now();
    loop {
        if next == eos || next == 128009 { finish_reason = "stop"; break; }
        generated.push(next);
        if generated.len() >= max_tokens { break; }
        next = if greedy { model.forward_argmax(next, pos) } else { sample(&model.forward(next, pos), sampler_cfg) };
        pos += 1;
    }
    let dec_s = t_decode.elapsed().as_secs_f64();
    tracing::info!(
        "Vulkan fast-lane: prefill {prompt_len} tok in {prefill_ms:.0} ms, decoded {} tok at {:.0} tok/s",
        generated.len(), generated.len() as f64 / dec_s.max(1e-6),
    );
    let text = s.tokenizer.read().unwrap().decode(&generated).unwrap_or_default();
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
) -> (String, usize, usize, &'static str) {
    // Spec-decode fast path: redirect to the dedicated handler if every
    // precondition holds. Keeps the main generate_blocking unchanged.
    // Hallucination detection forces the candle path (it needs full per-token
    // logits, one token per forward) — like inspection, it disables the fast lanes.
    let inspect_on = s.inspection_enabled.load(Ordering::Relaxed);
    let spec_on = s.spec_decode_enabled.load(Ordering::Relaxed)
        && !inspect_on
        && detect.is_none()
        && sampler_cfg.temperature == 0.0
        && fsm.is_none();
    if spec_on {
        // Peek at slot 0 for a draft — if any slot is missing the draft
        // we conservatively fall back to the normal path.
        let has_draft = s.pool.iter().all(|m| {
            m.try_lock().map(|g| g.draft.is_some()).unwrap_or(true)
        });
        if has_draft {
            return generate_spec_decode(s, prompt_tokens, max_tokens, sampler_cfg);
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
            && !s.pld_enabled.load(Ordering::Relaxed)
            && !s.early_exit_enabled.load(Ordering::Relaxed)
            && fsm.is_none()
            && (1..=crate::backend::gpu::MAX_PREFILL_M).contains(&prompt_tokens.len());
        if gpu_eligible {
            if let Ok(guard) = s.gpu.lock() {
                if let Some(model) = guard.as_ref() {
                    return generate_gpu(s, model, &prompt_tokens, max_tokens, sampler_cfg);
                }
            }
        }
    }
    // Raw-Vulkan decode fast-lane (same gate, prompt cap is the VkModel's).
    #[cfg(feature = "vulkan")]
    {
        let vk_eligible = !inspect_on
            && detect.is_none()
            && !s.pld_enabled.load(Ordering::Relaxed)
            && !s.early_exit_enabled.load(Ordering::Relaxed)
            && fsm.is_none()
            && (1..=crate::backend::vulkan::MAX_PREFILL_M).contains(&prompt_tokens.len());
        if vk_eligible {
            if let Ok(guard) = s.vk.lock() {
                if let Some(model) = guard.as_ref() {
                    return generate_vk(s, model, &prompt_tokens, max_tokens, sampler_cfg);
                }
            }
        }
    }
    let prompt_len = prompt_tokens.len();
    let eos = s.tokenizer.read().unwrap().eos_token_id().unwrap_or(128001);
    let mut all_tokens = prompt_tokens;
    let mut generated_ids: Vec<u32> = Vec::new();
    // Acquire any free backend slot — falls back to round-robin block
    // if everything is busy. Each slot has its own KV cache + prompt
    // cache, so cache hits only happen when we land on the same slot
    // as the previous matching request (try-lock policy makes this the
    // common case for sequential traffic).
    let mut slot = acquire_slot(&s.pool);
    let BackendSlot { backend, prompt_cache, .. } = &mut *slot;
    let prefill_start = prepare_prompt_cache(backend, prompt_cache, &all_tokens);
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

    for _ in 0..max_tokens {
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
        let next = sample(&logits, sampler_cfg);
        // Hallucination/uncertainty: observe the distribution `next` was drawn from
        // (post-grammar-mask, i.e. exactly what the model chose from).
        if let Some(d) = detect.as_deref_mut() {
            d.observe(&logits, next);
        }
        if inspect_on {
            let tok_text = s.tokenizer.read().unwrap().decode(&[next]).unwrap_or_default();
            observer.record_token(generated_ids.len(), next, tok_text, &logits, 5);
        }
        if next == eos || next == 128009 {
            break;
        }
        all_tokens.push(next);
        generated_ids.push(next);

        // ── Prompt-lookup decoding (PLD) ──
        // Greedy-only fast path: when temperature=0 + PLD enabled, look
        // up an n-gram from the prompt and verify a draft against the
        // main model in one batched forward. Skipped when inspection is
        // on (observer pipeline assumes one-token-at-a-time semantics)
        // and when sampling is non-greedy (we'd need rejection
        // sampling, which is out of scope for this version).
        let pld_on = s.pld_enabled.load(Ordering::Relaxed)
            && !inspect_on
            && detect.is_none()
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
                    if *d == eos || *d == 128009 { early_eos = true; break; }
                    all_tokens.push(*d);
                    generated_ids.push(*d);
                    if generated_ids.len() >= max_tokens { break; }
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
                    && verify.bonus != eos
                    && verify.bonus != 128009
                    && generated_ids.len() < max_tokens
                {
                    all_tokens.push(verify.bonus);
                    generated_ids.push(verify.bonus);
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
    let text = s.tokenizer.read().unwrap().decode(&generated_ids).unwrap_or_default();
    (text, prompt_len, generated_ids.len(), finish_reason)
}

/// Streaming generation via SSE. Spawns a blocking task that runs the
/// generation loop and pushes per-token chunks into a futures::channel,
/// which axum's Sse type consumes.
fn chat_stream(
    s: AppState,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampler_cfg: SamplerConfig,
    fsm: Option<LogitFSM>,
    id: String,
    model_id: String,
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
        let eos = s.tokenizer.read().unwrap().eos_token_id().unwrap_or(128001);

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
                        let greedy = sampler_cfg.temperature == 0.0;
                        let prompt_len = prompt_tokens.len();
                        let t_prefill = std::time::Instant::now();
                        let first_logits = model.prefill_forward(&prompt_tokens);
                        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
                        let mut finish_reason: &'static str = "length";
                        let mut pos = prompt_len;
                        let mut generated = 0usize;
                        let mut next = sample(&first_logits, &sampler_cfg);
                        let t_decode = std::time::Instant::now();
                        loop {
                            if next == eos || next == 128009 {
                                finish_reason = "stop";
                                break;
                            }
                            let text = s.tokenizer.read().unwrap().decode(&[next]).unwrap_or_default();
                            let chunk = json!({
                                "id": id, "object": "chat.completion.chunk",
                                "created": now, "model": model_id,
                                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                            });
                            if tx.unbounded_send(Ok(Event::default().data(chunk.to_string()))).is_err() {
                                break;
                            }
                            generated += 1;
                            if generated >= max_tokens {
                                break;
                            }
                            next = if greedy {
                                model.forward_argmax(next, pos)
                            } else {
                                sample(&model.forward(next, pos), &sampler_cfg)
                            };
                            pos += 1;
                        }
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
                        let greedy = sampler_cfg.temperature == 0.0;
                        let prompt_len = prompt_tokens.len();
                        let t_decode = std::time::Instant::now();
                        let mut next = if prompt_len > 32 {
                            sample(&model.prefill_forward(&prompt_tokens), &sampler_cfg)
                        } else {
                            for (i, &tk) in prompt_tokens[..prompt_len - 1].iter().enumerate() { model.prefill_step(tk, i); }
                            let last = prompt_tokens[prompt_len - 1];
                            if greedy { model.forward_argmax(last, prompt_len - 1) } else { sample(&model.forward(last, prompt_len - 1), &sampler_cfg) }
                        };
                        let mut finish_reason: &'static str = "length";
                        let mut pos = prompt_len;
                        let mut generated = 0usize;
                        loop {
                            if next == eos || next == 128009 { finish_reason = "stop"; break; }
                            let text = s.tokenizer.read().unwrap().decode(&[next]).unwrap_or_default();
                            let chunk = json!({
                                "id": id, "object": "chat.completion.chunk",
                                "created": now, "model": model_id,
                                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                            });
                            if tx.unbounded_send(Ok(Event::default().data(chunk.to_string()))).is_err() { break; }
                            generated += 1;
                            if generated >= max_tokens { break; }
                            next = if greedy { model.forward_argmax(next, pos) } else { sample(&model.forward(next, pos), &sampler_cfg) };
                            pos += 1;
                        }
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
        loop {
            if generated >= max_tokens {
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
            let next = sample(&logits, &sampler_cfg);
            if inspect_on {
                let tok_text = s.tokenizer.read().unwrap().decode(&[next]).unwrap_or_default();
                observer.record_token(generated, next, tok_text, &logits, 5);
            }
            if next == eos || next == 128009 {
                break;
            }
            all_tokens.push(next);
            generated += 1;
            // Helper: emit one token as an SSE delta chunk.
            let send_tok = |tok: u32, tx: &mut futures::channel::mpsc::UnboundedSender<Result<Event, Infallible>>| -> bool {
                if let Ok(text) = s.tokenizer.read().unwrap().decode(&[tok]) {
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

            // PLD fast-path (same logic as generate_blocking).
            let pld_on = s.pld_enabled.load(Ordering::Relaxed)
                && !inspect_on
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
                        if *d == eos || *d == 128009 { early_eos = true; break; }
                        all_tokens.push(*d); generated += 1;
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
                        && verify.bonus != eos
                        && verify.bonus != 128009
                        && generated < max_tokens
                    {
                        all_tokens.push(verify.bonus); generated += 1;
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
    let tokens = match s.tokenizer.read().unwrap().encode(&req.prompt) {
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
        temperature: 0.0, top_k: 0, top_p: 1.0,
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
        if final_token == s.tokenizer.read().unwrap().eos_token_id().unwrap_or(128001) { break; }
    }
    let n = generated.len() as f32;
    let pct: Vec<f64> = agreement.iter().map(|&c| 100.0 * c as f64 / n.max(1.0) as f64).collect();
    Json(json!({
        "n_tokens_measured": generated.len(),
        "n_layers": n_layers,
        "agreement_pct_per_layer": pct,
        "generated_preview": s.tokenizer.read().unwrap().decode(&generated).unwrap_or_default().chars().take(120).collect::<String>(),
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
    let store = s.memory.read().unwrap();
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
    let store = s.memory.read().unwrap();
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

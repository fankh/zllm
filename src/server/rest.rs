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
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::backend::candle::backend::CandleCpuBackend;
use crate::backend::candle::tokenizer::LlamaTokenizer;
use crate::config::EngineConfig;
use crate::control_plane::goal_manager::{GoalManager, TaskStatus};
use crate::engine::memory_store::MemoryStore;
use crate::engine::sampler::{SamplerConfig, sample};

const CHAT_UI_HTML: &str = include_str!("chat_ui.html");

#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<RwLock<CandleCpuBackend>>,
    pub tokenizer: Arc<LlamaTokenizer>,
    pub goals: Arc<GoalManager>,
    pub memory: Arc<RwLock<MemoryStore>>,
    pub engine: Arc<EngineConfig>,
    pub model_id: String,
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
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(text_completions))
        // Goal CRUD
        .route("/v1/goal/state", get(get_state))
        .route("/v1/goal/set", post(set_goal))
        .route("/v1/goal/list", get(list_goals))
        .route("/v1/goal/current", post(set_current_goal))
        .route("/v1/goal/task", post(add_task))
        .route("/v1/goal/task/{id}", patch(update_task))
        .route("/v1/goal/status", post(set_status))
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

async fn info() -> Json<Value> {
    Json(json!({
        "name": "ZLLM",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "White-box LLM inference engine — installed-app, REST + chat UI",
        "features": [
            "openai_compat_chat",
            "goal_manager",
            "memory_store",
            "latent_reasoning_runner",
            "logit_fsm",
            "paged_kv_cache"
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

async fn list_models(State(s): State<AppState>) -> Json<Value> {
    let now = unix_secs();
    Json(json!({
        "object": "list",
        "data": [{
            "id": s.model_id,
            "object": "model",
            "created": now,
            "owned_by": "local"
        }]
    }))
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
    let tokens = match s.tokenizer.encode(&prompt) {
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
    let model_id = s.model_id.clone();
    let max_tokens = req.max_tokens;
    let sampler_cfg = sampler_from_request(&s.engine, req.temperature, req.top_p, req.top_k);

    if req.stream {
        let stream = chat_stream(s.clone(), tokens, max_tokens, sampler_cfg, id, model_id);
        Sse::new(stream).into_response()
    } else {
        let (text, prompt_tokens, completion_tokens) =
            generate_blocking(&s, tokens, max_tokens, &sampler_cfg);
        let now = unix_secs();
        Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": now,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
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
}

async fn text_completions(
    State(s): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let prefix = s.goals.build_prompt_prefix();
    let prompt = format!("{prefix}{}", req.prompt);
    let tokens = match s.tokenizer.encode(&prompt) {
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
    let (text, p, c) = generate_blocking(&s, tokens, req.max_tokens, &sampler_cfg);
    let now = unix_secs();
    Json(json!({
        "id": id,
        "object": "text_completion",
        "created": now,
        "model": s.model_id,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": p + c
        }
    }))
    .into_response()
}

// --- Generation ---

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

/// Synchronous bypass-path generation. Returns (decoded_text, prompt_tok_count,
/// completion_tok_count). Holds the backend write lock for the full duration —
/// fine for single-user installed app, documented limitation.
fn generate_blocking(
    s: &AppState,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampler_cfg: &SamplerConfig,
) -> (String, usize, usize) {
    let prompt_len = prompt_tokens.len();
    let eos = s.tokenizer.eos_token_id().unwrap_or(128001);
    let mut all_tokens = prompt_tokens;
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut backend = s.backend.write().expect("backend poisoned");
    for _ in 0..max_tokens {
        let input = if generated_ids.is_empty() {
            &all_tokens[..]
        } else {
            &all_tokens[all_tokens.len() - 1..]
        };
        let logits = match backend.forward_logits(input) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("forward_logits failed: {e}");
                break;
            }
        };
        let next = sample(&logits, sampler_cfg);
        if next == eos || next == 128009 {
            break;
        }
        all_tokens.push(next);
        generated_ids.push(next);
    }
    let text = s.tokenizer.decode(&generated_ids).unwrap_or_default();
    (text, prompt_len, generated_ids.len())
}

/// Streaming generation via SSE. Spawns a blocking task that runs the
/// generation loop and pushes per-token chunks into a futures::channel,
/// which axum's Sse type consumes.
fn chat_stream(
    s: AppState,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampler_cfg: SamplerConfig,
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
        let eos = s.tokenizer.eos_token_id().unwrap_or(128001);
        let mut all_tokens = prompt_tokens;
        let mut backend = s.backend.write().expect("backend poisoned");
        let mut generated = 0usize;
        loop {
            if generated >= max_tokens {
                break;
            }
            let input = if generated == 0 {
                &all_tokens[..]
            } else {
                &all_tokens[all_tokens.len() - 1..]
            };
            let logits = match backend.forward_logits(input) {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("forward_logits failed: {e}");
                    break;
                }
            };
            let next = sample(&logits, &sampler_cfg);
            if next == eos || next == 128009 {
                break;
            }
            all_tokens.push(next);
            generated += 1;
            if let Ok(text) = s.tokenizer.decode(&[next]) {
                let chunk = json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": null
                    }]
                });
                if tx.unbounded_send(Ok(Event::default().data(chunk.to_string()))).is_err() {
                    break; // client disconnected
                }
            }
        }
        let final_chunk = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
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

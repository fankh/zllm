use axum::{Json, Router, routing::get};
use serde_json::{Value, json};

pub fn router() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/info", get(info))
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
        "description": "White-box LLM inference engine with zero-copy latent intercept",
        "backend": "dummy",
        "features": [
            "latent_reasoning",
            "activation_steering",
            "early_exit",
            "logit_fsm",
            "paged_kv_cache",
            "tenant_isolation"
        ]
    }))
}

use crate::backend::traits::Tensor;
use std::cell::Cell;

#[derive(Debug, Clone)]
pub enum HookAction {
    Continue,
    EarlyExit { reason: String },
    ModifyState,
    SkipRemaining,
}

/// Per-request context handed to every hook firing.
///
/// **Thread-safety invariant**: a `HookContext` is constructed per request and
/// not shared across threads concurrently. `stores_remaining` is `Cell<u32>` —
/// safely mutable through `&HookContext` only within a single thread. Do not
/// hand the same `HookContext` to spawned tasks; build a fresh one per task or
/// switch to atomics if that ever becomes a need.
#[derive(Debug, Clone)]
pub struct HookContext {
    pub request_id: String,
    pub tokens_generated: usize,
    /// Confidence signal updated by the runner before each hook firing.
    /// Wrapped in `Cell` for the same reason as `stores_remaining` —
    /// hooks see `&HookContext` and can read the latest value with
    /// `.get()` without forcing the `fire()` signature to take `&mut`.
    /// Source today: `ConfidenceHead::estimate(&hidden_state)` from
    /// `engine::confidence`.
    pub current_confidence: Cell<f32>,
    /// Number of memory writes (captures) a hook may still perform in this
    /// request. Hook implementations should `get()` to check and `set()` to
    /// decrement; `MemoryInjectHook` already does so.
    pub stores_remaining: Cell<u32>,
}

impl HookContext {
    /// Default write quota per request: 4 captures. Plenty for inject-once +
    /// capture-once-or-twice patterns; tight enough to stop a runaway hook
    /// from filling the store inside a single inference call.
    pub const DEFAULT_STORES_PER_REQUEST: u32 = 4;

    pub fn new(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            tokens_generated: 0,
            current_confidence: Cell::new(0.0),
            stores_remaining: Cell::new(Self::DEFAULT_STORES_PER_REQUEST),
        }
    }
}

pub trait Hook: Send + Sync {
    fn on_layer(
        &self,
        layer_idx: usize,
        loop_idx: usize,
        hidden_state: &mut Tensor,
        context: &HookContext,
    ) -> HookAction;

    fn target_layers(&self) -> Vec<usize>;
    fn name(&self) -> &str;
}

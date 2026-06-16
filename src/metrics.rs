use prometheus::{
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec,
    register_int_counter, register_int_counter_vec, register_int_gauge,
    register_int_gauge_vec,
};
use std::sync::OnceLock;

// --- v0.2 memory-subsystem metrics ---
//
// Lazy-initialized global handles. Call from anywhere:
//   metrics::memory_bytes_used().set(self.bytes_used as i64);
//   metrics::memory_evictions_total().with_label_values(&["budget"]).inc();
//
// Using OnceLock (stdlib) so initialization is thread-safe and idempotent.

static MEMORY_BYTES_USED: OnceLock<IntGauge> = OnceLock::new();
pub fn memory_bytes_used() -> &'static IntGauge {
    MEMORY_BYTES_USED.get_or_init(|| {
        register_int_gauge!("zllm_memory_bytes_used", "Memory store bytes used").unwrap()
    })
}

static MEMORY_ENTRIES: OnceLock<IntGaugeVec> = OnceLock::new();
pub fn memory_entries() -> &'static IntGaugeVec {
    MEMORY_ENTRIES.get_or_init(|| {
        register_int_gauge_vec!(
            "zllm_memory_entries",
            "Memory entries by category",
            &["category"]
        )
        .unwrap()
    })
}

static MEMORY_BYTES_BY_CATEGORY: OnceLock<IntGaugeVec> = OnceLock::new();
pub fn memory_bytes_by_category() -> &'static IntGaugeVec {
    MEMORY_BYTES_BY_CATEGORY.get_or_init(|| {
        register_int_gauge_vec!(
            "zllm_memory_bytes_by_category",
            "Memory bytes used by category",
            &["category"]
        )
        .unwrap()
    })
}

static MEMORY_EVICTIONS_TOTAL: OnceLock<IntCounterVec> = OnceLock::new();
pub fn memory_evictions_total() -> &'static IntCounterVec {
    MEMORY_EVICTIONS_TOTAL.get_or_init(|| {
        register_int_counter_vec!(
            "zllm_memory_evictions_total",
            "Memory evictions by reason",
            &["reason"]
        )
        .unwrap()
    })
}

static MEMORY_WRITE_QUOTA_REFUSALS: OnceLock<IntCounter> = OnceLock::new();
pub fn memory_write_quota_refusals() -> &'static IntCounter {
    MEMORY_WRITE_QUOTA_REFUSALS.get_or_init(|| {
        register_int_counter!(
            "zllm_memory_write_quota_refusals_total",
            "Hook writes refused because per-request quota was exhausted"
        )
        .unwrap()
    })
}

static MEMORY_EXPIRED_DROPS: OnceLock<IntCounter> = OnceLock::new();
pub fn memory_expired_drops() -> &'static IntCounter {
    MEMORY_EXPIRED_DROPS.get_or_init(|| {
        register_int_counter!(
            "zllm_memory_expired_drops_total",
            "Expired memory entries dropped under pressure"
        )
        .unwrap()
    })
}

static RUNNER_EARLY_EXITS: OnceLock<IntCounter> = OnceLock::new();
pub fn runner_early_exits() -> &'static IntCounter {
    RUNNER_EARLY_EXITS.get_or_init(|| {
        register_int_counter!(
            "zllm_runner_early_exits_total",
            "Chat prefills aborted early by a HookAction::EarlyExit firing"
        )
        .unwrap()
    })
}

static PREFIX_CACHE_HITS: OnceLock<IntCounter> = OnceLock::new();
pub fn prefix_cache_hits() -> &'static IntCounter {
    PREFIX_CACHE_HITS.get_or_init(|| {
        register_int_counter!(
            "zllm_prefix_cache_hits_total",
            "Chat requests that reused at least one cached prefix token"
        )
        .unwrap()
    })
}

static PREFIX_CACHE_MISSES: OnceLock<IntCounter> = OnceLock::new();
pub fn prefix_cache_misses() -> &'static IntCounter {
    PREFIX_CACHE_MISSES.get_or_init(|| {
        register_int_counter!(
            "zllm_prefix_cache_misses_total",
            "Chat requests whose prompt shared no prefix with the KV cache"
        )
        .unwrap()
    })
}

static PREFIX_CACHE_TOKENS_SAVED: OnceLock<IntCounter> = OnceLock::new();
pub fn prefix_cache_tokens_saved() -> &'static IntCounter {
    PREFIX_CACHE_TOKENS_SAVED.get_or_init(|| {
        register_int_counter!(
            "zllm_prefix_cache_tokens_saved_total",
            "Cumulative prompt tokens whose prefill was skipped via prefix cache reuse"
        )
        .unwrap()
    })
}

static PLD_DRAFT_ATTEMPTS: OnceLock<IntCounter> = OnceLock::new();
pub fn pld_draft_attempts() -> &'static IntCounter {
    PLD_DRAFT_ATTEMPTS.get_or_init(|| {
        register_int_counter!(
            "zllm_pld_draft_attempts_total",
            "Decode steps where prompt-lookup decoding found a candidate draft"
        )
        .unwrap()
    })
}

static PLD_TOKENS_ACCEPTED: OnceLock<IntCounter> = OnceLock::new();
pub fn pld_tokens_accepted() -> &'static IntCounter {
    PLD_TOKENS_ACCEPTED.get_or_init(|| {
        register_int_counter!(
            "zllm_pld_tokens_accepted_total",
            "Cumulative draft tokens accepted by the main model (verified == draft)"
        )
        .unwrap()
    })
}

static PLD_TOKENS_REJECTED: OnceLock<IntCounter> = OnceLock::new();
pub fn pld_tokens_rejected() -> &'static IntCounter {
    PLD_TOKENS_REJECTED.get_or_init(|| {
        register_int_counter!(
            "zllm_pld_tokens_rejected_total",
            "Cumulative draft tokens the main model overrode (wasted compute)"
        )
        .unwrap()
    })
}

static SPEC_DECODE_ITERS: OnceLock<IntCounter> = OnceLock::new();
pub fn spec_decode_iters() -> &'static IntCounter {
    SPEC_DECODE_ITERS.get_or_init(|| {
        register_int_counter!(
            "zllm_spec_decode_iters_total",
            "Speculative-decode iterations (one draft+verify cycle each)"
        )
        .unwrap()
    })
}

static SPEC_DECODE_ACCEPTED: OnceLock<IntCounter> = OnceLock::new();
pub fn spec_decode_accepted() -> &'static IntCounter {
    SPEC_DECODE_ACCEPTED.get_or_init(|| {
        register_int_counter!(
            "zllm_spec_decode_tokens_accepted_total",
            "Cumulative draft tokens the main model agreed with"
        )
        .unwrap()
    })
}

static SPEC_DECODE_REJECTED: OnceLock<IntCounter> = OnceLock::new();
pub fn spec_decode_rejected() -> &'static IntCounter {
    SPEC_DECODE_REJECTED.get_or_init(|| {
        register_int_counter!(
            "zllm_spec_decode_tokens_rejected_total",
            "Cumulative draft tokens the main model overrode"
        )
        .unwrap()
    })
}

static EARLY_EXIT_FIRES: OnceLock<IntCounter> = OnceLock::new();
pub fn early_exit_fires() -> &'static IntCounter {
    EARLY_EXIT_FIRES.get_or_init(|| {
        register_int_counter!(
            "zllm_early_exit_fires_total",
            "Per-token decode forwards that exited before the last layer"
        )
        .unwrap()
    })
}

static EARLY_EXIT_LAYER_SUM: OnceLock<IntCounter> = OnceLock::new();
pub fn early_exit_layer_sum() -> &'static IntCounter {
    EARLY_EXIT_LAYER_SUM.get_or_init(|| {
        register_int_counter!(
            "zllm_early_exit_layer_sum_total",
            "Sum of layer indices at which early exit fired (divide by fires for avg exit layer)"
        )
        .unwrap()
    })
}

static EARLY_EXIT_FULL: OnceLock<IntCounter> = OnceLock::new();
pub fn early_exit_full_forwards() -> &'static IntCounter {
    EARLY_EXIT_FULL.get_or_init(|| {
        register_int_counter!(
            "zllm_early_exit_full_forwards_total",
            "Per-token decode forwards that ran every layer (confidence below threshold)"
        )
        .unwrap()
    })
}

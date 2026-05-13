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

use prometheus::{Histogram, HistogramOpts, IntCounter, IntGauge, Registry, register_histogram, register_int_counter, register_int_gauge};

pub struct ZllmMetrics {
    pub requests_total: IntCounter,
    pub tokens_generated: IntCounter,
    pub early_exits: IntCounter,
    pub active_sequences: IntGauge,
    pub kv_blocks_used: IntGauge,
    pub kv_blocks_free: IntGauge,
    pub ttft_seconds: Histogram,
    pub intercept_latency_us: Histogram,
    pub tokens_per_second: Histogram,
}

impl ZllmMetrics {
    pub fn new() -> Self {
        Self {
            requests_total: register_int_counter!(
                "zllm_requests_total", "Total inference requests"
            ).unwrap(),
            tokens_generated: register_int_counter!(
                "zllm_tokens_generated", "Total tokens generated"
            ).unwrap(),
            early_exits: register_int_counter!(
                "zllm_early_exits_total", "Early exit events"
            ).unwrap(),
            active_sequences: register_int_gauge!(
                "zllm_active_sequences", "Currently active sequences"
            ).unwrap(),
            kv_blocks_used: register_int_gauge!(
                "zllm_kv_blocks_used", "KV cache blocks in use"
            ).unwrap(),
            kv_blocks_free: register_int_gauge!(
                "zllm_kv_blocks_free", "Free KV cache blocks"
            ).unwrap(),
            ttft_seconds: register_histogram!(
                "zllm_ttft_seconds", "Time to first token"
            ).unwrap(),
            intercept_latency_us: register_histogram!(
                HistogramOpts::new("zllm_intercept_latency_us", "Per-layer intercept latency in microseconds")
                    .buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0])
            ).unwrap(),
            tokens_per_second: register_histogram!(
                "zllm_tokens_per_second", "Generation throughput"
            ).unwrap(),
        }
    }
}

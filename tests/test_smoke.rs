use zllm::backend::dummy::DummyBackend;
use zllm::backend::traits::Backend;
use zllm::engine::hooks::registry::HookRegistry;
use zllm::engine::hooks::traits::{HookAction, HookContext};
use zllm::engine::memory_store::{MemoryCategory, MemoryMetadata, MemoryStore};
use zllm::engine::reasoning_budget::{ReasoningBudget, ReasoningState, TokenImportanceScorer};
use zllm::engine::runner::InferenceRunner;
use zllm::engine::sampler::{SamplerConfig, sample};
use zllm::memory::allocator::PagedAllocator;
use zllm::memory::isolation::TenantMemoryPool;
use std::sync::{Arc, RwLock};

#[test]
fn test_dummy_backend() {
    let mut backend = DummyBackend::new(32000, 4096, 32);
    let hidden = vec![0.0f32; 4096];
    let result = backend.forward_layer(0, &hidden, 1).unwrap();
    assert_eq!(result.len(), 4096);

    let logits = backend.compute_logits(&hidden).unwrap();
    assert_eq!(logits.len(), 32000);

    let block_id = backend.alloc_kv_block(16).unwrap();
    assert!(block_id > 0);
    backend.free_kv_block(block_id).unwrap();
}

#[test]
fn test_sampler() {
    let logits = vec![0.0f32; 100];
    let config = SamplerConfig::default();
    let token = sample(&logits, &config);
    assert!(token < 100);
}

#[test]
fn test_sampler_temperature() {
    let logits: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
    let config = SamplerConfig {
        temperature: 2.0,
        top_k: 10,
        top_p: 0.9,
    };
    let token = sample(&logits, &config);
    assert!(token < 1000);
}

#[test]
fn test_paged_allocator() {
    let mut alloc = PagedAllocator::new(100);
    let b1 = alloc.alloc().unwrap();
    let b2 = alloc.alloc().unwrap();
    assert_ne!(b1, b2);
    assert_eq!(alloc.used_count(), 2);
    assert_eq!(alloc.free_count(), 98);

    alloc.free(b1).unwrap();
    assert_eq!(alloc.used_count(), 1);
    assert_eq!(alloc.free_count(), 99);
}

#[test]
fn test_tenant_isolation() {
    let mut pool = TenantMemoryPool::new();
    let token = pool.create_session("tenant-a", 4096);
    assert!(pool.validate_access(&token));
    assert!(!pool.validate_access("invalid-token"));
    assert_eq!(pool.tenant_count(), 1);

    pool.destroy_session(&token);
    assert!(!pool.validate_access(&token));
    assert_eq!(pool.tenant_count(), 0);
}

#[test]
fn test_hook_registry() {
    let registry = HookRegistry::new();
    assert_eq!(registry.count(), 0);

    let context = HookContext {
        tenant_id: "test".into(),
        request_id: "req-1".into(),
        tokens_generated: 0,
        current_confidence: 0.5,
    };

    let mut hidden = vec![1.0f32; 4096];
    let action = registry.fire(0, 0, &mut hidden, &context);
    assert!(matches!(action, HookAction::Continue));
}

#[test]
fn test_config_parsing() {
    let config = zllm::config::ZllmConfig::load(std::path::Path::new("configs/default.toml")).unwrap();
    assert_eq!(config.server.rest_port, 8080);
    assert_eq!(config.server.grpc_port, 50051);
    assert_eq!(config.engine.max_loops, 16);
    assert_eq!(config.memory.block_size, 16);
}

// --- Reasoning Budget Tests ---

#[test]
fn test_reasoning_budget_tiers() {
    let free = ReasoningBudget::from_tenant_tier("free");
    assert_eq!(free.max_loops, 2);
    assert_eq!(free.max_memory_mb, 64);
    assert!(!free.per_token_adaptive);

    let standard = ReasoningBudget::from_tenant_tier("standard");
    assert_eq!(standard.max_loops, 8);
    assert!(standard.per_token_adaptive);

    let premium = ReasoningBudget::from_tenant_tier("premium");
    assert_eq!(premium.max_loops, 16);
    assert_eq!(premium.max_memory_mb, 512);
}

#[test]
fn test_reasoning_budget_should_continue() {
    let budget = ReasoningBudget::from_tenant_tier("standard");

    // Fresh state — should continue
    let state = ReasoningState::new(100);
    assert!(budget.should_continue(&state));

    // Hit loop limit
    let mut state = ReasoningState::new(100);
    state.loops_used = 8;
    assert!(!budget.should_continue(&state));

    // Hit memory limit
    let mut state = ReasoningState::new(100);
    state.memory_used_mb = 300;
    assert!(!budget.should_continue(&state));

    // Hit confidence threshold
    let mut state = ReasoningState::new(100);
    state.current_confidence = 0.95;
    assert!(!budget.should_continue(&state));

    // Below all limits — should continue
    let mut state = ReasoningState::new(100);
    state.loops_used = 3;
    state.memory_used_mb = 100;
    state.current_confidence = 0.5;
    assert!(budget.should_continue(&state));
}

#[test]
fn test_reasoning_state_record_loop() {
    let mut state = ReasoningState::new(50);
    assert_eq!(state.loops_used, 0);
    assert_eq!(state.memory_used_mb, 0);

    state.record_loop(32, 0.4);
    assert_eq!(state.loops_used, 1);
    assert_eq!(state.memory_used_mb, 32);
    assert_eq!(state.current_confidence, 0.4);

    state.record_loop(32, 0.7);
    assert_eq!(state.loops_used, 2);
    assert_eq!(state.memory_used_mb, 64);
    assert_eq!(state.current_confidence, 0.7);
}

#[test]
fn test_memory_estimation() {
    // 8 reasoning layers, 512 tokens, 4096 d_model, FP16
    let mb = ReasoningBudget::estimate_memory_per_loop(512, 4096, 8);
    // 8 * 512 * 4096 * 2 = 33,554,432 bytes = 32 MB
    assert_eq!(mb, 32);
}

#[test]
fn test_token_importance_scoring() {
    // Create hidden state with varying norms
    let d_model = 128;
    let seq_len = 512;
    let mut hidden = vec![0.1f32; seq_len * d_model];
    // Make middle tokens have lower activation
    for t in 200..300 {
        for d in 0..d_model {
            hidden[t * d_model + d] = 0.01;
        }
    }
    let scores = TokenImportanceScorer::score(&hidden, seq_len);
    assert_eq!(scores.len(), seq_len);

    // First tokens boosted as anchors
    assert!(scores[0] >= scores[250], "anchor token should score higher than middle");
    // All scores in valid range
    assert!(scores.iter().all(|&s| s >= 0.0 && s <= 1.0));
}

#[test]
fn test_token_importance_high_ratio() {
    // High importance tokens
    let scores = vec![0.9, 0.8, 0.3, 0.2, 0.95, 0.1, 0.85, 0.4];
    let high = TokenImportanceScorer::tokens_needing_deep_reasoning(&scores, 0.7);
    assert_eq!(high, vec![0, 1, 4, 6]); // indices with score >= 0.7
}

#[test]
fn test_runner_with_budget() {
    let backend = DummyBackend::new(32000, 4096, 32);
    let runner = InferenceRunner::new(Box::new(backend), 4096, 8);

    let prompt = vec![1u32; 10]; // 10 token prompt
    let config = SamplerConfig::default();
    let budget = ReasoningBudget::from_tenant_tier("free"); // max 2 loops

    let result = runner.generate(&prompt, 5, &config, &budget, "req-1", "tenant-a");
    assert!(result.tokens.len() <= 5);
    assert!(result.reasoning_loops_used <= 2); // budget enforced
    assert!(!result.early_exit);
    assert_eq!(result.memories_captured, 1); // always captures reasoning state
}

#[test]
fn test_runner_premium_budget() {
    let backend = DummyBackend::new(32000, 4096, 32);
    let runner = InferenceRunner::new(Box::new(backend), 4096, 8);

    let prompt = vec![1u32; 10];
    let config = SamplerConfig::default();
    let budget = ReasoningBudget::from_tenant_tier("premium");

    let result = runner.generate(&prompt, 5, &config, &budget, "req-2", "tenant-a");
    assert!(result.tokens.len() <= 5);
    assert!(result.reasoning_loops_used <= 16);
    assert!(result.avg_token_importance >= 0.0);
    assert_eq!(result.memories_captured, 1);
}

// --- Memory Store Tests ---

#[test]
fn test_memory_store_persist() {
    let mut store = MemoryStore::new(100, 50);

    let vector = vec![1.0f32; 128];
    let metadata = MemoryMetadata {
        source_request_id: "req-1".into(),
        tenant_id: "tenant-a".into(),
        layer_captured: 12,
        category: MemoryCategory::Finding,
        tags: vec!["sqli".into()],
        text_summary: "SQL injection found".into(),
    };

    store.store("finding-1".into(), vector, metadata);
    assert_eq!(store.entry_count(), 1);

    let entry = store.get("finding-1").unwrap();
    assert_eq!(entry.metadata.category, MemoryCategory::Finding);
    assert_eq!(entry.access_count, 1);
}

#[test]
fn test_memory_store_query_by_category() {
    let mut store = MemoryStore::new(100, 50);

    store.store("f1".into(), vec![1.0; 64], MemoryMetadata {
        source_request_id: "r1".into(),
        tenant_id: "t1".into(),
        layer_captured: 12,
        category: MemoryCategory::Finding,
        tags: vec![],
        text_summary: "Finding 1".into(),
    });
    store.store("c1".into(), vec![2.0; 64], MemoryMetadata {
        source_request_id: "r2".into(),
        tenant_id: "t1".into(),
        layer_captured: 8,
        category: MemoryCategory::Context,
        tags: vec![],
        text_summary: "Context 1".into(),
    });

    let findings = store.query_by_category(&MemoryCategory::Finding);
    assert_eq!(findings.len(), 1);
    let contexts = store.query_by_category(&MemoryCategory::Context);
    assert_eq!(contexts.len(), 1);
}

#[test]
fn test_memory_store_similarity_query() {
    let mut store = MemoryStore::new(100, 50);

    // Store two vectors: one similar to query, one different
    store.store("similar".into(), vec![1.0, 0.0, 1.0, 0.0], MemoryMetadata {
        source_request_id: "r1".into(),
        tenant_id: "t1".into(),
        layer_captured: 12,
        category: MemoryCategory::Finding,
        tags: vec![],
        text_summary: "Similar".into(),
    });
    store.store("different".into(), vec![0.0, 1.0, 0.0, 1.0], MemoryMetadata {
        source_request_id: "r2".into(),
        tenant_id: "t1".into(),
        layer_captured: 12,
        category: MemoryCategory::Context,
        tags: vec![],
        text_summary: "Different".into(),
    });

    let query = vec![1.0, 0.0, 1.0, 0.0]; // identical to "similar"
    let results = store.query_by_similarity(&query, 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0.key, "similar"); // most similar first
}

#[test]
fn test_memory_store_eviction() {
    let mut store = MemoryStore::new(3, 10); // max 3 entries

    for i in 0..5 {
        store.store(format!("entry-{i}"), vec![i as f32; 64], MemoryMetadata {
            source_request_id: format!("r{i}"),
            tenant_id: "t1".into(),
            layer_captured: 12,
            category: MemoryCategory::Context,
            tags: vec![],
            text_summary: format!("Entry {i}"),
        });
    }

    // Should never exceed max
    assert!(store.entry_count() <= 3);
}

#[test]
fn test_memory_injection_across_requests() {
    let memory = Arc::new(RwLock::new(MemoryStore::new(100, 50)));
    let backend1 = DummyBackend::new(32000, 4096, 32);
    let runner = InferenceRunner::new(Box::new(backend1), 4096, 8)
        .with_memory(memory.clone());

    let config = SamplerConfig::default();
    let budget = ReasoningBudget::from_tenant_tier("standard");

    // Request 1: generates and captures reasoning state
    let result1 = runner.generate(&vec![1u32; 10], 3, &config, &budget, "req-1", "tenant-a");
    assert_eq!(result1.memories_captured, 1);

    // Verify memory was stored
    let store = memory.read().unwrap();
    assert!(store.entry_count() >= 1);
    let tenant_mems = store.query_by_tenant("tenant-a");
    assert!(!tenant_mems.is_empty());
    drop(store);

    // Request 2: should inject memory from request 1
    let result2 = runner.generate(&vec![2u32; 10], 3, &config, &budget, "req-2", "tenant-a");
    assert_eq!(result2.memories_injected, 1); // injected from req-1
    assert_eq!(result2.memories_captured, 1); // captured its own state
}

#[test]
fn test_inspection_trace() {
    let backend = DummyBackend::new(32000, 4096, 32);
    let runner = InferenceRunner::new(Box::new(backend), 4096, 8)
        .with_inspection(true);

    let config = SamplerConfig::default();
    let budget = ReasoningBudget::from_tenant_tier("free");

    let result = runner.generate(&vec![1u32; 10], 3, &config, &budget, "req-trace", "tenant-a");
    assert!(result.inspection_trace.is_some());

    let trace = result.inspection_trace.unwrap();
    assert_eq!(trace.request_id, "req-trace");
    assert!(!trace.layers.is_empty());

    // Should have snapshots for zone 1 (8) + zone 2 (reasoning) + zone 3 (16)
    // Each layer in each zone produces a snapshot
    assert!(trace.layers.len() >= 8); // at least zone 1

    // Each snapshot should have valid data
    for snap in &trace.layers {
        assert!(snap.hidden_state_norm >= 0.0);
        assert!(!snap.top_activations.is_empty());
    }
}

#[test]
fn test_inspection_trace_stored_in_memory() {
    let memory = Arc::new(RwLock::new(MemoryStore::new(100, 50)));
    let backend = DummyBackend::new(32000, 4096, 32);
    let runner = InferenceRunner::new(Box::new(backend), 4096, 8)
        .with_memory(memory.clone())
        .with_inspection(true);

    let config = SamplerConfig::default();
    let budget = ReasoningBudget::from_tenant_tier("free");

    runner.generate(&vec![1u32; 10], 3, &config, &budget, "req-t1", "tenant-a");

    let store = memory.read().unwrap();
    assert_eq!(store.trace_count(), 1);
    let trace = store.get_trace_by_request("req-t1").unwrap();
    assert_eq!(trace.request_id, "req-t1");
}

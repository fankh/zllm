use zllm::backend::dummy::DummyBackend;
use zllm::backend::traits::Backend;
use zllm::engine::hooks::registry::HookRegistry;
use zllm::engine::hooks::traits::{HookAction, HookContext};
use zllm::engine::reasoning_budget::{ReasoningBudget, ReasoningState, TokenImportanceScorer};
use zllm::engine::runner::InferenceRunner;
use zllm::engine::sampler::{SamplerConfig, sample};
use zllm::memory::allocator::PagedAllocator;
use zllm::memory::isolation::TenantMemoryPool;

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

    let result = runner.generate(&prompt, 5, &config, &budget);
    assert!(result.tokens.len() <= 5);
    assert!(result.reasoning_loops_used <= 2); // budget enforced
    assert!(!result.early_exit);
}

#[test]
fn test_runner_premium_budget() {
    let backend = DummyBackend::new(32000, 4096, 32);
    let runner = InferenceRunner::new(Box::new(backend), 4096, 8);

    let prompt = vec![1u32; 10];
    let config = SamplerConfig::default();
    let budget = ReasoningBudget::from_tenant_tier("premium"); // max 16 loops, adaptive

    let result = runner.generate(&prompt, 5, &config, &budget);
    assert!(result.tokens.len() <= 5);
    // Premium with adaptive: actual loops depends on token importance
    assert!(result.reasoning_loops_used <= 16);
    assert!(result.avg_token_importance >= 0.0);
}

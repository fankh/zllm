use zllm::backend::dummy::DummyBackend;
use zllm::backend::traits::Backend;
use zllm::engine::hooks::registry::HookRegistry;
use zllm::engine::hooks::traits::{HookAction, HookContext};
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
    // High temperature should produce valid tokens
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

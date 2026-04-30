use std::path::Path;
use std::sync::{Arc, RwLock};
use zllm::backend::candle::backend::CandleCpuBackend;
use zllm::backend::candle::tokenizer::LlamaTokenizer;
use zllm::backend::traits::{Backend, QuantConfig};
use zllm::engine::hooks::registry::HookRegistry;
use zllm::engine::hooks::steering::SteeringHook;
use zllm::engine::hooks::early_exit::EarlyExitHook;
use zllm::engine::hooks::traits::{Hook, HookAction, HookContext};
use zllm::engine::memory_store::{MemoryStore, MemoryMetadata, MemoryCategory};
use zllm::engine::reasoning_budget::ReasoningBudget;
use zllm::engine::sampler::{SamplerConfig, sample};

const MODEL_PATH: &str = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
const TOKENIZER_PATH: &str = "models/tokenizer.json";

fn model_available() -> bool {
    Path::new(MODEL_PATH).exists() && Path::new(TOKENIZER_PATH).exists()
}

// --- Test 1: Real model loading and basic generation ---

#[test]
fn test_real_model_load() {
    if !model_available() {
        println!("SKIP: model not found at {MODEL_PATH}");
        return;
    }

    let mut backend = CandleCpuBackend::new();
    let result = backend.load_model(
        Path::new(MODEL_PATH),
        &QuantConfig { method: "gguf".into(), bits: 4 },
    );
    assert!(result.is_ok(), "Model should load: {:?}", result.err());

    let info = backend.device_info();
    assert_eq!(info.backend, "candle-cpu");
    println!("Device: {}", info.name);
}

// --- Test 2: Tokenizer encode/decode roundtrip ---

#[test]
fn test_real_tokenizer() {
    if !model_available() {
        println!("SKIP: tokenizer not found");
        return;
    }

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();

    let text = "What is SQL injection?";
    let tokens = tokenizer.encode(text).unwrap();
    assert!(!tokens.is_empty(), "Should produce tokens");
    println!("Encoded '{}' -> {} tokens: {:?}", text, tokens.len(), &tokens);

    let decoded = tokenizer.decode(&tokens).unwrap();
    println!("Decoded back: '{}'", decoded);
    assert!(decoded.contains("SQL injection"), "Roundtrip should preserve meaning");

    let vocab = tokenizer.vocab_size();
    assert!(vocab > 100000, "Llama 3 vocab should be 128K+, got {vocab}");
    println!("Vocab size: {vocab}");
}

// --- Test 3: Real token generation ---

#[test]
fn test_real_generate_tokens() {
    if !model_available() {
        println!("SKIP: model not found");
        return;
    }

    let mut backend = CandleCpuBackend::new();
    backend.load_model(
        Path::new(MODEL_PATH),
        &QuantConfig { method: "gguf".into(), bits: 4 },
    ).unwrap();

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let prompt_tokens = tokenizer.encode("The capital of France is").unwrap();

    let start = std::time::Instant::now();
    let (token_id, _hidden) = backend.generate_token(&prompt_tokens).unwrap();
    let elapsed = start.elapsed();

    let token_text = tokenizer.decode(&[token_id]).unwrap();
    println!("First token: {} (id={}) in {:.2}ms", token_text, token_id, elapsed.as_millis());

    // Should generate something related to "Paris"
    assert!(token_id > 0, "Should generate a valid token");
}

// --- Test 4: Multi-token generation with streaming ---

#[test]
fn test_real_multi_token_generation() {
    if !model_available() {
        println!("SKIP: model not found");
        return;
    }

    let mut backend = CandleCpuBackend::new();
    backend.load_model(
        Path::new(MODEL_PATH),
        &QuantConfig { method: "gguf".into(), bits: 4 },
    ).unwrap();

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let prompt = "1+1=";
    let prompt_tokens = tokenizer.encode(prompt).unwrap();
    let eos_id = tokenizer.eos_token_id().unwrap_or(128001);

    let mut all_tokens = prompt_tokens.clone();
    let mut generated = 0;
    let max_tokens = 5; // Keep small for test speed (debug mode is slow)
    let start = std::time::Instant::now();

    for _ in 0..max_tokens {
        let input = if generated == 0 {
            &all_tokens[..]
        } else {
            &all_tokens[all_tokens.len() - 1..]
        };

        let (token_id, _) = backend.generate_token(input).unwrap();
        if token_id == eos_id || token_id == 128009 {
            break;
        }
        all_tokens.push(token_id);
        generated += 1;
    }

    let elapsed = start.elapsed();
    let output_tokens = &all_tokens[prompt_tokens.len()..];
    let output_text = tokenizer.decode(output_tokens).unwrap();
    let tok_per_sec = generated as f64 / elapsed.as_secs_f64();

    println!("Prompt: '{prompt}'");
    println!("Output: '{output_text}'");
    println!("{generated} tokens in {:.2}s ({:.1} tok/s)", elapsed.as_secs_f64(), tok_per_sec);

    assert!(generated > 0, "Should generate at least 1 token");
    // Debug mode is ~100x slower than release; skip speed check in debug
    #[cfg(not(debug_assertions))]
    assert!(tok_per_sec > 1.0, "Should be faster than 1 tok/s");
}

// --- Test 5: Hook system with real backend ---

#[test]
fn test_hooks_on_real_backend() {
    if !model_available() {
        println!("SKIP: model not found");
        return;
    }

    let mut registry = HookRegistry::new();

    // Add a steering hook that modifies hidden state
    let steering = SteeringHook {
        vector: vec![0.01f32; 2048], // Llama 3.2 1B has hidden=2048
        alpha: 0.5,
        layer: 8,
    };
    registry.register(Box::new(steering));
    assert_eq!(registry.count(), 1);

    // Add early exit hook
    let early_exit = EarlyExitHook {
        threshold: 0.95,
        layer: 12,
    };
    registry.register(Box::new(early_exit));
    assert_eq!(registry.count(), 2);

    // Fire hooks with a dummy hidden state
    let context = HookContext {
        tenant_id: "test".into(),
        request_id: "req-hook-test".into(),
        tokens_generated: 10,
        current_confidence: 0.5,
    };

    // Test steering hook (layer 8)
    let mut hidden = vec![1.0f32; 2048];
    let original_sum: f32 = hidden.iter().sum();
    let action = registry.fire(8, 0, &mut hidden, &context);
    let modified_sum: f32 = hidden.iter().sum();
    assert!(modified_sum != original_sum, "Steering should modify hidden state");
    println!("Steering: sum changed from {original_sum:.2} to {modified_sum:.2}");
    // Registry returns last non-Continue action, or Continue if all hooks pass
    // Steering at layer 8 modifies state; early_exit at layer 12 doesn't fire here
    assert!(!matches!(action, HookAction::EarlyExit { .. }), "Should not early exit at layer 8");

    // Test early exit hook (layer 12, confidence below threshold)
    let mut hidden2 = vec![1.0f32; 2048];
    let action2 = registry.fire(12, 0, &mut hidden2, &context);
    assert!(matches!(action2, HookAction::Continue), "Should not exit at confidence 0.5");

    // Test early exit hook (layer 12, confidence above threshold)
    let high_conf_context = HookContext {
        current_confidence: 0.99,
        ..context.clone()
    };
    let mut hidden3 = vec![1.0f32; 2048];
    let action3 = registry.fire(12, 0, &mut hidden3, &high_conf_context);
    assert!(matches!(action3, HookAction::EarlyExit { .. }), "Should exit at confidence 0.99");
    println!("Early exit triggered correctly at confidence 0.99");
}

// --- Test 6: Memory store with real token vectors ---

#[test]
fn test_memory_store_with_real_tokens() {
    if !model_available() {
        println!("SKIP: model not found");
        return;
    }

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let mut store = MemoryStore::new(100, 50);

    // Simulate storing a finding from a security analysis
    let finding_vector = vec![0.5f32; 2048]; // simulated hidden state
    store.store(
        "finding-sqli-1".into(),
        finding_vector.clone(),
        MemoryMetadata {
            source_request_id: "req-1".into(),
            tenant_id: "security-team".into(),
            layer_captured: 12,
            category: MemoryCategory::Finding,
            tags: vec!["sqli".into(), "critical".into()],
            text_summary: "SQL injection found in login endpoint".into(),
        },
    );

    // Store another finding
    store.store(
        "finding-xss-1".into(),
        vec![0.3f32; 2048],
        MemoryMetadata {
            source_request_id: "req-2".into(),
            tenant_id: "security-team".into(),
            layer_captured: 12,
            category: MemoryCategory::Finding,
            tags: vec!["xss".into(), "medium".into()],
            text_summary: "Reflected XSS in search parameter".into(),
        },
    );

    // Query findings
    let findings = store.query_by_category(&MemoryCategory::Finding);
    assert_eq!(findings.len(), 2);
    println!("Stored {} findings", findings.len());

    // Query by tag
    let critical = store.query_by_tag("critical");
    assert_eq!(critical.len(), 1);
    assert_eq!(critical[0].key, "finding-sqli-1");

    // Similarity query
    let query = vec![0.5f32; 2048]; // similar to sqli finding
    let similar = store.query_by_similarity(&query, 2);
    assert_eq!(similar.len(), 2);
    assert_eq!(similar[0].0.key, "finding-sqli-1"); // most similar
    println!("Most similar memory: {} (score: {:.3})", similar[0].0.key, similar[0].1);

    // Build injection vector for tenant
    let injection = store.build_injection_vector(&query, "security-team", 5, 0.3);
    assert!(injection.is_some(), "Should build injection from 2 memories");
    let inj = injection.unwrap();
    assert_eq!(inj.len(), 2048);
    println!("Injection vector norm: {:.4}", inj.iter().map(|x| x * x).sum::<f32>().sqrt());
}

// --- Test 7: Reasoning budget with real model dimensions ---

#[test]
fn test_reasoning_budget_real_dimensions() {
    // Llama 3.2 1B: 16 layers, 2048 hidden, 8 reasoning layers
    let budget = ReasoningBudget::from_tenant_tier("standard"); // max 8 loops

    // Memory per loop for 512 tokens, 2048 hidden, 8 reasoning layers
    let mem_per_loop = ReasoningBudget::estimate_memory_per_loop(512, 2048, 8);
    println!("Memory per reasoning loop (512 tokens, 2048 hidden, 8 layers): {} MB", mem_per_loop);
    // 8 * 512 * 2048 * 2 = 16,777,216 bytes = 16 MB
    assert_eq!(mem_per_loop, 16);

    // Total for max loops
    let total_mb = mem_per_loop * budget.max_loops;
    println!("Max reasoning memory ({} loops): {} MB", budget.max_loops, total_mb);
    assert!(total_mb <= budget.max_memory_mb, "Total should fit within budget");

    // Free tier: 2 loops * 16 MB = 32 MB (fits in 64 MB budget)
    let free = ReasoningBudget::from_tenant_tier("free");
    let free_total = mem_per_loop * free.max_loops;
    println!("Free tier reasoning memory: {} MB (budget: {} MB)", free_total, free.max_memory_mb);
    assert!(free_total <= free.max_memory_mb);
}

// --- Test 8: Full pipeline — generate + store memory + retrieve ---

#[test]
fn test_full_pipeline_with_memory() {
    if !model_available() {
        println!("SKIP: model not found");
        return;
    }

    let mut backend = CandleCpuBackend::new();
    backend.load_model(
        Path::new(MODEL_PATH),
        &QuantConfig { method: "gguf".into(), bits: 4 },
    ).unwrap();

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let memory = Arc::new(RwLock::new(MemoryStore::new(100, 50)));

    // Request 1: Generate response and store in memory
    let prompt1 = "What is a buffer overflow?";
    let tokens1 = tokenizer.encode(prompt1).unwrap();

    let (first_token, _) = backend.generate_token(&tokens1).unwrap();
    let first_word = tokenizer.decode(&[first_token]).unwrap();
    println!("Request 1 prompt: '{prompt1}'");
    println!("Request 1 first token: '{first_word}'");

    // Store the "analysis result" in memory
    {
        let mut store = memory.write().unwrap();
        store.store(
            "analysis-1".into(),
            vec![0.42f32; 2048], // simulated hidden state from analysis
            MemoryMetadata {
                source_request_id: "req-1".into(),
                tenant_id: "team-a".into(),
                layer_captured: 8,
                category: MemoryCategory::Finding,
                tags: vec!["buffer-overflow".into()],
                text_summary: format!("Analysis of buffer overflow, first token: {first_word}"),
            },
        );
    }

    // Request 2: Check that memory exists and can be queried
    backend.reset_position();
    let prompt2 = "How to prevent buffer overflow?";
    let tokens2 = tokenizer.encode(prompt2).unwrap();

    let (second_token, _) = backend.generate_token(&tokens2).unwrap();
    let second_word = tokenizer.decode(&[second_token]).unwrap();
    println!("Request 2 prompt: '{prompt2}'");
    println!("Request 2 first token: '{second_word}'");

    // Retrieve memory
    {
        let store = memory.read().unwrap();
        let tenant_memories = store.query_by_tenant("team-a");
        assert_eq!(tenant_memories.len(), 1);
        println!("Retrieved memory: {}", tenant_memories[0].metadata.text_summary);

        let tag_memories = store.query_by_tag("buffer-overflow");
        assert_eq!(tag_memories.len(), 1);

        // Build injection from previous analysis
        let query = vec![0.4f32; 2048];
        let injection = store.build_injection_vector(&query, "team-a", 5, 0.3);
        assert!(injection.is_some());
        println!("Memory injection vector built from {} entries", tenant_memories.len());
    }
}

// --- Test 9: Throughput measurement ---

#[test]
fn test_throughput_benchmark() {
    if !model_available() {
        println!("SKIP: model not found");
        return;
    }

    let mut backend = CandleCpuBackend::new();
    backend.load_model(
        Path::new(MODEL_PATH),
        &QuantConfig { method: "gguf".into(), bits: 4 },
    ).unwrap();

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let prompt_tokens = tokenizer.encode("Hello").unwrap();

    // Prefill timing
    let prefill_start = std::time::Instant::now();
    let (_, _) = backend.generate_token(&prompt_tokens).unwrap();
    let prefill_time = prefill_start.elapsed();

    // Decode timing (single tokens)
    let mut times = Vec::new();
    let mut last_token = prompt_tokens[prompt_tokens.len() - 1];
    for _ in 0..3 {  // 3 tokens for benchmark (debug mode is slow)
        let start = std::time::Instant::now();
        let (token_id, _) = backend.generate_token(&[last_token]).unwrap();
        times.push(start.elapsed());
        last_token = token_id;
    }

    let avg_decode_ms = times.iter().map(|t| t.as_millis() as f64).sum::<f64>() / times.len() as f64;
    let tok_per_sec = 1000.0 / avg_decode_ms;

    println!("=== ZLLM Throughput Benchmark ===");
    println!("Model: Llama 3.2 1B Q4_K_M");
    println!("Backend: Candle CPU (x86_64)");
    println!("Prefill ({} tokens): {:.0}ms", prompt_tokens.len(), prefill_time.as_millis());
    println!("Decode (avg per token): {:.0}ms", avg_decode_ms);
    println!("Throughput: {:.1} tok/s", tok_per_sec);
    println!("================================");

    // Debug mode is ~100x slower; only assert speed in release
    #[cfg(not(debug_assertions))]
    assert!(tok_per_sec > 1.0, "Should be faster than 1 tok/s, got {:.1}", tok_per_sec);
}

use std::path::Path;
use std::sync::{Arc, RwLock};
use zllm::backend::candle::backend::CandleCpuBackend;
use zllm::backend::candle::tokenizer::LlamaTokenizer;
use zllm::backend::traits::{Backend, Tensor};
use zllm::engine::hooks::registry::HookRegistry;
use zllm::engine::hooks::steering::SteeringHook;
use zllm::engine::hooks::traits::{Hook, HookAction, HookContext};
use zllm::engine::memory_store::{MemoryStore, MemoryMetadata, MemoryCategory};
use zllm::engine::reasoning_budget::ReasoningBudget;

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
    let result = backend.load_model(Path::new(MODEL_PATH));
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
    backend.load_model(Path::new(MODEL_PATH)).unwrap();

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let prompt_tokens = tokenizer.encode("The capital of France is").unwrap();

    let start = std::time::Instant::now();
    let token_id = backend.generate_token(&prompt_tokens).unwrap();
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
    backend.load_model(Path::new(MODEL_PATH)).unwrap();

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let prompt = "1+1=";
    let prompt_tokens = tokenizer.encode(prompt).unwrap();
    let stops = tokenizer.stop_token_ids();

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

        let token_id = backend.generate_token(input).unwrap();
        if stops.contains(&token_id) {
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

/// Local confidence-gated exit hook. The shipped `EarlyExitHook` was
/// removed (production early exit runs a `ConfidenceHead` closure over
/// `forward_logits_early_exit`); this stand-in keeps the registry's
/// `HookAction::EarlyExit` mechanics covered.
struct ConfidenceGate {
    threshold: f32,
    layer: usize,
}

impl Hook for ConfidenceGate {
    fn on_layer(
        &self,
        _layer_idx: usize,
        _loop_idx: usize,
        _hidden_state: &mut Tensor,
        context: &HookContext,
    ) -> HookAction {
        let c = context.current_confidence.get();
        if c > self.threshold {
            HookAction::EarlyExit {
                reason: format!("confidence {:.3} > threshold {:.3}", c, self.threshold),
            }
        } else {
            HookAction::Continue
        }
    }

    fn target_layers(&self) -> Vec<usize> {
        vec![self.layer]
    }

    fn name(&self) -> &str {
        "test-confidence-gate"
    }
}

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

    // Add the confidence-gated exit hook
    registry.register(Box::new(ConfidenceGate { threshold: 0.95, layer: 12 }));
    assert_eq!(registry.count(), 2);

    // Fire hooks with a dummy hidden state
    let mut context = HookContext::new("req-hook-test");
    context.tokens_generated = 10;
    context.current_confidence.set(0.5);

    // Test steering hook (layer 8). Steering edits the live residual
    // stream via the write-back channel (`residual_delta`), not by
    // mutating the pooled observe-path copy passed to `fire`.
    let mut hidden = vec![1.0f32; 2048];
    let action = registry.fire(8, 0, &mut hidden, &context);
    let delta = registry
        .residual_delta(8, &hidden, &context)
        .expect("steering should produce a residual delta at its target layer");
    assert_eq!(delta.len(), 2048);
    assert!(
        delta.iter().all(|&d| (d - 0.005).abs() < 1e-6),
        "delta should be alpha * vector = 0.5 * 0.01"
    );
    assert!(
        registry.residual_delta(9, &hidden, &context).is_none(),
        "no steering delta off the target layer"
    );
    println!("Steering delta at layer 8: {} dims of {:.3}", delta.len(), delta[0]);
    // Registry returns last non-Continue action, or Continue if all hooks pass
    // Steering at layer 8 is write-back only; early_exit at layer 12 doesn't fire here
    assert!(!matches!(action, HookAction::EarlyExit { .. }), "Should not early exit at layer 8");

    // Test early exit hook (layer 12, confidence below threshold)
    let mut hidden2 = vec![1.0f32; 2048];
    let action2 = registry.fire(12, 0, &mut hidden2, &context);
    assert!(matches!(action2, HookAction::Continue), "Should not exit at confidence 0.5");

    // Test early exit hook (layer 12, confidence above threshold)
    let high_conf_context = context.clone();
    high_conf_context.current_confidence.set(0.99);
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

    let _tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let mut store = MemoryStore::new(100, 50);

    // Simulate storing a finding from a security analysis
    let finding_vector = vec![0.5f32; 2048]; // simulated hidden state
    store.store(
        "finding-sqli-1".into(),
        finding_vector.clone(),
        MemoryMetadata {
            source_request_id: "req-1".into(),
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

    // Build injection vector from all live memories (similarity-scored
    // internally — the standalone query_by_similarity API was removed).
    let query = vec![0.5f32; 2048]; // similar to sqli finding
    let injection = store.build_injection_vector(&query, 5, 0.3);
    assert!(injection.is_some(), "Should build injection from 2 memories");
    let inj = injection.unwrap();
    assert_eq!(inj.len(), 2048);
    println!("Injection vector norm: {:.4}", inj.iter().map(|x| x * x).sum::<f32>().sqrt());
}

// --- Test 7: Reasoning budget with real model dimensions ---

#[test]
fn test_reasoning_budget_real_dimensions() {
    // Llama 3.2 1B: 16 layers, 2048 hidden, 8 reasoning layers
    let budget = ReasoningBudget::from_tier("standard"); // max 8 loops

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
    let free = ReasoningBudget::from_tier("free");
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
    backend.load_model(Path::new(MODEL_PATH)).unwrap();

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let memory = Arc::new(RwLock::new(MemoryStore::new(100, 50)));

    // Request 1: Generate response and store in memory
    let prompt1 = "What is a buffer overflow?";
    let tokens1 = tokenizer.encode(prompt1).unwrap();

    let first_token = backend.generate_token(&tokens1).unwrap();
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

    let second_token = backend.generate_token(&tokens2).unwrap();
    let second_word = tokenizer.decode(&[second_token]).unwrap();
    println!("Request 2 prompt: '{prompt2}'");
    println!("Request 2 first token: '{second_word}'");

    // Retrieve memory
    {
        let store = memory.read().unwrap();
        let findings = store.query_by_category(&MemoryCategory::Finding);
        assert_eq!(findings.len(), 1);
        println!("Retrieved memory: {}", findings[0].metadata.text_summary);

        let tag_memories = store.query_by_tag("buffer-overflow");
        assert_eq!(tag_memories.len(), 1);

        // Build injection from previous analysis
        let query = vec![0.4f32; 2048];
        let injection = store.build_injection_vector(&query, 5, 0.3);
        assert!(injection.is_some());
        println!("Memory injection vector built from {} entries", findings.len());
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
    backend.load_model(Path::new(MODEL_PATH)).unwrap();

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let prompt_tokens = tokenizer.encode("Hello").unwrap();

    // Prefill timing
    let prefill_start = std::time::Instant::now();
    let _ = backend.generate_token(&prompt_tokens).unwrap();
    let prefill_time = prefill_start.elapsed();

    // Decode timing (single tokens)
    let mut times = Vec::new();
    let mut last_token = prompt_tokens[prompt_tokens.len() - 1];
    for _ in 0..3 {  // 3 tokens for benchmark (debug mode is slow)
        let start = std::time::Instant::now();
        let token_id = backend.generate_token(&[last_token]).unwrap();
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

// --- Test 10: Manual layer drive matches the fused forward pass ---
//
// The Backend trait's step-by-step surface (embed_tokens → forward_layer
// per block → compute_logits) must reproduce what the fused
// forward_logits pass computes: same embeddings, same causal mask, same
// blocks, same final norm + LM head. Guards the per-layer path the
// InferenceRunner drives.

#[test]
fn test_manual_layer_drive_matches_fused_forward() {
    if !model_available() {
        println!("SKIP: model not found");
        return;
    }

    let mut backend = CandleCpuBackend::new();
    backend.load_model(Path::new(MODEL_PATH)).unwrap();

    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let tokens = tokenizer.encode("The capital of France is").unwrap();
    let seq_len = tokens.len();

    // Reference: fused single-shot forward.
    let fused = backend.forward_logits(&tokens).unwrap();
    backend.reset_position();

    // Manual drive over the trait surface.
    let n_layers = Backend::n_layers(&backend);
    assert!(n_layers > 0, "n_layers should be reported after load");
    let mut hidden = backend.embed_tokens(&tokens).unwrap();
    assert_eq!(hidden.len() % seq_len, 0, "embedding width must divide evenly");
    for layer_idx in 0..n_layers {
        hidden = backend.forward_layer(layer_idx, &hidden, seq_len).unwrap();
    }
    let manual = backend.compute_logits(&hidden).unwrap();

    assert_eq!(fused.len(), manual.len(), "vocab widths must match");
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    let (fused_top, manual_top) = (argmax(&fused), argmax(&manual));
    let max_abs_diff = fused
        .iter()
        .zip(&manual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "fused top: {} '{}', manual top: {} '{}', max |Δlogit| = {max_abs_diff}",
        fused_top,
        tokenizer.decode(&[fused_top as u32]).unwrap_or_default(),
        manual_top,
        tokenizer.decode(&[manual_top as u32]).unwrap_or_default(),
    );
    assert_eq!(fused_top, manual_top, "top-1 token must agree");
    assert!(
        max_abs_diff < 1e-3,
        "logits should match the fused pass, max |Δ| = {max_abs_diff}"
    );

    // Out-of-range layer must be a hard error, not a silent identity.
    assert!(backend.forward_layer(n_layers, &hidden, seq_len).is_err());
}

// --- Test 11: Runner decode is real autoregression ---
//
// With reasoning_layers = 0 the 3-zone program reduces to a plain
// full-depth forward, so the runner's greedy decode must reproduce the
// fused KV-path greedy continuation token-for-token. The old decode
// loop sampled every token from one frozen logit vector (and stopped on
// the Llama-2 id 2) — this test pins the fix.

#[test]
fn test_runner_decode_matches_greedy_continuation() {
    if !model_available() {
        println!("SKIP: model not found");
        return;
    }
    let tokenizer = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let prompt = tokenizer.encode("The capital of France is").unwrap();
    let n_new = 4usize;

    // Reference: greedy continuation via stateless full re-forwards on the
    // fused path — the same prefill kernels the runner's per-layer surface
    // uses, so equality is exact. (The KV-cache decode path uses a different
    // CPU SDPA kernel whose accumulation order can flip near-tied logits —
    // e.g. " The capital" vs " The Eiffel" here — so it is not a bit-exact
    // oracle for this comparison.)
    let mut backend = CandleCpuBackend::new();
    backend.load_model(Path::new(MODEL_PATH)).unwrap();
    let stops = tokenizer.stop_token_ids();
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap()
    };
    let mut all = prompt.clone();
    let mut reference = Vec::new();
    for _ in 0..n_new {
        backend.reset_position();
        let logits = backend.forward_logits(&all).unwrap();
        let t = argmax(&logits);
        reference.push(t);
        if stops.contains(&t) {
            break;
        }
        all.push(t);
    }

    // Runner path: zones over the trait surface, then autoregressive decode.
    let mut runner_backend = CandleCpuBackend::new();
    runner_backend.load_model(Path::new(MODEL_PATH)).unwrap();
    let d_model = 2048; // Llama 3.2 1B
    let mut runner = zllm::engine::runner::InferenceRunner::new(
        Box::new(runner_backend), d_model, 0,
    )
    .with_eos_tokens(tokenizer.stop_token_ids());
    let config = zllm::engine::sampler::SamplerConfig {
        temperature: 0.0, top_k: 0, top_p: 1.0,
    };
    let budget = ReasoningBudget::from_tier("free");
    let result = runner
        .generate(&prompt, n_new, &config, &budget, "req-ar")
        .expect("runner generate");

    println!(
        "reference: {:?} '{}'\nrunner:    {:?} '{}'",
        reference,
        tokenizer.decode(&reference).unwrap_or_default(),
        result.tokens,
        tokenizer.decode(&result.tokens).unwrap_or_default(),
    );
    assert_eq!(
        result.tokens, reference,
        "runner decode must match the fused greedy continuation"
    );
}

// --- Test 12: Stop-token set derived from the tokenizer ---

#[test]
fn test_stop_token_ids_from_vocab() {
    if !model_available() {
        println!("SKIP: tokenizer not found");
        return;
    }
    let tok = LlamaTokenizer::from_file(TOKENIZER_PATH).unwrap();
    let stops = tok.stop_token_ids();
    // Llama-3 vocab: both <|end_of_text|> and <|eot_id|> must come out of
    // the vocab probe — 128009 used to be hardcoded at every call site.
    assert!(stops.contains(&128001), "missing <|end_of_text|>: {stops:?}");
    assert!(stops.contains(&128009), "missing <|eot_id|>: {stops:?}");
    assert!(!stops.contains(&2), "Llama-2 </s> id must not appear for a Llama-3 vocab");
    println!("stop set: {stops:?}");
}

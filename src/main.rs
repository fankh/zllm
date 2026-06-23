use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod backend;
mod config;
mod control_plane;
mod engine;
mod error;
mod memory;
mod metrics;
mod server;

#[derive(Parser)]
#[command(name = "zllm")]
#[command(about = "ZLLM — White-box LLM inference engine")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
    Serve {
        #[arg(short, long, default_value = "configs/default.toml")]
        config: PathBuf,
    },
    /// Generate text from a prompt (CPU inference)
    Generate {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long, default_value = "")]
        tokenizer: String,
        #[arg(short, long)]
        prompt: String,
        #[arg(long, default_value = "128")]
        max_tokens: usize,
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        #[arg(long, default_value = "50")]
        top_k: usize,
        #[arg(long, default_value = "0.9")]
        top_p: f32,
    },
}

/// Build the global rayon pool with one worker per **physical** core,
/// each pinned to a distinct core. Decode is memory-bandwidth bound, and
/// letting rayon spread across SMT siblings oversubscribes the cores that
/// actually move memory — measured ~10% slower than one-thread-per-core
/// on this box (Strix Halo, Zen5). Pinning to physical cores is what
/// llama.cpp does by default and is the single cheapest decode win.
///
/// SMT siblings enumerate adjacently (logical 0,1 = core 0; 2,3 = core 1;
/// …), so every-other logical id is one thread per physical core. Set
/// `ZLLM_NO_PIN=1` to fall back to rayon's default (unpinned, all logical)
/// for A/B comparison. `RAYON_NUM_THREADS`, if set, caps the worker count.
fn configure_thread_pool() {
    if std::env::var("ZLLM_NO_PIN").map(|v| v == "1").unwrap_or(false) {
        return;
    }
    let ids = core_affinity::get_core_ids().unwrap_or_default();
    if ids.is_empty() {
        return;
    }
    // One logical thread per physical core (skip SMT siblings).
    let mut phys: Vec<core_affinity::CoreId> = ids.iter().step_by(2).cloned().collect();
    if let Ok(n) = std::env::var("RAYON_NUM_THREADS").map(|v| v.parse::<usize>()) {
        if let Ok(n) = n {
            if n > 0 && n < phys.len() {
                phys.truncate(n);
            }
        }
    }
    let n = phys.len().max(1);
    let pins = phys.clone();
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .start_handler(move |i| {
            if let Some(id) = pins.get(i) {
                core_affinity::set_for_current(*id);
            }
        })
        .build_global();
    tracing::info!("rayon pool: {} workers pinned to physical cores", n);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "zllm=info".into()),
        )
        .init();

    configure_thread_pool();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { config } => {
            use backend::candle::backend::CandleCpuBackend;
            use backend::candle::tokenizer::LlamaTokenizer;
            use backend::traits::{Backend, QuantConfig};
            use server::rest::{AppState, BackendSlot};
            use std::sync::{Arc, Mutex, RwLock};

            let cfg = config::ZllmConfig::load(&config)?;
            tracing::info!("Starting ZLLM server (REST: {})", cfg.server.rest_port);

            // Tokenizer: explicit path wins; otherwise look next to the model.
            let tokenizer = if !cfg.model.tokenizer_path.is_empty() {
                LlamaTokenizer::from_file(&cfg.model.tokenizer_path)?
            } else {
                let model_path = std::path::PathBuf::from(&cfg.model.path);
                let next_to = model_path
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .join("tokenizer.json");
                if next_to.exists() {
                    LlamaTokenizer::from_file(next_to.to_str().unwrap())?
                } else {
                    return Err(anyhow::anyhow!(
                        "tokenizer.json not found at model.tokenizer_path or next to model.path ({})",
                        next_to.display()
                    ));
                }
            };

            // Backend pool. Each slot holds its own model weights + KV
            // cache + prompt cache, so N slots = N×model RAM. Reading
            // from cfg.engine.backend_pool_size; default 2 — single
            // slot serialises like the pre-pool code, 2 lets us serve
            // a second client without contention.
            let pool_size = cfg.engine.backend_pool_size.unwrap_or(2).max(1);
            let model_path = std::path::PathBuf::from(&cfg.model.path);
            let model_exists = model_path.exists();
            if !model_exists {
                tracing::warn!(
                    "model file {} not found — server will start but /v1/chat/completions will fail until a model is loaded",
                    model_path.display()
                );
            }
            let draft_model_path: Option<std::path::PathBuf> = cfg
                .engine
                .draft_model_path
                .as_ref()
                .filter(|p| !p.trim().is_empty())
                .map(std::path::PathBuf::from);
            let draft_exists = draft_model_path
                .as_ref()
                .map(|p| p.exists())
                .unwrap_or(false);
            if let Some(p) = &draft_model_path {
                if !draft_exists {
                    tracing::warn!(
                        "draft_model_path = {} not found — spec-decode disabled",
                        p.display()
                    );
                }
            }
            let mut pool_slots: Vec<Mutex<BackendSlot>> = Vec::with_capacity(pool_size);
            for i in 0..pool_size {
                let mut be = CandleCpuBackend::new();
                if model_exists {
                    tracing::info!("loading main model into pool slot {}/{}", i + 1, pool_size);
                    be.load_model(&model_path, &QuantConfig {
                        method: "gguf".into(),
                        bits: 4,
                    })?;
                    // Pre-warm: do one cheap forward so subsequent cold
                    // requests don't pay the page-fault + JIT + scratch-
                    // allocation cost. The dummy run mmaps weight pages,
                    // populates Candle's lazy buffers, and warms the
                    // tokenizer-adjacent paths. Reset position right
                    // after so the KV cache is clean.
                    let warm_t = std::time::Instant::now();
                    if let Err(e) = be.forward_logits(&[1u32, 2, 3]) {
                        tracing::warn!("pre-warm forward failed in slot {}: {}", i + 1, e);
                    } else {
                        tracing::info!(
                            "pre-warmed main slot {}/{} in {} ms",
                            i + 1, pool_size, warm_t.elapsed().as_millis()
                        );
                    }
                    be.reset_position();
                }
                let draft = if draft_exists {
                    let p = draft_model_path.as_ref().unwrap();
                    tracing::info!("loading draft model into pool slot {}/{}", i + 1, pool_size);
                    let mut db = CandleCpuBackend::new();
                    db.load_model(p, &QuantConfig { method: "gguf".into(), bits: 4 })?;
                    let warm_t = std::time::Instant::now();
                    if let Err(e) = db.forward_logits(&[1u32, 2, 3]) {
                        tracing::warn!("pre-warm draft forward failed in slot {}: {}", i + 1, e);
                    } else {
                        tracing::info!(
                            "pre-warmed draft slot {}/{} in {} ms",
                            i + 1, pool_size, warm_t.elapsed().as_millis()
                        );
                    }
                    db.reset_position();
                    Some(db)
                } else {
                    None
                };
                pool_slots.push(Mutex::new(BackendSlot {
                    backend: be,
                    prompt_cache: Vec::new(),
                    draft,
                    draft_prompt_cache: Vec::new(),
                }));
            }

            let memory_store = Arc::new(RwLock::new(
                engine::memory_store::MemoryStore::new(1024, 256),
            ));

            // GoalManager persistence: save next to the config file by
            // default so the snapshot travels with the install.
            let goals_path = config
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .join("goals.json");
            let goal_manager = Arc::new(
                control_plane::goal_manager::GoalManager::new(memory_store.clone())
                    .with_save_path(goals_path),
            );
            // Restore prior state if a snapshot exists. No-op on first run.
            goal_manager.load_from_disk();

            // Resolve current_model id from the loaded path's stem (e.g.
            // "Llama-3.2-1B-Instruct-Q4_K_M") so /v1/models reports
            // what's actually in memory. Falls back to "zllm" when no
            // model loaded.
            let current_model = std::path::PathBuf::from(&cfg.model.path)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
                .filter(|_| model_path.exists())
                .unwrap_or_else(|| "zllm".to_string());

            let models_dir = if cfg.model.dir.is_empty() {
                None
            } else {
                Some(std::path::PathBuf::from(&cfg.model.dir))
            };

            // Build the hook registry the chat path consults on every
            // prefill. Default contents:
            //   - MemoryInjectHook (capture-only effective in v0.8 — see
            //     RunnerObserver doc; inject branch wakes up in v0.9).
            // The capture/inject layer indices match the InferenceRunner
            // defaults so test parity holds.
            let mut hook_registry = engine::hooks::registry::HookRegistry::new();
            let inject_layer = 8usize.saturating_sub(1);
            let capture_layer = 8 + cfg.engine.reasoning_layers.saturating_sub(1);
            hook_registry.register(Box::new(
                engine::hooks::memory_inject::MemoryInjectHook {
                    memory: memory_store.clone(),
                    inject_layer,
                    capture_layer,
                    alpha: 0.3,
                    max_memories: 5,
                },
            ));

            // Optional resident iGPU engine, loaded once at startup when the
            // model exists and `ZLLM_GPU=1`. The chat fast-lane (inspection
            // off, no spec/PLD/early-exit/grammar, prompt ≤512) routes whole
            // requests here — batched prefill + GPU decode. On any failure we
            // log and run CPU-only. Reloaded on model swap (see rest.rs).
            #[cfg(feature = "gpu")]
            let gpu_engine: Arc<Mutex<Option<backend::gpu::GpuModel>>> = {
                let loaded = if model_exists && std::env::var("ZLLM_GPU").is_ok() {
                    let t = std::time::Instant::now();
                    match backend::gpu::GpuContext::new().and_then(|ctx| {
                        backend::gpu::GpuModel::load(model_path.to_str().unwrap_or(""), ctx)
                    }) {
                        Ok(gm) => {
                            tracing::info!(
                                "GPU engine loaded in {} ms — chat fast-lane enabled (ZLLM_GPU=1)",
                                t.elapsed().as_millis()
                            );
                            Some(gm)
                        }
                        Err(e) => {
                            tracing::warn!("GPU engine load failed ({e}); running CPU-only");
                            None
                        }
                    }
                } else {
                    None
                };
                Arc::new(Mutex::new(loaded))
            };

            // GPU continuous-batching server — enabled via ZLLM_CB=1. Loads its
            // OWN GpuModel onto a dedicated serving thread and batches all
            // in-flight /v1/cb/completions requests together (vLLM-style). Slots
            // / max context configurable via ZLLM_CB_SLOTS (default 16) and
            // ZLLM_CB_SEQ (default 2048). Independent of the ZLLM_GPU fast-lane.
            #[cfg(feature = "gpu")]
            let cb_server: Option<Arc<backend::gpu::GpuBatchServer>> = {
                if model_exists && std::env::var("ZLLM_CB").is_ok() {
                    let slots = std::env::var("ZLLM_CB_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
                    let max_seq = std::env::var("ZLLM_CB_SEQ").ok().and_then(|v| v.parse().ok()).unwrap_or(2048usize);
                    let t = std::time::Instant::now();
                    match backend::gpu::GpuContext::new().and_then(|ctx| {
                        backend::gpu::GpuModel::load(model_path.to_str().unwrap_or(""), ctx)
                    }) {
                        Ok(gm) => {
                            let srv = backend::gpu::GpuBatchServer::spawn(gm, slots, max_seq);
                            tracing::info!(
                                "GPU continuous-batching server up in {} ms — {} slots x {} ctx (ZLLM_CB=1)",
                                t.elapsed().as_millis(), slots, max_seq
                            );
                            Some(Arc::new(srv))
                        }
                        Err(e) => { tracing::warn!("CB server load failed ({e}); /v1/cb disabled"); None }
                    }
                } else {
                    None
                }
            };

            // Raw-Vulkan (ash) decode engine — enabled via ZLLM_VK=1. Validated
            // bit-exact vs candle; beats CPU/wgpu on decode. Reloaded on swap.
            #[cfg(feature = "vulkan")]
            let vk_engine: Arc<Mutex<Option<backend::vulkan::VkModel>>> = {
                let loaded = if model_exists && std::env::var("ZLLM_VK").is_ok() {
                    let t = std::time::Instant::now();
                    match backend::vulkan::VkContext::new().and_then(|ctx| {
                        backend::vulkan::VkModel::load(model_path.to_str().unwrap_or(""), ctx)
                    }) {
                        Ok(m) => {
                            tracing::info!("Vulkan engine loaded in {} ms — chat fast-lane enabled (ZLLM_VK=1)", t.elapsed().as_millis());
                            Some(m)
                        }
                        Err(e) => { tracing::warn!("Vulkan engine load failed ({e}); not using it"); None }
                    }
                } else {
                    None
                };
                Arc::new(Mutex::new(loaded))
            };

            let state = AppState {
                pool: Arc::new(pool_slots),
                tokenizer: Arc::new(RwLock::new(tokenizer)),
                goals: goal_manager,
                memory: memory_store,
                engine: Arc::new(cfg.engine.clone()),
                models_dir,
                current_model: Arc::new(RwLock::new(current_model)),
                arch_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
                hooks: Arc::new(hook_registry),
                inspection_enabled: Arc::new(std::sync::atomic::AtomicBool::new(true)),
                pld_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                // Default OFF even when draft is loaded — on small
                // CPU model pairs (1B/3B) spec-decode is slower than
                // plain decode. User opts in via /v1/spec_decode/enabled
                // or the settings UI once they're on a model pair
                // (e.g. 8B+1B) where it actually wins.
                spec_decode_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                early_exit_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                early_exit_min_layer: Arc::new(std::sync::atomic::AtomicUsize::new(12)),
                early_exit_threshold_bits: Arc::new(std::sync::atomic::AtomicU32::new(0.30_f32.to_bits())),
                #[cfg(feature = "gpu")]
                gpu: gpu_engine,
                #[cfg(feature = "gpu")]
                cb: cb_server,
                #[cfg(feature = "vulkan")]
                vk: vk_engine,
            };

            let rest_addr = format!("0.0.0.0:{}", cfg.server.rest_port);
            let router = server::rest::router(state);

            let rest_handle = tokio::spawn(async move {
                let listener = tokio::net::TcpListener::bind(&rest_addr).await.unwrap();
                tracing::info!("REST server listening on {rest_addr}");
                axum::serve(listener, router).await.unwrap();
            });

            tokio::select! {
                _ = rest_handle => {},
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Shutting down...");
                },
            }
        }
        Commands::Generate {
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
        } => {
            use backend::candle::backend::CandleCpuBackend;
            use backend::candle::tokenizer::LlamaTokenizer;
            use backend::traits::{Backend, QuantConfig};

            // Load tokenizer
            let tok = if tokenizer.is_empty() {
                // Try to find tokenizer.json next to model file
                let tok_path = model.parent().unwrap_or(std::path::Path::new(".")).join("tokenizer.json");
                if tok_path.exists() {
                    LlamaTokenizer::from_file(tok_path.to_str().unwrap())?
                } else {
                    tracing::info!("Downloading tokenizer from HuggingFace...");
                    LlamaTokenizer::from_hf("meta-llama/Meta-Llama-3.1-8B-Instruct")?
                }
            } else {
                LlamaTokenizer::from_file(&tokenizer)?
            };

            // GPU fast-lane: ZLLM_VK=1 decodes on the raw-Vulkan VkModel engine (the optimized
            // forward — barrier-lean SDPA + tree combine + 2-pass partial; bit-exact vs candle).
            // This is the path that beats llama.cpp on short-context decode (all-Q4 model).
            #[cfg(feature = "vulkan")]
            if std::env::var("ZLLM_VK").is_ok() {
                let prompt_tokens = tok.encode(&prompt)?;
                tracing::info!("Prompt: {} tokens", prompt_tokens.len());
                let eos_id = tok.eos_token_id().unwrap_or(128001);
                let t_load = std::time::Instant::now();
                match backend::vulkan::VkContext::new()
                    .and_then(|ctx| backend::vulkan::VkModel::load(model.to_str().unwrap_or(""), ctx))
                {
                    Ok(vmodel) => {
                        tracing::info!("VkModel loaded in {} ms (ZLLM_VK)", t_load.elapsed().as_millis());
                        use std::io::Write;
                        print!("{prompt}");
                        std::io::stdout().flush()?;
                        // Prefill: feed the prompt; the last argmax is the first generated token.
                        let mut next = 0u32;
                        for (i, &t) in prompt_tokens.iter().enumerate() { next = vmodel.forward_argmax(t, i); }
                        let mut pos = prompt_tokens.len();
                        let start = std::time::Instant::now(); // time decode (steady-state)
                        let mut generated = 0usize;
                        while generated < max_tokens && next != eos_id && next != 128009 {
                            if let Ok(text) = tok.decode(&[next]) { print!("{text}"); std::io::stdout().flush()?; }
                            generated += 1;
                            next = vmodel.forward_argmax(next, pos); pos += 1;
                        }
                        let elapsed = start.elapsed();
                        println!("\n\n--- {} tokens in {:.2}s ({:.1} tok/s) [ZLLM_VK decode] ---",
                            generated, elapsed.as_secs_f64(), generated as f64 / elapsed.as_secs_f64());
                        return Ok(());
                    }
                    Err(e) => tracing::warn!("VkModel load failed ({e}); falling back to candle CPU"),
                }
            }

            // Load model
            let mut candle_backend = CandleCpuBackend::new();
            candle_backend.load_model(&model, &QuantConfig {
                method: "gguf".into(),
                bits: 4,
            })?;

            // Tokenize prompt
            let prompt_tokens = tok.encode(&prompt)?;
            tracing::info!("Prompt: {} tokens", prompt_tokens.len());

            // Generate tokens
            let eos_id = tok.eos_token_id().unwrap_or(128001);
            let mut all_tokens = prompt_tokens.clone();
            let start = std::time::Instant::now();
            let mut generated = 0usize;

            print!("{prompt}");
            use std::io::Write;
            std::io::stdout().flush()?;

            for _ in 0..max_tokens {
                let input_tokens = if generated == 0 {
                    &all_tokens[..]
                } else {
                    &all_tokens[all_tokens.len() - 1..]
                };

                let (token_id, _hidden) = candle_backend.generate_token(input_tokens)?;

                // Apply sampling (temperature + top_k + top_p would be applied to logits)
                // For now, generate_token returns greedy argmax
                // TODO: expose logits and use our sampler

                if token_id == eos_id || token_id == 128009 {
                    break;
                }

                all_tokens.push(token_id);
                generated += 1;

                // Decode and print token
                if let Ok(text) = tok.decode(&[token_id]) {
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
            }

            let elapsed = start.elapsed();
            let tok_per_sec = generated as f64 / elapsed.as_secs_f64();
            println!();
            println!();
            println!("--- {} tokens in {:.2}s ({:.1} tok/s) ---", generated, elapsed.as_secs_f64(), tok_per_sec);

            #[cfg(feature = "profile")]
            {
                let snap = crate::backend::candle::quantized_llama_fork::TIMING.snapshot();
                let nf = snap.n_forwards.max(1) as f64;
                eprintln!("--- profile (per forward, n={}) ---", snap.n_forwards);
                eprintln!("  total      {:.3} ms", snap.total_ms as f64 / nf);
                eprintln!("  attention  {:.3} ms  (qmm {:.3} ms)", snap.attention_ms as f64 / nf, snap.qmm_attn_ms as f64 / nf);
                eprintln!("  ffn        {:.3} ms  (qmm {:.3} ms)", snap.ffn_ms as f64 / nf, snap.qmm_ffn_ms as f64 / nf);
                eprintln!("  norm       {:.3} ms", snap.norm_ms as f64 / nf);
                eprintln!("  lm_head    {:.3} ms  (qmm {:.3} ms)", snap.lm_head_ms as f64 / nf, snap.qmm_lm_ms as f64 / nf);
                let qmm_total = (snap.qmm_attn_ms + snap.qmm_ffn_ms + snap.qmm_lm_ms) as f64 / nf;
                let sect_total = (snap.attention_ms + snap.ffn_ms + snap.norm_ms + snap.lm_head_ms) as f64 / nf;
                eprintln!("  -> matmul {:.3} ms / sections {:.3} ms / forward {:.3} ms", qmm_total, sect_total, snap.total_ms as f64 / nf);
                eprintln!("  -> non-matmul in sections: {:.3} ms; outside-forward (sampling/kv/convert): {:.3} ms",
                    sect_total - qmm_total, (tok_per_sec.recip() * 1000.0) - (snap.total_ms as f64 / nf));
            }
        }
    }

    Ok(())
}

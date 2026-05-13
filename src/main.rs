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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "zllm=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { config } => {
            let cfg = config::ZllmConfig::load(&config)?;
            tracing::info!("Starting ZLLM server (REST: {}, gRPC: {})", cfg.server.rest_port, cfg.server.grpc_port);

            // Shared memory store + goal manager (in-memory, no persistence).
            let memory_store = std::sync::Arc::new(std::sync::RwLock::new(
                engine::memory_store::MemoryStore::new(1024, 256),
            ));
            let goal_manager = std::sync::Arc::new(
                control_plane::goal_manager::GoalManager::new(memory_store.clone()),
            );

            // Start REST server
            let rest_router = server::rest::router();
            let rest_addr = format!("0.0.0.0:{}", cfg.server.rest_port);

            // Start gRPC server
            let grpc_addr = format!("0.0.0.0:{}", cfg.server.grpc_port).parse()?;
            let grpc_service = server::grpc::ZllmInferenceService;
            let goal_service = server::grpc::ZllmGoalService::new(goal_manager.clone());

            let grpc_handle = tokio::spawn(async move {
                tonic::transport::Server::builder()
                    .add_service(
                        server::grpc::inference_proto::inference_service_server::InferenceServiceServer::new(grpc_service),
                    )
                    .add_service(
                        server::grpc::control_proto::goal_service_server::GoalServiceServer::new(goal_service),
                    )
                    .serve(grpc_addr)
                    .await
                    .unwrap();
            });

            let rest_handle = tokio::spawn(async move {
                let listener = tokio::net::TcpListener::bind(&rest_addr).await.unwrap();
                tracing::info!("REST server listening on {rest_addr}");
                axum::serve(listener, rest_router).await.unwrap();
            });

            tokio::select! {
                _ = grpc_handle => {},
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
        }
    }

    Ok(())
}

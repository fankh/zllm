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
    /// Run performance benchmarks
    Bench {
        #[arg(short, long, default_value = "configs/default.toml")]
        config: PathBuf,
    },
    /// Manage hooks
    Hooks {
        #[command(subcommand)]
        action: HookAction,
    },
    /// Manage tenants
    Tenants {
        #[command(subcommand)]
        action: TenantAction,
    },
    /// Show live metrics
    Metrics,
}

#[derive(Subcommand)]
enum HookAction {
    /// List registered hooks
    List,
    /// Add a hook
    Add {
        #[arg(long)]
        r#type: String,
        #[arg(long)]
        layer: usize,
        #[arg(long, default_value = "0.9")]
        threshold: f32,
    },
}

#[derive(Subcommand)]
enum TenantAction {
    /// List tenants
    List,
    /// Create a tenant
    Create {
        #[arg(long)]
        name: String,
        #[arg(long, default_value = "4096")]
        quota_mb: u64,
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

            // Start REST server
            let rest_router = server::rest::router();
            let rest_addr = format!("0.0.0.0:{}", cfg.server.rest_port);

            // Start gRPC server
            let grpc_addr = format!("0.0.0.0:{}", cfg.server.grpc_port).parse()?;
            let grpc_service = server::grpc::ZllmInferenceService;

            let grpc_handle = tokio::spawn(async move {
                tonic::transport::Server::builder()
                    .add_service(
                        server::grpc::inference_proto::inference_service_server::InferenceServiceServer::new(grpc_service),
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
        Commands::Bench { config: _ } => {
            tracing::info!("Benchmark mode (stub)");
            println!("Benchmark not yet implemented. Use 'zllm serve' first.");
        }
        Commands::Hooks { action } => match action {
            HookAction::List => {
                println!("No hooks registered (engine not running).");
            }
            HookAction::Add { r#type, layer, threshold } => {
                println!("Hook added (stub): type={}, layer={layer}, threshold={threshold}", r#type);
            }
        },
        Commands::Tenants { action } => match action {
            TenantAction::List => {
                println!("No tenants (engine not running).");
            }
            TenantAction::Create { name, quota_mb } => {
                println!("Tenant created (stub): name={name}, quota={quota_mb}MB");
            }
        },
        Commands::Metrics => {
            println!("Metrics endpoint: http://localhost:8080/metrics (start server first)");
        }
    }

    Ok(())
}

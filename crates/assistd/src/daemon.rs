use anyhow::Result;
use assistd_core::{AppState, Config};
use assistd_llm::{LlamaService, ModelSpec, ServerSpec};
use clap::Args;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;
use tracing::info;

#[derive(Args)]
pub struct DaemonArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

pub async fn run(args: DaemonArgs) -> Result<()> {
    init_tracing();

    let config_path = match args.config {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;

    info!(
        "assistd v{} — local model agent OS assistant daemon",
        assistd_core::version()
    );
    info!("  core  v{}", assistd_core::version());
    info!("  llm   v{}", assistd_llm::version());
    info!("  voice v{}", assistd_voice::version());
    info!("  tools v{}", assistd_tools::version());
    info!("  wm    v{}", assistd_wm::version());
    info!("loaded config from {}", config_path.display());

    // One watch channel is the single source of truth for "shutdown requested":
    // the signal task flips it, the supervisor and the socket listener both
    // subscribe.
    let (shutdown_tx, _) = watch::channel(false);

    {
        let signal_tx = shutdown_tx.clone();
        tokio::spawn(async move {
            let mut term = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("failed to install SIGTERM handler: {e}");
                    return;
                }
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => info!("received SIGINT"),
                _ = term.recv() => info!("received SIGTERM"),
            }
            let _ = signal_tx.send(true);
        });
    }

    // Bring up llama-server and block until /health returns 200.
    let llama = LlamaService::start(
        ServerSpec {
            binary_path: config.llama_server.binary_path.clone(),
            host: config.llama_server.host.clone(),
            port: config.llama_server.port,
            gpu_layers: config.llama_server.gpu_layers,
        },
        ModelSpec {
            path: config.model.path.clone(),
            vram_budget_mb: config.model.vram_budget_mb,
            context_length: config.model.context_length,
        },
        shutdown_tx.subscribe(),
    )
    .await?;
    info!(
        "llama-server ready on {}:{}",
        config.llama_server.host, config.llama_server.port
    );

    let state = Arc::new(AppState::new(config, Arc::new(assistd_llm::EchoBackend)));

    let mut socket_shutdown_rx = shutdown_tx.subscribe();
    let socket_shutdown = async move {
        let _ = socket_shutdown_rx.wait_for(|v| *v).await;
    };

    let serve_result = assistd_core::socket::serve(state, socket_shutdown).await;

    if let Err(e) = llama.shutdown().await {
        tracing::error!("llama-server shutdown error: {e}");
    }

    serve_result?;
    info!("assistd stopped");
    Ok(())
}

pub fn init_config() -> Result<()> {
    init_tracing();
    let path = Config::default_path()?;
    Config::write_default(&path)?;
    info!("wrote default config to {}", path.display());
    Ok(())
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
}

use anyhow::Result;
use assistd_core::{AppState, Config, PresenceManager};
use assistd_llm::LlamaChatClient;
use assistd_tools::DenyAllGate;
use clap::Args;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;
use tracing::info;

use crate::{gpu_monitor, hotkey, idle_monitor};

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
    hotkey::validate(&config.presence)?;
    gpu_monitor::validate(&config.sleep)?;
    idle_monitor::validate(&config.sleep)?;

    let overflow_dir = PathBuf::from(&config.tools.output.overflow_dir);

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

    // Bring the daemon up in Active: PresenceManager cold-starts llama-server
    // (supervisor spawns, /health = 200, /models/load completes) before it
    // returns. Socket serving starts only after this succeeds.
    let presence = PresenceManager::new_active(
        config.to_server_spec(),
        config.to_model_spec(),
        shutdown_tx.subscribe(),
    )
    .await?;
    info!(
        "presence: Active (llama-server ready on {}:{})",
        config.llama_server.host, config.llama_server.port
    );

    let chat = LlamaChatClient::new(config.to_chat_spec())?;
    let hotkey_handle =
        hotkey::spawn_listener(&config.presence, presence.clone(), shutdown_tx.subscribe());
    let gpu_monitor_handle =
        gpu_monitor::spawn_monitor(&config.sleep, presence.clone(), shutdown_tx.subscribe());
    let idle_monitor_handle =
        idle_monitor::spawn_monitor(&config.sleep, presence.clone(), shutdown_tx.subscribe());

    // IPC-connected clients (including `assistd query`) have no interactive
    // channel, so destructive bash commands are denied by default. To
    // approve such commands, run them from the chat TUI where the modal
    // overlay can prompt the user.
    let tools = assistd_core::build_tools(&config, overflow_dir.clone(), Arc::new(DenyAllGate))?;
    info!(
        "tools: registered {} (overflow dir {})",
        tools.len(),
        overflow_dir.display()
    );

    let state = Arc::new(AppState::new(
        config,
        Arc::new(chat),
        presence.clone(),
        tools,
    ));

    let mut socket_shutdown_rx = shutdown_tx.subscribe();
    let socket_shutdown = async move {
        let _ = socket_shutdown_rx.wait_for(|v| *v).await;
    };

    let serve_result = assistd_core::socket::serve(state, socket_shutdown).await;

    // Drop to Sleeping on shutdown: tears down the supervisor and the child.
    if let Err(e) = presence.sleep().await {
        tracing::error!("presence shutdown error: {e:#}");
    }

    if let Some(h) = hotkey_handle {
        let _ = h.await;
    }
    if let Some(h) = gpu_monitor_handle {
        let _ = h.await;
    }
    if let Some(h) = idle_monitor_handle {
        let _ = h.await;
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

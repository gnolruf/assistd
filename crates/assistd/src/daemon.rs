use anyhow::Result;
use assistd_core::{AppState, Config, PresenceManager};
use assistd_llm::LlamaChatClient;
use assistd_tools::{
    CommandRegistry, PresentSpec, RunTool, ToolRegistry,
    commands::{
        BashCommand, CatCommand, EchoCommand, GrepCommand, LsCommand, SeeCommand, WcCommand,
        WebCommand, WriteCommand,
    },
};
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

    // Reset the Layer 2 overflow directory on every startup. Clearing it
    // means the per-daemon monotonic counter always starts at 1 with no
    // stale `cmd-<n>.txt` files left over from a previous run.
    let overflow_dir = PathBuf::from(&config.tools.output.overflow_dir);
    if overflow_dir.exists() {
        std::fs::remove_dir_all(&overflow_dir).map_err(|e| {
            anyhow::anyhow!(
                "failed to clear tools.output.overflow_dir {}: {e}",
                overflow_dir.display()
            )
        })?;
    }
    std::fs::create_dir_all(&overflow_dir).map_err(|e| {
        anyhow::anyhow!(
            "failed to create tools.output.overflow_dir {}: {e}",
            overflow_dir.display()
        )
    })?;

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

    let mut commands = CommandRegistry::new();
    commands.register(CatCommand);
    commands.register(LsCommand);
    commands.register(GrepCommand);
    commands.register(WcCommand);
    commands.register(EchoCommand);
    commands.register(WriteCommand);
    commands.register(SeeCommand);
    commands.register(WebCommand::new());
    commands.register(BashCommand::default());
    let commands = Arc::new(commands);

    let present_spec = PresentSpec {
        max_lines: config.tools.output.max_lines as usize,
        max_bytes: (config.tools.output.max_kb as usize) * 1024,
        overflow_dir: overflow_dir.clone(),
    };
    let mut tools = ToolRegistry::new();
    tools.register(RunTool::new(commands.clone(), present_spec));
    let tools = Arc::new(tools);
    info!(
        "tools: registered {} ({} commands, overflow dir {})",
        tools.len(),
        commands.len(),
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

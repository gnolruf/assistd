//! Daemon entrypoint: bring up every subsystem, serve the IPC socket,
//! tear down in order.

use anyhow::{Context, Result};
use assistd_core::{AppState, Config, MemoryStack, PresenceManager, RuntimeState, Subsystems};
use assistd_llm::LlamaChatClient;
use assistd_tools::{IpcConfirmationGate, MemoryOps};
use clap::Args;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::info;

use crate::{
    embed_init, gpu_monitor, hotkey, idle_monitor, ipc_voice_proxy, listen_dispatcher, mcp_init,
    memory_init, voice_init, wm_init,
};

/// Command-line arguments for the `daemon` subcommand.
#[derive(Args)]
pub struct DaemonArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
    /// Defer the global PTT hotkey to a connected client (e.g. the
    /// chat TUI). Set automatically when the daemon is auto-spawned by
    /// `assistd chat`.
    #[arg(long, default_value_t = false)]
    pub client_mode: bool,
}

/// Run the daemon until shutdown.
pub async fn run(args: DaemonArgs) -> Result<()> {
    init_tracing();

    if args.client_mode {
        match rustix::process::setsid() {
            Ok(_) => info!("detached: became session leader"),
            Err(e) => tracing::warn!("setsid() failed (continuing): {e}"),
        }
    }

    let config_path = match args.config {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;
    hotkey::validate(&config.presence, &config.voice)?;
    idle_monitor::validate(&config.sleep)?;
    assistd_voice::mic_validate(&config.voice)?;

    info!(
        "assistd v{}: local model agent OS assistant daemon",
        assistd_core::version()
    );
    info!("  core  v{}", assistd_core::version());
    info!("  llm   v{}", assistd_llm::version());
    info!("  voice v{}", assistd_voice::version());
    info!("  tools v{}", assistd_tools::version());
    info!("  wm    v{}", assistd_wm::version());
    info!("loaded config from {}", config_path.display());

    let (shutdown_tx, _) = watch::channel(false);
    spawn_signal_handler(&shutdown_tx);

    let mut startup_shutdown_rx = shutdown_tx.subscribe();
    let started = tokio::select! {
        biased;
        _ = startup_shutdown_rx.wait_for(|v| *v) => None,
        started = start(config, &config_path, args.client_mode, &shutdown_tx) => Some(started?),
    };
    let Some((state, subsystems)) = started else {
        info!("shutdown requested during startup; assistd stopped");
        return Ok(());
    };

    let mut socket_shutdown_rx = shutdown_tx.subscribe();
    let socket_shutdown = async move {
        let _ = socket_shutdown_rx.wait_for(|v| *v).await;
    };

    let serve_result = assistd_core::socket::serve(state, socket_shutdown).await;

    shutdown_subsystems(subsystems).await;

    serve_result?;
    info!("assistd stopped");
    Ok(())
}

async fn start(
    config: Config,
    config_path: &std::path::Path,
    client_mode: bool,
    shutdown_tx: &watch::Sender<bool>,
) -> Result<(Arc<AppState>, DaemonShutdown)> {
    let overflow_dir = PathBuf::from(&config.tools.output.overflow_dir);

    let presence = PresenceManager::new_active(
        config.llama_server.clone(),
        config.model.clone(),
        config.timeouts.clone(),
        shutdown_tx.subscribe(),
    )
    .await?;
    info!(
        "presence: Active (llama-server ready on {}:{})",
        config.llama_server.host, config.llama_server.port
    );

    assistd_core::install_panic_hook(Arc::downgrade(&presence));

    let vision_revalidator = probe_vision(&config, &presence).await?;

    let health_probe: Arc<dyn assistd_llm::LlmHealthProbe> = Arc::new(
        assistd_core::presence::PresenceLlmHealthProbe::new(presence.clone()),
    );

    let voice = voice_init::init(&config, &presence).await;

    let hotkey_handle = if client_mode {
        info!("hotkey: deferred to client (--client-mode)");
        None
    } else {
        spawn_hotkeys(&config, &presence, &voice, shutdown_tx.subscribe())
    };
    let gpu_monitor_handle =
        gpu_monitor::spawn_monitor(&config.sleep, presence.clone(), shutdown_tx.subscribe());
    let idle_monitor_handle =
        idle_monitor::spawn_monitor(&config.sleep, presence.clone(), shutdown_tx.subscribe());

    let mut memory = memory_init::init(&config, shutdown_tx).await;
    let memory_store = memory.memory_store.clone();
    let conversation_store = memory.conversation_store.clone();
    let session_id_for_state = memory.session_id.clone();
    let branch_id_for_state = memory.branch_id;
    let resumed_history = std::mem::take(&mut memory.resumed_history);
    let sqlite_handle = memory.sqlite_handle.clone();

    let memory_ops = Arc::new(MemoryOps::new(memory_store.clone(), conversation_store));

    let embed = embed_init::init(&config, sqlite_handle.as_ref(), shutdown_tx).await;
    let embedder = embed.embedder.clone();
    let semantic_store = embed.semantic_store.clone();
    let embed_tx = embed.embed_tx.clone();
    let embedding_model_name = embed.model_name.clone();

    let window = wm_init::init(&config, shutdown_tx).await;
    let window_manager = window.manager.clone();

    let mut mcp = mcp_init::init(&config, shutdown_tx).await;
    let mcp_tools = std::mem::take(&mut mcp.tools);
    let mcp_startup_failures = mcp.startup_failures.clone();

    let conversation_ctx = Arc::new(assistd_core::ConversationContext::from_arc(
        session_id_for_state,
        branch_id_for_state,
    ));

    let tools = assistd_core::build_tools(assistd_core::BuildToolsDeps {
        config: &config,
        config_path,
        overflow_dir: overflow_dir.clone(),
        confirmation_gate: Arc::new(IpcConfirmationGate),
        vision_gate: vision_revalidator.gate(),
        memory_ops,
        embedder: embedder.clone(),
        semantic: semantic_store.clone(),
        embed_tx: embed_tx.clone(),
        embedding_model: embedding_model_name,
        current_session: conversation_ctx.session_updates(),
        window_manager: window_manager.clone(),
        mcp_tools,
    })?;
    info!(
        "tools: registered {} (overflow dir {})",
        tools.len(),
        overflow_dir.display()
    );

    let chat = LlamaChatClient::new(
        &config.chat,
        &config.llama_server,
        &config.model,
        &config.timeouts,
        Some(health_probe.clone()),
    )?;

    let continuous_enabled = config.voice.enabled && config.voice.continuous.enabled;
    let continuous_start_on_launch = config.voice.continuous.start_on_launch;

    let embedding_cfg_for_state = config.embedding.clone();
    let chat: Arc<dyn assistd_llm::LlmBackend> = Arc::new(chat);

    replay_history(chat.as_ref(), &resumed_history).await;

    let subsystems = Subsystems::new(
        chat,
        presence.clone(),
        tools,
        voice.input.clone(),
        voice.listener.clone(),
        voice.output,
    )
    .with_window_manager(window_manager)
    .with_vision_revalidator(vision_revalidator)
    .with_mcp_startup_failures(mcp_startup_failures);

    let mut memory_stack = MemoryStack::disabled(embedding_cfg_for_state)
        .with_memory(memory_store)
        .with_conversations(memory.conversation_store.clone())
        .with_embedder(embedder)
        .with_semantic(semantic_store)
        .with_embed_tx(embed_tx);
    if let Some(handle) = sqlite_handle {
        memory_stack = memory_stack.with_chunks(handle);
    }

    let runtime = RuntimeState::new().with_conversation_ctx(conversation_ctx);

    let state = Arc::new(AppState {
        config,
        subsystems,
        memory: memory_stack,
        runtime,
    });
    let persistence_tracker = state.runtime.persistence_tracker_handle();

    let listen_handles = if continuous_enabled {
        Some(listen_dispatcher::spawn_dispatcher(
            state.clone(),
            voice.listener.clone(),
            presence.clone(),
            continuous_start_on_launch,
            shutdown_tx.subscribe(),
        ))
    } else {
        None
    };

    Ok((
        state,
        DaemonShutdown {
            persistence_tracker,
            presence,
            memory,
            embed,
            window,
            mcp,
            hotkey_handle,
            gpu_monitor_handle,
            idle_monitor_handle,
            listen_handles,
        },
    ))
}

/// Write a default config file to the platform config directory.
pub fn init_config() -> Result<()> {
    init_tracing();
    let path = Config::default_path()?;
    Config::write_default(&path)?;
    info!("wrote default config to {}", path.display());
    Ok(())
}

/// Probe llama-server for vision support once and build the revalidator
/// whose gate it seeds.
async fn probe_vision(
    config: &Config,
    presence: &PresenceManager,
) -> Result<Arc<assistd_core::VisionRevalidator>> {
    let control = assistd_llm::LlamaServerControl::new(
        &config.llama_server.host.to_string(),
        config.llama_server.port.get(),
    )
    .context("failed to construct llama-server control client for vision probe")?;
    let revalidator =
        assistd_core::VisionRevalidator::new(control, config.model.name.clone(), presence).await;
    if revalidator.gate().supported() {
        info!("vision: enabled (model has mmproj)");
    } else {
        tracing::warn!("Vision not available: mmproj not loaded.");
    }
    Ok(revalidator)
}

/// The daemon's own hotkeys route push-to-talk through its IPC socket,
/// so the daemon and the chat TUI share one PTT path.
fn spawn_hotkeys(
    config: &Config,
    presence: &Arc<PresenceManager>,
    voice: &voice_init::VoiceSubsystem,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    let voice_proxy: Arc<dyn assistd_voice::VoiceInput> = Arc::new(
        ipc_voice_proxy::IpcVoiceProxy::new(Arc::new(assistd_ipc::IpcClient::new()), None),
    );
    hotkey::spawn_listener(
        &config.presence,
        &config.voice,
        hotkey::Subsystems {
            presence: Some(presence.clone()),
            voice: voice_proxy,
            listener: Some(voice.listener.clone()),
            voice_output: Some(voice.output.clone()),
        },
        shutdown,
    )
}

fn spawn_signal_handler(shutdown_tx: &watch::Sender<bool>) {
    let signal_tx = shutdown_tx.clone();
    assistd_core::spawn_supervised(
        "signal_handler",
        assistd_core::Component::Daemon,
        async move {
            let (mut int, mut term) = match (
                signal(SignalKind::interrupt()),
                signal(SignalKind::terminate()),
            ) {
                (Ok(int), Ok(term)) => (int, term),
                (Err(e), _) | (_, Err(e)) => {
                    tracing::error!("failed to install signal handlers: {e}");
                    return;
                }
            };
            loop {
                let (name, exit_code) = tokio::select! {
                    _ = int.recv() => ("SIGINT", 130),
                    _ = term.recv() => ("SIGTERM", 143),
                };
                if signal_tx.send_replace(true) {
                    tracing::warn!("received {name} again; exiting without cleanup");
                    std::process::exit(exit_code);
                }
                info!("received {name}; shutting down (send again to force exit)");
            }
        },
    );
}

async fn replay_history(chat: &dyn assistd_llm::LlmBackend, rows: &[assistd_memory::HistoryRow]) {
    if rows.is_empty() {
        return;
    }
    let count = rows.len();
    if let Err(e) = chat
        .replace_history(assistd_core::history_entries(rows))
        .await
    {
        tracing::warn!("memory: resume replay failed: {e:#}");
    } else {
        info!("memory: resumed {count} message(s) from prior branch");
    }
}

struct DaemonShutdown {
    persistence_tracker: tokio_util::task::TaskTracker,
    presence: Arc<PresenceManager>,
    memory: memory_init::MemorySubsystem,
    embed: embed_init::EmbeddingSubsystem,
    window: wm_init::WindowSubsystem,
    mcp: mcp_init::McpSubsystem,
    hotkey_handle: Option<JoinHandle<()>>,
    gpu_monitor_handle: Option<JoinHandle<()>>,
    idle_monitor_handle: Option<JoinHandle<()>>,
    listen_handles: Option<listen_dispatcher::ListenDispatcherHandles>,
}

async fn shutdown_subsystems(s: DaemonShutdown) {
    s.persistence_tracker.close();
    let drain_budget = Duration::from_secs(5);
    if tokio::time::timeout(drain_budget, s.persistence_tracker.wait())
        .await
        .is_err()
    {
        tracing::warn!(
            target: "assistd::memory",
            in_flight = s.persistence_tracker.len(),
            "persistence task drain timed out at shutdown; abandoning remaining tasks"
        );
    }

    if let Err(e) = s.presence.sleep().await {
        tracing::error!("presence shutdown error: {e:#}");
    }

    s.memory.shutdown().await;

    if let Some(h) = s.hotkey_handle {
        let _ = h.await;
    }
    if let Some(h) = s.gpu_monitor_handle {
        let _ = h.await;
    }
    if let Some(h) = s.idle_monitor_handle {
        let _ = h.await;
    }
    if let Some(handles) = s.listen_handles {
        let _ = handles.forwarder.await;
        let _ = handles.presence_gate.await;
    }
    s.window.shutdown().await;
    s.mcp.shutdown().await;
    s.embed.shutdown().await;
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
}

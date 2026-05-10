//! Daemon entrypoint: parses args, loads/validates config, brings up
//! every subsystem (each via its own `*_init` sibling module), assembles
//! [`AppState`], serves the IPC socket, and orchestrates an ordered
//! shutdown when the run loop exits.
//!
//! Each subsystem (voice, wm, mcp, memory, embed) lives in its own
//! `<name>_init.rs` module and exposes a composite "Subsystem" struct
//! plus an `init(...)` async constructor. The daemon is a thin
//! orchestrator: it composes the subsystems' Arc'd trait handles into
//! [`AppState`] and retains the shutdown-relevant pieces in
//! [`DaemonShutdown`] for ordered teardown.

use anyhow::Result;
use assistd_core::{AppState, Config, PresenceManager};
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
    embed_init, gpu_monitor, hotkey, idle_monitor, listen_dispatcher, mcp_init, memory_init,
    voice_init, wm_init,
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

/// Start the assistd daemon: load config, init subsystems, serve the IPC socket.
///
/// # Errors
///
/// Returns an error if config loading, validation, or the IPC socket fails.
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
    gpu_monitor::validate(&config.sleep)?;
    idle_monitor::validate(&config.sleep)?;
    assistd_voice::mic_validate(&config.voice)?;

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

    let (shutdown_tx, _) = watch::channel(false);
    spawn_signal_handler(&shutdown_tx);

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

    let initial_vision_state =
        assistd_llm::probe_capabilities(&config.llama_server.host, config.llama_server.port).await;
    if initial_vision_state.vision_supported {
        info!("vision: enabled (model has mmproj)");
    } else {
        tracing::warn!("Vision not available: mmproj not loaded.");
    }
    let vision_gate = assistd_tools::VisionGate::new(initial_vision_state.vision_supported);
    let vision_revalidator = assistd_core::VisionRevalidator::new(
        vision_gate.clone(),
        initial_vision_state.model_id,
        config.llama_server.host.clone(),
        config.llama_server.port,
    );

    let health_probe: std::sync::Arc<dyn assistd_llm::LlmHealthProbe> = std::sync::Arc::new(
        assistd_core::presence::PresenceLlmHealthProbe::new(presence.clone()),
    );
    let chat = LlamaChatClient::new(
        &config.chat,
        &config.llama_server,
        &config.model,
        &config.timeouts,
        Some(health_probe.clone()),
    )?;

    let voice = voice_init::init(&config, &presence).await;

    let hotkey_handle = if args.client_mode {
        info!("hotkey: deferred to client (--client-mode)");
        None
    } else {
        hotkey::spawn_listener(
            &config.presence,
            &config.voice,
            hotkey::Subsystems {
                presence: Some(presence.clone()),
                voice: voice.input.clone(),
                listener: Some(voice.listener.clone()),
                voice_output: Some(voice.output.clone()),
            },
            shutdown_tx.subscribe(),
        )
    };
    let gpu_monitor_handle =
        gpu_monitor::spawn_monitor(&config.sleep, presence.clone(), shutdown_tx.subscribe());
    let idle_monitor_handle =
        idle_monitor::spawn_monitor(&config.sleep, presence.clone(), shutdown_tx.subscribe());

    let mut memory = memory_init::init(&config, &shutdown_tx).await;
    let memory_store = memory.memory_store.clone();
    let conversation_store = memory.conversation_store.clone();
    let session_id_for_state = memory.session_id.clone();
    let branch_id_for_state = memory.branch_id;
    let resumed_history = std::mem::take(&mut memory.resumed_history);
    let sqlite_handle = memory.sqlite_handle.clone();

    let memory_ops = Arc::new(MemoryOps::new(memory_store.clone(), conversation_store));

    let embed = embed_init::init(&config, sqlite_handle.as_ref(), &shutdown_tx).await;
    let embedder = embed.embedder.clone();
    let semantic_store = embed.semantic_store.clone();
    let embed_tx = embed.embed_tx.clone();
    let embedding_model_name = embed.model_name.clone();

    let window = wm_init::init(&config, &shutdown_tx).await;
    let window_manager = window.manager.clone();

    let mut mcp = mcp_init::init(&config, &shutdown_tx).await;
    let mcp_tools = std::mem::take(&mut mcp.tools);

    let tools = assistd_core::build_tools(assistd_core::BuildToolsDeps {
        config: &config,
        overflow_dir: overflow_dir.clone(),
        confirmation_gate: Arc::new(IpcConfirmationGate),
        vision_gate: vision_gate.clone(),
        memory_ops,
        embedder: embedder.clone(),
        semantic: semantic_store.clone(),
        embed_tx: embed_tx.clone(),
        embedding_model: embedding_model_name,
        window_manager: window_manager.clone(),
        mcp_tools,
    })?;
    info!(
        "tools: registered {} (overflow dir {})",
        tools.len(),
        overflow_dir.display()
    );

    let continuous_enabled = config.voice.enabled && config.voice.continuous.enabled;
    let continuous_start_on_launch = config.voice.continuous.start_on_launch;

    let embedding_cfg_for_state = config.embedding.clone();
    let conversation_ctx = Arc::new(assistd_core::ConversationContext::from_arc(
        session_id_for_state,
        branch_id_for_state,
    ));
    let chat: Arc<dyn assistd_llm::LlmBackend> = Arc::new(chat);

    replay_history(chat.as_ref(), resumed_history).await;
    let mut state_builder = AppState::new(
        config,
        chat,
        presence.clone(),
        tools,
        voice.input.clone(),
        voice.listener.clone(),
        voice.output,
    )
    .with_vision_revalidator(vision_revalidator)
    .with_memory(memory_store)
    .with_conversations(memory.conversation_store.clone())
    .with_conversation_ctx(conversation_ctx)
    .with_embedder(embedder)
    .with_semantic(semantic_store)
    .with_embed_tx(embed_tx)
    .with_embedding_cfg(embedding_cfg_for_state)
    .with_window_manager(window_manager);
    if let Some(handle) = sqlite_handle {
        state_builder = state_builder.with_chunks(handle);
    }
    let state = Arc::new(state_builder);
    let persistence_tracker = state.persistence_tracker();

    let listen_handles = if continuous_enabled {
        Some(listen_dispatcher::spawn(
            state.clone(),
            voice.listener.clone(),
            presence.clone(),
            continuous_start_on_launch,
            /* pause_when_sleeping = */ true,
            shutdown_tx.subscribe(),
        ))
    } else {
        None
    };

    let mut socket_shutdown_rx = shutdown_tx.subscribe();
    let socket_shutdown = async move {
        let _ = socket_shutdown_rx.wait_for(|v| *v).await;
    };

    let serve_result = assistd_core::socket::serve(state, socket_shutdown).await;

    shutdown_subsystems(DaemonShutdown {
        persistence_tracker,
        presence: presence.clone(),
        memory,
        embed,
        window,
        mcp,
        hotkey_handle,
        gpu_monitor_handle,
        idle_monitor_handle,
        listen_handles,
    })
    .await;

    serve_result?;
    info!("assistd stopped");
    Ok(())
}

/// Write a default config file to the platform config directory.
///
/// # Errors
///
/// Returns an error if the config path cannot be determined or the file cannot be written.
pub fn init_config() -> Result<()> {
    init_tracing();
    let path = Config::default_path()?;
    Config::write_default(&path)?;
    info!("wrote default config to {}", path.display());
    Ok(())
}

fn spawn_signal_handler(shutdown_tx: &watch::Sender<bool>) {
    let signal_tx = shutdown_tx.clone();
    assistd_core::spawn_supervised(
        "signal_handler",
        assistd_core::Component::Daemon,
        async move {
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
        },
    );
}

async fn replay_history(chat: &dyn assistd_llm::LlmBackend, rows: Vec<assistd_memory::HistoryRow>) {
    if rows.is_empty() {
        return;
    }
    let entries: Vec<assistd_llm::HistoryEntry> = rows
        .into_iter()
        .map(|r| assistd_llm::HistoryEntry {
            role: persisted_role_to_history_role(r.role),
            content: r.content,
            tool_calls_json: r.tool_calls,
            tool_call_id: r.tool_call_id,
            tool_name: r.tool_name,
        })
        .collect();
    let count = entries.len();
    if let Err(e) = chat.replace_history(entries).await {
        tracing::warn!("memory: resume replay failed: {e:#}");
    } else {
        info!("memory: resumed {count} message(s) from prior branch");
    }
}

fn persisted_role_to_history_role(role: assistd_memory::PersistedRole) -> assistd_llm::HistoryRole {
    match role {
        assistd_memory::PersistedRole::System => assistd_llm::HistoryRole::System,
        assistd_memory::PersistedRole::User => assistd_llm::HistoryRole::User,
        assistd_memory::PersistedRole::Assistant => assistd_llm::HistoryRole::Assistant,
        assistd_memory::PersistedRole::Tool => assistd_llm::HistoryRole::Tool,
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

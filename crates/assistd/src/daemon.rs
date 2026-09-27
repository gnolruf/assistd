//! Daemon entrypoint: bring up every subsystem, serve the IPC socket,
//! tear down in order.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_core::presence::PresenceLlmHealthProbe;
use assistd_core::socket::StartupLock;
use assistd_core::{
    AppState, BuildToolsDeps, Component, Config, ContinuousListener, ConversationContext,
    MemoryStack, PresenceManager, RuntimeState, Subsystems, VisionRevalidator, spawn_supervised,
};
use assistd_ipc::IpcClient;
use assistd_llm::{LlamaChatClient, LlamaServerControl, LlmBackend, LlmHealthProbe};
use assistd_memory::HistoryRow;
use assistd_tools::{IpcConfirmationGate, MemoryOps};
use assistd_utils::tracing_init::env_filter_or;
use clap::Args;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::task::TaskTracker;
use tracing::info;

use crate::embed_init::EmbeddingSubsystem;
use crate::ipc_voice_proxy::IpcVoiceProxy;
use crate::listen_dispatcher::ListenDispatcherHandles;
use crate::mcp_init::McpSubsystem;
use crate::memory_init::MemorySubsystem;
use crate::voice_init::VoiceSubsystem;
use crate::wm_init::WindowSubsystem;
use crate::{
    embed_init, gpu_monitor, hotkey, idle_monitor, listen_dispatcher, mcp_init, memory_init,
    voice_init, wm_init,
};

const PERSISTENCE_DRAIN_BUDGET: Duration = Duration::from_secs(5);

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

/// One watch per teardown stage. A signal flips only `intake`; the rest
/// flip in dependency order once in-flight work has drained.
struct ShutdownStages {
    intake: watch::Sender<bool>,
    llm: watch::Sender<bool>,
    tools: watch::Sender<bool>,
    embed_worker: watch::Sender<bool>,
    embed_server: watch::Sender<bool>,
    memory_writer: watch::Sender<bool>,
}

impl ShutdownStages {
    fn new() -> Self {
        Self {
            intake: watch::channel(false).0,
            llm: watch::channel(false).0,
            tools: watch::channel(false).0,
            embed_worker: watch::channel(false).0,
            embed_server: watch::channel(false).0,
            memory_writer: watch::channel(false).0,
        }
    }

    /// Flip every stage at once, for a startup abandoned mid-way.
    fn cancel_all(&self) {
        for stage in [
            &self.intake,
            &self.llm,
            &self.tools,
            &self.embed_worker,
            &self.embed_server,
            &self.memory_writer,
        ] {
            stage.send_replace(true);
        }
    }
}

struct DaemonShutdown {
    persistence_tracker: TaskTracker,
    presence: Arc<PresenceManager>,
    memory: MemorySubsystem,
    embed: EmbeddingSubsystem,
    window: WindowSubsystem,
    mcp: McpSubsystem,
    intake_tasks: IntakeTasks,
}

impl DaemonShutdown {
    /// Tear down after the socket has drained: finish intake tasks and
    /// persistence, then stop each subsystem before the ones it depends on.
    async fn shutdown(self, stages: &ShutdownStages) {
        join_intake_tasks(self.intake_tasks).await;
        drain_persistence(&self.persistence_tracker).await;

        stages.llm.send_replace(true);
        if let Err(e) = self.presence.sleep().await {
            tracing::error!("presence shutdown error: {e:#}");
        }

        stages.tools.send_replace(true);
        self.window.shutdown().await;
        self.mcp.shutdown().await;

        self.embed
            .shutdown(&stages.embed_worker, &stages.embed_server)
            .await;
        self.memory.shutdown(&stages.memory_writer).await;
    }
}

/// Tasks that start new work and stop on the `intake` stage.
struct IntakeTasks {
    hotkey: Option<JoinHandle<()>>,
    gpu_monitor: Option<JoinHandle<()>>,
    idle_monitor: Option<JoinHandle<()>>,
    listen: Option<ListenDispatcherHandles>,
}

/// Run the daemon until shutdown.
pub async fn run(args: DaemonArgs) -> Result<()> {
    init_tracing();
    assistd_core::install_panic_hook();

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

    let _startup_lock = StartupLock::acquire()?;

    let stages = ShutdownStages::new();
    spawn_signal_handler(&stages.intake);

    let mut startup_shutdown_rx = stages.intake.subscribe();
    let started = tokio::select! {
        biased;
        _ = startup_shutdown_rx.wait_for(|v| *v) => None,
        started = start(config, &config_path, args.client_mode, &stages) => Some(started?),
    };
    let Some((state, subsystems)) = started else {
        stages.cancel_all();
        info!("shutdown requested during startup; assistd stopped");
        return Ok(());
    };

    let mut socket_shutdown_rx = stages.intake.subscribe();
    let socket_shutdown = async move {
        let _ = socket_shutdown_rx.wait_for(|v| *v).await;
    };

    let serve_result = assistd_core::socket::serve(state, socket_shutdown).await;

    subsystems.shutdown(&stages).await;

    serve_result?;
    info!("assistd stopped");
    Ok(())
}

async fn start(
    config: Config,
    config_path: &Path,
    client_mode: bool,
    stages: &ShutdownStages,
) -> Result<(Arc<AppState>, DaemonShutdown)> {
    let presence = start_presence(&config, &stages.llm).await?;
    let vision_revalidator = probe_vision(&config, &presence).await?;
    let health_probe: Arc<dyn LlmHealthProbe> =
        Arc::new(PresenceLlmHealthProbe::new(presence.clone()));

    let voice = voice_init::init(&config, &presence).await;

    let hotkey_handle = if client_mode {
        info!("hotkey: deferred to client (--client-mode)");
        None
    } else {
        spawn_hotkeys(&config, &presence, &voice, stages.intake.subscribe())
    };
    let gpu_monitor_handle =
        gpu_monitor::spawn_monitor(&config.sleep, presence.clone(), stages.intake.subscribe());
    let idle_monitor_handle =
        idle_monitor::spawn_monitor(&config.sleep, presence.clone(), stages.intake.subscribe());

    let mut memory = memory_init::init(&config, &stages.memory_writer).await;
    let embed = embed_init::init(
        &config,
        memory.sqlite_handle.as_ref(),
        &stages.embed_worker,
        &stages.embed_server,
    )
    .await;
    let window = wm_init::init(&config, &stages.tools).await;
    let mut mcp = mcp_init::init(&config, &stages.tools).await;

    let conversation_ctx = Arc::new(ConversationContext::from_arc(
        memory.session_id.clone(),
        memory.branch_id,
    ));

    let overflow_dir = PathBuf::from(&config.tools.output.overflow_dir);
    let tools = assistd_core::build_tools(BuildToolsDeps {
        config: &config,
        config_path,
        overflow_dir: overflow_dir.clone(),
        confirmation_gate: Arc::new(IpcConfirmationGate),
        vision_gate: vision_revalidator.gate(),
        memory_ops: Arc::new(MemoryOps::new(
            memory.memory_store.clone(),
            memory.conversation_store.clone(),
        )),
        embedder: embed.embedder.clone(),
        semantic: embed.semantic_store.clone(),
        embed_tx: embed.embed_tx.clone(),
        embedding_model: embed.model_name.clone(),
        current_session: conversation_ctx.session_updates(),
        window_manager: window.manager.clone(),
        mcp_tools: std::mem::take(&mut mcp.tools),
    })?;
    info!(
        "tools: registered {} (overflow dir {})",
        tools.len(),
        overflow_dir.display()
    );

    let chat = build_chat_backend(&config, health_probe)?;
    let resumed_history = std::mem::take(&mut memory.resumed_history);
    replay_history(chat.as_ref(), &resumed_history).await;

    let subsystems = Subsystems::new(
        chat,
        presence.clone(),
        tools,
        voice.input.clone(),
        voice.listener.clone(),
        voice.output,
    )
    .with_window_manager(window.manager.clone())
    .with_vision_revalidator(vision_revalidator)
    .with_mcp_startup_failures(mcp.startup_failures.clone());
    let memory_stack = build_memory_stack(&config, &memory, &embed);

    let state = Arc::new(AppState {
        config,
        subsystems,
        memory: memory_stack,
        runtime: RuntimeState::new().with_conversation_ctx(conversation_ctx),
    });
    let persistence_tracker = state.runtime.persistence_tracker_handle();
    let listen_handles =
        spawn_listen_dispatcher(&state, &voice.listener, &presence, &stages.intake);

    Ok((
        state,
        DaemonShutdown {
            persistence_tracker,
            presence,
            memory,
            embed,
            window,
            mcp,
            intake_tasks: IntakeTasks {
                hotkey: hotkey_handle,
                gpu_monitor: gpu_monitor_handle,
                idle_monitor: idle_monitor_handle,
                listen: listen_handles,
            },
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

async fn start_presence(
    config: &Config,
    llm_shutdown: &watch::Sender<bool>,
) -> Result<Arc<PresenceManager>> {
    let presence = PresenceManager::new_active(
        config.llama_server.clone(),
        config.model.clone(),
        config.timeouts.clone(),
        llm_shutdown.subscribe(),
    )
    .await?;
    info!(
        "presence: Active (llama-server ready on {}:{})",
        config.llama_server.host, config.llama_server.port
    );
    Ok(presence)
}

/// Probe llama-server for vision support once and build the revalidator
/// whose gate it seeds.
async fn probe_vision(
    config: &Config,
    presence: &PresenceManager,
) -> Result<Arc<VisionRevalidator>> {
    let control = LlamaServerControl::new(
        &config.llama_server.host.to_string(),
        config.llama_server.port.get(),
    )
    .context("failed to construct llama-server control client for vision probe")?;
    let revalidator = VisionRevalidator::new(control, config.model.name.clone(), presence).await;
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
    voice: &VoiceSubsystem,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    let voice_proxy: Arc<dyn assistd_voice::VoiceInput> =
        Arc::new(IpcVoiceProxy::new(Arc::new(IpcClient::new()), None));
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

fn build_chat_backend(
    config: &Config,
    health_probe: Arc<dyn LlmHealthProbe>,
) -> Result<Arc<dyn LlmBackend>> {
    let chat = LlamaChatClient::new(
        &config.chat,
        &config.llama_server,
        &config.model,
        &config.timeouts,
        Some(health_probe),
    )?;
    Ok(Arc::new(chat))
}

fn build_memory_stack(
    config: &Config,
    memory: &MemorySubsystem,
    embed: &EmbeddingSubsystem,
) -> MemoryStack {
    let stack = MemoryStack::disabled(config.embedding.clone())
        .with_memory(memory.memory_store.clone())
        .with_conversations(memory.conversation_store.clone())
        .with_embedder(embed.embedder.clone())
        .with_semantic(embed.semantic_store.clone())
        .with_embed_tx(embed.embed_tx.clone());
    match memory.sqlite_handle.clone() {
        Some(handle) => stack.with_chunks(handle),
        None => stack,
    }
}

fn spawn_listen_dispatcher(
    state: &Arc<AppState>,
    listener: &Arc<dyn ContinuousListener>,
    presence: &Arc<PresenceManager>,
    intake_shutdown: &watch::Sender<bool>,
) -> Option<ListenDispatcherHandles> {
    let voice = &state.config.voice;
    (voice.enabled && voice.continuous.enabled).then(|| {
        listen_dispatcher::spawn_dispatcher(
            state.clone(),
            listener.clone(),
            presence.clone(),
            voice.continuous.start_on_launch,
            intake_shutdown.subscribe(),
        )
    })
}

fn spawn_signal_handler(shutdown_tx: &watch::Sender<bool>) {
    spawn_supervised(
        "signal_handler",
        Component::Daemon,
        forward_signals(shutdown_tx.clone()),
    );
}

/// Flag shutdown on the first SIGINT/SIGTERM; a second signal exits
/// immediately without cleanup.
async fn forward_signals(shutdown_tx: watch::Sender<bool>) {
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
        if shutdown_tx.send_replace(true) {
            tracing::warn!("received {name} again; exiting without cleanup");
            std::process::exit(exit_code);
        }
        info!("received {name}; shutting down (send again to force exit)");
    }
}

async fn join_intake_tasks(tasks: IntakeTasks) {
    for handle in [tasks.hotkey, tasks.gpu_monitor, tasks.idle_monitor]
        .into_iter()
        .flatten()
    {
        let _ = handle.await;
    }
    if let Some(listen) = tasks.listen {
        let _ = listen.forwarder.await;
        let _ = listen.presence_gate.await;
    }
}

/// Wait for fire-and-forget persistence tasks, abandoning them after
/// [`PERSISTENCE_DRAIN_BUDGET`].
async fn drain_persistence(tracker: &TaskTracker) {
    tracker.close();
    if tokio::time::timeout(PERSISTENCE_DRAIN_BUDGET, tracker.wait())
        .await
        .is_err()
    {
        tracing::warn!(
            target: "assistd::memory",
            in_flight = tracker.len(),
            "persistence task drain timed out at shutdown; abandoning remaining tasks"
        );
    }
}

async fn replay_history(chat: &dyn LlmBackend, rows: &[HistoryRow]) {
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

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(env_filter_or("info"))
        .init();
}

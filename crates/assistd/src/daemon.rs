use anyhow::Result;
use assistd_core::{
    AppState, CompositorType, Config, ContinuousListener, NoContinuousListener, NoVoiceInput,
    NoVoiceOutput, NoWindowManager, PresenceManager, VoiceInput, VoiceOutput, WindowManager,
};
use assistd_embed::{
    EmbedJob, EmbedService, Embedder, LlamaEmbedder, NoEmbedder, spawn_embedder_task,
};
use assistd_llm::LlamaChatClient;
use assistd_memory::{
    ConversationStore, MemoryStore, NoConversationStore, NoMemoryStore, NoSemanticStore,
    SemanticStore, SqliteConversationStore, SqliteHandle, SqliteMemoryStore, SqliteSemanticStore,
};
use assistd_tools::{IpcConfirmationGate, MemoryOps};
use assistd_voice::{
    MicContinuousListener, MicVoiceInput, QueueConfig, QueuedTranscriber, Transcriber,
    WhisperTranscriberBuilder, build_cpu_fallback,
};
use assistd_wm::{I3Backend, SwayBackend, WmHandle};
use clap::Args;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::info;

use assistd_core::{McpServerConfig, McpTransport};
use assistd_mcp::{
    McpServerHandle, SseConfig, StdioConfig, TransportConfig, adapt_handle_as_tools,
};

use crate::voice_probe::PresenceGpuProbe;
use crate::{gpu_monitor, hotkey, idle_monitor, listen_dispatcher};

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

pub async fn run(args: DaemonArgs) -> Result<()> {
    init_tracing();

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

    {
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

    let VoiceSubsystem {
        input: voice,
        listener,
        output: voice_output,
    } = init_voice(&config, &presence).await;

    let hotkey_handle = if args.client_mode {
        info!("hotkey: deferred to client (--client-mode)");
        None
    } else {
        hotkey::spawn_listener(
            &config.presence,
            &config.voice,
            Some(presence.clone()),
            voice.clone(),
            Some(listener.clone()),
            Some(voice_output.clone()),
            shutdown_tx.subscribe(),
        )
    };
    let gpu_monitor_handle =
        gpu_monitor::spawn_monitor(&config.sleep, presence.clone(), shutdown_tx.subscribe());
    let idle_monitor_handle =
        idle_monitor::spawn_monitor(&config.sleep, presence.clone(), shutdown_tx.subscribe());

    let MemorySubsystem {
        memory_store,
        conversation_store,
        writer_handle: memory_writer_handle,
        session_id: session_id_for_state,
        branch_id: branch_id_for_state,
        resumed_history,
        sqlite_handle,
    } = init_memory(&config, &shutdown_tx).await;
    let conv_store_for_shutdown = conversation_store.clone();
    let session_for_shutdown = session_id_for_state.clone();

    let memory_ops = Arc::new(MemoryOps::new(
        memory_store.clone(),
        conversation_store.clone(),
    ));

    let EmbeddingSubsystem {
        embedder,
        semantic_store,
        embed_tx,
        service_handle: embed_service_handle,
        task_handle: embedder_task_handle,
        model_name: embedding_model_name,
    } = init_embedding(&config, sqlite_handle.as_ref(), &shutdown_tx).await;

    let WindowSubsystem {
        manager: window_manager,
        handle: wm_handle,
    } = init_window_manager(&config, &shutdown_tx).await;

    let McpSubsystem {
        handles: mcp_handles,
        tools: mcp_tools,
    } = init_mcp(&config, &shutdown_tx).await;

    let tools = assistd_core::build_tools(
        &config,
        overflow_dir.clone(),
        Arc::new(IpcConfirmationGate),
        vision_gate.clone(),
        memory_ops,
        embedder.clone(),
        semantic_store.clone(),
        embed_tx.clone(),
        embedding_model_name.clone(),
        window_manager.clone(),
        mcp_tools,
    )?;
    info!(
        "tools: registered {} (overflow dir {})",
        tools.len(),
        overflow_dir.display()
    );

    let continuous_enabled = config.voice.enabled && config.voice.continuous.enabled;
    let continuous_start_on_launch = config.voice.continuous.start_on_launch;

    let embedding_cfg_for_state = config.embedding.clone();
    let conversation_ctx = Arc::new(assistd_core::ConversationContext::from_arc(
        session_id_for_state.clone(),
        branch_id_for_state,
    ));
    let chat: Arc<dyn assistd_llm::LlmBackend> = Arc::new(chat);

    replay_history(chat.as_ref(), resumed_history).await;
    let mut state_builder = AppState::new(
        config,
        chat,
        presence.clone(),
        tools,
        voice.clone(),
        listener.clone(),
        voice_output,
    )
    .with_vision_revalidator(vision_revalidator)
    .with_memory(memory_store)
    .with_conversations(conversation_store)
    .with_conversation_ctx(conversation_ctx)
    .with_embedder(embedder)
    .with_semantic(semantic_store)
    .with_embed_tx(embed_tx)
    .with_embedding_cfg(embedding_cfg_for_state)
    .with_window_manager(window_manager);
    if let Some(handle) = sqlite_handle.clone() {
        state_builder = state_builder.with_chunks(handle);
    }
    let state = Arc::new(state_builder);
    let persistence_tracker = state.persistence_tracker();

    let listen_handles = if continuous_enabled {
        Some(listen_dispatcher::spawn(
            state.clone(),
            listener.clone(),
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
        conv_store: conv_store_for_shutdown,
        session_id: session_for_shutdown,
        hotkey_handle,
        gpu_monitor_handle,
        idle_monitor_handle,
        listen_handles,
        wm_handle,
        mcp_handles,
        embedder_task_handle,
        embed_service_handle,
        memory_writer_handle,
    })
    .await;

    serve_result?;
    info!("assistd stopped");
    Ok(())
}

struct VoiceSubsystem {
    input: Arc<dyn VoiceInput>,
    listener: Arc<dyn ContinuousListener>,
    output: Arc<assistd_voice::VoiceOutputController>,
}

async fn init_voice(config: &Config, presence: &Arc<PresenceManager>) -> VoiceSubsystem {
    let (input, listener) = init_voice_input(config, presence).await;
    let output_inner = init_voice_output(config).await;
    let output =
        assistd_voice::VoiceOutputController::new(output_inner, config.voice.synthesis.enabled);
    VoiceSubsystem {
        input,
        listener,
        output,
    }
}

async fn init_voice_input(
    config: &Config,
    presence: &Arc<PresenceManager>,
) -> (Arc<dyn VoiceInput>, Arc<dyn ContinuousListener>) {
    fn disabled_voice() -> (Arc<dyn VoiceInput>, Arc<dyn ContinuousListener>) {
        (
            Arc::new(NoVoiceInput::new()),
            Arc::new(NoContinuousListener::new()),
        )
    }

    if !config.voice.enabled {
        info!("voice: disabled in config (voice.enabled = false)");
        return disabled_voice();
    }

    info!(
        "voice: building mic input ({})",
        config.voice.transcription.model
    );
    let primary = match WhisperTranscriberBuilder::from_config(&config.voice.transcription)
        .build()
        .await
    {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("voice input failed to initialize: {e:#}; PTT commands will error");
            return disabled_voice();
        }
    };

    let is_gpu = primary.is_gpu();
    let primary: Arc<dyn Transcriber> = Arc::new(primary);

    let transcriber: Arc<dyn Transcriber> =
        if is_gpu && config.voice.transcription.cpu_fallback_enabled {
            let probe = Arc::new(PresenceGpuProbe::new(
                presence.clone(),
                config.sleep.gpu_allowlist.clone(),
            ));
            let queue_cfg = QueueConfig {
                gpu_busy_timeout_ms: config.voice.transcription.gpu_busy_timeout_ms,
                cpu_fallback_enabled: config.voice.transcription.cpu_fallback_enabled,
            };
            let cpu_cfg = config.voice.transcription.clone();
            let cpu_factory: assistd_voice::CpuFallbackFactory = Arc::new(move || {
                let cfg = cpu_cfg.clone();
                Box::pin(async move {
                    let t = build_cpu_fallback(&cfg, None).await?;
                    Ok(Arc::new(t) as Arc<dyn Transcriber>)
                })
            });
            info!(
                "voice: GPU transcription active; CPU fallback armed \
                 (gpu_busy_timeout_ms={})",
                queue_cfg.gpu_busy_timeout_ms
            );
            Arc::new(QueuedTranscriber::new(
                primary.clone(),
                true,
                cpu_factory,
                probe,
                queue_cfg,
            ))
        } else {
            if is_gpu {
                info!("voice: GPU transcription active; CPU fallback disabled by config");
            } else {
                info!("voice: CPU transcription active");
            }
            primary.clone()
        };

    let mic = MicVoiceInput::new(
        transcriber.clone(),
        config.voice.mic_device.clone(),
        config.voice.max_recording_secs.max(1),
    );
    let listener: Arc<dyn ContinuousListener> = if config.voice.continuous.enabled {
        info!(
            "voice.continuous: enabled (hotkey={:?}, start_on_launch={})",
            config.voice.continuous.hotkey, config.voice.continuous.start_on_launch
        );
        Arc::new(MicContinuousListener::new(transcriber, &config.voice))
    } else {
        info!("voice.continuous: disabled in config");
        Arc::new(NoContinuousListener::new())
    };
    (Arc::new(mic), listener)
}

async fn init_voice_output(config: &Config) -> Arc<dyn VoiceOutput> {
    if !config.voice.synthesis.enabled {
        info!("voice.synthesis: disabled in config (voice.synthesis.enabled = false)");
        return Arc::new(NoVoiceOutput) as Arc<dyn VoiceOutput>;
    }
    info!(
        "voice.synthesis: starting Piper ({})",
        config.voice.synthesis.voice
    );
    match assistd_voice::PiperVoiceOutput::start(config.voice.synthesis.clone()).await {
        Ok(p) => {
            info!("voice.synthesis: Piper ready");
            Arc::new(p) as Arc<dyn VoiceOutput>
        }
        Err(e) => {
            tracing::warn!("voice.synthesis failed to initialize: {e:#}; speech output disabled");
            Arc::new(NoVoiceOutput) as Arc<dyn VoiceOutput>
        }
    }
}

struct WindowSubsystem {
    manager: Arc<dyn WindowManager>,
    handle: Option<WmHandle>,
}

async fn init_window_manager(
    config: &Config,
    shutdown_tx: &watch::Sender<bool>,
) -> WindowSubsystem {
    let resolved = match config.compositor.compositor_type {
        CompositorType::Auto => match assistd_core::config::compositor::detect_from_env(
            std::env::var_os("SWAYSOCK").is_some(),
            std::env::var_os("I3SOCK").is_some(),
            std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some(),
            std::env::var("XDG_CURRENT_DESKTOP").ok().as_deref(),
        ) {
            Some(c) => {
                info!("wm: auto-detected compositor = {:?}", c);
                c
            }
            None => {
                info!(
                    "wm: auto-detect found no supported compositor (no $SWAYSOCK/$I3SOCK/$HYPRLAND_INSTANCE_SIGNATURE/$XDG_CURRENT_DESKTOP); window ops disabled"
                );
                CompositorType::Auto
            }
        },
        explicit => explicit,
    };

    match resolved {
        CompositorType::I3 => match I3Backend::start(shutdown_tx.subscribe()).await {
            Ok(handle) => {
                info!("wm: i3 backend connected");
                WindowSubsystem {
                    manager: handle.backend.clone(),
                    handle: Some(WmHandle::I3(handle)),
                }
            }
            Err(e) => {
                tracing::warn!("wm: i3 backend unavailable ({e:#}); window ops disabled");
                WindowSubsystem {
                    manager: Arc::new(NoWindowManager),
                    handle: None,
                }
            }
        },
        CompositorType::Sway => match SwayBackend::start(shutdown_tx.subscribe()).await {
            Ok(handle) => {
                info!("wm: sway backend connected");
                WindowSubsystem {
                    manager: handle.backend.clone(),
                    handle: Some(WmHandle::Sway(handle)),
                }
            }
            Err(e) => {
                tracing::warn!("wm: sway backend unavailable ({e:#}); window ops disabled");
                WindowSubsystem {
                    manager: Arc::new(NoWindowManager),
                    handle: None,
                }
            }
        },
        CompositorType::Hyprland => {
            info!("wm: hyprland backend not yet implemented; window ops disabled");
            WindowSubsystem {
                manager: Arc::new(NoWindowManager),
                handle: None,
            }
        }
        CompositorType::Auto => WindowSubsystem {
            manager: Arc::new(NoWindowManager),
            handle: None,
        },
    }
}

struct McpSubsystem {
    handles: Vec<McpServerHandle>,
    tools: Vec<Box<dyn assistd_tools::Tool>>,
}

async fn init_mcp(config: &Config, shutdown_tx: &watch::Sender<bool>) -> McpSubsystem {
    if !config.mcp.enabled {
        info!("mcp: disabled in config (mcp.enabled = false)");
        return McpSubsystem {
            handles: Vec::new(),
            tools: Vec::new(),
        };
    }

    let mut handles: Vec<McpServerHandle> = Vec::new();
    let mut tools: Vec<Box<dyn assistd_tools::Tool>> = Vec::new();
    for s_cfg in &config.mcp.servers {
        let transport_cfg = build_transport_config(s_cfg);
        let label = s_cfg.name.clone();
        match McpServerHandle::start(label.clone(), transport_cfg, shutdown_tx.subscribe()).await {
            Ok(handle) => {
                let prefix = format!("mcp__{}", handle.name);
                match adapt_handle_as_tools(&handle, &prefix).await {
                    Ok(t) => {
                        info!(
                            "mcp: {} ready ({} tools, transport={:?})",
                            handle.name,
                            t.len(),
                            s_cfg.transport
                        );
                        tools.extend(t);
                        handles.push(handle);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "mcp: {} discovery failed ({e:#}); shutting down server",
                            handle.name
                        );
                        handle.shutdown().await;
                    }
                }
            }
            Err(e) => {
                tracing::warn!("mcp: {label} failed to start ({e:#}); skipping");
            }
        }
    }
    McpSubsystem { handles, tools }
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

struct DaemonShutdown {
    persistence_tracker: tokio_util::task::TaskTracker,
    presence: Arc<PresenceManager>,
    conv_store: Arc<dyn ConversationStore>,
    session_id: Arc<assistd_memory::SessionId>,
    hotkey_handle: Option<JoinHandle<()>>,
    gpu_monitor_handle: Option<JoinHandle<()>>,
    idle_monitor_handle: Option<JoinHandle<()>>,
    listen_handles: Option<listen_dispatcher::ListenDispatcherHandles>,
    wm_handle: Option<WmHandle>,
    mcp_handles: Vec<McpServerHandle>,
    embedder_task_handle: Option<JoinHandle<()>>,
    embed_service_handle: Option<EmbedService>,
    memory_writer_handle: Option<JoinHandle<()>>,
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

    if let Err(e) = s.conv_store.end_session(&s.session_id).await {
        tracing::warn!("memory: end_session failed at shutdown: {e:#}");
    }

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
    if let Some(h) = s.wm_handle {
        h.shutdown().await;
    }
    for handle in s.mcp_handles {
        handle.shutdown().await;
    }
    if let Some(h) = s.embedder_task_handle {
        let _ = h.await;
    }
    if let Some(svc) = s.embed_service_handle {
        if let Err(e) = svc.shutdown().await {
            tracing::warn!("embed-server shutdown error: {e:#}");
        }
    }
    if let Some(h) = s.memory_writer_handle {
        let _ = h.await;
    }
}

struct MemorySubsystem {
    memory_store: Arc<dyn MemoryStore>,
    conversation_store: Arc<dyn ConversationStore>,
    writer_handle: Option<JoinHandle<()>>,
    session_id: Arc<assistd_memory::SessionId>,
    branch_id: assistd_memory::BranchId,
    resumed_history: Vec<assistd_memory::HistoryRow>,
    sqlite_handle: Option<Arc<SqliteHandle>>,
}

impl MemorySubsystem {
    fn disabled() -> Self {
        Self {
            memory_store: Arc::new(NoMemoryStore),
            conversation_store: Arc::new(NoConversationStore),
            writer_handle: None,
            session_id: Arc::new(assistd_memory::SessionId::new()),
            branch_id: assistd_memory::BranchId(0),
            resumed_history: Vec::new(),
            sqlite_handle: None,
        }
    }
}

async fn init_memory(config: &Config, shutdown_tx: &watch::Sender<bool>) -> MemorySubsystem {
    if !config.memory.enabled {
        info!("memory: disabled in config (memory.enabled = false)");
        return MemorySubsystem::disabled();
    }

    let db_path = std::path::PathBuf::from(&config.memory.db_path);
    let (handle, writer_handle) = match SqliteHandle::open(&db_path, shutdown_tx.subscribe()).await
    {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!(
                "memory: failed to open {} ({e:#}); persistence disabled this run",
                db_path.display()
            );
            return MemorySubsystem::disabled();
        }
    };

    let handle = Arc::new(handle);
    let conv_store = Arc::new(SqliteConversationStore::new(handle.clone()));
    let mem_store = Arc::new(SqliteMemoryStore::new(handle.clone()));
    let mut resumed_history: Vec<assistd_memory::HistoryRow> = Vec::new();

    let (session, branch) = match conv_store.find_resumable_session().await {
        Ok(Some(cand)) if !pid_is_alive(cand.daemon_pid) => {
            info!(
                "memory: resuming prior session {} (branch={})",
                cand.session_id, cand.current_branch_id.0
            );
            match conv_store.load_branch_history(cand.current_branch_id).await {
                Ok(rows) => resumed_history = rows,
                Err(e) => {
                    tracing::warn!("memory: load_branch_history failed for resume ({e:#})")
                }
            }
            (Arc::new(cand.session_id), cand.current_branch_id)
        }
        Ok(_) => match conv_store
            .begin_session_with_main_branch(std::process::id())
            .await
        {
            Ok((s, b)) => {
                info!(
                    "memory: SQLite ready at {} (session={}, branch={})",
                    db_path.display(),
                    s,
                    b.0
                );
                (Arc::new(s), b)
            }
            Err(e) => {
                tracing::warn!(
                    "memory: begin_session_with_main_branch failed: {e:#}; \
                     continuing without session row"
                );
                (
                    Arc::new(assistd_memory::SessionId::new()),
                    assistd_memory::BranchId(0),
                )
            }
        },
        Err(e) => {
            tracing::warn!("memory: find_resumable_session failed: {e:#}; starting fresh");
            match conv_store
                .begin_session_with_main_branch(std::process::id())
                .await
            {
                Ok((s, b)) => (Arc::new(s), b),
                Err(e) => {
                    tracing::warn!("memory: begin_session_with_main_branch failed: {e:#}");
                    (
                        Arc::new(assistd_memory::SessionId::new()),
                        assistd_memory::BranchId(0),
                    )
                }
            }
        }
    };

    MemorySubsystem {
        memory_store: mem_store,
        conversation_store: conv_store,
        writer_handle: Some(writer_handle),
        session_id: session,
        branch_id: branch,
        resumed_history,
        sqlite_handle: Some(handle),
    }
}

struct EmbeddingSubsystem {
    embedder: Arc<dyn Embedder>,
    semantic_store: Arc<dyn SemanticStore>,
    embed_tx: mpsc::Sender<EmbedJob>,
    service_handle: Option<EmbedService>,
    task_handle: Option<JoinHandle<()>>,
    model_name: String,
}

impl EmbeddingSubsystem {
    fn disabled(service_handle: Option<EmbedService>) -> Self {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        Self {
            embedder: Arc::new(NoEmbedder),
            semantic_store: Arc::new(NoSemanticStore),
            embed_tx: tx,
            service_handle,
            task_handle: None,
            model_name: String::new(),
        }
    }
}

async fn init_embedding(
    config: &Config,
    sqlite_handle: Option<&Arc<SqliteHandle>>,
    shutdown_tx: &watch::Sender<bool>,
) -> EmbeddingSubsystem {
    if !config.embedding.enabled {
        info!("embedding: disabled in config (embedding.enabled = false)");
        return EmbeddingSubsystem::disabled(None);
    }

    let svc = match EmbedService::start(config.embedding.clone(), shutdown_tx.subscribe()).await {
        Ok(svc) => svc,
        Err(e) => {
            tracing::warn!("embedding: failed to start ({e:#}); semantic search disabled this run");
            return EmbeddingSubsystem::disabled(None);
        }
    };

    let client = match LlamaEmbedder::new(
        &config.embedding.host,
        config.embedding.port,
        config.embedding.model.clone(),
        Duration::from_secs(config.embedding.request_timeout_secs),
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                "embedding: client probe failed ({e:#}); semantic search disabled this run"
            );
            return EmbeddingSubsystem::disabled(Some(svc));
        }
    };

    let model_name = config.embedding.model.clone();
    let embedder: Arc<dyn Embedder> = Arc::new(client);
    let semantic: Arc<dyn SemanticStore> = match sqlite_handle {
        Some(h) => Arc::new(SqliteSemanticStore::new(h.clone())),
        None => Arc::new(NoSemanticStore),
    };
    let writer_tx = sqlite_handle.map(|h| h.writer_tx()).unwrap_or_else(|| {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        Arc::new(tx)
    });
    let (embed_tx, embed_rx) = mpsc::channel(256);
    let task = spawn_embedder_task(
        embedder.clone(),
        writer_tx,
        embed_rx,
        shutdown_tx.subscribe(),
    );

    info!(
        "embedding: ready (model={}, dim={}, port={})",
        embedder.model(),
        embedder.dim(),
        config.embedding.port,
    );

    match semantic.count_stale(&model_name).await {
        Ok((n, models)) if n > 0 => {
            tracing::warn!(
                "embedding: {n} rows exist under non-current model(s) {models:?}; \
                 run `assistd memory reindex` to rebuild against {model_name}"
            );
        }
        Ok(_) => {}
        Err(e) => {
            tracing::debug!("embedding: count_stale check failed ({e:#}); skipping diagnostic");
        }
    }

    EmbeddingSubsystem {
        embedder,
        semantic_store: semantic,
        embed_tx,
        service_handle: Some(svc),
        task_handle: Some(task),
        model_name,
    }
}

#[allow(unsafe_code)] // libc::kill probe — see SAFETY comment below
fn pid_is_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    // SAFETY: `kill(pid, 0)` reads no memory and writes no signal; it
    // only probes for the pid's existence. The libc binding is safe
    // to call from any thread.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc == 0 {
        return true;
    }
    // EPERM: process exists but we lack permission. EINVAL: bad sig
    // (won't happen with 0). ESRCH: no such process.
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    errno == libc::EPERM
}

fn persisted_role_to_history_role(role: assistd_memory::PersistedRole) -> assistd_llm::HistoryRole {
    match role {
        assistd_memory::PersistedRole::System => assistd_llm::HistoryRole::System,
        assistd_memory::PersistedRole::User => assistd_llm::HistoryRole::User,
        assistd_memory::PersistedRole::Assistant => assistd_llm::HistoryRole::Assistant,
        assistd_memory::PersistedRole::Tool => assistd_llm::HistoryRole::Tool,
    }
}

fn build_transport_config(s: &McpServerConfig) -> TransportConfig {
    match s.transport {
        McpTransport::Stdio => {
            let mut cfg = StdioConfig::new(s.name.clone(), s.command.clone().unwrap_or_default());
            cfg.args = s.args.clone();
            cfg.env = s.env.clone();
            cfg.request_timeout = Duration::from_secs(s.request_timeout_secs);
            TransportConfig::Stdio(cfg)
        }
        McpTransport::Sse => {
            let mut cfg = SseConfig::new(s.name.clone(), s.url.clone().unwrap_or_default());
            cfg.headers = s.headers.clone();
            cfg.request_timeout = Duration::from_secs(s.request_timeout_secs);
            cfg.read_timeout = Duration::from_secs(s.sse_read_timeout_secs);
            cfg.ping_interval = Duration::from_secs(s.sse_ping_interval_secs);
            TransportConfig::Sse(cfg)
        }
    }
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

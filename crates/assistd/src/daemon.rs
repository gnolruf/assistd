use anyhow::Result;
use assistd_core::{
    AppState, CompositorType, Config, ContinuousListener, NoContinuousListener, NoVoiceInput,
    NoVoiceOutput, NoWindowManager, PresenceManager, VoiceInput, VoiceOutput, WindowManager,
};
use assistd_embed::{EmbedService, Embedder, LlamaEmbedder, NoEmbedder, spawn_embedder_task};
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
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;
use tracing::info;

use crate::voice_probe::PresenceGpuProbe;
use crate::{gpu_monitor, hotkey, idle_monitor, listen_dispatcher};

#[derive(Args)]
pub struct DaemonArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
    /// Defer the global PTT hotkey to a connected client (e.g. the
    /// chat TUI). Set automatically when the daemon is auto-spawned by
    /// `assistd chat`. With this flag the daemon does not register a
    /// global hotkey itself; voice capture is driven exclusively by
    /// IPC `PttStart`/`PttStop` requests. Useful when the operator
    /// wants the TUI's hotkey grab to win without a key-binding race.
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
        config.llama_server.clone(),
        config.model.clone(),
        shutdown_tx.subscribe(),
    )
    .await?;
    info!(
        "presence: Active (llama-server ready on {}:{})",
        config.llama_server.host, config.llama_server.port
    );

    // Capability probe: ask the now-ready llama-server for the loaded
    // model id and whether it has a vision encoder. The shared
    // VisionGate drives see/screenshot/attach_image; the
    // VisionRevalidator re-probes at the top of each query so a model
    // swap on the running llama-server flips the gate transparently.
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

    let chat = LlamaChatClient::new(&config.chat, &config.llama_server, &config.model)?;

    // Voice input: build the mic-backed implementation when the user
    // has enabled it. The whisper transcriber is built once and shared
    // between the PTT `MicVoiceInput` and the continuous
    // `MicContinuousListener` so only one whisper/VAD model is loaded.
    // Eager download + GPU probe happens here; failures surface at
    // startup rather than on the first mic press.
    let (voice, listener): (Arc<dyn VoiceInput>, Arc<dyn ContinuousListener>) = if config
        .voice
        .enabled
    {
        info!(
            "voice: building mic input ({})",
            config.voice.transcription.model
        );
        match WhisperTranscriberBuilder::from_config(&config.voice.transcription)
            .build()
            .await
        {
            Ok(primary) => {
                let is_gpu = primary.is_gpu();
                let primary: Arc<dyn Transcriber> = Arc::new(primary);

                // When the primary context runs on the GPU and CPU
                // fallback is enabled, wrap it in a QueuedTranscriber
                // that consults PresenceManager + NVML before each
                // transcription. In the CPU-only case we hand the
                // primary directly to the PTT and listen pipelines
                // (no contention window, no Queued state flash).
                let transcriber: Arc<dyn Transcriber> = if is_gpu
                    && config.voice.transcription.cpu_fallback_enabled
                {
                    let probe = Arc::new(PresenceGpuProbe::new(presence.clone()));
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
                let listener_impl: Arc<dyn ContinuousListener> = if config.voice.continuous.enabled
                {
                    info!(
                        "voice.continuous: enabled (hotkey={:?}, start_on_launch={})",
                        config.voice.continuous.hotkey, config.voice.continuous.start_on_launch
                    );
                    Arc::new(MicContinuousListener::new(transcriber, &config.voice))
                } else {
                    info!("voice.continuous: disabled in config");
                    Arc::new(NoContinuousListener::new())
                };
                (Arc::new(mic), listener_impl)
            }
            Err(e) => {
                tracing::warn!("voice input failed to initialize: {e:#}; PTT commands will error");
                (
                    Arc::new(NoVoiceInput::new()),
                    Arc::new(NoContinuousListener::new()),
                )
            }
        }
    } else {
        info!("voice: disabled in config (voice.enabled = false)");
        (
            Arc::new(NoVoiceInput::new()),
            Arc::new(NoContinuousListener::new()),
        )
    };

    // Voice output (Piper TTS): try-warn-fallback, identical pattern
    // to voice input above. On any error — missing binary, voice
    // download fails, audio device unavailable, health-check synth
    // fails — log a warning and substitute `NoVoiceOutput` so LLM
    // streaming continues silently.
    let voice_output_inner: Arc<dyn VoiceOutput> = if config.voice.synthesis.enabled {
        info!(
            "voice.synthesis: starting Piper ({})",
            config.voice.synthesis.voice
        );
        match assistd_voice::PiperVoiceOutput::start(config.voice.synthesis.clone()).await {
            Ok(p) => {
                info!("voice.synthesis: Piper ready");
                Arc::new(p)
            }
            Err(e) => {
                tracing::warn!(
                    "voice.synthesis failed to initialize: {e:#}; speech output disabled"
                );
                Arc::new(NoVoiceOutput)
            }
        }
    } else {
        info!("voice.synthesis: disabled in config (voice.synthesis.enabled = false)");
        Arc::new(NoVoiceOutput)
    };
    // Wrap in a runtime controller so the toggle/skip/PTT-interrupt
    // controls all flow through one shared object. Initial enabled
    // state mirrors config; runtime changes do not persist.
    let voice_output = assistd_voice::VoiceOutputController::new(
        voice_output_inner,
        config.voice.synthesis.enabled,
    );

    // In `--client-mode` the daemon defers the global hotkey to the
    // attached client (the chat TUI grabs it instead). Two processes
    // can't both grab the same X11/Wayland key, and the TUI is what
    // the user is looking at, so it wins. Daemon-internal voice
    // capture in this mode flows exclusively through IPC PttStart /
    // PttStop requests.
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

    // Persistent memory: open the SQLite store if enabled. The
    // `try-warn-fallback` shape mirrors voice input/output above —
    // any failure (bad path, FS readonly, etc.) downgrades to the
    // `NoMemoryStore` placeholder so the daemon still starts and the
    // user gets a startup warning instead of a hard error.
    //
    // Opened BEFORE `build_tools` because the LLM-callable `remember`
    // and `recall` tools need an `Arc<MemoryOps>` to register against;
    // the placeholder fallback keeps the registration unconditional so
    // a startup error doesn't change the tool surface visible to the
    // model.
    let (
        memory_store,
        conversation_store,
        memory_writer_handle,
        session_id_for_state,
        sqlite_handle,
    ) = if config.memory.enabled {
        let db_path = std::path::PathBuf::from(&config.memory.db_path);
        match SqliteHandle::open(&db_path, shutdown_tx.subscribe()).await {
            Ok((handle, writer_handle)) => {
                let handle = Arc::new(handle);
                let conv_store = Arc::new(SqliteConversationStore::new(handle.clone()));
                let mem_store = Arc::new(SqliteMemoryStore::new(handle.clone()));
                let session = match conv_store.begin_session(std::process::id()).await {
                    Ok(s) => Arc::new(s),
                    Err(e) => {
                        tracing::warn!(
                            "memory: begin_session failed: {e:#}; continuing without session row"
                        );
                        Arc::new(assistd_memory::SessionId::new())
                    }
                };
                info!(
                    "memory: SQLite ready at {} (session={})",
                    db_path.display(),
                    session
                );
                (
                    mem_store as Arc<dyn MemoryStore>,
                    conv_store as Arc<dyn ConversationStore>,
                    Some(writer_handle),
                    session,
                    Some(handle),
                )
            }
            Err(e) => {
                tracing::warn!(
                    "memory: failed to open {} ({e:#}); persistence disabled this run",
                    db_path.display()
                );
                (
                    Arc::new(NoMemoryStore) as Arc<dyn MemoryStore>,
                    Arc::new(NoConversationStore) as Arc<dyn ConversationStore>,
                    None,
                    Arc::new(assistd_memory::SessionId::new()),
                    None,
                )
            }
        }
    } else {
        info!("memory: disabled in config (memory.enabled = false)");
        (
            Arc::new(NoMemoryStore) as Arc<dyn MemoryStore>,
            Arc::new(NoConversationStore) as Arc<dyn ConversationStore>,
            None,
            Arc::new(assistd_memory::SessionId::new()),
            None,
        )
    };
    // Keep a clone of the conversation store for the shutdown path
    // below — `state` will own the canonical handle, but we want to
    // call `end_session` after the socket has drained without
    // reaching back through `Arc<AppState>`.
    let conv_store_for_shutdown = conversation_store.clone();
    let session_for_shutdown = session_id_for_state.clone();
    // Combined façade handed to `build_tools` so the LLM-callable
    // `remember` / `recall` tools can read & write through the same
    // single SQLite writer used by chat-turn persistence.
    let memory_ops = Arc::new(MemoryOps::new(
        memory_store.clone(),
        conversation_store.clone(),
    ));

    // Embedding subsystem: try-warn-fallback like memory and voice. A
    // failed start downgrades to NoEmbedder + NoSemanticStore + closed
    // channel, so retrieval-touching tools register but no-op rather
    // than crashing the daemon. Independent of memory.enabled — but if
    // memory is off, we wire NoSemanticStore even on success since
    // there's no DB to read embeddings from.
    let (
        embedder,
        semantic_store,
        embed_tx,
        embed_service_handle,
        embedder_task_handle,
        embedding_model_name,
    ) = if config.embedding.enabled {
        match EmbedService::start(config.embedding.clone(), shutdown_tx.subscribe()).await {
            Ok(svc) => {
                // Build the HTTP client; probe dim. Failure here downgrades
                // the whole subsystem to fallback so tools still register.
                match LlamaEmbedder::new(
                    &config.embedding.host,
                    config.embedding.port,
                    config.embedding.model.clone(),
                    std::time::Duration::from_secs(config.embedding.request_timeout_secs),
                )
                .await
                {
                    Ok(client) => {
                        let model_name = config.embedding.model.clone();
                        let embedder_arc: Arc<dyn Embedder> = Arc::new(client);
                        let semantic: Arc<dyn SemanticStore> = match sqlite_handle.as_ref() {
                            Some(h) => Arc::new(SqliteSemanticStore::new(h.clone())),
                            None => Arc::new(NoSemanticStore),
                        };
                        let writer_tx = sqlite_handle
                            .as_ref()
                            .map(|h| h.writer_tx())
                            .unwrap_or_else(|| {
                                // No DB — embedder task has nowhere to ack
                                // back to. Build a closed sender so any
                                // try_send still no-ops and the worker
                                // exits cleanly on shutdown.
                                let (tx, rx) = tokio::sync::mpsc::channel(1);
                                drop(rx);
                                Arc::new(tx)
                            });
                        let (etx, erx) = tokio::sync::mpsc::channel(256);
                        let task = spawn_embedder_task(
                            embedder_arc.clone(),
                            writer_tx,
                            erx,
                            shutdown_tx.subscribe(),
                        );
                        info!(
                            "embedding: ready (model={}, dim={}, port={})",
                            embedder_arc.model(),
                            embedder_arc.dim(),
                            config.embedding.port,
                        );
                        // Surface stranded embeddings (rows under a
                        // different model than the current config) at
                        // startup so a model swap is visible instead of
                        // silently producing `(no memories)` from
                        // `recall` / `reminisce`. Failures are best-effort
                        // — a warning here must not gate daemon startup.
                        match semantic.count_stale(&model_name).await {
                            Ok((n, models)) if n > 0 => {
                                tracing::warn!(
                                    "embedding: {n} rows exist under non-current model(s) {models:?}; \
                                     run `assistd memory reindex` to rebuild against {model_name}"
                                );
                            }
                            Ok(_) => {}
                            Err(e) => {
                                tracing::debug!(
                                    "embedding: count_stale check failed ({e:#}); skipping diagnostic"
                                );
                            }
                        }
                        (
                            embedder_arc,
                            semantic,
                            etx,
                            Some(svc),
                            Some(task),
                            model_name,
                        )
                    }
                    Err(e) => {
                        tracing::warn!(
                            "embedding: client probe failed ({e:#}); semantic search disabled this run"
                        );
                        // Still hold svc so its supervisor task gets
                        // awaited on shutdown — abandoning it would leak
                        // the child until the daemon exits.
                        let (tx, rx) = tokio::sync::mpsc::channel(1);
                        drop(rx);
                        (
                            Arc::new(NoEmbedder) as Arc<dyn Embedder>,
                            Arc::new(NoSemanticStore) as Arc<dyn SemanticStore>,
                            tx,
                            Some(svc),
                            None,
                            String::new(),
                        )
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    "embedding: failed to start ({e:#}); semantic search disabled this run"
                );
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                drop(rx);
                (
                    Arc::new(NoEmbedder) as Arc<dyn Embedder>,
                    Arc::new(NoSemanticStore) as Arc<dyn SemanticStore>,
                    tx,
                    None,
                    None,
                    String::new(),
                )
            }
        }
    } else {
        info!("embedding: disabled in config (embedding.enabled = false)");
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        (
            Arc::new(NoEmbedder) as Arc<dyn Embedder>,
            Arc::new(NoSemanticStore) as Arc<dyn SemanticStore>,
            tx,
            None,
            None,
            String::new(),
        )
    };

    // Window-manager backend: try-warn-fallback like memory/embedding
    // above. Each backend opens two Unix sockets (one for commands,
    // one for the event stream) — connect failure (e.g. macOS dev box,
    // non-matching session) degrades to `NoWindowManager` so the
    // daemon still starts and `wm.focus(...)` simply errors at call
    // time. Auto resolves the configured `auto` to a concrete
    // compositor via $SWAYSOCK / $I3SOCK / $HYPRLAND_INSTANCE_SIGNATURE
    // / $XDG_CURRENT_DESKTOP — see assistd_config::compositor::detect_from_env.
    let resolved_compositor = match config.compositor.compositor_type {
        CompositorType::Auto => {
            let detected = assistd_core::config::compositor::detect_from_env(
                std::env::var_os("SWAYSOCK").is_some(),
                std::env::var_os("I3SOCK").is_some(),
                std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some(),
                std::env::var("XDG_CURRENT_DESKTOP").ok().as_deref(),
            );
            match detected {
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
            }
        }
        explicit => explicit,
    };
    let (window_manager, wm_handle): (Arc<dyn WindowManager>, Option<WmHandle>) =
        match resolved_compositor {
            CompositorType::I3 => match I3Backend::start(shutdown_tx.subscribe()).await {
                Ok(handle) => {
                    info!("wm: i3 backend connected");
                    (
                        handle.backend.clone() as Arc<dyn WindowManager>,
                        Some(WmHandle::I3(handle)),
                    )
                }
                Err(e) => {
                    tracing::warn!("wm: i3 backend unavailable ({e:#}); window ops disabled");
                    (Arc::new(NoWindowManager), None)
                }
            },
            CompositorType::Sway => match SwayBackend::start(shutdown_tx.subscribe()).await {
                Ok(handle) => {
                    info!("wm: sway backend connected");
                    (
                        handle.backend.clone() as Arc<dyn WindowManager>,
                        Some(WmHandle::Sway(handle)),
                    )
                }
                Err(e) => {
                    tracing::warn!("wm: sway backend unavailable ({e:#}); window ops disabled");
                    (Arc::new(NoWindowManager), None)
                }
            },
            CompositorType::Hyprland => {
                info!("wm: hyprland backend not yet implemented; window ops disabled");
                (Arc::new(NoWindowManager), None)
            }
            CompositorType::Auto => {
                // Auto-detection failed above — fall through silently.
                (Arc::new(NoWindowManager), None)
            }
        };

    // Destructive bash commands prompt the active IPC client through
    // `IpcConfirmationGate`: the per-connection `ConfirmRouter`
    // (installed by `assistd-core::socket::handle_connection` into the
    // CONFIRM_ROUTER task-local) emits `Event::ConfirmRequest` and
    // awaits a `Request::ConfirmResponse` on the same connection.
    // Connections with no router in scope (autonomous daemon-internal
    // dispatch — e.g. continuous-listener-driven queries with no TUI
    // attached) fall through to deny, mirroring the previous
    // `DenyAllGate` behavior.
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
    )?;
    info!(
        "tools: registered {} (overflow dir {})",
        tools.len(),
        overflow_dir.display()
    );

    // Snapshot the continuous listen config before `config` is moved
    // into `AppState`. Controls start_on_launch and pause semantics.
    let continuous_enabled = config.voice.enabled && config.voice.continuous.enabled;
    let continuous_start_on_launch = config.voice.continuous.start_on_launch;

    let embedding_cfg_for_state = config.embedding.clone();
    let mut state_builder = AppState::new(
        config,
        Arc::new(chat),
        presence.clone(),
        tools,
        voice.clone(),
        listener.clone(),
        voice_output,
    )
    .with_vision_revalidator(vision_revalidator)
    .with_memory(memory_store)
    .with_conversations(conversation_store)
    .with_session(session_id_for_state)
    .with_embedder(embedder)
    .with_semantic(semantic_store)
    .with_embed_tx(embed_tx)
    .with_embedding_cfg(embedding_cfg_for_state)
    .with_window_manager(window_manager);
    if let Some(handle) = sqlite_handle.clone() {
        state_builder = state_builder.with_chunks(handle);
    }
    let state = Arc::new(state_builder);

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

    // Drop to Sleeping on shutdown: tears down the supervisor and the child.
    if let Err(e) = presence.sleep().await {
        tracing::error!("presence shutdown error: {e:#}");
    }

    // Mark the persisted session as ended before draining the writer
    // so the row's `ended_at` is set in the same final flush. Failures
    // are logged but never block shutdown — the session row simply
    // stays open and a future pass can heuristically close it.
    if let Err(e) = conv_store_for_shutdown
        .end_session(&session_for_shutdown)
        .await
    {
        tracing::warn!("memory: end_session failed at shutdown: {e:#}");
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
    if let Some(handles) = listen_handles {
        let _ = handles.forwarder.await;
        let _ = handles.presence_gate.await;
    }
    if let Some(h) = wm_handle {
        h.shutdown().await;
    }
    // Embedder task runs BEFORE the memory writer drains so any
    // in-flight EmbedJob lands as a StoreChunkEmbedding/StoreMemoryEmbedding
    // op in the writer queue, which then drains.
    if let Some(h) = embedder_task_handle {
        let _ = h.await;
    }
    // Then tear down the embed-server process. The supervisor task
    // observes the shared shutdown watch (already flipped) and exits.
    if let Some(svc) = embed_service_handle {
        if let Err(e) = svc.shutdown().await {
            tracing::warn!("embed-server shutdown error: {e:#}");
        }
    }
    if let Some(h) = memory_writer_handle {
        // Awaiting the writer ensures the in-flight queue is drained
        // before the daemon process exits — otherwise a SIGTERM could
        // truncate the last few `append_message` writes.
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

use anyhow::Result;
use assistd_core::{
    AppState, Config, ContinuousListener, NoContinuousListener, NoVoiceInput, NoVoiceOutput,
    PresenceManager, VoiceInput, VoiceOutput,
};
use assistd_llm::LlamaChatClient;
use assistd_tools::DenyAllGate;
use assistd_voice::{
    MicContinuousListener, MicVoiceInput, QueueConfig, QueuedTranscriber, Transcriber,
    WhisperTranscriberBuilder, build_cpu_fallback,
};
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

    let hotkey_handle = hotkey::spawn_listener(
        &config.presence,
        &config.voice,
        Some(presence.clone()),
        voice.clone(),
        Some(listener.clone()),
        Some(voice_output.clone()),
        shutdown_tx.subscribe(),
    );
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

    // Snapshot the continuous listen config before `config` is moved
    // into `AppState`. Controls start_on_launch and pause semantics.
    let continuous_enabled = config.voice.enabled && config.voice.continuous.enabled;
    let continuous_start_on_launch = config.voice.continuous.start_on_launch;

    let state = Arc::new(AppState::new(
        config,
        Arc::new(chat),
        presence.clone(),
        tools,
        voice.clone(),
        listener.clone(),
        voice_output,
    ));

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

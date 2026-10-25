//! Daemon entrypoint: bring up the subsystems, serve the IPC socket while
//! the model, voice, embedding, search and MCP servers start in the
//! background, tear down in order.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use assistd_core::presence::PresenceLlmHealthProbe;
use assistd_core::socket::StartupLock;
use assistd_core::{
    AppState, Config, MemoryStack, PresenceManager, RuntimeState, SessionContext, Subsystems,
    VisionRevalidator, VoiceManager,
};
use assistd_ipc::IpcClient;
use assistd_llm::{HistoryReplay, LlamaChatClient, LlamaServerControl, LlmBackend, LlmHealthProbe};
use assistd_utils::child_server::ApiKey;
use assistd_utils::tracing_init::env_filter_or;
use clap::Args;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::info;

use crate::hotkey;
use crate::ipc_voice_proxy::IpcVoiceProxy;
use embed_init::{EmbeddingHandles, EmbeddingService, EmbeddingStages, EmbeddingStartup};
use mcp_init::{McpService, McpStartup};
use memory_init::MemorySubsystem;
use search_init::{SearchService, SearchStartup};
use shutdown::{DaemonShutdown, IntakeTasks, ShutdownStages, spawn_signal_handler};
use tools_init::ToolDeps;
use vram::VramLedger;
use warmup::Warmup;

mod embed_init;
mod gpu_monitor;
mod idle_monitor;
mod listen_dispatcher;
mod mcp_init;
mod memory_init;
mod search_init;
mod shutdown;
mod tools_init;
mod voice_init;
mod voice_probe;
mod vram;
mod warmup;
mod wm_init;

/// Log filter when `RUST_LOG` is unset.
const DEFAULT_LOG_DIRECTIVES: &str = "info";

/// Command-line arguments for the `daemon` subcommand.
#[derive(Args)]
pub(crate) struct DaemonArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

/// Run the daemon until shutdown.
pub(crate) async fn run(args: DaemonArgs) -> Result<()> {
    init_tracing();
    assistd_core::install_panic_hook();

    let config_path = match args.config {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;
    hotkey::validate(&config.hotkeys, &config.voice)?;
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
        started = start(config, &config_path, &stages) => Some(started?),
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
    if let Err(e) = &serve_result {
        tracing::error!("socket server failed: {e}; shutting down");
    }

    subsystems.shutdown(&stages).await;

    serve_result?;
    info!("assistd stopped");
    Ok(())
}

async fn start(
    config: Config,
    config_path: &Path,
    stages: &ShutdownStages,
) -> Result<(Arc<AppState>, DaemonShutdown)> {
    let api_key = ApiKey::generate().context("failed to write the model servers' API key")?;
    let presence = PresenceManager::new_sleeping(
        config.model.clone(),
        config.timeouts.clone(),
        api_key.clone(),
        stages.llm.subscribe(),
    )?;
    let health_probe: Arc<dyn LlmHealthProbe> =
        Arc::new(PresenceLlmHealthProbe::new(presence.clone()));
    let vision_revalidator = build_vision_revalidator(&config, &api_key, &presence, &health_probe)?;
    let voice = VoiceManager::new(config.voice.synthesis.enabled);

    let vram = VramLedger::open(&config.sleep);
    if let Some(vram) = &vram {
        presence.set_wake_room(vram.clone());
    }

    let hotkey_handle = spawn_hotkeys(&config, &voice, stages.intake.subscribe());
    let gpu_monitor_handle = gpu_monitor::spawn_monitor(
        &config.sleep,
        vram.clone(),
        presence.clone(),
        stages.intake.subscribe(),
    );
    let idle_monitor_handle =
        idle_monitor::spawn_monitor(&config.sleep, presence.clone(), stages.intake.subscribe());

    let mut memory = memory_init::init().await;
    let (embed, embed_startup) =
        embed_init::prepare(&config, &api_key, memory.sqlite_handle.as_ref());
    let (web_search, search_startup) = search_init::prepare(&config);
    let window = wm_init::init(&config, &stages.tools).await;

    let session_ctx = Arc::new(SessionContext::new(memory.session_id, memory.branch_id));

    let tools = tools_init::init(
        &config,
        config_path,
        ToolDeps {
            vision_gate: vision_revalidator.gate(),
            memory: &memory,
            embed: &embed,
            current_session: session_ctx.session_updates(),
            window_manager: window.manager.clone(),
            web_search: web_search.clone(),
        },
    )?;

    let chat = build_chat_backend(&config, &api_key, health_probe)?;
    let resumed_history = std::mem::take(&mut memory.resumed_history);
    replay_history(chat.as_ref(), resumed_history).await;

    let subsystems = Subsystems::new(chat, presence.clone(), tools.registry, voice.clone())
        .with_window_manager(window.manager.clone())
        .with_vision_revalidator(vision_revalidator.clone())
        .with_mcp_servers(tools.mcp_servers)
        .with_web_search(web_search)
        .with_tools_disabled(tools.disabled);
    let memory_stack = build_memory_stack(&config, &memory, &embed);

    let state = Arc::new(AppState {
        config,
        subsystems,
        memory: memory_stack,
        runtime: RuntimeState::new().with_session_ctx(session_ctx),
    });
    let persistence_tracker = state.runtime.persistence_tracker_handle();
    let background = spawn_background_starts(
        &state,
        BackgroundStarts {
            vision: vision_revalidator,
            vram,
            embed: embed_startup,
            search: search_startup,
            mcp: tools.mcp_startup,
        },
        stages,
    );

    Ok((
        state,
        DaemonShutdown {
            persistence_tracker,
            presence,
            memory,
            embed: background.embed,
            search: background.search,
            window,
            mcp: background.mcp,
            intake_tasks: IntakeTasks {
                hotkey: hotkey_handle,
                gpu_monitor: gpu_monitor_handle,
                idle_monitor: idle_monitor_handle,
                warmup: background.warmup,
            },
        },
    ))
}

struct BackgroundStarts {
    vision: Arc<VisionRevalidator>,
    vram: Option<Arc<VramLedger>>,
    embed: Option<EmbeddingStartup>,
    search: Option<SearchStartup>,
    mcp: Option<McpStartup>,
}

struct BackgroundServices {
    warmup: JoinHandle<()>,
    embed: EmbeddingService,
    search: SearchService,
    mcp: McpService,
}

/// Start the search and MCP servers at once, and the embedding server once
/// the model load has settled so the two never load onto the GPU together.
fn spawn_background_starts(
    state: &Arc<AppState>,
    starts: BackgroundStarts,
    stages: &ShutdownStages,
) -> BackgroundServices {
    let (model_settled_tx, model_settled_rx) = watch::channel(false);
    let warmup = warmup::spawn(
        Warmup {
            state: state.clone(),
            vision: starts.vision,
            vram: starts.vram,
            model_settled: model_settled_tx,
        },
        stages.intake.subscribe(),
    );
    let embed = starts
        .embed
        .map_or_else(EmbeddingService::disabled, |startup| {
            startup.spawn(
                state.clone(),
                model_settled_rx,
                EmbeddingStages {
                    worker: stages.embed_worker.subscribe(),
                    server: stages.embed_server.subscribe(),
                },
            )
        });
    let search = starts
        .search
        .map_or_else(SearchService::disabled, |startup| {
            startup.spawn(state.clone(), stages.search.subscribe())
        });
    let mcp = starts.mcp.map_or_else(McpService::disabled, |startup| {
        startup.spawn(state.clone(), stages.tools.subscribe())
    });
    BackgroundServices {
        warmup,
        embed,
        search,
        mcp,
    }
}

/// Write a default config file to the platform config directory.
pub(crate) fn init_config() -> Result<()> {
    init_tracing();
    let path = Config::default_path()?;
    Config::write_default(&path)?;
    info!("wrote default config to {}", path.display());
    Ok(())
}

/// The revalidator that keeps the vision gate in step with the model,
/// closed until the model first loads.
fn build_vision_revalidator(
    config: &Config,
    api_key: &ApiKey,
    presence: &PresenceManager,
    health_probe: &Arc<dyn LlmHealthProbe>,
) -> Result<Arc<VisionRevalidator>> {
    let control = LlamaServerControl::new(
        SocketAddr::new(config.model.host, config.model.port.get()),
        Some(api_key),
        Some(Arc::clone(health_probe)),
    )
    .context("failed to construct llama-server control client for vision probe")?;
    Ok(VisionRevalidator::new(
        control,
        config.model.name.clone(),
        presence,
    ))
}

/// The daemon's own hotkeys route push-to-talk through its IPC socket,
/// so the daemon and the chat TUI share one PTT path.
fn spawn_hotkeys(
    config: &Config,
    voice: &Arc<VoiceManager>,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    let voice_proxy: Arc<dyn assistd_voice::VoiceInput> =
        Arc::new(IpcVoiceProxy::new(Arc::new(IpcClient::new())));
    hotkey::spawn_listener(
        &config.hotkeys,
        &config.voice,
        config.compositor.resolve(),
        hotkey::Subsystems {
            ptt: voice_proxy,
            voice: Some(voice.clone()),
        },
        shutdown,
    )
}

fn build_chat_backend(
    config: &Config,
    api_key: &ApiKey,
    health_probe: Arc<dyn LlmHealthProbe>,
) -> Result<Arc<dyn LlmBackend>> {
    let chat = LlamaChatClient::new(
        &config.chat,
        &config.model,
        &config.timeouts,
        Some(api_key),
        Some(health_probe),
    )?;
    Ok(Arc::new(chat))
}

fn build_memory_stack(
    config: &Config,
    memory: &MemorySubsystem,
    embed: &EmbeddingHandles,
) -> MemoryStack {
    MemoryStack::disabled(config.embedding.clone())
        .with_memory(memory.memory_store.clone())
        .with_transcripts(memory.transcript_store.clone())
        .with_embedder(embed.embedder.clone())
        .with_semantic(embed.semantic_store.clone())
        .with_embed_tx(embed.embed_tx.clone())
}

async fn replay_history(chat: &dyn LlmBackend, replay: HistoryReplay) {
    if replay.entries.is_empty() {
        return;
    }
    let count = replay.entries.len();
    if let Err(e) = chat.replace_history(replay).await {
        tracing::warn!("memory: resume replay failed: {e:#}");
    } else {
        info!("memory: resumed {count} message(s) from prior branch");
    }
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(env_filter_or(DEFAULT_LOG_DIRECTIVES))
        .init();
}

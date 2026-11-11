//! Daemon entrypoint: bring up the subsystems, serve IPC while the model, voice,
//! embedding, search and MCP servers start in the background, tear down in order.

use std::collections::HashSet;
use std::fs::DirBuilder;
use std::io::ErrorKind;
use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_core::presence::PresenceLlmHealthProbe;
use assistd_core::socket::StartupLock;
use assistd_core::{
    AppState, Config, MemoryStack, PresenceManager, RelayDeps, RuntimeState, SessionContext,
    SubAgentRunner, Subsystems, VisionRevalidator, VoiceManager,
};
use assistd_llm::{
    HistoryReplay, LlamaChatClient, LlamaServerControl, LlmBackend, LlmHealthProbe, SlotDir,
    SlotStash,
};
use assistd_utils::child_server::ApiKey;
use clap::Args;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use embed_init::{EmbeddingHandles, EmbeddingService, EmbeddingStages, EmbeddingStartup};
use mcp_init::{McpFleet, McpService};
use memory_init::MemorySubsystem;
use search_init::{SearchService, SearchStartup};
use shutdown::{DaemonShutdown, IntakeTasks, ShutdownStages, spawn_signal_handler};
use tools_init::ToolDeps;
use voice_init::VoiceService;
use vram::VramLedger;
use warmup::Warmup;

mod embed_init;
mod gpu_monitor;
mod idle_monitor;
mod listen_dispatcher;
mod mcp_init;
mod memory_init;
mod relay_answers;
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

const RELAY_DIR_MODE: u32 = 0o700;

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

    info!(
        "assistd v{}: local model agent OS assistant daemon",
        crate::VERSION
    );
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
    let mut memory = memory_init::init().await?;
    let api_key = ApiKey::generate().context("failed to write the model servers' API key")?;
    let slot_dir = prepare_slot_dir()?;
    let presence = PresenceManager::new_sleeping(
        config.model.clone(),
        config.timeouts.clone(),
        api_key.clone(),
        slot_dir.clone(),
        stages.llm.subscribe(),
    )?;
    let health_probe: Arc<dyn LlmHealthProbe> =
        Arc::new(PresenceLlmHealthProbe::new(presence.clone()));
    let vision_revalidator = build_vision_revalidator(&config, &api_key, &presence, &health_probe)?;
    let voice = VoiceManager::new(config.voice.synthesis.enabled);

    let vram = open_vram_ledger(&config, &presence);

    let gpu_monitor_handle = gpu_monitor::spawn_monitor(
        &config.sleep,
        vram.clone(),
        presence.clone(),
        stages.intake.subscribe(),
    );
    let idle_monitor_handle =
        idle_monitor::spawn_monitor(&config.sleep, presence.clone(), stages.intake.subscribe());

    let (embed, embed_startup) = embed_init::prepare(&config, &api_key, &memory.sqlite_handle);
    let (web_search, search_startup) = search_init::prepare(&config);
    let window = wm_init::init(&config, &stages.tools).await;

    let session_ctx = Arc::new(SessionContext::new(memory.session_id, memory.branch_id));

    let (chat, relay, slot_stash) = build_chat_and_relay(
        &config,
        &api_key,
        slot_dir,
        &health_probe,
        &session_ctx,
        memory.relay_answer_files.take(),
        vision_revalidator.slot_count(),
    )?;
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
            relay,
        },
    )?;

    replay_history(chat.as_ref(), std::mem::take(&mut memory.resumed_history)).await;

    let subsystems = Subsystems::new(chat, presence.clone(), tools.registry, voice.clone())
        .with_window_manager(window.manager.clone())
        .with_vision_revalidator(vision_revalidator.clone())
        .with_mcp_servers(tools.mcp_servers, tools.mcp_fleet.clone())
        .with_web_search(web_search)
        .with_tools_disabled(tools.disabled)
        .with_planning_tools(tools.planning)
        .with_slot_stash(slot_stash);

    let state = Arc::new(AppState {
        memory: build_memory_stack(&config, &memory, &embed),
        config,
        subsystems,
        runtime: build_runtime(session_ctx, &vision_revalidator),
    });
    state.start_task_queue().await;
    let task_runner = Arc::clone(&state.runtime.task_runner);
    let background = spawn_background_starts(
        &state,
        BackgroundStarts {
            vision: vision_revalidator,
            vram,
            embed: embed_startup,
            search: search_startup,
            mcp: tools.mcp_fleet,
        },
        stages,
    );

    Ok((
        state.clone(),
        DaemonShutdown {
            state,
            task_runner,
            presence,
            memory,
            voice: background.voice,
            embed: background.embed,
            search: background.search,
            window,
            mcp: background.mcp,
            intake_tasks: IntakeTasks {
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
    mcp: Option<Arc<McpFleet>>,
}

struct BackgroundServices {
    warmup: JoinHandle<()>,
    voice: VoiceService,
    embed: EmbeddingService,
    search: SearchService,
    mcp: McpService,
}

/// Start the search and MCP servers at once, and the voice worker and the
/// embedding server once the model load has settled so none of them loads
/// onto the GPU beside it.
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
            model_settled: model_settled_tx,
        },
        stages.intake.subscribe(),
    );
    let voice = VoiceService::spawn(
        state.clone(),
        starts.vram,
        model_settled_rx.clone(),
        stages.voice.subscribe(),
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
    let mcp = starts
        .mcp
        .map_or_else(McpService::disabled, |fleet| fleet.spawn(state));
    BackgroundServices {
        warmup,
        voice,
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

/// The chat server's slot directory under the user cache dir, emptied of
/// slots a previous daemon left behind.
fn prepare_slot_dir() -> Result<SlotDir> {
    let cache_dir = dirs::cache_dir().context("no user cache directory for llama-server slots")?;
    let slot_dir = SlotDir::new(cache_dir.join("assistd").join("slots"));
    slot_dir.reset().with_context(|| {
        format!(
            "failed to prepare the slot directory {}",
            slot_dir.path().display()
        )
    })?;
    Ok(slot_dir)
}

/// The chat backend, what the `relay` tool runs sub-agents with beside it on the same
/// slot, its saved answers swept of those not in `kept_answers`, and that slot's stash.
fn build_chat_and_relay(
    config: &Config,
    api_key: &ApiKey,
    slot_dir: SlotDir,
    health_probe: &Arc<dyn LlmHealthProbe>,
    session_ctx: &SessionContext,
    kept_answers: Option<HashSet<PathBuf>>,
    slots: watch::Receiver<NonZeroU32>,
) -> Result<(Arc<dyn LlmBackend>, RelayDeps, Arc<SlotStash>)> {
    let slot_stash = build_slot_stash(config, api_key, slot_dir, health_probe, slots)?;
    let chat = build_chat_backend(
        config,
        api_key,
        Arc::clone(health_probe),
        Arc::clone(&slot_stash),
    )?;
    let relay = build_relay_deps(
        config,
        Arc::clone(&slot_stash),
        &chat,
        health_probe,
        session_ctx,
        kept_answers,
    )?;
    Ok((chat, relay, slot_stash))
}

/// The stash that keeps each conversation's state of the chat server's slot.
fn build_slot_stash(
    config: &Config,
    api_key: &ApiKey,
    slot_dir: SlotDir,
    health_probe: &Arc<dyn LlmHealthProbe>,
    slots: watch::Receiver<NonZeroU32>,
) -> Result<Arc<SlotStash>> {
    let slot_stash = SlotStash::new(
        SocketAddr::new(config.model.host, config.model.port.get()),
        Some(api_key),
        config.model.name.clone(),
        slot_dir,
        Some(Arc::clone(health_probe)),
    )
    .context("failed to construct the slot stash client")?;
    Ok(Arc::new(slot_stash.with_slot_count(slots)))
}

/// What the `relay` tool runs sub-agents with: the chat backend they share, the stash of its
/// slot, an emptied directory for their quarantines, and where their answers are saved,
/// swept of those not in `kept_answers`.
fn build_relay_deps(
    config: &Config,
    slot_stash: Arc<SlotStash>,
    chat: &Arc<dyn LlmBackend>,
    health_probe: &Arc<dyn LlmHealthProbe>,
    session_ctx: &SessionContext,
    kept_answers: Option<HashSet<PathBuf>>,
) -> Result<RelayDeps> {
    Ok(RelayDeps {
        runner: SubAgentRunner::new(
            Arc::clone(chat),
            Some(Arc::clone(health_probe)),
            Duration::from_secs(config.timeouts.tool_call_secs),
        ),
        slot_stash,
        dir: prepare_relay_dir()?,
        subagent_context_tokens: config
            .chat
            .max_history_tokens
            .get()
            .min(config.model.context_budget()),
        answers_dir: relay_answers::prepare(kept_answers)?,
        session: session_ctx.session_updates(),
    })
}

/// The owner-only directory relay quarantines are created in, under the
/// user cache dir, cleared of any a previous daemon left behind.
fn prepare_relay_dir() -> Result<PathBuf> {
    let cache_dir = dirs::cache_dir().context("no user cache directory for relay quarantines")?;
    let relay_dir = cache_dir.join("assistd").join("relays");
    if let Err(error) = std::fs::remove_dir_all(&relay_dir)
        && error.kind() != ErrorKind::NotFound
    {
        warn!(
            "cannot clear stale relay quarantines in {}: {error}",
            relay_dir.display()
        );
    }
    DirBuilder::new()
        .recursive(true)
        .mode(RELAY_DIR_MODE)
        .create(&relay_dir)
        .with_context(|| {
            format!(
                "failed to create the relay directory {}",
                relay_dir.display()
            )
        })?;
    Ok(relay_dir)
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

fn build_chat_backend(
    config: &Config,
    api_key: &ApiKey,
    health_probe: Arc<dyn LlmHealthProbe>,
    slot_stash: Arc<SlotStash>,
) -> Result<Arc<dyn LlmBackend>> {
    let chat = LlamaChatClient::new(
        &config.chat,
        &config.model,
        &config.timeouts,
        Some(api_key),
        Some(health_probe),
    )?
    .with_slot_stash(slot_stash);
    Ok(Arc::new(chat))
}

/// Request bookkeeping over the resumed session, with model lanes sized by the
/// server's slot count.
fn build_runtime(session_ctx: Arc<SessionContext>, vision: &VisionRevalidator) -> RuntimeState {
    RuntimeState::new()
        .with_session_ctx(session_ctx)
        .with_slot_count(vision.slot_count())
}

/// The VRAM ledger, checked by `presence` before every wake; `None` when the GPU
/// monitor is off or cannot read the GPU.
fn open_vram_ledger(config: &Config, presence: &PresenceManager) -> Option<Arc<VramLedger>> {
    let vram = VramLedger::open(&config.sleep)?;
    presence.set_wake_room(vram.clone());
    Some(vram)
}

fn build_memory_stack(
    config: &Config,
    memory: &MemorySubsystem,
    embed: &EmbeddingHandles,
) -> MemoryStack {
    MemoryStack::disabled(config.embedding.clone())
        .with_memory(memory.memory_store.clone())
        .with_transcripts(memory.transcript_store.clone())
        .with_tasks(memory.task_store.clone())
        .with_embedder(embed.embedder.clone())
        .with_semantic(embed.semantic_store.clone())
        .with_embed_wake(embed.embed_wake.clone())
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
        .with_env_filter(crate::env_filter_or(DEFAULT_LOG_DIRECTIVES))
        .init();
}

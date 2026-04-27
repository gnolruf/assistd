#![allow(unsafe_code)] // libc / env / fd primitives — each unsafe block is locally justified

//! Interactive ratatui-based chat TUI.
//!
//! `assistd chat` loads the existing config, starts its own llama-server
//! via `LlamaService`, creates a `LlamaChatClient`, and drives a
//! three-region terminal UI (output / status / input) with live streaming
//! tokens, readline-style input, and a VRAM/throughput status bar.

mod app;
mod gate;
mod input;
mod output;
mod throughput;
mod ui;
mod voice;
mod vram;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_core::{Config, PresenceManager, SleepConfig, ToolRegistry};
use assistd_llm::{FailedBackend, LlamaChatClient, LlmBackend};

use crate::idle_monitor;
use clap::Args;
use crossterm::event::{self, Event, EventStream};
use crossterm::{cursor, execute, terminal};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui_image::picker::{Picker, ProtocolType};
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::{mpsc, watch};
use tracing::info;

use self::app::{App, ChatEvent};
use self::gate::{PendingConfirmation, TuiGate};

#[derive(Args)]
pub struct ChatArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

pub async fn run(args: ChatArgs) -> Result<()> {
    // Redirect process-level stderr to a sibling log file BEFORE any
    // subsystem initialization runs. C libraries (ALSA's JACK/OSS
    // fallback probing, whisper.cpp when hooks aren't yet installed,
    // future TTS backends) write diagnostics straight to fd 2; without
    // this they leak into the ratatui draw surface and glitch the TUI.
    let _stderr_redirect = redirect_stderr_to_log()?;

    let config_path = match args.config {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;
    idle_monitor::validate(&config.sleep)?;
    assistd_voice::mic_validate(&config.voice)?;

    let _log_guard = init_file_tracing()?;

    info!("assistd chat v{}", assistd_core::version());
    info!("loaded config from {}", config_path.display());

    let (shutdown_tx, _) = watch::channel(false);
    install_signal_handler(shutdown_tx.clone());

    // Process-scoped overflow dir: the chat TUI and the daemon must not
    // share a path, otherwise their startup resets race each other.
    let tui_overflow_dir =
        std::env::temp_dir().join(format!("assistd-chat-{}", std::process::id()));

    // The TUI gate forwards destructive-command prompts to the modal
    // overlay via this channel. Capacity 8 is generous — we process one
    // at a time and the agent loop is blocked waiting for the response.
    let (confirm_tx, confirm_rx) = mpsc::channel::<PendingConfirmation>(8);

    println!("loading model {} ...", config.model.name);

    let (presence, client, startup_error) = match PresenceManager::new_active(
        config.llama_server.clone(),
        config.model.clone(),
        shutdown_tx.subscribe(),
    )
    .await
    {
        Ok(presence) => {
            info!(
                "llama-server ready on {}:{}",
                config.llama_server.host, config.llama_server.port
            );
            let client: Arc<dyn LlmBackend> = Arc::new(LlamaChatClient::new(
                &config.chat,
                &config.llama_server,
                &config.model,
            )?);
            (Some(presence), client, None)
        }
        Err(e) => {
            let msg = e.to_string();
            tracing::error!("llama-server failed to start: {msg}");
            let client: Arc<dyn LlmBackend> = Arc::new(FailedBackend::new(msg.clone()));
            (None, client, Some(msg))
        }
    };

    // Capability probe — only meaningful when llama-server actually
    // came up. The FailedBackend path already surfaces the startup
    // error to the TUI, so skip the misleading mmproj warning in that
    // case. Held in a VisionGate so a daemon-side revalidation flips
    // see/screenshot/attach_image without a registry rebuild.
    let initial_vision_state = if presence.is_some() {
        let s =
            assistd_llm::probe_capabilities(&config.llama_server.host, config.llama_server.port)
                .await;
        if s.vision_supported {
            info!("vision: enabled (model has mmproj)");
        } else {
            tracing::warn!("Vision not available: mmproj not loaded.");
        }
        s
    } else {
        assistd_llm::VisionState::default()
    };
    let vision_enabled = initial_vision_state.vision_supported;
    let vision_gate = assistd_tools::VisionGate::new(vision_enabled);

    let tools = assistd_core::build_tools(
        &config,
        tui_overflow_dir.clone(),
        Arc::new(TuiGate::new(confirm_tx)),
        vision_gate.clone(),
    )?;
    info!(
        "tools: registered {} (overflow dir {})",
        tools.len(),
        tui_overflow_dir.display()
    );

    let mut resource_rx = vram::spawn_probe(shutdown_tx.subscribe());

    let idle_monitor_handle = presence.as_ref().and_then(|p| {
        idle_monitor::spawn_monitor(&config.sleep, p.clone(), shutdown_tx.subscribe())
    });

    // Voice pipeline: builds a `MicVoiceInput`, spawns the hotkey
    // listener (press/release), and gives us a channel of
    // VoiceEvents to plumb into the event loop. Always returns a
    // handle — even when voice is disabled, which just gives a
    // `NoVoiceInput` + no hotkey bound.
    let (voice_tx, voice_rx) = mpsc::channel::<voice::VoiceEvent>(32);
    // Keep the pipeline's handles alive for the lifetime of the TUI
    // so the hotkey listener and state forwarder don't drop their
    // `Arc<dyn VoiceInput>` reference. Dropped at the end of
    // `run()` after the shutdown signal fires.
    let _voice_pipeline = voice::spawn(&config, voice_tx, shutdown_tx.subscribe()).await;

    let model_name = config
        .model
        .name
        .rsplit_once('/')
        .map(|(_, rest)| rest.to_string())
        .unwrap_or_else(|| config.model.name.clone());

    let run_result = run_tui(
        client,
        tools,
        config.agent.max_iterations,
        model_name,
        config.sleep.clone(),
        presence.clone(),
        vision_enabled,
        &mut resource_rx,
        shutdown_tx.clone(),
        startup_error,
        confirm_rx,
        voice_rx,
    )
    .await;

    let _ = shutdown_tx.send(true);
    if let Some(h) = idle_monitor_handle {
        let _ = h.await;
    }
    if let Some(p) = presence {
        if let Err(e) = p.sleep().await {
            tracing::error!("presence shutdown error: {e:#}");
        }
    }
    info!("assistd chat stopped");
    run_result
}

#[allow(clippy::too_many_arguments)]
async fn run_tui(
    client: Arc<dyn LlmBackend>,
    tools: Arc<ToolRegistry>,
    max_iterations: u32,
    model_name: String,
    sleep_cfg: SleepConfig,
    presence: Option<Arc<PresenceManager>>,
    vision_enabled: bool,
    resource_rx: &mut watch::Receiver<vram::ResourceState>,
    shutdown_tx: watch::Sender<bool>,
    startup_error: Option<String>,
    mut confirmation_rx: mpsc::Receiver<PendingConfirmation>,
    mut voice_rx: mpsc::Receiver<voice::VoiceEvent>,
) -> Result<()> {
    terminal::enable_raw_mode().context("enable_raw_mode")?;
    if let Err(e) = execute!(
        std::io::stdout(),
        terminal::EnterAlternateScreen,
        event::EnableMouseCapture
    ) {
        let _ = terminal::disable_raw_mode();
        return Err(e).context("EnterAlternateScreen");
    }

    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = TerminalGuard::cleanup();
        previous_hook(info);
    }));
    let _guard = TerminalGuard;

    // Probe terminal graphics support immediately after entering the
    // alternate screen and before reading any events (per
    // `Picker::from_query_stdio` docs). Halfblocks is treated as "no
    // graphics" so `/attach` falls back to filename-only display on
    // non-graphics terminals — chunky ASCII thumbnails are gimmicky.
    let picker = match Picker::from_query_stdio() {
        Ok(p)
            if matches!(
                p.protocol_type(),
                ProtocolType::Kitty | ProtocolType::Sixel | ProtocolType::Iterm2
            ) =>
        {
            tracing::info!(
                "terminal graphics: {:?} (font_size {:?})",
                p.protocol_type(),
                p.font_size()
            );
            Some(p)
        }
        Ok(p) => {
            tracing::info!(
                "terminal graphics: {:?} → /attach will display filenames only",
                p.protocol_type()
            );
            None
        }
        Err(e) => {
            tracing::info!(
                "terminal graphics probe failed ({e}); /attach will display filenames only"
            );
            None
        }
    };

    let backend = CrosstermBackend::new(std::io::stdout());
    let mut terminal = Terminal::new(backend).context("Terminal::new")?;

    let (chat_tx, mut chat_rx) = mpsc::channel::<ChatEvent>(64);
    let mut app = App::new(
        client,
        tools,
        max_iterations,
        chat_tx,
        model_name,
        sleep_cfg,
        presence.clone(),
        vision_enabled,
        picker,
    );

    if let Some(err) = startup_error {
        app.output
            .push_error(&format!("llama-server failed to start: {err}"));
        app.output
            .push_error("check config and model path, then restart");
    }

    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(250));
    let mut shutdown_rx = shutdown_tx.subscribe();
    let mut presence_rx = presence.as_ref().map(|p| p.subscribe());

    terminal.draw(|f| ui::render(f, &mut app))?;

    loop {
        if app.should_quit() {
            break;
        }
        tokio::select! {
            maybe_ev = events.next() => {
                match maybe_ev {
                    Some(Ok(Event::Key(k))) => app.on_key(k),
                    Some(Ok(Event::Resize(_, _))) => {}
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::error!("terminal event error: {e}");
                        break;
                    }
                    None => break,
                }
            }
            Some(ev) = chat_rx.recv() => {
                app.on_llm_event(ev);
            }
            _ = tick.tick() => {
                app.on_tick();
            }
            Ok(_) = resource_rx.changed() => {
                let v = resource_rx.borrow_and_update().clone();
                app.on_resources(v);
            }
            Some(pending) = confirmation_rx.recv() => {
                app.open_confirmation_modal(pending);
            }
            Some(v) = voice_rx.recv() => {
                match v {
                    voice::VoiceEvent::State(s) => app.on_voice_state(s),
                    voice::VoiceEvent::Transcription(text) => app.on_transcription(text),
                    voice::VoiceEvent::Error(msg) => app.on_voice_error(msg),
                }
            }
            presence_changed = async {
                match presence_rx.as_mut() {
                    Some(rx) => rx.changed().await.map(|_| true),
                    None => std::future::pending().await,
                }
            } => {
                if presence_changed.is_ok() {
                    if let Some(rx) = presence_rx.as_mut() {
                        let s = *rx.borrow_and_update();
                        app.on_presence(s);
                    }
                }
            }
            _ = shutdown_rx.changed() => {
                break;
            }
        }
        terminal.draw(|f| ui::render(f, &mut app))?;
    }

    drop(terminal);
    Ok(())
}

fn install_signal_handler(shutdown_tx: watch::Sender<bool>) {
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
        let _ = shutdown_tx.send(true);
    });
}

fn log_dir() -> Result<PathBuf> {
    let dir = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .map(|p| p.join("assistd"))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state/assistd")))
        .unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating log dir {}", dir.display()))?;
    Ok(dir)
}

fn init_file_tracing() -> Result<tracing_appender::non_blocking::WorkerGuard> {
    use tracing_subscriber::{EnvFilter, fmt};

    let file_appender = tracing_appender::rolling::daily(log_dir()?, "chat.log");
    let (writer, guard) = tracing_appender::non_blocking(file_appender);

    fmt()
        .with_writer(writer)
        .with_ansi(false)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    Ok(guard)
}

/// Redirect the process's stderr (fd 2) to a dedicated log file next to
/// `chat.log`. Returns the owning `File` — keep it alive for the TUI
/// lifetime; dropping it doesn't close fd 2 because `dup2` has already
/// linked the kernel descriptor. Panics and C-library stderr writes all
/// land in `chat-stderr.log` instead of the ratatui draw surface.
fn redirect_stderr_to_log() -> Result<std::fs::File> {
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;

    let path = log_dir()?.join("chat-stderr.log");
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening stderr log {}", path.display()))?;
    // SAFETY: dup2 on fd 2 is a standard POSIX operation; the target fd
    // is a valid, owned descriptor from `OpenOptions::open`. On failure
    // we surface the errno and leave stderr untouched.
    let rc = unsafe { libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO) };
    if rc < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("dup2 stderr → {}", path.display()));
    }
    Ok(file)
}

struct TerminalGuard;

impl TerminalGuard {
    fn cleanup() -> std::io::Result<()> {
        terminal::disable_raw_mode()?;
        execute!(
            std::io::stdout(),
            event::DisableMouseCapture,
            terminal::LeaveAlternateScreen,
            cursor::Show,
        )?;
        Ok(())
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = Self::cleanup();
    }
}

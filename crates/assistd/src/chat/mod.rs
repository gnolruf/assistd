//! Interactive ratatui-based chat TUI.
//!
//! `assistd chat` is a thin window onto the running daemon. It loads
//! the existing config, probes the daemon's Unix socket (auto-spawning
//! `assistd daemon` when nothing is listening), and drives a
//! three-region terminal UI (output / status / input) by streaming
//! `Event`s back from the daemon over IPC.
//!
//! No LLM service, voice pipeline, presence manager, tool registry,
//! or memory store is constructed in this process: all of that lives
//! in the daemon. The TUI only owns: ratatui rendering, key handling,
//! the local hotkey grab (PTT keystrokes need to arrive at the
//! foreground process), VRAM/throughput probes, and attachment
//! staging.

mod app;
mod input;
mod output;
mod throughput;
mod ui;
mod voice;
mod vram;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_core::{Config, SleepConfig};
use assistd_ipc::{Event, IpcClient, Request};
use clap::Args;
use crossterm::event::{self, Event as TermEvent, EventStream};
use crossterm::{cursor, execute, terminal};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui_image::picker::{Picker, ProtocolType};
use tokio::net::UnixStream;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::{mpsc, watch};
use tracing::info;
use uuid::Uuid;

use self::app::{App, ChatEvent};

/// Arguments for the `chat` subcommand.
#[derive(Args)]
pub struct ChatArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

struct TuiContext {
    ipc: Arc<IpcClient>,
    chat_tx: mpsc::Sender<ChatEvent>,
    chat_rx: mpsc::Receiver<ChatEvent>,
    resource_rx: watch::Receiver<vram::ResourceState>,
    shutdown_rx: watch::Receiver<bool>,
    model_name: String,
    sleep_cfg: SleepConfig,
    vision_enabled: bool,
    startup_error: Option<String>,
}

/// Launch the interactive chat TUI, auto-spawning the daemon if needed.
///
/// # Errors
///
/// Returns an error if config loading fails, the terminal cannot be set up,
/// or a fatal I/O error occurs during the session.
pub async fn run(args: ChatArgs) -> Result<()> {
    let _stderr_redirect = redirect_stderr_to_log()?;

    let config_path = match args.config.clone() {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;

    let _log_guard = init_file_tracing()?;

    info!("assistd chat v{}", assistd_core::version());
    info!("loaded config from {}", config_path.display());

    let (shutdown_tx, _) = watch::channel(false);
    install_signal_handler(shutdown_tx.clone());

    let ipc = Arc::new(IpcClient::new());
    let mut startup_error: Option<String> = None;
    if UnixStream::connect(ipc.socket_path()).await.is_err() {
        info!(
            "daemon not reachable at {}; auto-spawning",
            ipc.socket_path().display()
        );
        match spawn_daemon_detached(args.config.as_deref()) {
            Ok(()) => {
                if let Err(e) = wait_for_socket(ipc.socket_path(), Duration::from_secs(30)).await {
                    startup_error =
                        Some(format!("daemon spawned but socket never became ready: {e}"));
                }
            }
            Err(e) => {
                startup_error = Some(format!(
                    "could not auto-start daemon: {e}; run `assistd daemon` manually then retry"
                ));
            }
        }
    }

    let (vision_enabled, daemon_model_name) = if startup_error.is_none() {
        match get_capabilities(&ipc).await {
            Ok((vision, name)) => (vision, name),
            Err(e) => {
                info!("get_capabilities failed: {e:#}");
                (false, String::new())
            }
        }
    } else {
        (false, String::new())
    };
    let model_name = if daemon_model_name.is_empty() {
        config
            .model
            .name
            .rsplit_once('/')
            .map(|(_, rest)| rest.to_string())
            .unwrap_or_else(|| config.model.name.clone())
    } else {
        daemon_model_name
    };

    let resource_rx = vram::spawn_probe(shutdown_tx.subscribe());

    let (chat_tx, chat_rx) = mpsc::channel::<ChatEvent>(64);
    let _voice_pipeline = voice::spawn(
        &config,
        ipc.clone(),
        chat_tx.clone(),
        shutdown_tx.subscribe(),
    )
    .await;

    let _polling_handle =
        spawn_status_polling(ipc.clone(), chat_tx.clone(), shutdown_tx.subscribe());

    let run_result = run_tui(TuiContext {
        ipc: ipc.clone(),
        chat_tx,
        chat_rx,
        resource_rx,
        shutdown_rx: shutdown_tx.subscribe(),
        model_name,
        sleep_cfg: config.sleep.clone(),
        vision_enabled,
        startup_error,
    })
    .await;

    let _ = shutdown_tx.send(true);
    info!("assistd chat stopped");
    run_result
}

async fn run_tui(ctx: TuiContext) -> Result<()> {
    let TuiContext {
        ipc,
        chat_tx,
        mut chat_rx,
        mut resource_rx,
        mut shutdown_rx,
        model_name,
        sleep_cfg,
        vision_enabled,
        startup_error,
    } = ctx;

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

    let mut app = App::new(ipc, chat_tx, model_name, sleep_cfg, vision_enabled, picker);

    if let Some(err) = startup_error {
        app.output.push_error(&format!("daemon startup: {err}"));
        app.output
            .push_error("once the daemon is reachable, retry your query");
    } else {
        // 10s mirrors the "follow-up window" smart-speakers use: a
        // conversation triggered via voice (or any other daemon
        // channel) within the last few seconds stays threaded, but
        // opening the TUI cold gives a fresh chat.
        app.spawn_resume_or_new(10);
    }

    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(250));

    terminal.draw(|f| ui::render(f, &mut app))?;

    loop {
        if app.should_quit() {
            break;
        }
        tokio::select! {
            maybe_ev = events.next() => {
                match maybe_ev {
                    Some(Ok(TermEvent::Key(k))) => app.on_key(k),
                    Some(Ok(TermEvent::Mouse(m))) => app.on_mouse(m),
                    Some(Ok(TermEvent::Resize(_, _))) => {}
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::error!("terminal event error: {e}");
                        break;
                    }
                    None => break,
                }
            }
            Some(ev) = chat_rx.recv() => {
                app.on_chat_event(ev);
            }
            _ = tick.tick() => {
                app.on_tick();
            }
            Ok(_) = resource_rx.changed() => {
                let v = resource_rx.borrow_and_update().clone();
                app.on_resources(v);
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

/// Auto-spawn a detached daemon child. We exec the same binary
/// (`current_exe`) so a `cargo run` build doesn't accidentally fork
/// off a stale system install. The child calls `setsid()` itself on
/// startup (gated on `--client-mode`) so it becomes its own session
/// leader and survives the TUI's controlling-terminal closing; we
/// drop the `Child` handle without reaping, which is fine because
/// the daemon already self-handles SIGTERM.
fn spawn_daemon_detached(config: Option<&Path>) -> Result<()> {
    use std::process::{Command, Stdio};

    let exe = std::env::current_exe().context("std::env::current_exe()")?;
    let mut cmd = Command::new(&exe);
    cmd.arg("daemon").arg("--client-mode");
    if let Some(p) = config {
        cmd.arg("--config").arg(p);
    }

    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = cmd
        .spawn()
        .with_context(|| format!("could not spawn daemon binary at {}", exe.display()))?;
    info!("spawned daemon pid {}", child.id());
    drop(child);
    Ok(())
}

async fn wait_for_socket(path: &Path, deadline: Duration) -> Result<()> {
    let start = std::time::Instant::now();
    loop {
        if UnixStream::connect(path).await.is_ok() {
            return Ok(());
        }
        if start.elapsed() >= deadline {
            anyhow::bail!(
                "timed out after {:?} waiting for daemon socket at {}",
                deadline,
                path.display()
            );
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn get_capabilities(ipc: &IpcClient) -> Result<(bool, String)> {
    let req = Request::GetCapabilities {
        id: Uuid::new_v4().to_string(),
    };
    let mut stream = ipc.one_shot(req).await?;
    let mut vision = false;
    let mut model_name = String::new();
    loop {
        match stream.next_event().await? {
            Some(Event::Capabilities {
                vision: v,
                model_name: m,
                ..
            }) => {
                vision = v;
                model_name = m;
            }
            Some(Event::Done { .. }) => return Ok((vision, model_name)),
            Some(Event::Error { message, .. }) => anyhow::bail!("{message}"),
            Some(_) => {} // ignore stray events
            None => anyhow::bail!("daemon closed without responding"),
        }
    }
}

fn spawn_status_polling(
    ipc: Arc<IpcClient>,
    chat_tx: mpsc::Sender<ChatEvent>,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(2));
        loop {
            tokio::select! {
                _ = shutdown.changed() => break,
                _ = tick.tick() => {
                    poll_one(&ipc, &chat_tx, Request::GetPresence { id: Uuid::new_v4().to_string() }).await;
                    poll_one(&ipc, &chat_tx, Request::GetVoiceState { id: Uuid::new_v4().to_string() }).await;
                    poll_one(&ipc, &chat_tx, Request::GetListenState { id: Uuid::new_v4().to_string() }).await;
                }
            }
        }
    })
}

async fn poll_one(ipc: &IpcClient, chat_tx: &mpsc::Sender<ChatEvent>, req: Request) {
    let kind = req.kind();
    let mut stream = match ipc.one_shot(req).await {
        Ok(s) => s,
        Err(e) => {
            tracing::debug!("status poll {kind} failed: {e}");
            return;
        }
    };
    while let Ok(Some(ev)) = stream.next_event().await {
        if ev.is_terminal() {
            break;
        }
        let _ = chat_tx.send(ChatEvent::Wire(ev)).await;
    }
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

fn redirect_stderr_to_log() -> Result<std::fs::File> {
    use std::fs::OpenOptions;

    let path = log_dir()?.join("chat-stderr.log");
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening stderr log {}", path.display()))?;
    rustix::stdio::dup2_stderr(&file)
        .with_context(|| format!("dup2 stderr → {}", path.display()))?;
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

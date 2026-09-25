//! `assistd chat`: a terminal window onto the running daemon. Owns
//! rendering, key handling, the local hotkey grab, resource probes and
//! attachment staging; every service lives in the daemon.

use std::fs::{File, OpenOptions};
use std::io;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use assistd_core::{Config, SleepConfig};
use assistd_ipc::{Event, EventKind, IpcClient, Request, SubscribeFilter};
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
use tokio::task::JoinHandle;
use tokio_util::task::AbortOnDropHandle;
use tracing::info;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, fmt};
use uuid::Uuid;

use self::app::{App, ChatEvent, WireStream};

mod app;
mod input;
mod output;
mod throughput;
mod ui;
mod voice;
mod vram;

const CHAT_CHANNEL_CAPACITY: usize = 64;
const DAEMON_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const RESUME_RECENCY_SECS: u64 = 10;
const TICK_INTERVAL: Duration = Duration::from_millis(250);
const STATUS_POLL_INTERVAL: Duration = Duration::from_secs(2);
const TITLE_RECONNECT_DELAY: Duration = Duration::from_secs(2);

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

/// Restores the terminal on drop.
struct TerminalGuard;

impl TerminalGuard {
    /// Enter raw mode and the alternate screen, and install a panic hook
    /// that restores the terminal before the previous hook runs.
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode().context("enable_raw_mode")?;
        if let Err(e) = execute!(
            io::stdout(),
            terminal::EnterAlternateScreen,
            event::EnableMouseCapture
        ) {
            let _ = terminal::disable_raw_mode();
            return Err(e).context("EnterAlternateScreen");
        }
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = Self::cleanup();
            previous_hook(info);
        }));
        Ok(Self)
    }

    fn cleanup() -> io::Result<()> {
        terminal::disable_raw_mode()?;
        execute!(
            io::stdout(),
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

/// Run the TUI, auto-spawning the daemon when nothing is listening.
pub async fn run(args: ChatArgs) -> Result<()> {
    let _stderr_redirect = redirect_stderr_to_log()?;

    let _log_guard = init_file_tracing()?;

    let config_path = match args.config.clone() {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;

    info!("assistd chat v{}", assistd_core::version());
    info!("loaded config from {}", config_path.display());

    let (shutdown_tx, _) = watch::channel(false);
    let _signal_handler = AbortOnDropHandle::new(install_signal_handler(shutdown_tx.clone()));

    let ipc = Arc::new(IpcClient::new());
    let startup_error = ensure_daemon(&ipc, args.config.as_deref()).await.err();
    let (vision_enabled, model_name) =
        resolve_capabilities(&ipc, &config, startup_error.is_none()).await;

    let (resource_rx, resource_probe) = vram::spawn_probe(shutdown_tx.subscribe());
    let _resource_probe = AbortOnDropHandle::new(resource_probe);

    let (chat_tx, chat_rx) = mpsc::channel::<ChatEvent>(CHAT_CHANNEL_CAPACITY);
    let voice_pipeline = voice::spawn_pipeline(
        &config,
        ipc.clone(),
        chat_tx.clone(),
        shutdown_tx.subscribe(),
    )
    .await;

    let _polling_handle = AbortOnDropHandle::new(spawn_status_polling(
        ipc.clone(),
        chat_tx.clone(),
        shutdown_tx.subscribe(),
    ));
    let _title_handle = AbortOnDropHandle::new(spawn_title_subscription(
        ipc.clone(),
        chat_tx.clone(),
        shutdown_tx.subscribe(),
    ));

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
    voice_pipeline.shutdown().await;
    info!("assistd chat stopped");
    run_result
}

/// Spawn the daemon and wait for its socket when nothing is listening.
/// `Err` is the message shown in the output pane.
async fn ensure_daemon(ipc: &IpcClient, config: Option<&Path>) -> Result<(), String> {
    if UnixStream::connect(ipc.socket_path()).await.is_ok() {
        return Ok(());
    }
    info!(
        "daemon not reachable at {}; auto-spawning",
        ipc.socket_path().display()
    );
    spawn_daemon_detached(config).map_err(|e| {
        format!("could not auto-start daemon: {e}; run `assistd daemon` manually then retry")
    })?;
    wait_for_socket(ipc.socket_path(), DAEMON_STARTUP_TIMEOUT)
        .await
        .map_err(|e| format!("daemon spawned but socket never became ready: {e}"))
}

/// Vision support and the display model name: the daemon's when it
/// answers, else the configured model's basename.
async fn resolve_capabilities(ipc: &IpcClient, config: &Config, daemon_up: bool) -> (bool, String) {
    let (vision_enabled, daemon_model_name) = if daemon_up {
        get_capabilities(ipc).await.unwrap_or_else(|e| {
            info!("get_capabilities failed: {e:#}");
            (false, String::new())
        })
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
    (vision_enabled, model_name)
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

    let _guard = TerminalGuard::enter()?;
    let picker = probe_graphics();
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend).context("Terminal::new")?;

    let mut app = App::new(ipc, chat_tx, model_name, sleep_cfg, vision_enabled, picker);
    match startup_error {
        Some(err) => {
            app.output.push_error(&format!("daemon startup: {err}"));
            app.output
                .push_error("once the daemon is reachable, retry your query");
        }
        None => app.spawn_resume_or_new(RESUME_RECENCY_SECS),
    }

    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(TICK_INTERVAL);

    terminal.draw(|f| ui::render(f, &mut app))?;

    while !app.should_quit() {
        tokio::select! {
            maybe_ev = events.next() => {
                if apply_terminal_event(&mut app, maybe_ev).is_break() {
                    break;
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
        drain_queued(&mut app, &mut chat_rx, &mut resource_rx);
        terminal.draw(|f| ui::render(f, &mut app))?;
    }

    drop(terminal);
    Ok(())
}

/// A picker for terminals that can draw images inline; `None` means
/// `/attach` shows filenames only.
fn probe_graphics() -> Option<Picker> {
    match Picker::from_query_stdio() {
        Ok(p)
            if matches!(
                p.protocol_type(),
                ProtocolType::Kitty | ProtocolType::Sixel | ProtocolType::Iterm2
            ) =>
        {
            info!(
                "terminal graphics: {:?} (font_size {:?})",
                p.protocol_type(),
                p.font_size()
            );
            Some(p)
        }
        Ok(p) => {
            info!(
                "terminal graphics: {:?} → /attach will display filenames only",
                p.protocol_type()
            );
            None
        }
        Err(e) => {
            info!("terminal graphics probe failed ({e}); /attach will display filenames only");
            None
        }
    }
}

/// Breaks when the terminal event stream ends or fails.
fn apply_terminal_event(app: &mut App, maybe_ev: Option<io::Result<TermEvent>>) -> ControlFlow<()> {
    match maybe_ev {
        Some(Ok(TermEvent::Key(k))) => app.on_key(k),
        Some(Ok(TermEvent::Mouse(m))) => app.on_mouse(m),
        Some(Ok(_)) => {}
        Some(Err(e)) => {
            tracing::error!("terminal event error: {e}");
            return ControlFlow::Break(());
        }
        None => return ControlFlow::Break(()),
    }
    ControlFlow::Continue(())
}

/// Apply channel events that are already queued so a burst of deltas
/// costs one frame, bounded so a producer that keeps pace cannot starve
/// redraws. Terminal events stay with the select loop: polling
/// `EventStream` outside it would drop its waker.
fn drain_queued(
    app: &mut App,
    chat_rx: &mut mpsc::Receiver<ChatEvent>,
    resource_rx: &mut watch::Receiver<vram::ResourceState>,
) {
    for _ in 0..CHAT_CHANNEL_CAPACITY {
        let Ok(ev) = chat_rx.try_recv() else {
            break;
        };
        app.on_chat_event(ev);
    }
    if resource_rx.has_changed().unwrap_or(false) {
        let v = resource_rx.borrow_and_update().clone();
        app.on_resources(v);
    }
}

fn spawn_daemon_detached(config: Option<&Path>) -> Result<()> {
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
    let start = Instant::now();
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
            Some(_) => {}
            None => anyhow::bail!("daemon closed without responding"),
        }
    }
}

fn spawn_status_polling(
    ipc: Arc<IpcClient>,
    chat_tx: mpsc::Sender<ChatEvent>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(STATUS_POLL_INTERVAL);
        loop {
            tokio::select! {
                _ = shutdown.changed() => break,
                _ = tick.tick() => poll_status(&ipc, &chat_tx).await,
            }
        }
    })
}

async fn poll_status(ipc: &IpcClient, chat_tx: &mpsc::Sender<ChatEvent>) {
    let requests = [
        Request::GetPresence {
            id: Uuid::new_v4().to_string(),
        },
        Request::GetVoiceState {
            id: Uuid::new_v4().to_string(),
        },
        Request::GetListenState {
            id: Uuid::new_v4().to_string(),
        },
    ];
    for req in requests {
        poll_one(ipc, chat_tx, req).await;
    }
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
        let _ = chat_tx
            .send(ChatEvent::Wire {
                stream: WireStream::Status,
                event: ev,
            })
            .await;
    }
}

/// Session titles arrive on the daemon's broadcast bus after the turn that
/// produced them has closed. Reconnects on a fixed delay.
fn spawn_title_subscription(
    ipc: Arc<IpcClient>,
    chat_tx: mpsc::Sender<ChatEvent>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown.changed() => break,
                () = pump_titles(&ipc, &chat_tx) => {
                    tokio::select! {
                        _ = shutdown.changed() => break,
                        _ = tokio::time::sleep(TITLE_RECONNECT_DELAY) => {}
                    }
                }
            }
        }
    })
}

async fn pump_titles(ipc: &IpcClient, chat_tx: &mpsc::Sender<ChatEvent>) {
    let req = Request::Subscribe {
        id: Uuid::new_v4().to_string(),
        filter: SubscribeFilter {
            kinds: vec![EventKind::SessionTitle],
        },
    };
    let mut stream = match ipc.one_shot(req).await {
        Ok(s) => s,
        Err(e) => {
            tracing::debug!("session-title subscribe failed: {e}");
            return;
        }
    };
    while let Ok(Some(ev)) = stream.next_event().await {
        if chat_tx
            .send(ChatEvent::Wire {
                stream: WireStream::Status,
                event: ev,
            })
            .await
            .is_err()
        {
            return;
        }
    }
}

fn install_signal_handler(shutdown_tx: watch::Sender<bool>) -> JoinHandle<()> {
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
    })
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

fn init_file_tracing() -> Result<WorkerGuard> {
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

fn redirect_stderr_to_log() -> Result<File> {
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

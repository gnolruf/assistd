//! `assistd chat`: a terminal window onto the running daemon. Owns
//! rendering, key handling, resource probes and attachment staging; every
//! service lives in the daemon. One chat may be open per user, since every
//! chat drives the daemon's one session.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Write};
use std::ops::ControlFlow;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_core::{Config, SleepConfig};
use assistd_ipc::{Event, EventKind, IpcClient, Request, SubscribeFilter};
use assistd_utils::tracing_init::env_filter_or;
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
use tracing_subscriber::fmt;
use uuid::Uuid;

use self::app::{App, ChatEvent, WireStream};

mod app;
mod focus;
mod input;
mod markdown;
mod output;
mod throughput;
mod ui;
mod vram;

const CHAT_CHANNEL_CAPACITY: usize = 64;
const RESUME_RECENCY_SECS: u64 = 10;
const TICK_INTERVAL: Duration = Duration::from_millis(250);
const STATUS_POLL_INTERVAL: Duration = Duration::from_secs(2);
const BUS_RECONNECT_DELAY: Duration = Duration::from_secs(2);
const DAEMON_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(500);
const INSTANCE_LOCK_FILE: &str = "chat.lock";
const INSTANCE_LOCK_MODE: u32 = 0o600;

#[derive(Args)]
pub(crate) struct ChatArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
    /// Wait for the daemon's socket instead of exiting when no daemon is
    /// listening yet (e.g. when launched at login beside the daemon)
    #[arg(long)]
    pub wait: bool,
}

struct TuiContext {
    ipc: Arc<IpcClient>,
    focus_tx: watch::Sender<bool>,
    chat_tx: mpsc::Sender<ChatEvent>,
    chat_rx: mpsc::Receiver<ChatEvent>,
    resource_rx: watch::Receiver<vram::ResourceState>,
    shutdown_rx: watch::Receiver<bool>,
    model_name: String,
    sleep_cfg: SleepConfig,
    vision_enabled: bool,
}

/// Exclusive hold on the per-user chat lock beside the daemon socket,
/// released when dropped or when the process exits.
struct InstanceLock {
    _file: File,
}

impl InstanceLock {
    /// Fails when another `assistd chat` holds the lock.
    fn acquire(socket_path: &Path) -> Result<Self> {
        let lock_path = socket_path.with_file_name(INSTANCE_LOCK_FILE);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(INSTANCE_LOCK_MODE)
            .open(&lock_path)
            .with_context(|| format!("opening chat lock {}", lock_path.display()))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => anyhow::bail!(
                "another `assistd chat` is already open (lock held at {}); only one chat may \
                 run at a time",
                lock_path.display()
            ),
            Err(TryLockError::Error(e)) => {
                Err(e).with_context(|| format!("locking {}", lock_path.display()))
            }
        }
    }
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
            event::EnableMouseCapture,
            event::EnableFocusChange
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
            event::DisableFocusChange,
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

/// Points fd 2 at a log file, restoring the original stderr on drop so
/// errors returned after the TUI exits still reach the terminal.
struct StderrRedirect {
    original: OwnedFd,
}

impl StderrRedirect {
    fn to_log() -> Result<Self> {
        let path = log_dir()?.join("chat-stderr.log");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening stderr log {}", path.display()))?;
        let original = rustix::io::dup(io::stderr().as_fd()).context("dup stderr")?;
        rustix::stdio::dup2_stderr(&file)
            .with_context(|| format!("dup2 stderr → {}", path.display()))?;
        Ok(Self { original })
    }
}

impl Drop for StderrRedirect {
    fn drop(&mut self) {
        let _ = rustix::stdio::dup2_stderr(&self.original);
    }
}

/// Run the TUI against the running daemon. Fails before touching the
/// terminal when no daemon is listening or another chat is open.
pub(crate) async fn run(args: ChatArgs) -> Result<()> {
    let ipc = Arc::new(IpcClient::new());
    if args.wait {
        wait_for_daemon(&ipc).await?;
    } else {
        require_daemon(&ipc).await?;
    }
    let _instance_lock = InstanceLock::acquire(ipc.socket_path())?;

    let _stderr_redirect = StderrRedirect::to_log()?;

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

    let (vision_enabled, model_name) = resolve_capabilities(&ipc, &config).await;

    let (resource_rx, resource_probe) = vram::spawn_probe(shutdown_tx.subscribe());
    let _resource_probe = AbortOnDropHandle::new(resource_probe);

    let (chat_tx, chat_rx) = mpsc::channel::<ChatEvent>(CHAT_CHANNEL_CAPACITY);

    let _polling_handle = AbortOnDropHandle::new(spawn_status_polling(
        ipc.clone(),
        chat_tx.clone(),
        shutdown_tx.subscribe(),
    ));
    let _bus_handle = AbortOnDropHandle::new(spawn_bus_subscription(
        ipc.clone(),
        chat_tx.clone(),
        shutdown_tx.subscribe(),
    ));

    let (focus_tx, focus_rx) = watch::channel(true);
    let focus_reporter = AbortOnDropHandle::new(focus::spawn_reporter(
        ipc.clone(),
        focus_rx,
        shutdown_tx.subscribe(),
    ));

    let run_result = run_tui(TuiContext {
        ipc: ipc.clone(),
        focus_tx,
        chat_tx,
        chat_rx,
        resource_rx,
        shutdown_rx: shutdown_tx.subscribe(),
        model_name,
        sleep_cfg: config.sleep.clone(),
        vision_enabled,
    })
    .await;

    let _ = shutdown_tx.send(true);
    drop(focus_reporter);
    focus::report_closed(&ipc).await;
    info!("assistd chat stopped");
    run_result
}

/// Fails with start-up instructions when nothing is listening on the
/// daemon socket.
async fn require_daemon(ipc: &IpcClient) -> Result<()> {
    if UnixStream::connect(ipc.socket_path()).await.is_ok() {
        return Ok(());
    }
    anyhow::bail!(
        "no assistd daemon is listening at {}. Start it with `systemctl --user start assistd` \
         or `assistd daemon`; if it is already starting, wait for the model to load (see \
         `journalctl --user -u assistd`) and retry",
        ipc.socket_path().display()
    )
}

/// Poll the daemon socket until something accepts, noting once on stderr
/// that the chat is waiting.
async fn wait_for_daemon(ipc: &IpcClient) -> Result<()> {
    if UnixStream::connect(ipc.socket_path()).await.is_ok() {
        return Ok(());
    }
    writeln!(
        io::stderr(),
        "waiting for the assistd daemon at {} (Ctrl-C to give up)",
        ipc.socket_path().display()
    )?;
    while UnixStream::connect(ipc.socket_path()).await.is_err() {
        tokio::time::sleep(DAEMON_WAIT_POLL_INTERVAL).await;
    }
    Ok(())
}

/// Vision support and the display model name: the daemon's when it
/// answers, else the configured model's basename.
async fn resolve_capabilities(ipc: &IpcClient, config: &Config) -> (bool, String) {
    let (vision_enabled, daemon_model_name) = get_capabilities(ipc).await.unwrap_or_else(|e| {
        info!("get_capabilities failed: {e:#}");
        (false, String::new())
    });
    let model_name = if daemon_model_name.is_empty() {
        config
            .model
            .name
            .rsplit_once('/')
            .map_or_else(|| config.model.name.clone(), |(_, rest)| rest.to_string())
    } else {
        daemon_model_name
    };
    (vision_enabled, model_name)
}

async fn run_tui(ctx: TuiContext) -> Result<()> {
    let TuiContext {
        ipc,
        focus_tx,
        chat_tx,
        mut chat_rx,
        mut resource_rx,
        mut shutdown_rx,
        model_name,
        sleep_cfg,
        vision_enabled,
    } = ctx;

    let _guard = TerminalGuard::enter()?;
    let picker = probe_graphics();
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend).context("Terminal::new")?;

    let mut app = App::new(ipc, chat_tx, model_name, sleep_cfg, vision_enabled, picker);
    app.spawn_resume_or_new(RESUME_RECENCY_SECS);

    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(TICK_INTERVAL);

    terminal.draw(|f| ui::render(f, &mut app))?;

    while !app.should_quit() {
        tokio::select! {
            maybe_ev = events.next() => {
                if apply_terminal_event(&mut app, &focus_tx, maybe_ev).is_break() {
                    break;
                }
            }
            Some(ev) = chat_rx.recv() => {
                app.on_chat_event(ev);
            }
            _ = tick.tick() => {
                app.on_tick();
            }
            Ok(()) = resource_rx.changed() => {
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
fn apply_terminal_event(
    app: &mut App,
    focus: &watch::Sender<bool>,
    maybe_ev: Option<io::Result<TermEvent>>,
) -> ControlFlow<()> {
    match maybe_ev {
        Some(Ok(ev)) => {
            focus::track(focus, &ev);
            match ev {
                TermEvent::Key(k) => app.on_key(k),
                TermEvent::Mouse(m) => app.on_mouse(m),
                _ => {}
            }
        }
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

/// Follow the daemon's broadcast bus for session titles and for turns this
/// chat did not start (push-to-talk, continuous listening, other clients).
/// Reconnects on a fixed delay.
fn spawn_bus_subscription(
    ipc: Arc<IpcClient>,
    chat_tx: mpsc::Sender<ChatEvent>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown.changed() => break,
                () = pump_bus(&ipc, &chat_tx) => {
                    tokio::select! {
                        _ = shutdown.changed() => break,
                        () = tokio::time::sleep(BUS_RECONNECT_DELAY) => {}
                    }
                }
            }
        }
    })
}

async fn pump_bus(ipc: &IpcClient, chat_tx: &mpsc::Sender<ChatEvent>) {
    let req = Request::Subscribe {
        id: Uuid::new_v4().to_string(),
        filter: SubscribeFilter {
            kinds: vec![
                EventKind::SessionTitle,
                EventKind::Transcription,
                EventKind::Delta,
                EventKind::ReasoningDelta,
                EventKind::ToolCall,
                EventKind::ToolResult,
                EventKind::Done,
                EventKind::Error,
            ],
        },
    };
    let mut stream = match ipc.one_shot(req).await {
        Ok(s) => s,
        Err(e) => {
            tracing::debug!("bus subscribe failed: {e}");
            return;
        }
    };
    while let Ok(Some(ev)) = stream.next_event().await {
        if chat_tx.send(ChatEvent::Bus(ev)).await.is_err() {
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
    let dir = assistd_utils::xdg::state_home()
        .map_or_else(std::env::temp_dir, |state| state.join("assistd"));
    std::fs::create_dir_all(&dir).with_context(|| format!("creating log dir {}", dir.display()))?;
    Ok(dir)
}

fn init_file_tracing() -> Result<WorkerGuard> {
    let file_appender = tracing_appender::rolling::daily(log_dir()?, "chat.log");
    let (writer, guard) = tracing_appender::non_blocking(file_appender);

    fmt()
        .with_writer(writer)
        .with_ansi(false)
        .with_env_filter(env_filter_or("info"))
        .init();

    Ok(guard)
}

#[cfg(test)]
mod tests;

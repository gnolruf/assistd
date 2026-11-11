//! `assistd chat`: a terminal window onto the running daemon, which owns every service.
//! One chat may be open per user, since every chat drives the daemon's one session.

use std::fs::{DirBuilder, File, Permissions, TryLockError};
use std::future;
use std::io;
use std::ops::ControlFlow;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_core::Config;
use assistd_core::socket::open_private_lock_file;
use assistd_ipc::{Event, EventKind, IpcClient, Request, SubscribeFilter};
use assistd_utils::tracing_init::env_filter_or;
use clap::Args;
use crossterm::event::{self, Event as TermEvent, EventStream, MouseEventKind};
use crossterm::{cursor, execute, style, terminal};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui_image::picker::{Picker, ProtocolType};
use rustix::fs::{Mode, OFlags};
use tokio::net::UnixStream;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::{Notify, mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::task::AbortOnDropHandle;
use tracing::info;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::fmt;
use uuid::Uuid;

use self::app::{App, ChatEvent};
use self::theme::Theme;

mod app;
mod focus;
mod input;
mod markdown;
mod output;
pub(crate) mod settings;
mod tasks;
mod theme;
mod ui;

const CHAT_CHANNEL_CAPACITY: usize = 64;
const TICK_INTERVAL: Duration = Duration::from_millis(250);
/// Frame interval of the live thinking and compaction glyphs.
const GLYPH_INTERVAL: Duration = Duration::from_millis(150);
/// Frame interval of the `/wakeword` computer's audio and bit lanes.
const SCENE_INTERVAL: Duration = Duration::from_millis(30);
/// Shortest time between two frames, so a burst of events costs one draw.
const FRAME_INTERVAL: Duration = Duration::from_millis(33);
const BUS_RECONNECT_DELAY: Duration = Duration::from_secs(2);
const INSTANCE_LOCK_FILE: &str = "chat.lock";
/// Owner-only modes for the log directory and the stderr log.
const LOG_DIR_MODE: u32 = 0o700;
const LOG_FILE_MODE: u32 = 0o600;
const WINDOW_TITLE: &str = "assistd";
/// xterm window-ops sequences that save and restore the terminal's title.
const PUSH_WINDOW_TITLE: &str = "\x1b[22;0t";
const POP_WINDOW_TITLE: &str = "\x1b[23;0t";

#[derive(Args)]
pub(crate) struct ChatArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

struct TuiContext {
    ipc: Arc<IpcClient>,
    focus_tx: watch::Sender<bool>,
    chat_tx: mpsc::Sender<ChatEvent>,
    chat_rx: mpsc::Receiver<ChatEvent>,
    shutdown_rx: watch::Receiver<bool>,
    capabilities: StartupCapabilities,
    daemon_reachable: bool,
    config_path: PathBuf,
    theme: Theme,
}

/// What the daemon reports about itself when the chat starts.
#[derive(Debug, Default)]
struct StartupCapabilities {
    vision_enabled: bool,
    model_name: String,
    auto_mode: bool,
    /// Each MCP server that is unavailable, with why.
    mcp_unavailable: Vec<(String, String)>,
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
        let file = open_private_lock_file(&lock_path)?;
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
    /// Enter raw mode and the alternate screen with modifier-aware key reports
    /// and bracketed paste, saving the window title, and install a panic hook that restores the
    /// terminal before the previous hook runs.
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode().context("enable_raw_mode")?;
        if let Err(e) = execute!(
            io::stdout(),
            terminal::EnterAlternateScreen,
            style::Print(PUSH_WINDOW_TITLE),
            event::EnableMouseCapture,
            event::EnableFocusChange,
            event::EnableBracketedPaste,
            event::PushKeyboardEnhancementFlags(
                event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
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
            event::PopKeyboardEnhancementFlags,
            event::DisableBracketedPaste,
            event::DisableFocusChange,
            event::DisableMouseCapture,
            style::Print(POP_WINDOW_TITLE),
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
        let file = open_private_log(&path)
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

/// Run the TUI against the daemon, showing it offline until its socket accepts.
/// Fails before touching the terminal when another chat is open.
pub(crate) async fn run(args: ChatArgs) -> Result<()> {
    let ipc = Arc::new(IpcClient::new());
    let daemon_reachable = UnixStream::connect(ipc.socket_path()).await.is_ok();
    let _instance_lock = InstanceLock::acquire(ipc.socket_path())?;

    let _stderr_redirect = StderrRedirect::to_log()?;

    let _log_guard = init_file_tracing()?;

    let config_path = match args.config.clone() {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;

    info!("assistd chat v{}", crate::VERSION);
    info!("loaded config from {}", config_path.display());

    let (shutdown_tx, _) = watch::channel(false);
    let _signal_handler = AbortOnDropHandle::new(install_signal_handler(shutdown_tx.clone()));

    let capabilities = resolve_capabilities(&ipc, &config).await;

    let (chat_tx, chat_rx) = mpsc::channel::<ChatEvent>(CHAT_CHANNEL_CAPACITY);

    let bus_attached = Arc::new(Notify::new());
    let _bus_handle = AbortOnDropHandle::new(spawn_bus_subscription(
        BusFollower {
            ipc: ipc.clone(),
            chat_tx: chat_tx.clone(),
            attached: bus_attached.clone(),
        },
        shutdown_tx.subscribe(),
        !daemon_reachable,
    ));

    let (focus_tx, focus_rx) = watch::channel(true);
    let focus_reporter = AbortOnDropHandle::new(focus::spawn_reporter(
        ipc.clone(),
        focus_rx,
        bus_attached,
        shutdown_tx.subscribe(),
    ));

    let run_result = run_tui(TuiContext {
        ipc: ipc.clone(),
        focus_tx,
        chat_tx,
        chat_rx,
        shutdown_rx: shutdown_tx.subscribe(),
        capabilities,
        daemon_reachable,
        config_path,
        theme: Theme::new(&config.chat.theme),
    })
    .await;

    let _ = shutdown_tx.send(true);
    drop(focus_reporter);
    focus::report_closed(&ipc).await;
    info!("assistd chat stopped");
    run_result
}

/// The daemon's capabilities, with the configured model's basename as the display name when
/// the daemon gives none.
async fn resolve_capabilities(ipc: &IpcClient, config: &Config) -> StartupCapabilities {
    let mut capabilities = get_capabilities(ipc).await.unwrap_or_else(|e| {
        info!("get_capabilities failed: {e:#}");
        StartupCapabilities::default()
    });
    capabilities.mcp_unavailable = get_mcp_unavailable(ipc).await.unwrap_or_else(|e| {
        info!("mcp_status failed: {e:#}");
        Vec::new()
    });
    if capabilities.model_name.is_empty() {
        capabilities.model_name = config
            .model
            .name
            .rsplit_once('/')
            .map_or_else(|| config.model.name.clone(), |(_, rest)| rest.to_string());
    }
    capabilities
}

async fn run_tui(ctx: TuiContext) -> Result<()> {
    let TuiContext {
        ipc,
        focus_tx,
        chat_tx,
        mut chat_rx,
        mut shutdown_rx,
        capabilities,
        daemon_reachable,
        config_path,
        theme,
    } = ctx;

    let _guard = TerminalGuard::enter()?;
    let picker = probe_graphics();
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend).context("Terminal::new")?;

    let mut app = App::new(
        ipc,
        chat_tx,
        capabilities.model_name,
        capabilities.vision_enabled,
        capabilities.auto_mode,
        picker,
        config_path,
    )
    .with_theme(theme)
    .with_mcp_unavailable(capabilities.mcp_unavailable)
    .with_daemon_reachable(daemon_reachable);
    if daemon_reachable {
        app.spawn_resume_or_new();
    }

    let mut events = EventStream::new().filter(|ev| future::ready(is_acted_on(ev)));
    let mut tick = tokio::time::interval(TICK_INTERVAL);
    let mut glyph_tick = tokio::time::interval(GLYPH_INTERVAL);
    glyph_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut scene_tick = tokio::time::interval(SCENE_INTERVAL);
    scene_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut window_title = String::new();
    let mut redraw = true;
    let mut frame_due = Instant::now();

    while !app.should_quit() {
        tokio::select! {
            maybe_ev = events.next() => {
                if apply_terminal_event(&mut app, &focus_tx, maybe_ev).is_break() {
                    break;
                }
                redraw = true;
            }
            Some(ev) = chat_rx.recv() => {
                app.on_chat_event(ev);
                redraw = true;
            }
            _ = tick.tick() => {
                redraw |= app.on_tick().context("a chat task panicked")?;
            }
            _ = glyph_tick.tick(), if app.glyph_live() => {
                app.advance_live_glyphs();
                redraw = true;
            }
            _ = scene_tick.tick(), if app.wake_word_scene_live() => {
                app.advance_wake_word_scene();
                redraw = true;
            }
            () = tokio::time::sleep_until(frame_due), if redraw => {
                sync_window_title(&mut window_title, app.session_title.as_deref())?;
                terminal.draw(|f| ui::render(f, &mut app))?;
                redraw = false;
                frame_due = Instant::now() + FRAME_INTERVAL;
            }
            _ = shutdown_rx.changed() => {
                break;
            }
        }
    }

    drop(terminal);
    Ok(())
}

/// Retitle the terminal window after the open session when its title
/// differs from `shown`, the title last set. Control characters are dropped
/// so a model-written title cannot end the escape sequence early.
fn sync_window_title(shown: &mut String, session_title: Option<&str>) -> io::Result<()> {
    let wanted = session_title.map_or_else(
        || WINDOW_TITLE.to_string(),
        |title| {
            let printable: String = title.chars().filter(|c| !c.is_control()).collect();
            format!("{WINDOW_TITLE} — {printable}")
        },
    );
    if *shown != wanted {
        execute!(io::stdout(), terminal::SetTitle(&wanted))?;
        *shown = wanted;
    }
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

/// Whether the chat acts on `ev`. Mouse capture also reports pointer motion, clicks and
/// drags, which nothing reads.
fn is_acted_on(ev: &io::Result<TermEvent>) -> bool {
    match ev {
        Ok(TermEvent::Mouse(mouse)) => matches!(
            mouse.kind,
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
        ),
        _ => true,
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
                TermEvent::Paste(text) => app.on_paste(&text),
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

/// Each configured MCP server that is not usable, with why.
async fn get_mcp_unavailable(ipc: &IpcClient) -> Result<Vec<(String, String)>> {
    let req = Request::McpStatus {
        id: Uuid::new_v4().to_string(),
    };
    let mut stream = ipc.one_shot(req).await?;
    let mut unavailable = Vec::new();
    loop {
        match stream.next_event().await? {
            Some(Event::McpServer { server, state, .. }) => {
                if let Some(reason) = state.unavailable_reason(&server) {
                    unavailable.push((server, reason));
                }
            }
            Some(Event::Done { .. }) => return Ok(unavailable),
            Some(Event::Error { message, .. }) => anyhow::bail!("{message}"),
            Some(_) => {}
            None => anyhow::bail!("daemon closed without responding"),
        }
    }
}

async fn get_capabilities(ipc: &IpcClient) -> Result<StartupCapabilities> {
    let req = Request::GetCapabilities {
        id: Uuid::new_v4().to_string(),
    };
    let mut stream = ipc.one_shot(req).await?;
    let mut capabilities = StartupCapabilities::default();
    loop {
        match stream.next_event().await? {
            Some(Event::Capabilities {
                vision,
                model_name,
                auto_mode,
                ..
            }) => {
                capabilities = StartupCapabilities {
                    vision_enabled: vision,
                    model_name,
                    auto_mode,
                    ..StartupCapabilities::default()
                };
            }
            Some(Event::Done { .. }) => return Ok(capabilities),
            Some(Event::Error { message, .. }) => anyhow::bail!("{message}"),
            Some(_) => {}
            None => anyhow::bail!("daemon closed without responding"),
        }
    }
}

/// What the bus subscription task reports to.
struct BusFollower {
    ipc: Arc<IpcClient>,
    chat_tx: mpsc::Sender<ChatEvent>,
    /// Notified on every successful subscribe.
    attached: Arc<Notify>,
}

/// Follow the daemon's broadcast bus for the daemon's state, session titles and turns this
/// chat did not start (push-to-talk, continuous listening, other clients).
/// Reconnects on a fixed delay, reporting the loss and the reconnect to the app;
/// `start_lost` reports the first subscribe as a reconnect too.
fn spawn_bus_subscription(
    follower: BusFollower,
    mut shutdown: watch::Receiver<bool>,
    start_lost: bool,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lost = start_lost;
        loop {
            tokio::select! {
                _ = shutdown.changed() => break,
                () = pump_bus(&follower, &mut lost) => {
                    tokio::select! {
                        _ = shutdown.changed() => break,
                        () = tokio::time::sleep(BUS_RECONNECT_DELAY) => {}
                    }
                }
            }
        }
    })
}

/// Forward bus events until the subscription ends. `lost` says a previous
/// subscription ended, so a successful subscribe sends `BusRestored`.
async fn pump_bus(follower: &BusFollower, lost: &mut bool) {
    let BusFollower {
        ipc,
        chat_tx,
        attached,
    } = follower;
    let Some(mut stream) = subscribe_bus(ipc).await else {
        return;
    };
    attached.notify_one();
    if std::mem::take(lost) && chat_tx.send(ChatEvent::BusRestored).await.is_err() {
        return;
    }
    while let Ok(Some(ev)) = stream.next_event().await {
        if chat_tx.send(ChatEvent::Bus(ev)).await.is_err() {
            return;
        }
    }
    *lost = true;
    let _ = chat_tx.send(ChatEvent::BusLost).await;
}

async fn subscribe_bus(ipc: &IpcClient) -> Option<assistd_ipc::EventStream> {
    let req = Request::Subscribe {
        id: Uuid::new_v4().to_string(),
        filter: SubscribeFilter {
            kinds: vec![
                EventKind::SessionTitle,
                EventKind::Transcription,
                EventKind::Presence,
                EventKind::VoiceState,
                EventKind::ListenState,
                EventKind::WakeWordDetected,
                EventKind::Delta,
                EventKind::ReasoningDelta,
                EventKind::ToolCall,
                EventKind::ToolResult,
                EventKind::Relay,
                EventKind::ContextSize,
                EventKind::TaskQueue,
                EventKind::Readiness,
                EventKind::Done,
                EventKind::Error,
            ],
        },
    };
    ipc.one_shot(req)
        .await
        .inspect_err(|e| tracing::debug!("bus subscribe failed: {e}"))
        .ok()
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

/// The owner-only chat log directory under the XDG state dir, else the
/// runtime dir. Fails when neither is known or the directory is not a real
/// directory owned by this user.
fn log_dir() -> Result<PathBuf> {
    let dir = dirs::state_dir()
        .or_else(dirs::runtime_dir)
        .context("no XDG state or runtime directory for chat logs")?
        .join("assistd");
    DirBuilder::new()
        .recursive(true)
        .mode(LOG_DIR_MODE)
        .create(&dir)
        .with_context(|| format!("creating log dir {}", dir.display()))?;
    restrict_log_dir(&dir).with_context(|| format!("securing log dir {}", dir.display()))?;
    Ok(dir)
}

fn restrict_log_dir(dir: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(dir)?;
    anyhow::ensure!(metadata.is_dir(), "not a directory");
    anyhow::ensure!(
        metadata.uid() == rustix::process::geteuid().as_raw(),
        "owned by another user"
    );
    if metadata.mode() & 0o077 != 0 {
        std::fs::set_permissions(dir, Permissions::from_mode(LOG_DIR_MODE))?;
    }
    Ok(())
}

fn open_private_log(path: &Path) -> io::Result<OwnedFd> {
    let flags =
        OFlags::WRONLY | OFlags::CREATE | OFlags::APPEND | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let fd = rustix::fs::open(path, flags, Mode::from_raw_mode(LOG_FILE_MODE))?;
    rustix::fs::fchmod(&fd, Mode::from_raw_mode(LOG_FILE_MODE))?;
    Ok(fd)
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

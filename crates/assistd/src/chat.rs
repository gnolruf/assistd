//! `assistd chat`: a terminal window onto the running daemon, which owns every service.
//! One chat may be open per user, since every chat drives the daemon's one session.

use std::fs::{DirBuilder, File, Permissions, TryLockError};
use std::future;
use std::io::{self, Write};
use std::ops::ControlFlow;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_config::ConfigIssue;
use assistd_core::Config;
use assistd_core::socket::open_private_lock_file;
use assistd_ipc::{
    EventBody, EventKind, IpcClient, IpcClientError, Request, RequestBody, SubscribeFilter,
};
use clap::Args;
use crossterm::event::{self, Event as TermEvent, EventStream, MouseEventKind};
use crossterm::{cursor, execute, queue, style, terminal};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui_image::picker::{Picker, ProtocolType};
use rustix::fs::{Mode, OFlags};
use tokio::net::UnixStream;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::{Notify, mpsc, watch};
use tokio::task::JoinSet;
use tokio::time::{Instant, MissedTickBehavior};
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
mod selection;
pub(crate) mod settings;
mod tasks;
mod text;
mod text_editor;
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
    config_issues: Vec<ConfigIssue>,
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

/// Which input the terminal reports under a [`TerminalGuard`].
#[derive(Clone, Copy)]
enum TerminalInput {
    /// Keys and pastes; the mouse stays with the terminal, which keeps its own text selection.
    Keyboard,
    /// Keys, pastes, mouse events and focus changes.
    KeyboardMouseFocus,
}

/// Restores the terminal on drop.
struct TerminalGuard {
    input: TerminalInput,
}

impl TerminalGuard {
    /// Enter raw mode and the alternate screen with modifier-aware key reports
    /// and bracketed paste, saving the window title, and install a panic hook that restores the
    /// terminal before the previous hook runs.
    fn enter(input: TerminalInput) -> Result<Self> {
        terminal::enable_raw_mode().context("enable_raw_mode")?;
        if let Err(e) = Self::enable_reports(input) {
            let _ = terminal::disable_raw_mode();
            return Err(e).context("EnterAlternateScreen");
        }
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = Self::cleanup(input);
            previous_hook(info);
        }));
        Ok(Self { input })
    }

    fn enable_reports(input: TerminalInput) -> io::Result<()> {
        let mut stdout = io::stdout();
        queue!(
            stdout,
            terminal::EnterAlternateScreen,
            style::Print(PUSH_WINDOW_TITLE),
            event::EnableBracketedPaste,
            event::PushKeyboardEnhancementFlags(
                event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
        )?;
        if let TerminalInput::KeyboardMouseFocus = input {
            queue!(stdout, event::EnableMouseCapture, event::EnableFocusChange)?;
        }
        stdout.flush()
    }

    fn cleanup(input: TerminalInput) -> io::Result<()> {
        terminal::disable_raw_mode()?;
        let mut stdout = io::stdout();
        if let TerminalInput::KeyboardMouseFocus = input {
            queue!(
                stdout,
                event::DisableFocusChange,
                event::DisableMouseCapture
            )?;
        }
        queue!(
            stdout,
            event::PopKeyboardEnhancementFlags,
            event::DisableBracketedPaste,
            style::Print(POP_WINDOW_TITLE),
            terminal::LeaveAlternateScreen,
            cursor::Show,
        )?;
        stdout.flush()
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = Self::cleanup(self.input);
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

/// Run the TUI against the daemon, showing it offline until its socket accepts, until it
/// quits or SIGINT or SIGTERM arrives. Fails before touching the terminal when another chat
/// is open.
pub(crate) async fn run(args: ChatArgs) -> Result<()> {
    let ipc = Arc::new(IpcClient::new());
    let daemon_reachable = UnixStream::connect(ipc.socket_path()).await.is_ok();
    let _instance_lock = InstanceLock::acquire(ipc.socket_path())?;

    let _stderr_redirect = StderrRedirect::to_log()?;

    let _log_guard = init_file_tracing()?;

    let config_path = Config::path_or_default(args.config)?;
    let (config, config_issues) = Config::load_with_issues(&config_path)?;

    info!("assistd chat v{}", crate::VERSION);
    info!("loaded config from {}", config_path.display());

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let mut background = JoinSet::new();
    let signalled = shutdown_signal().context("installing signal handlers")?;
    let signal_shutdown_tx = shutdown_tx.clone();
    background.spawn(async move {
        signalled.await;
        let _ = signal_shutdown_tx.send(true);
    });

    let capabilities = tokio::select! {
        resolved = resolve_capabilities(&ipc, &config) => resolved?,
        _ = shutdown_rx.wait_for(|&stopping| stopping) => return Ok(()),
    };

    let (chat_tx, chat_rx) = mpsc::channel::<ChatEvent>(CHAT_CHANNEL_CAPACITY);

    let bus_attached = Arc::new(Notify::new());
    background.spawn(follow_bus(
        BusFollower {
            ipc: ipc.clone(),
            chat_tx: chat_tx.clone(),
            attached: bus_attached.clone(),
        },
        shutdown_tx.subscribe(),
        !daemon_reachable,
    ));

    let (focus_tx, focus_rx) = watch::channel(true);
    background.spawn(focus::report_changes(
        ipc.clone(),
        focus_rx,
        bus_attached,
        shutdown_tx.subscribe(),
    ));

    let run_result = run_tui(
        TuiContext {
            ipc: ipc.clone(),
            focus_tx,
            chat_tx,
            chat_rx,
            shutdown_rx,
            capabilities,
            daemon_reachable,
            config_path,
            config_issues,
            theme: Theme::new(&config.chat.theme),
        },
        &mut background,
    )
    .await;

    let _ = shutdown_tx.send(true);
    background.shutdown().await;
    focus::report_closed(&ipc).await;
    info!("assistd chat stopped");
    run_result
}

/// The daemon's capabilities, with the configured model's basename as the display name when
/// the daemon gives none. Fails only when the daemon is from another version.
async fn resolve_capabilities(ipc: &IpcClient, config: &Config) -> Result<StartupCapabilities> {
    let mut capabilities = match get_capabilities(ipc).await {
        Ok(capabilities) => capabilities,
        Err(e @ IpcClientError::VersionMismatch { .. }) => return Err(e.into()),
        Err(e) => {
            info!("get_capabilities failed: {e:#}");
            StartupCapabilities::default()
        }
    };
    capabilities.mcp_unavailable = get_mcp_unavailable(ipc).await.unwrap_or_else(|e| {
        info!("mcp_status failed: {e:#}");
        Vec::new()
    });
    if capabilities.model_name.is_empty() {
        let configured = config.model.name.as_str();
        capabilities.model_name = configured
            .rsplit_once('/')
            .map_or(configured, |(_, rest)| rest)
            .to_string();
    }
    Ok(capabilities)
}

/// Draw the chat until it quits, a signal arrives or the terminal closes. Fails when a task
/// in `background` panics.
async fn run_tui(ctx: TuiContext, background: &mut JoinSet<()>) -> Result<()> {
    let TuiContext {
        ipc,
        focus_tx,
        chat_tx,
        mut chat_rx,
        mut shutdown_rx,
        capabilities,
        daemon_reachable,
        config_path,
        config_issues,
        theme,
    } = ctx;

    let _guard = TerminalGuard::enter(TerminalInput::KeyboardMouseFocus)?;
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
    .with_config_issues(&config_issues)
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
            _ = shutdown_rx.wait_for(|&stopping| stopping) => {
                break;
            }
            Some(finished) = background.join_next() => {
                finished.context("a chat task panicked")?;
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
    let req = Request::new(Uuid::new_v4().to_string(), RequestBody::McpStatus);
    let mut unavailable = Vec::new();
    ipc.request(req, |event| {
        if let EventBody::McpServer { server, state, .. } = &event.body
            && let Some(reason) = state.unavailable_reason(server)
        {
            unavailable.push((server.clone(), reason));
        }
        Ok::<(), IpcClientError>(())
    })
    .await?;
    Ok(unavailable)
}

async fn get_capabilities(ipc: &IpcClient) -> Result<StartupCapabilities, IpcClientError> {
    let req = Request::new(Uuid::new_v4().to_string(), RequestBody::GetCapabilities);
    let mut capabilities = StartupCapabilities::default();
    ipc.request(req, |event| {
        if let EventBody::Capabilities {
            vision,
            model_name,
            auto_mode,
        } = &event.body
        {
            capabilities = StartupCapabilities {
                vision_enabled: *vision,
                model_name: model_name.clone(),
                auto_mode: *auto_mode,
                ..StartupCapabilities::default()
            };
        }
        Ok::<(), IpcClientError>(())
    })
    .await?;
    Ok(capabilities)
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
/// `start_lost` reports the first subscribe as a reconnect too. Returns when `shutdown` flips.
async fn follow_bus(follower: BusFollower, mut shutdown: watch::Receiver<bool>, start_lost: bool) {
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
    let req = Request::new(
        Uuid::new_v4().to_string(),
        RequestBody::Subscribe {
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
        },
    );
    ipc.one_shot(req)
        .await
        .inspect_err(|e| tracing::debug!("bus subscribe failed: {e}"))
        .ok()
}

/// Install the SIGINT and SIGTERM handlers; the returned future resolves on the first of
/// either signal.
fn shutdown_signal() -> io::Result<impl Future<Output = ()>> {
    let mut interrupt = signal(SignalKind::interrupt())?;
    let mut terminate = signal(SignalKind::terminate())?;
    Ok(async move {
        tokio::select! {
            _ = interrupt.recv() => info!("received SIGINT"),
            _ = terminate.recv() => info!("received SIGTERM"),
        }
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
        .with_env_filter(crate::env_filter_or("info"))
        .init();

    Ok(guard)
}

#[cfg(test)]
mod tests;

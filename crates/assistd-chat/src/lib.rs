//! `assistd chat`: a terminal window onto the running daemon, which owns every service.
//! One chat may be open per user, since every chat drives the daemon's one session.

use std::fs::{File, TryLockError};
use std::future;
use std::io;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_config::{Config, ConfigIssue};
use assistd_ipc::socket::open_private_lock_file;
use assistd_ipc::{Event, EventBody, IpcClient, IpcClientError, Request, RequestBody};
use clap::Args;
use crossterm::event::{Event as TermEvent, EventStream, MouseEventKind};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::net::UnixStream;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::{Notify, mpsc, watch};
use tokio::task::JoinSet;
use tokio::time::{Instant, MissedTickBehavior};
use tracing::info;
use uuid::Uuid;

use self::app::{App, ChatEvent, request_capabilities};
use self::bus::{BusFollower, follow_bus};
use self::logging::{StderrRedirect, init_file_tracing};
use self::terminal::{TerminalGuard, TerminalInput, probe_graphics, sync_window_title};
use self::theme::Theme;

mod app;
mod bus;
mod focus;
mod input;
mod logging;
mod markdown;
mod output;
mod selection;
pub mod settings;
mod tasks;
mod terminal;
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
const INSTANCE_LOCK_FILE: &str = "chat.lock";

#[derive(Debug, Args)]
pub struct ChatArgs {
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

/// Run the TUI against the daemon, showing it offline until its socket accepts, until it
/// quits or SIGINT or SIGTERM arrives. Fails before touching the terminal when another chat
/// is open.
pub async fn run(args: ChatArgs) -> Result<()> {
    let ipc = Arc::new(IpcClient::new());
    let daemon_reachable = UnixStream::connect(ipc.socket_path()).await.is_ok();
    let _instance_lock = InstanceLock::acquire(ipc.socket_path())?;

    let _stderr_redirect = StderrRedirect::to_log()?;

    let _log_guard = init_file_tracing()?;

    let config_path = Config::path_or_default(args.config)?;
    let (config, config_issues) = Config::load_with_issues(&config_path)?;

    info!("assistd chat v{}", assistd_ipc::VERSION);
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
    let Some(Event {
        body:
            EventBody::Capabilities {
                vision,
                model_name,
                auto_mode,
            },
        ..
    }) = request_capabilities(ipc).await?
    else {
        return Ok(StartupCapabilities::default());
    };
    Ok(StartupCapabilities {
        vision_enabled: vision,
        model_name,
        auto_mode,
        ..StartupCapabilities::default()
    })
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

#[cfg(test)]
mod tests;

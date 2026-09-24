//! Connection machinery shared by the i3 and Sway backends: a command
//! socket guarded by a per-call timeout, an event socket that keeps the
//! focus snapshot current, and a supervisor that reconnects with backoff
//! when a socket drops. [`IpcProtocol`] captures what differs between
//! the two IPC client crates.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{Mutex, Notify, RwLock, broadcast, watch};
use tokio::task::JoinHandle;

use crate::criteria::format_place_floating_pixels;
use crate::snapshot::{self, Snapshot, WindowChangeKind};
use crate::{
    FocusedWindowContext, PlacementAnchor, PlacementCriteria, Rect, TransportError, WM_IPC_TIMEOUT,
    Window, WindowEvent, WindowId, WmError, WmResult, WorkspaceInfo,
};

/// Window events buffered per subscriber. A lagged subscriber falls
/// back to polling, so a small buffer is fine.
const WINDOW_EVENTS_CAPACITY: usize = 32;

/// How long `find_window_rect_by_criteria` waits for a matching
/// `WindowEvent::Opened` after its first tree poll misses.
const WINDOW_EVENT_WAIT: Duration = Duration::from_millis(500);

/// Labels for the IPC calls the shared machinery makes, reported as the
/// `op` of [`WmError::Ipc`].
pub(crate) struct OpLabels {
    pub run_command: &'static str,
    pub get_tree: &'static str,
    pub get_tree_window_rect: &'static str,
    pub get_workspaces: &'static str,
    pub get_workspaces_focused_rect: &'static str,
}

/// The focus-relevant identity of a tree node.
#[derive(Default)]
pub(crate) struct NodeIdentity {
    pub id: Option<WindowId>,
    pub class: Option<String>,
    pub title: Option<String>,
}

/// A compositor event projected onto what the shared machinery acts on.
pub(crate) enum IpcEvent {
    /// `focus` updates the snapshot; `event` is broadcast to subscribers.
    Window {
        focus: Option<(WindowChangeKind, NodeIdentity)>,
        event: Option<WindowEvent>,
    },
    WorkspaceFocused(Option<String>),
    Ignored,
}

/// One workspace from a `GET_WORKSPACES` reply.
pub(crate) struct Workspace {
    pub info: WorkspaceInfo,
    pub rect: Rect,
}

/// What differs between the i3 and Sway IPC client crates.
pub(crate) trait IpcProtocol: Send + Sync + Sized + 'static {
    type Cmd: Send + 'static;
    type Events: Send + 'static;
    type Node: Send;

    /// Compositor name used in log messages.
    const NAME: &'static str;
    const OPS: OpLabels;

    /// Open a command connection and an event stream subscribed to
    /// window and workspace events.
    fn connect(&self) -> impl Future<Output = WmResult<(Self::Cmd, Self::Events)>> + Send;

    /// `None` when the stream has ended. Must be cancel-safe.
    fn next_event(
        events: &mut Self::Events,
    ) -> impl Future<Output = Option<Result<IpcEvent, TransportError>>> + Send;

    /// The inner error is the compositor's message for the first
    /// command it rejected.
    fn run_command(
        cmd: &mut Self::Cmd,
        payload: &str,
    ) -> impl Future<Output = Result<Result<(), String>, TransportError>> + Send;

    fn get_tree(
        cmd: &mut Self::Cmd,
    ) -> impl Future<Output = Result<Self::Node, TransportError>> + Send;

    fn get_workspaces(
        cmd: &mut Self::Cmd,
    ) -> impl Future<Output = Result<Vec<Workspace>, TransportError>> + Send;

    /// Tiled children, then floating children.
    fn children(node: &Self::Node) -> impl Iterator<Item = &Self::Node>;

    fn is_focused(node: &Self::Node) -> bool;

    fn identity(node: &Self::Node) -> NodeIdentity;

    /// `node`'s rect when it is a window matching `criteria`; does not
    /// descend into children.
    fn window_rect(node: &Self::Node, criteria: &PlacementCriteria) -> Option<Rect>;

    fn collect_windows(tree: &Self::Node) -> Vec<Window>;
}

/// State shared between a backend's IPC calls and its supervisor task.
pub(crate) struct IpcBackend<P: IpcProtocol> {
    protocol: P,
    cmd: Mutex<Option<P::Cmd>>,
    /// Mirrors whether `cmd` holds a connection, readable without the lock.
    connected: AtomicBool,
    snapshot: RwLock<Snapshot>,
    reconnect: Notify,
    window_events: broadcast::Sender<WindowEvent>,
}

impl<P: IpcProtocol> IpcBackend<P> {
    /// Connect, seed the focus snapshot, and spawn the supervisor that
    /// drives events and reconnects on socket drops. Errors only when
    /// the initial connect fails.
    pub(crate) async fn start(
        protocol: P,
        shutdown: watch::Receiver<bool>,
    ) -> WmResult<(Arc<Self>, JoinHandle<()>)> {
        let (mut cmd, events) = protocol.connect().await?;
        let initial = seed_snapshot::<P>(&mut cmd).await.unwrap_or_else(|e| {
            tracing::warn!(
                "{} seed_snapshot failed: {e:#}; starting with empty focus state",
                P::NAME
            );
            Snapshot::default()
        });
        let (window_events, _) = broadcast::channel(WINDOW_EVENTS_CAPACITY);
        let backend = Arc::new(Self {
            protocol,
            cmd: Mutex::new(Some(cmd)),
            connected: AtomicBool::new(true),
            snapshot: RwLock::new(initial),
            reconnect: Notify::new(),
            window_events,
        });
        let supervisor_task = tokio::spawn(supervise(backend.clone(), events, shutdown));
        Ok((backend, supervisor_task))
    }

    /// Whether the command socket is connected. Never blocks.
    pub(crate) fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    fn set_conn(&self, slot: &mut Option<P::Cmd>, conn: Option<P::Cmd>) {
        self.connected.store(conn.is_some(), Ordering::Relaxed);
        *slot = conn;
    }

    /// Run one IPC call on the command socket under [`WM_IPC_TIMEOUT`].
    /// A timeout or transport error drops the connection and wakes the
    /// supervisor to reconnect.
    pub(crate) async fn with_conn<T, E: Into<TransportError>>(
        &self,
        ctx: &'static str,
        op: impl AsyncFnOnce(&mut P::Cmd) -> Result<T, E>,
    ) -> WmResult<T> {
        let mut guard = self.cmd.lock().await;
        let conn = guard.as_mut().ok_or(WmError::Disconnected)?;
        let outcome = tokio::time::timeout(WM_IPC_TIMEOUT, op(conn)).await;
        let err = match outcome {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(e)) => WmError::ipc(ctx, e),
            Err(_) => WmError::Timeout(WM_IPC_TIMEOUT),
        };
        self.set_conn(&mut guard, None);
        self.reconnect.notify_one();
        Err(err)
    }

    pub(crate) async fn run_command(&self, payload: &str) -> WmResult<()> {
        self.with_conn(P::OPS.run_command, async |conn| {
            P::run_command(conn, payload).await
        })
        .await?
        .map_err(|e| WmError::Rejected(format!("{payload}: {e}")))
    }

    pub(crate) async fn workspaces(&self, ctx: &'static str) -> WmResult<Vec<Workspace>> {
        self.with_conn(ctx, async |conn| P::get_workspaces(conn).await)
            .await
    }

    async fn tree(&self, ctx: &'static str) -> WmResult<P::Node> {
        self.with_conn(ctx, async |conn| P::get_tree(conn).await)
            .await
    }

    pub(crate) async fn focused_window(&self) -> Option<WindowId> {
        snapshot::read_focused_id(&self.snapshot).await
    }

    pub(crate) async fn focused_context(&self) -> Option<FocusedWindowContext> {
        snapshot::read_focused_context(&self.snapshot).await
    }

    pub(crate) async fn list_windows(&self) -> WmResult<Vec<Window>> {
        let tree = self.tree(P::OPS.get_tree).await?;
        Ok(P::collect_windows(&tree))
    }

    pub(crate) async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        Ok(self
            .workspaces(P::OPS.get_workspaces)
            .await?
            .into_iter()
            .map(|w| w.info)
            .collect())
    }

    pub(crate) async fn focused_workspace_rect(&self) -> WmResult<Rect> {
        self.workspaces(P::OPS.get_workspaces_focused_rect)
            .await?
            .into_iter()
            .find(|w| w.info.focused)
            .map(|w| w.rect)
            .ok_or_else(|| WmError::Rejected("no focused workspace".into()))
    }

    /// Float and place the window matching `criteria`, sized from its
    /// actual rect when it can be found: DPI scaling can map a
    /// 360-logical-px request to 420 physical px.
    pub(crate) async fn place_floating(
        &self,
        criteria: &PlacementCriteria,
        anchor: PlacementAnchor,
    ) -> WmResult<()> {
        let workspace = self.focused_workspace_rect().await?;
        let effective = match self.find_window_rect_by_criteria(criteria).await {
            Ok(actual) => {
                tracing::info!(
                    target: "tray",
                    "popup: actual window rect = {}x{} (configured {}x{}); placing accordingly",
                    actual.width, actual.height, anchor.width, anchor.height
                );
                PlacementAnchor {
                    width: actual.width,
                    height: actual.height,
                    ..anchor
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "tray",
                    "popup: could not query window rect ({e}); falling back to configured size"
                );
                anchor
            }
        };
        self.run_command(&format_place_floating_pixels(
            criteria, effective, workspace,
        ))
        .await
    }

    /// Current rect of the window matching `criteria`. Subscribes to
    /// window events before the first tree poll so a `window::new`
    /// that lands between the poll and the wait is not missed.
    async fn find_window_rect_by_criteria(&self, criteria: &PlacementCriteria) -> WmResult<Rect> {
        let mut events = self.window_events.subscribe();

        if let Ok(rect) = self.find_window_rect_once(criteria).await {
            return Ok(rect);
        }

        let waited = tokio::time::timeout(WINDOW_EVENT_WAIT, async {
            loop {
                match events.recv().await {
                    Ok(ev) => {
                        if ev.matches_opened(criteria).is_some() {
                            return true;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => return false,
                    Err(broadcast::error::RecvError::Lagged(_)) => return false,
                }
            }
        })
        .await
        .unwrap_or(false);

        let result = self.find_window_rect_once(criteria).await;
        if waited && result.is_ok() {
            tracing::debug!(
                target: "tray",
                "popup: window appeared via {} window::new event",
                P::NAME
            );
        }
        result
    }

    async fn find_window_rect_once(&self, criteria: &PlacementCriteria) -> WmResult<Rect> {
        let tree = self.tree(P::OPS.get_tree_window_rect).await?;
        find_map_node::<P, _>(&tree, &|node| P::window_rect(node, criteria))
            .ok_or_else(|| WmError::Rejected(format!("no window matches {criteria:?}")))
    }

    /// Drive one event stream. Returns `true` when the caller should
    /// reconnect, `false` on shutdown.
    async fn drive_events(
        &self,
        mut events: P::Events,
        shutdown: &mut watch::Receiver<bool>,
    ) -> bool {
        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() { return false; }
                }
                _ = self.reconnect.notified() => {
                    return true;
                }
                evt = P::next_event(&mut events) => {
                    match evt {
                        Some(Ok(event)) => self.apply_event(event).await,
                        Some(Err(e)) => {
                            tracing::warn!("{} event stream error: {e}", P::NAME);
                            return true;
                        }
                        None => return true,
                    }
                }
            }
        }
    }

    async fn apply_event(&self, event: IpcEvent) {
        match event {
            IpcEvent::Window { focus, event } => {
                if let Some((kind, NodeIdentity { id, class, title })) = focus {
                    snapshot::apply_window_event(&self.snapshot, kind, id, class, title).await;
                }
                if let Some(ev) = event {
                    let _ = self.window_events.send(ev);
                }
            }
            IpcEvent::WorkspaceFocused(name) => {
                snapshot::apply_workspace_focus(&self.snapshot, name).await;
            }
            IpcEvent::Ignored => {}
        }
    }
}

async fn supervise<P: IpcProtocol>(
    backend: Arc<IpcBackend<P>>,
    initial_events: P::Events,
    mut shutdown: watch::Receiver<bool>,
) {
    let name = P::NAME;
    if !backend.drive_events(initial_events, &mut shutdown).await {
        tracing::info!("{name} supervisor exited (shutdown during initial events stream)");
        return;
    }

    let mut attempt: u32 = 0;
    loop {
        backend.set_conn(&mut *backend.cmd.lock().await, None);
        tracing::warn!(
            "{name} disconnected; reconnecting (attempt {})",
            attempt + 1
        );

        let delay = crate::backoff::backoff_delay(attempt);
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    tracing::info!("{name} supervisor exited (shutdown during backoff)");
                    return;
                }
            }
            _ = tokio::time::sleep(delay) => {}
        }

        match backend.protocol.connect().await {
            Ok((mut cmd, events)) => {
                if let Ok(s) = seed_snapshot::<P>(&mut cmd).await {
                    *backend.snapshot.write().await = s;
                }
                backend.set_conn(&mut *backend.cmd.lock().await, Some(cmd));
                attempt = 0;
                tracing::info!("{name} backend reconnected");
                if !backend.drive_events(events, &mut shutdown).await {
                    tracing::info!("{name} supervisor exited (shutdown during events stream)");
                    return;
                }
            }
            Err(e) => {
                tracing::warn!("{name} reconnect failed: {e}; will retry after backoff");
                attempt = attempt.saturating_add(1);
            }
        }
    }
}

async fn seed_snapshot<P: IpcProtocol>(cmd: &mut P::Cmd) -> WmResult<Snapshot> {
    let tree = P::get_tree(cmd)
        .await
        .map_err(|e| WmError::ipc(P::OPS.get_tree, e))?;
    let NodeIdentity { id, class, title } = find_map_node::<P, _>(&tree, &|node| {
        P::is_focused(node).then(|| P::identity(node))
    })
    .unwrap_or_default();

    let active_workspace = match P::get_workspaces(cmd).await {
        Ok(ws) => ws.into_iter().find(|w| w.info.focused).map(|w| w.info.name),
        Err(e) => {
            tracing::warn!("{} GET_WORKSPACES on seed failed: {e:#}", P::NAME);
            None
        }
    };
    Ok(Snapshot {
        focused_id: id,
        focused_class: class,
        focused_title: title,
        active_workspace,
    })
}

/// Pre-order search of the tree for the first node `f` maps to `Some`.
fn find_map_node<P: IpcProtocol, T>(
    node: &P::Node,
    f: &impl Fn(&P::Node) -> Option<T>,
) -> Option<T> {
    f(node).or_else(|| P::children(node).find_map(|child| find_map_node::<P, T>(child, f)))
}

#[cfg(test)]
mod tests;

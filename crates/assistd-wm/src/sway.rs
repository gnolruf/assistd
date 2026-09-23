//! Sway backend for [`crate::WindowManager`], over `swayipc-async`.
//! Same shape as the i3 backend: one command socket, one event socket.
//! `swayipc-async` runs on `async-io`, which costs one extra reactor
//! thread alongside tokio. Views carry either `app_id` (Wayland-native)
//! or `window_properties.class` (XWayland); whichever is present is
//! surfaced as the window's app.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use swayipc_async::{
    Connection, Event, EventStream, EventType, Node, NodeType, WindowChange, WorkspaceChange,
};
use tokio::sync::{Mutex, RwLock, broadcast, watch};
use tokio::task::JoinHandle;

use crate::criteria::{
    format_focus, format_layout, format_move_to_workspace, format_place_floating_pixels,
    format_resize_width,
};
use crate::snapshot::{
    self, Snapshot, WindowChangeKind, apply_window_event, apply_workspace_focus,
};
use crate::{
    FocusedWindowContext, Layout, OutputInfo, PlacementAnchor, PlacementCriteria, Rect, ResizeDir,
    WM_IPC_TIMEOUT, Window, WindowEvent, WindowId, WindowManager, WmError, WmResult, WorkspaceId,
    WorkspaceInfo,
};

const WINDOW_EVENTS_CAPACITY: usize = 32;
const WINDOW_EVENT_WAIT: Duration = Duration::from_millis(500);

fn sway_id(raw: i64) -> Option<WindowId> {
    if raw <= 0 {
        return None;
    }
    WindowId::new(raw as u64)
}

/// [`WindowManager`] over a single Sway IPC command socket.
pub struct SwayBackend {
    cmd: Arc<Mutex<Option<Connection>>>,
    snapshot: Arc<RwLock<Snapshot>>,
    reconnect: Arc<tokio::sync::Notify>,
    window_events: broadcast::Sender<WindowEvent>,
}

/// The backend plus its supervisor task, returned by [`SwayBackend::start`].
pub struct SwayHandle {
    pub backend: Arc<SwayBackend>,
    supervisor_task: JoinHandle<()>,
}

impl SwayHandle {
    /// Awaits the supervisor task. Flip the shutdown watch first or
    /// this blocks until the socket drops.
    pub async fn shutdown(self) {
        let _ = self.supervisor_task.await;
    }
}

impl SwayBackend {
    /// Connect to the Sway IPC sockets, seed the focus snapshot, and
    /// spawn the supervisor that drives events and reconnects on
    /// socket drops. Errors only when the initial connect fails.
    pub async fn start(shutdown: watch::Receiver<bool>) -> WmResult<SwayHandle> {
        let (mut cmd, stream) = connect_pair().await?;
        let initial = match seed_snapshot(&mut cmd).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("sway seed_snapshot failed: {e:#}; starting with empty focus state");
                Snapshot::default()
            }
        };
        let snapshot = Arc::new(RwLock::new(initial));
        let reconnect = Arc::new(tokio::sync::Notify::new());
        let (window_events, _) = broadcast::channel(WINDOW_EVENTS_CAPACITY);
        let backend = Arc::new(Self {
            cmd: Arc::new(Mutex::new(Some(cmd))),
            snapshot: snapshot.clone(),
            reconnect: reconnect.clone(),
            window_events,
        });

        let supervisor_task = tokio::spawn(supervisor_loop(
            backend.clone(),
            stream,
            snapshot,
            reconnect,
            shutdown,
        ));

        Ok(SwayHandle {
            backend,
            supervisor_task,
        })
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
                "popup: window appeared via sway window::new event"
            );
        }
        result
    }

    async fn find_window_rect_once(&self, criteria: &PlacementCriteria) -> WmResult<Rect> {
        let tree = self.tree("sway GET_TREE (window rect)").await?;
        find_sway_node_rect(&tree, criteria)
            .ok_or_else(|| WmError::Rejected(format!("no window matches {criteria:?}")))
    }

    /// Run one IPC call on the command socket under [`WM_IPC_TIMEOUT`].
    /// A timeout or transport error drops the connection and wakes the
    /// supervisor to reconnect.
    async fn with_conn<T>(
        &self,
        ctx: &'static str,
        op: impl AsyncFnOnce(&mut Connection) -> swayipc_async::Fallible<T>,
    ) -> WmResult<T> {
        let mut guard = self.cmd.lock().await;
        let conn = guard.as_mut().ok_or(WmError::Disconnected)?;
        let outcome = tokio::time::timeout(WM_IPC_TIMEOUT, op(conn)).await;
        let err = match outcome {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(e)) => WmError::ipc(ctx, e),
            Err(_) => WmError::Timeout(WM_IPC_TIMEOUT),
        };
        *guard = None;
        self.reconnect.notify_one();
        Err(err)
    }

    async fn run_command(&self, payload: &str) -> WmResult<()> {
        let outcomes = self
            .with_conn("sway RUN_COMMAND", async |conn| {
                conn.run_command(payload).await
            })
            .await?;
        for r in outcomes {
            r.map_err(|e| WmError::Rejected(format!("{payload}: {e}")))?;
        }
        Ok(())
    }

    async fn workspaces(&self, ctx: &'static str) -> WmResult<Vec<swayipc_async::Workspace>> {
        self.with_conn(ctx, async |conn| conn.get_workspaces().await)
            .await
    }

    async fn outputs(&self, ctx: &'static str) -> WmResult<Vec<swayipc_async::Output>> {
        self.with_conn(ctx, async |conn| conn.get_outputs().await)
            .await
    }

    async fn tree(&self, ctx: &'static str) -> WmResult<Node> {
        self.with_conn(ctx, async |conn| conn.get_tree().await)
            .await
    }
}

#[async_trait]
impl WindowManager for SwayBackend {
    async fn focus(&self, window: &WindowId) -> WmResult<()> {
        self.run_command(&format_focus(window)).await
    }

    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        self.run_command(&format_move_to_workspace(window, workspace))
            .await
    }

    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        Ok(snapshot::read_focused_id(&self.snapshot).await)
    }

    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        Ok(snapshot::read_focused_context(&self.snapshot).await)
    }

    async fn list_windows(&self) -> WmResult<Vec<Window>> {
        let tree = self.tree("sway GET_TREE").await?;
        let mut out = Vec::new();
        collect_windows(&tree, None, &mut out);
        Ok(out)
    }

    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        Ok(self
            .workspaces("sway GET_WORKSPACES")
            .await?
            .into_iter()
            .map(|w| WorkspaceInfo {
                num: w.num,
                name: w.name,
                focused: w.focused,
                output: w.output,
            })
            .collect())
    }

    async fn resize_width(
        &self,
        window: &WindowId,
        direction: ResizeDir,
        pixels: u32,
    ) -> WmResult<()> {
        self.run_command(&format_resize_width(window, direction, pixels))
            .await
    }

    async fn set_layout(&self, layout: Layout) -> WmResult<()> {
        self.run_command(&format_layout(layout)).await
    }

    async fn focused_workspace_rect(&self) -> WmResult<Rect> {
        self.workspaces("sway GET_WORKSPACES (focused rect)")
            .await?
            .into_iter()
            .find(|w| w.focused)
            .map(|w| Rect {
                x: w.rect.x,
                y: w.rect.y,
                width: w.rect.width.max(0) as u32,
                height: w.rect.height.max(0) as u32,
            })
            .ok_or_else(|| WmError::Rejected("no focused workspace".into()))
    }

    async fn place_floating(
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

    async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
        Ok(self
            .outputs("sway GET_OUTPUTS")
            .await?
            .into_iter()
            .map(|o| OutputInfo {
                name: o.name,
                active: o.active,
                primary: o.primary,
                current_mode: o.current_mode.map(|m| {
                    (
                        m.width.max(0) as u32,
                        m.height.max(0) as u32,
                        m.refresh.max(0) as u32,
                    )
                }),
                scale: o.scale,
                focused_workspace: o.current_workspace,
            })
            .collect())
    }

    async fn focused_output_scale(&self) -> WmResult<f64> {
        Ok(self
            .outputs("sway GET_OUTPUTS (focused scale)")
            .await?
            .into_iter()
            .find(|o| o.focused)
            .and_then(|o| o.scale)
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(1.0))
    }
}

async fn connect_pair() -> WmResult<(Connection, EventStream)> {
    let cmd = Connection::new()
        .await
        .map_err(|e| WmError::ipc("connect to sway IPC (cmd socket)", e))?;
    let events_conn = Connection::new()
        .await
        .map_err(|e| WmError::ipc("connect to sway IPC (events socket)", e))?;
    let stream = events_conn
        .subscribe([EventType::Window, EventType::Workspace])
        .await
        .map_err(|e| WmError::ipc("subscribe to sway window+workspace events", e))?;
    Ok((cmd, stream))
}

/// Drive one events stream. Returns `true` when the caller should
/// reconnect, `false` on shutdown.
async fn drive_events(
    mut stream: EventStream,
    snapshot: Arc<RwLock<Snapshot>>,
    reconnect: Arc<tokio::sync::Notify>,
    window_events: broadcast::Sender<WindowEvent>,
    shutdown: &mut watch::Receiver<bool>,
) -> bool {
    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() { return false; }
            }
            _ = reconnect.notified() => {
                return true;
            }
            evt = stream.next() => {
                match evt {
                    Some(Ok(Event::Window(w))) => {
                        handle_window_event(&w, &snapshot).await;
                        if let Some(ev) = window_event_from_sway(&w) {
                            let _ = window_events.send(ev);
                        }
                    }
                    Some(Ok(Event::Workspace(data))) => {
                        if matches!(data.change, WorkspaceChange::Focus) {
                            let name = data.current.as_ref().and_then(|n| n.name.clone());
                            apply_workspace_focus(&snapshot, name).await;
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::warn!("sway event stream error: {e}");
                        return true;
                    }
                    None => return true,
                }
            }
        }
    }
}

fn window_event_from_sway(w: &swayipc_async::WindowEvent) -> Option<WindowEvent> {
    let id = sway_id(w.container.id)?;
    match w.change {
        WindowChange::New => {
            let props = w.container.window_properties.as_ref();
            let class = props.and_then(|p| p.class.clone());
            let title = w
                .container
                .name
                .clone()
                .or_else(|| props.and_then(|p| p.title.clone()));
            Some(WindowEvent::Opened {
                id,
                title,
                class,
                app_id: w.container.app_id.clone(),
            })
        }
        WindowChange::Title => Some(WindowEvent::TitleChanged {
            id,
            new_title: w.container.name.clone(),
        }),
        WindowChange::Close => Some(WindowEvent::Closed { id }),
        _ => None,
    }
}

async fn supervisor_loop(
    backend: Arc<SwayBackend>,
    initial_stream: EventStream,
    snapshot: Arc<RwLock<Snapshot>>,
    reconnect: Arc<tokio::sync::Notify>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !drive_events(
        initial_stream,
        snapshot.clone(),
        reconnect.clone(),
        backend.window_events.clone(),
        &mut shutdown,
    )
    .await
    {
        tracing::info!("sway supervisor exited (shutdown during initial events stream)");
        return;
    }

    let mut attempt: u32 = 0;
    loop {
        *backend.cmd.lock().await = None;
        tracing::warn!("sway disconnected; reconnecting (attempt {})", attempt + 1);

        let delay = crate::backoff::backoff_delay(attempt);
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    tracing::info!("sway supervisor exited (shutdown during backoff)");
                    return;
                }
            }
            _ = tokio::time::sleep(delay) => {}
        }

        match connect_pair().await {
            Ok((mut cmd, stream)) => {
                if let Ok(s) = seed_snapshot(&mut cmd).await {
                    *snapshot.write().await = s;
                }
                *backend.cmd.lock().await = Some(cmd);
                attempt = 0;
                tracing::info!("sway backend reconnected");
                if !drive_events(
                    stream,
                    snapshot.clone(),
                    reconnect.clone(),
                    backend.window_events.clone(),
                    &mut shutdown,
                )
                .await
                {
                    tracing::info!("sway supervisor exited (shutdown during events stream)");
                    return;
                }
            }
            Err(e) => {
                tracing::warn!("sway reconnect failed: {e}; will retry after backoff");
                attempt = attempt.saturating_add(1);
            }
        }
    }
}

fn find_sway_node_rect(node: &Node, criteria: &PlacementCriteria) -> Option<Rect> {
    if matches!(node.node_type, NodeType::Con | NodeType::FloatingCon)
        && sway_node_matches(node, criteria)
    {
        return Some(Rect {
            x: node.rect.x,
            y: node.rect.y,
            width: node.rect.width.max(0) as u32,
            height: node.rect.height.max(0) as u32,
        });
    }
    for child in node.nodes.iter().chain(node.floating_nodes.iter()) {
        if let Some(r) = find_sway_node_rect(child, criteria) {
            return Some(r);
        }
    }
    None
}

fn sway_node_matches(node: &Node, criteria: &PlacementCriteria) -> bool {
    let props = node.window_properties.as_ref();
    match criteria {
        PlacementCriteria::AppId(want) => node.app_id.as_deref().is_some_and(|a| a == want),
        PlacementCriteria::Class(want) => props
            .and_then(|p| p.class.as_deref())
            .is_some_and(|c| c == want),
        PlacementCriteria::Title(want) => node
            .name
            .as_deref()
            .or_else(|| props.and_then(|p| p.title.as_deref()))
            .is_some_and(|t| t == want),
        // ConId is matched by id elsewhere. No-match so a misuse fails loudly.
        PlacementCriteria::ConId(_) => false,
    }
}

fn collect_windows(node: &Node, current_ws: Option<&str>, out: &mut Vec<Window>) {
    let next_ws = if matches!(node.node_type, NodeType::Workspace) {
        node.name.as_deref()
    } else {
        current_ws
    };

    if matches!(node.node_type, NodeType::Con | NodeType::FloatingCon)
        && let Some(id) = sway_id(node.id)
    {
        let class = node
            .window_properties
            .as_ref()
            .and_then(|p| p.class.clone());
        let app = node.app_id.clone().or(class);
        let title = node.name.clone().or_else(|| {
            node.window_properties
                .as_ref()
                .and_then(|p| p.title.clone())
        });
        out.push(Window {
            id,
            app,
            title,
            workspace: next_ws.map(|s| s.to_string()),
        });
    }

    for child in node.nodes.iter().chain(node.floating_nodes.iter()) {
        collect_windows(child, next_ws, out);
    }
}

async fn seed_snapshot(cmd: &mut Connection) -> WmResult<Snapshot> {
    let tree = cmd
        .get_tree()
        .await
        .map_err(|e| WmError::ipc("sway GET_TREE", e))?;
    let focused = walk_focused(&tree);
    let focused_id = focused.and_then(|n| sway_id(n.id));
    let focused_class = focused.and_then(|n| {
        n.app_id
            .clone()
            .or_else(|| n.window_properties.as_ref().and_then(|p| p.class.clone()))
    });
    let focused_title = focused.and_then(|n| {
        n.name
            .clone()
            .or_else(|| n.window_properties.as_ref().and_then(|p| p.title.clone()))
    });

    let active_workspace = match cmd.get_workspaces().await {
        Ok(ws) => ws.into_iter().find(|w| w.focused).map(|w| w.name),
        Err(e) => {
            tracing::warn!("sway GET_WORKSPACES on seed failed: {e}");
            None
        }
    };
    Ok(Snapshot {
        focused_id,
        focused_class,
        focused_title,
        active_workspace,
    })
}

async fn handle_window_event(w: &swayipc_async::WindowEvent, snap: &Arc<RwLock<Snapshot>>) {
    let kind = match w.change {
        WindowChange::Focus => WindowChangeKind::Focus,
        WindowChange::Title => WindowChangeKind::Title,
        WindowChange::Close => WindowChangeKind::Close,
        _ => return,
    };
    let id = sway_id(w.container.id);
    let class = w.container.app_id.clone().or_else(|| {
        w.container
            .window_properties
            .as_ref()
            .and_then(|p| p.class.clone())
    });
    let title = w.container.name.clone().or_else(|| {
        w.container
            .window_properties
            .as_ref()
            .and_then(|p| p.title.clone())
    });
    apply_window_event(snap, kind, id, class, title).await;
}

fn walk_focused(node: &Node) -> Option<&Node> {
    if node.focused {
        return Some(node);
    }
    for child in node.nodes.iter().chain(node.floating_nodes.iter()) {
        if let Some(n) = walk_focused(child) {
            return Some(n);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sway_id_rejects_non_positive() {
        for raw in [0, -1, -12345] {
            assert_eq!(sway_id(raw), None, "{raw}");
        }
        assert_eq!(sway_id(42), WindowId::new(42));
    }
}

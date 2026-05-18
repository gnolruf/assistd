//! i3 backend for [`crate::WindowManager`].
//!
//! Speaks the i3 IPC protocol via `tokio-i3ipc`. Two Unix-socket
//! connections are opened at startup: one held under a `Mutex` for
//! command sends, one consumed by [`tokio_i3ipc::I3::listen`] for the
//! event stream. A background task subscribes to `Window` events and
//! keeps a snapshot of the focused window's class up to date so
//! [`I3Backend::focused_window`] reads from memory rather than round-
//! tripping over the socket.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use tokio::sync::{Mutex, RwLock, broadcast, watch};
use tokio::task::JoinHandle;
use tokio_i3ipc::{
    I3,
    event::{Event, Subscribe, WindowChange, WorkspaceChange},
    reply,
};

use crate::criteria::{format_place_floating_pixels, format_workspace_target};
use crate::error::ipc_ctx;
use crate::snapshot::{
    self, Snapshot, WindowChangeKind, apply_window_event, apply_workspace_focus,
};
use crate::{
    FocusedWindowContext, Layout, PlacementAnchor, PlacementCriteria, Rect, ResizeDir, Window,
    WindowEvent, WindowId, WindowManager, WmError, WmResult, WorkspaceId, WorkspaceInfo,
};

/// How many window events to buffer per subscriber. Subscribers that
/// fall this far behind get a `Lagged` error and must reconnect — we
/// treat that as a no-op in the popup placement path (just fall back
/// to polling), so a small buffer is fine.
const WINDOW_EVENTS_CAPACITY: usize = 32;

/// Max time `find_window_rect_by_criteria` waits for a matching
/// `WindowEvent::Opened` after its initial tree poll comes up empty.
/// Bounded so the popup never blocks indefinitely if the window
/// somehow never appears.
const WINDOW_EVENT_WAIT: Duration = Duration::from_millis(500);

/// `WindowManager` impl wrapping a single i3 IPC command socket.
/// Held inside `Arc<dyn WindowManager>` by the daemon's `AppState`.
pub struct I3Backend {
    cmd: Arc<Mutex<Option<I3>>>,
    snapshot: Arc<RwLock<Snapshot>>,
    /// Bumped by call-site failures (`run_command`, `get_tree`, …) so
    /// the supervisor task reconnects without waiting for the next
    /// event-stream error to fire.
    reconnect: Arc<tokio::sync::Notify>,
    /// Broadcasts every i3 `window::*` event so consumers (currently
    /// just the tray popup's placement path) can react synchronously
    /// to a freshly mapped window instead of polling `GET_TREE`.
    /// Receivers are created on demand via `subscribe`; lagged
    /// receivers get a typed `Lagged` error and fall back to polling.
    window_events: broadcast::Sender<WindowEvent>,
}

/// Returned by [`I3Backend::start`] alongside the backend itself. The
/// daemon awaits [`I3Handle::shutdown`] in its graceful-shutdown block
/// so the supervisor task drains before the process exits, matching
/// how other long-lived subsystems (`embedder_task_handle`,
/// `hotkey_handle`, etc.) are awaited in `daemon.rs`.
pub struct I3Handle {
    pub backend: Arc<I3Backend>,
    supervisor_task: JoinHandle<()>,
}

impl I3Handle {
    /// Awaits the supervisor task. The daemon should flip `shutdown_tx` before
    /// calling this so the supervisor exits cleanly rather than blocking.
    pub async fn shutdown(self) {
        let _ = self.supervisor_task.await;
    }
}

impl I3Backend {
    /// Connect to the i3 IPC sockets, seed the focused-window snapshot,
    /// and spawn the supervisor task that drives the event stream and
    /// reconnects on socket drops.
    ///
    /// Returns `Err` when the i3 socket isn't reachable on first try
    /// (i3 not running, `I3SOCK` unset, non-Linux dev box, …). The
    /// daemon catches that and substitutes `NoWindowManager` so the
    /// rest of startup proceeds. After the initial connect, transient
    /// socket failures are handled in-process by the supervisor;
    /// `i3-msg restart` no longer requires a daemon restart.
    pub async fn start(shutdown: watch::Receiver<bool>) -> WmResult<I3Handle> {
        let (mut cmd, events_conn) = connect_pair().await?;
        let initial = match seed_snapshot(&mut cmd).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("i3 seed_snapshot failed: {e:#}; starting with empty focus state");
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
            events_conn,
            snapshot,
            reconnect,
            shutdown,
        ));

        Ok(I3Handle {
            backend,
            supervisor_task,
        })
    }

    async fn run(&self, payload: &str) -> WmResult<()> {
        let mut guard = self.cmd.lock().await;
        let conn = guard.as_mut().ok_or(WmError::Disconnected)?;
        let results =
            match tokio::time::timeout(crate::WM_IPC_TIMEOUT, conn.run_command(payload)).await {
                Err(_) => {
                    // Wedged i3: drop the conn so the supervisor reconnects
                    // instead of holding the broken socket forever.
                    *guard = None;
                    self.reconnect.notify_one();
                    return Err(WmError::Timeout(crate::WM_IPC_TIMEOUT));
                }
                Ok(Err(e)) => {
                    *guard = None;
                    self.reconnect.notify_one();
                    return Err(ipc_ctx(e, "i3 RUN_COMMAND"));
                }
                Ok(Ok(v)) => v,
            };
        for r in results {
            if !r.success {
                return Err(WmError::Rejected(format!(
                    "{payload}: {}",
                    r.error.unwrap_or_else(|| "unknown error".into())
                )));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl WindowManager for I3Backend {
    async fn focus(&self, window: &WindowId) -> WmResult<()> {
        let cmd = format!(r#"[con_id="{}"] focus"#, window.get());
        self.run(&cmd).await
    }

    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        let target = format_workspace_target(workspace);
        let cmd = format!(r#"[con_id="{}"] move container to {target}"#, window.get(),);
        self.run(&cmd).await
    }

    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        Ok(snapshot::read_focused_id(&self.snapshot).await)
    }

    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        Ok(snapshot::read_focused_context(&self.snapshot).await)
    }

    async fn list_windows(&self) -> WmResult<Vec<Window>> {
        let mut guard = self.cmd.lock().await;
        let conn = guard.as_mut().ok_or(WmError::Disconnected)?;
        let tree = match tokio::time::timeout(crate::WM_IPC_TIMEOUT, conn.get_tree()).await {
            Err(_) => {
                *guard = None;
                self.reconnect.notify_one();
                return Err(WmError::Timeout(crate::WM_IPC_TIMEOUT));
            }
            Ok(Err(e)) => {
                *guard = None;
                self.reconnect.notify_one();
                return Err(ipc_ctx(e, "i3 GET_TREE"));
            }
            Ok(Ok(t)) => t,
        };
        drop(guard);
        let mut out = Vec::new();
        collect_windows(&tree, None, &mut out);
        Ok(out)
    }

    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        let mut guard = self.cmd.lock().await;
        let conn = guard.as_mut().ok_or(WmError::Disconnected)?;
        let ws = match tokio::time::timeout(crate::WM_IPC_TIMEOUT, conn.get_workspaces()).await {
            Err(_) => {
                *guard = None;
                self.reconnect.notify_one();
                return Err(WmError::Timeout(crate::WM_IPC_TIMEOUT));
            }
            Ok(Err(e)) => {
                *guard = None;
                self.reconnect.notify_one();
                return Err(ipc_ctx(e, "i3 GET_WORKSPACES"));
            }
            Ok(Ok(w)) => w,
        };
        Ok(ws
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
        self.run(&i3_resize_payload(window, direction, pixels))
            .await
    }

    async fn set_layout(&self, layout: Layout) -> WmResult<()> {
        self.run(&i3_layout_payload(layout)).await
    }

    async fn focused_workspace_rect(&self) -> WmResult<Rect> {
        self.focused_workspace_rect_inner().await
    }

    async fn place_floating(
        &self,
        criteria: &PlacementCriteria,
        anchor: PlacementAnchor,
    ) -> WmResult<()> {
        let translated = translate_criteria_for_i3(criteria);
        let workspace = self.focused_workspace_rect_inner().await?;
        // eframe / winit / X11 don't always honour the size we passed
        // to `with_inner_size` exactly — DPI scaling especially can
        // turn a configured 360-logical-px window into a 420-physical-
        // px one. Compute placement using the window's actual rect so
        // the anchor lands precisely regardless of scale.
        let effective = match self.find_window_rect_by_criteria(&translated).await {
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
        self.run(&format_place_floating_pixels(
            &translated,
            effective,
            workspace,
        ))
        .await
    }
}

impl I3Backend {
    /// Walk the i3 tree looking for a window matching the given
    /// criteria, returning its current rect. Used by
    /// [`Self::place_floating`] to feed the actual on-screen
    /// dimensions into the placement formula. Only the
    /// title/class/app_id variants are searched here; `con_id`
    /// short-circuits via the tree's `id` field.
    ///
    /// Subscribes to the backend's window-event broadcast first, then
    /// does a single tree poll. If the window is already there we
    /// return immediately. Otherwise we wait (up to
    /// [`WINDOW_EVENT_WAIT`]) for a matching `WindowEvent::Opened`
    /// from i3's `window::new` event stream and then re-poll the tree
    /// to get the rect. Subscribing before polling closes the race
    /// where the supervisor processes the new-window event after our
    /// `GET_TREE` reply but before we get a chance to wait.
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
                    // Sender dropped (backend shutting down) — bail.
                    Err(broadcast::error::RecvError::Closed) => return false,
                    // Lagged: we missed events. The window might
                    // already be in the tree, so break out and let the
                    // final poll find it.
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
                "popup: window appeared via i3 window::new event"
            );
        }
        result
    }

    async fn find_window_rect_once(&self, criteria: &PlacementCriteria) -> WmResult<Rect> {
        let mut guard = self.cmd.lock().await;
        let conn = guard.as_mut().ok_or(WmError::Disconnected)?;
        let tree = match tokio::time::timeout(crate::WM_IPC_TIMEOUT, conn.get_tree()).await {
            Err(_) => {
                *guard = None;
                self.reconnect.notify_one();
                return Err(WmError::Timeout(crate::WM_IPC_TIMEOUT));
            }
            Ok(Err(e)) => {
                *guard = None;
                self.reconnect.notify_one();
                return Err(ipc_ctx(e, "i3 GET_TREE (window rect)"));
            }
            Ok(Ok(t)) => t,
        };
        drop(guard);
        find_node_rect(&tree, criteria)
            .ok_or_else(|| WmError::Rejected(format!("no window matches {criteria:?}")))
    }

    /// Query the focused workspace's pixel rect via `GET_WORKSPACES`.
    /// Used by [`Self::place_floating`] to compute absolute pixel
    /// coordinates for the popup, bypassing i3's silent clamping of
    /// off-screen ppt-based positions. Exposed through the
    /// `WindowManager` trait method of the same name; the private
    /// `_inner` suffix avoids the trait/inherent collision.
    async fn focused_workspace_rect_inner(&self) -> WmResult<Rect> {
        let mut guard = self.cmd.lock().await;
        let conn = guard.as_mut().ok_or(WmError::Disconnected)?;
        let workspaces =
            match tokio::time::timeout(crate::WM_IPC_TIMEOUT, conn.get_workspaces()).await {
                Err(_) => {
                    *guard = None;
                    self.reconnect.notify_one();
                    return Err(WmError::Timeout(crate::WM_IPC_TIMEOUT));
                }
                Ok(Err(e)) => {
                    *guard = None;
                    self.reconnect.notify_one();
                    return Err(ipc_ctx(e, "i3 GET_WORKSPACES (focused rect)"));
                }
                Ok(Ok(w)) => w,
            };
        drop(guard);
        workspaces
            .into_iter()
            .find(|w| w.focused)
            .map(|w| Rect {
                // tokio-i3ipc reports rect coordinates as `isize`; sway
                // uses `i32`. Normalise both to `i32` at this seam.
                x: w.rect.x as i32,
                y: w.rect.y as i32,
                width: w.rect.width.max(0) as u32,
                height: w.rect.height.max(0) as u32,
            })
            .ok_or_else(|| WmError::Rejected("no focused workspace".into()))
    }
}

/// i3's IPC grammar predates Wayland; its criteria block accepts
/// `class` / `instance` / `con_id` / `id` / `title` etc. but not
/// `app_id`. Map the protocol-neutral [`PlacementCriteria::AppId`]
/// onto `title` — egui-winit 0.34 (and presumably other toolkits)
/// doesn't reliably populate X11 `WM_CLASS` from the same identifier
/// they pass to `with_app_id` on Wayland, but the title is always
/// settable. Callers that own their popup window must set its title
/// to the same string they pass as `AppId(...)`.
fn translate_criteria_for_i3(c: &PlacementCriteria) -> PlacementCriteria {
    match c {
        PlacementCriteria::AppId(s) => PlacementCriteria::Title(s.clone()),
        other => other.clone(),
    }
}

/// Recursively search the i3 tree for a window matching the given
/// criteria and return its rendered rect. The rect is i3's
/// `window_rect` (the client area inside any decorations), which is
/// what the placement formula needs to anchor exactly. Container
/// nodes (no `window`) are skipped.
fn find_node_rect(node: &reply::Node, criteria: &PlacementCriteria) -> Option<Rect> {
    if node.window.is_some() && node_matches(node, criteria) {
        return Some(Rect {
            x: node.rect.x as i32,
            y: node.rect.y as i32,
            width: node.rect.width.max(0) as u32,
            height: node.rect.height.max(0) as u32,
        });
    }
    for child in node.nodes.iter().chain(node.floating_nodes.iter()) {
        if let Some(r) = find_node_rect(child, criteria) {
            return Some(r);
        }
    }
    None
}

fn node_matches(node: &reply::Node, criteria: &PlacementCriteria) -> bool {
    let props = node.window_properties.as_ref();
    match criteria {
        PlacementCriteria::Title(want) => node
            .name
            .as_deref()
            .or_else(|| props.and_then(|p| p.title.as_deref()))
            .is_some_and(|t| t == want),
        PlacementCriteria::Class(want) => props
            .and_then(|p| p.class.as_deref())
            .is_some_and(|c| c == want),
        // AppId is rewritten to Title before reaching here; ConId is
        // matched by id at a different layer. Leave as no-match so a
        // misuse fails loudly rather than silently picking the wrong
        // window.
        _ => false,
    }
}

/// Format the i3 RUN_COMMAND payload for `resize_width`. Factored out
/// so unit tests can assert on the literal string without a live i3.
fn i3_resize_payload(window: &WindowId, direction: ResizeDir, pixels: u32) -> String {
    format!(
        r#"[con_id="{}"] resize {} width {} px or 0 ppt"#,
        window.get(),
        direction.as_str(),
        pixels,
    )
}

/// Format the i3 RUN_COMMAND payload for `set_layout`. The bare
/// `layout <name>` form acts on the focused container, the same shape
/// `i3-msg layout …` produces.
fn i3_layout_payload(layout: Layout) -> String {
    format!("layout {}", layout.as_str())
}

/// Walk the i3 tree recursively, emitting one [`Window`] per leaf node
/// that has an X11 window backing (i.e. real, mapped clients, not
/// containers). Tracks the most recent `NodeType::Workspace` ancestor
/// in `current_ws` so each window can be tagged with the workspace it
/// lives on.
fn collect_windows(node: &reply::Node, current_ws: Option<&str>, out: &mut Vec<Window>) {
    let next_ws = if matches!(node.node_type, reply::NodeType::Workspace) {
        node.name.as_deref()
    } else {
        current_ws
    };

    if node.window.is_some()
        && let Some(id) = WindowId::new(node.id as u64)
    {
        let (app, title) = match node.window_properties.as_ref() {
            Some(props) => (props.class.clone(), props.title.clone()),
            None => (None, None),
        };
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

async fn seed_snapshot(cmd: &mut I3) -> Result<Snapshot> {
    let tree = cmd.get_tree().await.context("i3 GET_TREE")?;
    let focused = walk_focused(&tree);
    let focused_id = focused.and_then(|n| WindowId::new(n.id as u64));
    let focused_class = focused
        .and_then(|n| n.window_properties.as_ref())
        .and_then(|p| p.class.clone());
    let focused_title = focused.and_then(|n| n.name.clone());

    let active_workspace = match cmd.get_workspaces().await {
        Ok(ws) => ws.into_iter().find(|w| w.focused).map(|w| w.name),
        Err(e) => {
            tracing::warn!("i3 GET_WORKSPACES on seed failed: {e:#}");
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

/// Open the cmd + events socket pair and subscribe events. Pulled out
/// of `start()` so the supervisor can reuse it on each reconnect.
async fn connect_pair() -> WmResult<(I3, I3)> {
    let cmd = I3::connect()
        .await
        .map_err(|e| ipc_ctx(e, "connect to i3 IPC (cmd socket)"))?;
    let mut events_conn = I3::connect()
        .await
        .map_err(|e| ipc_ctx(e, "connect to i3 IPC (events socket)"))?;
    events_conn
        .subscribe([Subscribe::Window, Subscribe::Workspace])
        .await
        .map_err(|e| ipc_ctx(e, "subscribe to i3 window+workspace events"))?;
    Ok((cmd, events_conn))
}

/// Drive one events connection until it errors or the supervisor
/// signals a forced reconnect. Returns when the inner loop should
/// fall through to the reconnect branch, or `false` if shutdown was
/// observed (in which case the supervisor exits cleanly).
async fn drive_events(
    events_conn: I3,
    snapshot: Arc<RwLock<Snapshot>>,
    reconnect: Arc<tokio::sync::Notify>,
    window_events: broadcast::Sender<WindowEvent>,
    shutdown: &mut watch::Receiver<bool>,
) -> bool {
    let mut stream = events_conn.listen();
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
                        if let Some(ev) = window_event_from_i3(&w) {
                            // Best-effort: receivers may not exist yet
                            // (no popup running), and that's fine.
                            let _ = window_events.send(ev);
                        }
                    }
                    Some(Ok(Event::Workspace(d))) => {
                        if matches!(d.change, WorkspaceChange::Focus) {
                            let ws = d.current.as_ref().and_then(|n| n.name.clone());
                            apply_workspace_focus(&snapshot, ws).await;
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::warn!("i3 event stream error: {e}");
                        return true;
                    }
                    None => return true, // socket closed
                }
            }
        }
    }
}

/// Project an i3 `WindowData` event onto our backend-neutral
/// [`WindowEvent`]. Returns `None` for change kinds the broadcast
/// channel doesn't care about (focus, move, urgent, mark, fullscreen,
/// floating).
fn window_event_from_i3(w: &tokio_i3ipc::event::WindowData) -> Option<WindowEvent> {
    let id = WindowId::new(w.container.id as u64)?;
    match w.change {
        WindowChange::New => {
            let props = w.container.window_properties.as_ref();
            let class = props.and_then(|p| p.class.clone());
            // i3 is X11-only; Wayland app_id doesn't apply. Setting
            // `app_id` to `None` keeps the field shape uniform with
            // sway, where the same enum variant carries both.
            Some(WindowEvent::Opened {
                id,
                title: w.container.name.clone(),
                class,
                app_id: None,
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

/// Outer reconnect loop. Drives events through `drive_events`; on
/// fall-through (socket error, forced-reconnect, or initial events-
/// stream creation failure), drops `cmd` to `None`, sleeps with
/// exponential backoff, and reconnects. Exits cleanly on shutdown.
async fn supervisor_loop(
    backend: Arc<I3Backend>,
    initial_events: I3,
    snapshot: Arc<RwLock<Snapshot>>,
    reconnect: Arc<tokio::sync::Notify>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !drive_events(
        initial_events,
        snapshot.clone(),
        reconnect.clone(),
        backend.window_events.clone(),
        &mut shutdown,
    )
    .await
    {
        tracing::info!("i3 supervisor exited (shutdown during initial events stream)");
        return;
    }

    let mut attempt: u32 = 0;
    loop {
        *backend.cmd.lock().await = None;
        tracing::warn!("i3 disconnected; reconnecting (attempt {})", attempt + 1);

        let delay = crate::backoff::backoff_delay(attempt);
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    tracing::info!("i3 supervisor exited (shutdown during backoff)");
                    return;
                }
            }
            _ = tokio::time::sleep(delay) => {}
        }

        match connect_pair().await {
            Ok((mut cmd, events_conn)) => {
                if let Ok(s) = seed_snapshot(&mut cmd).await {
                    *snapshot.write().await = s;
                }
                *backend.cmd.lock().await = Some(cmd);
                attempt = 0;
                tracing::info!("i3 backend reconnected");
                if !drive_events(
                    events_conn,
                    snapshot.clone(),
                    reconnect.clone(),
                    backend.window_events.clone(),
                    &mut shutdown,
                )
                .await
                {
                    tracing::info!("i3 supervisor exited (shutdown during events stream)");
                    return;
                }
            }
            Err(e) => {
                tracing::warn!("i3 reconnect failed: {e}; will retry after backoff");
                attempt = attempt.saturating_add(1);
            }
        }
    }
}

/// Project an i3 `WindowData` event into the shared snapshot's
/// `(id, class, title, kind)` tuple and dispatch to
/// [`apply_window_event`]. The compositor-specific projection (reading
/// `container.id`, `window_properties.class`, etc.) lives here; the
/// "what changes when" rules live in `crate::snapshot`.
async fn handle_window_event(w: &tokio_i3ipc::event::WindowData, snap: &Arc<RwLock<Snapshot>>) {
    let kind = match w.change {
        WindowChange::Focus => WindowChangeKind::Focus,
        WindowChange::Title => WindowChangeKind::Title,
        WindowChange::Close => WindowChangeKind::Close,
        _ => return,
    };
    let id = WindowId::new(w.container.id as u64);
    let class = w
        .container
        .window_properties
        .as_ref()
        .and_then(|p| p.class.clone());
    let title = w.container.name.clone();
    apply_window_event(snap, kind, id, class, title).await;
}

fn walk_focused(node: &reply::Node) -> Option<&reply::Node> {
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

    fn id(n: u64) -> WindowId {
        WindowId::new(n).expect("test ids are non-zero")
    }

    #[test]
    fn resize_payload_uses_con_id_criteria() {
        let p = i3_resize_payload(&id(42), ResizeDir::Grow, 50);
        assert_eq!(p, r#"[con_id="42"] resize grow width 50 px or 0 ppt"#);
    }

    #[test]
    fn resize_payload_renders_id_in_decimal() {
        let p = i3_resize_payload(&id(1234567890), ResizeDir::Shrink, 5);
        assert_eq!(
            p,
            r#"[con_id="1234567890"] resize shrink width 5 px or 0 ppt"#
        );
    }

    #[test]
    fn translate_criteria_rewrites_app_id_to_title() {
        let out = translate_criteria_for_i3(&PlacementCriteria::AppId("dev.assistd.popup".into()));
        assert_eq!(out, PlacementCriteria::Title("dev.assistd.popup".into()));
    }

    #[test]
    fn translate_criteria_preserves_class_title_and_con_id() {
        assert_eq!(
            translate_criteria_for_i3(&PlacementCriteria::Class("Firefox".into())),
            PlacementCriteria::Class("Firefox".into())
        );
        assert_eq!(
            translate_criteria_for_i3(&PlacementCriteria::Title("Inbox".into())),
            PlacementCriteria::Title("Inbox".into())
        );
        let con = WindowId::new(42).unwrap();
        assert_eq!(
            translate_criteria_for_i3(&PlacementCriteria::ConId(con)),
            PlacementCriteria::ConId(con)
        );
    }

    #[test]
    fn layout_payload_emits_bare_form() {
        // i3 / sway treat `layout <name>` as acting on the focused
        // container; no criteria prefix.
        for (l, expected) in [
            (Layout::Default, "layout default"),
            (Layout::Tabbed, "layout tabbed"),
            (Layout::Stacking, "layout stacking"),
            (Layout::SplitH, "layout splith"),
            (Layout::SplitV, "layout splitv"),
        ] {
            assert_eq!(i3_layout_payload(l), expected);
        }
    }
}

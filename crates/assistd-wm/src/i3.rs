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

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use tokio::sync::{Mutex, RwLock, watch};
use tokio::task::JoinHandle;
use tokio_i3ipc::{
    I3,
    event::{Event, Subscribe, WindowChange, WorkspaceChange},
    reply,
};

use crate::criteria::format_workspace_target;
use crate::error::ipc_ctx;
use crate::snapshot::{
    self, Snapshot, WindowChangeKind, apply_window_event, apply_workspace_focus,
};
use crate::{
    FocusedWindowContext, Layout, ResizeDir, WindowId, WindowManager, WmError, WmResult, Window,
    WorkspaceId, WorkspaceInfo,
};

// PR 4: the Snapshot struct + apply-event race rules now live in
// `crate::snapshot` so the i3 and Sway backends share them. This file
// only owns the i3-specific connection plumbing and reply-to-snapshot
// projection.

/// `WindowManager` impl wrapping a single i3 IPC command socket.
/// Held inside `Arc<dyn WindowManager>` by the daemon's `AppState`.
///
/// PR 5: the cmd connection is now `Option<I3>`. The supervisor task
/// sets it to `None` while reconnecting after a socket drop or a
/// timeout strike, and `Some` once the new connection is seeded. The
/// trait-method helpers return [`WmError::Disconnected`] for the None
/// state so callers can render the right hint without blocking.
pub struct I3Backend {
    cmd: Arc<Mutex<Option<I3>>>,
    snapshot: Arc<RwLock<Snapshot>>,
    /// Bumped by call-site failures (`run_command`, `get_tree`, …) so
    /// the supervisor task reconnects without waiting for the next
    /// event-stream error to fire.
    reconnect: Arc<tokio::sync::Notify>,
}

/// Returned by [`I3Backend::start`] alongside the backend itself. The
/// daemon awaits [`I3Handle::shutdown`] in its graceful-shutdown block
/// so the supervisor task drains before the process exits — matching
/// how other long-lived subsystems (`embedder_task_handle`,
/// `hotkey_handle`, etc.) are awaited in `daemon.rs`.
pub struct I3Handle {
    pub backend: Arc<I3Backend>,
    supervisor_task: JoinHandle<()>,
}

impl I3Handle {
    pub async fn shutdown(self) {
        // The daemon flips `shutdown_tx` first; the supervisor task
        // selects on `shutdown.changed()` and exits.
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
    /// socket failures are handled in-process by the supervisor —
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
        let backend = Arc::new(Self {
            cmd: Arc::new(Mutex::new(Some(cmd))),
            snapshot: snapshot.clone(),
            reconnect: reconnect.clone(),
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
        let results = match tokio::time::timeout(
            crate::WM_IPC_TIMEOUT,
            conn.run_command(payload),
        )
        .await
        {
            Err(_) => {
                // Wedged i3 — drop the conn so the supervisor reconnects
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
        // Drop the lock before walking the tree so concurrent calls
        // can proceed against the shared cmd socket.
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
        self.run(&i3_resize_payload(window, direction, pixels)).await
    }

    async fn set_layout(&self, layout: Layout) -> WmResult<()> {
        self.run(&i3_layout_payload(layout)).await
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
/// `layout <name>` form acts on the focused container — the same shape
/// `i3-msg layout …` produces.
fn i3_layout_payload(layout: Layout) -> String {
    format!("layout {}", layout.as_str())
}

/// Walk the i3 tree recursively, emitting one [`Window`] per leaf node
/// that has an X11 window backing (i.e. real, mapped clients — not
/// containers). Tracks the most recent `NodeType::Workspace` ancestor
/// in `current_ws` so each window can be tagged with the workspace it
/// lives on.
///
/// PR 3b: the [`Window::id`] is the i3 con_id (`reply::Node.id`),
/// which is unique. The X11 class moves into [`Window::app`] for human
/// display (`wm list`'s second column).
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
    // Active workspace seeded from a separate query: walking the
    // tree to the focused leaf's workspace ancestor would work, but
    // GET_WORKSPACES is one round-trip and gives an authoritative
    // `focused == true` row that already accounts for multi-output.
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
                    Some(Ok(Event::Window(w))) => handle_window_event(&w, &snapshot).await,
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
    // First pass: drive the events conn we got at startup. cmd is
    // already in `Some(_)` state, no reconnect needed yet.
    if !drive_events(initial_events, snapshot.clone(), reconnect.clone(), &mut shutdown).await {
        tracing::info!("i3 supervisor exited (shutdown during initial events stream)");
        return;
    }

    // Subsequent passes: clear cmd, reconnect, re-seed, drive events.
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
                if !drive_events(events_conn, snapshot.clone(), reconnect.clone(), &mut shutdown)
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
        // No need for escape — con_id is a decimal integer literal,
        // never a free-form string.
        let p = i3_resize_payload(&id(1234567890), ResizeDir::Shrink, 5);
        assert_eq!(p, r#"[con_id="1234567890"] resize shrink width 5 px or 0 ppt"#);
    }

    #[test]
    fn layout_payload_emits_bare_form() {
        // i3 / sway treat `layout <name>` as acting on the focused
        // container — no criteria prefix.
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

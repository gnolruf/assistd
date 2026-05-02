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
pub struct I3Backend {
    cmd: Arc<Mutex<I3>>,
    snapshot: Arc<RwLock<Snapshot>>,
}

/// Returned by [`I3Backend::start`] alongside the backend itself. The
/// daemon awaits [`I3Handle::shutdown`] in its graceful-shutdown block
/// so the event task drains before the process exits — matching how
/// other long-lived subsystems (`embedder_task_handle`, `hotkey_handle`,
/// etc.) are awaited in `daemon.rs`.
pub struct I3Handle {
    pub backend: Arc<I3Backend>,
    event_task: JoinHandle<()>,
}

impl I3Handle {
    pub async fn shutdown(self) {
        // The daemon flips `shutdown_tx` first; the event task selects
        // on `shutdown.changed()` and exits. Awaiting here just drains.
        let _ = self.event_task.await;
    }
}

impl I3Backend {
    /// Connect to the i3 IPC sockets, seed the focused-window snapshot,
    /// and spawn the background event task.
    ///
    /// Returns `Err` when the i3 socket isn't reachable (i3 not running,
    /// `I3SOCK` unset, non-Linux dev box, …). The daemon catches that
    /// and substitutes `NoWindowManager` so the rest of startup proceeds.
    pub async fn start(mut shutdown: watch::Receiver<bool>) -> WmResult<I3Handle> {
        // Two sockets: `listen()` consumes its connection by value, so a
        // single socket can't multiplex command + event traffic.
        let mut cmd = I3::connect()
            .await
            .map_err(|e| ipc_ctx(e, "connect to i3 IPC (cmd socket)"))?;
        let mut events_conn = I3::connect()
            .await
            .map_err(|e| ipc_ctx(e, "connect to i3 IPC (events socket)"))?;
        events_conn
            .subscribe([Subscribe::Window, Subscribe::Workspace])
            .await
            .map_err(|e| ipc_ctx(e, "subscribe to i3 window+workspace events"))?;

        // Seed snapshot before wrapping cmd in the Mutex — reuses the
        // command socket since `get_tree()` needs `&mut`.
        let initial = match seed_snapshot(&mut cmd).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("i3 seed_snapshot failed: {e:#}; starting with empty focus state");
                Snapshot::default()
            }
        };
        let snapshot = Arc::new(RwLock::new(initial));

        let snap_for_task = snapshot.clone();
        let event_task = tokio::spawn(async move {
            let mut stream = events_conn.listen();
            loop {
                tokio::select! {
                    _ = shutdown.changed() => {
                        if *shutdown.borrow() { break; }
                    }
                    evt = stream.next() => {
                        match evt {
                            Some(Ok(Event::Window(w))) => {
                                handle_window_event(&w, &snap_for_task).await;
                            }
                            Some(Ok(Event::Workspace(d))) => {
                                if matches!(d.change, WorkspaceChange::Focus) {
                                    let ws = d
                                        .current
                                        .as_ref()
                                        .and_then(|n| n.name.clone());
                                    apply_workspace_focus(&snap_for_task, ws).await;
                                }
                            }
                            Some(Ok(_)) => {}
                            Some(Err(e)) => {
                                tracing::warn!("i3 event stream error: {e}");
                                break;
                            }
                            None => break, // socket closed
                        }
                    }
                }
            }
            tracing::info!("i3 event task exited");
        });

        let backend = Arc::new(Self {
            cmd: Arc::new(Mutex::new(cmd)),
            snapshot,
        });
        Ok(I3Handle {
            backend,
            event_task,
        })
    }

    async fn run(&self, payload: &str) -> WmResult<()> {
        let results = self
            .cmd
            .lock()
            .await
            .run_command(payload)
            .await
            .map_err(|e| ipc_ctx(e, "i3 RUN_COMMAND"))?;
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
        let tree = self
            .cmd
            .lock()
            .await
            .get_tree()
            .await
            .map_err(|e| ipc_ctx(e, "i3 GET_TREE"))?;
        let mut out = Vec::new();
        collect_windows(&tree, None, &mut out);
        Ok(out)
    }

    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        let ws = self
            .cmd
            .lock()
            .await
            .get_workspaces()
            .await
            .map_err(|e| ipc_ctx(e, "i3 GET_WORKSPACES"))?;
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

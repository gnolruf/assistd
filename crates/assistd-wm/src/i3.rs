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

use crate::criteria::{escape_for_criteria, format_workspace_target};
use crate::error::ipc_ctx;
use crate::{
    FocusedWindowContext, WindowId, WindowManager, WmError, WmResult, Window, WorkspaceId,
    WorkspaceInfo,
};

/// Cached focus state. Seeded from `get_tree()` + `get_workspaces()`
/// at startup, then updated by the event task on each `Window::Focus`,
/// `Window::Title`, `Window::Close`, and `Workspace::Focus` event so
/// the synchronous reads in [`I3Backend::focused_window`] and
/// [`I3Backend::focused_context`] hit memory rather than IPC.
#[derive(Default, Clone)]
struct Snapshot {
    focused_class: Option<WindowId>,
    focused_title: Option<String>,
    active_workspace: Option<String>,
}

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
                                    snap_for_task.write().await.active_workspace = ws;
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
        let cmd = format!(r#"[class="{}"] focus"#, escape_for_criteria(window));
        self.run(&cmd).await
    }

    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        let target = format_workspace_target(workspace);
        let cmd = format!(
            r#"[class="{}"] move container to {}"#,
            escape_for_criteria(window),
            target,
        );
        self.run(&cmd).await
    }

    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        Ok(self.snapshot.read().await.focused_class.clone())
    }

    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        let s = self.snapshot.read().await;
        if s.focused_class.is_none() && s.focused_title.is_none() && s.active_workspace.is_none() {
            return Ok(None);
        }
        Ok(Some(FocusedWindowContext {
            class: s.focused_class.clone(),
            title: s.focused_title.clone(),
            workspace: s.active_workspace.clone(),
        }))
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

    async fn run_raw(&self, payload: &str) -> WmResult<()> {
        self.run(payload).await
    }
}

/// Walk the i3 tree recursively, emitting one [`Window`] per leaf node
/// that has both an X11 window id and a class. Tracks the most recent
/// `NodeType::Workspace` ancestor in `current_ws` so each window can be
/// tagged with the workspace it lives on.
///
/// Children of a non-workspace node inherit the parent's workspace
/// context; children of a workspace node use that workspace's name.
/// Floating nodes are walked alongside tiling nodes — both are real
/// windows the user can focus.
fn collect_windows(node: &reply::Node, current_ws: Option<&str>, out: &mut Vec<Window>) {
    let next_ws = if matches!(node.node_type, reply::NodeType::Workspace) {
        node.name.as_deref()
    } else {
        current_ws
    };

    if node.window.is_some()
        && let Some(props) = node.window_properties.as_ref()
        && let Some(class) = props.class.clone()
    {
        out.push(Window {
            id: class,
            title: props.title.clone(),
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
        focused_class,
        focused_title,
        active_workspace,
    })
}

/// Apply one i3 `Window` event to the cached focus snapshot.
///
/// Race rules:
/// - `Focus` overwrites both class and title from the new container.
/// - `Title` only takes effect when the event's class matches the
///   currently-focused class — title events from background windows
///   must not overwrite the foreground title.
/// - `Close` clears class + title only when the closed container is
///   the focused one. Workspace is left intact.
async fn handle_window_event(w: &tokio_i3ipc::event::WindowData, snap: &Arc<RwLock<Snapshot>>) {
    let class = w
        .container
        .window_properties
        .as_ref()
        .and_then(|p| p.class.clone());
    let title = w.container.name.clone();
    match w.change {
        WindowChange::Focus => {
            let mut s = snap.write().await;
            s.focused_class = class;
            s.focused_title = title;
        }
        WindowChange::Title => {
            let mut s = snap.write().await;
            // Only honor title updates for the currently-focused class.
            // (i3 emits Title events for background windows too.)
            if s.focused_class == class && class.is_some() {
                s.focused_title = title;
            }
        }
        WindowChange::Close => {
            let mut s = snap.write().await;
            if s.focused_class == class && class.is_some() {
                s.focused_class = None;
                s.focused_title = None;
            }
        }
        _ => {}
    }
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

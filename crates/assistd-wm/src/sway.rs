//! Sway backend for [`crate::WindowManager`].
//!
//! Sway implements the i3 IPC protocol with Wayland-specific extensions
//! (the `app_id` field on views, and a richer output reply). This
//! backend mirrors [`crate::i3`] in shape — two IPC connections (one
//! held under a `Mutex` for commands, one consumed by `subscribe()` to
//! drive the event stream), a snapshot updated on every `Window` and
//! `Workspace::Focus` event so synchronous reads from the daemon's
//! per-turn context injection don't round-trip the socket.
//!
//! The IPC client is `swayipc-async`, which is built on `async-io`
//! rather than `tokio`. Its futures coexist with the workspace tokio
//! runtime — the cost is one extra reactor thread per process, which
//! is acceptable since the WM event path is not on the LLM hot loop.
//!
//! Wayland-native vs XWayland identifiers: every Sway view has either
//! `Node::app_id` (xdg-shell, the Wayland-native case) or
//! `Node::window_properties.class` (XWayland fallback). We surface
//! whichever is present as [`crate::Window::id`], and dispatch
//! [`SwayBackend::focus`] / [`SwayBackend::move_to_workspace`] with a
//! composite criteria string so the caller doesn't have to know which
//! kind a given window uses. Sway treats a 0-match criteria as a
//! silent success — same as i3 — so `wm focus X` is best-effort: the
//! caller verifies via `wm active` if it cares.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures_util::StreamExt;
use swayipc_async::{Connection, Event, EventType, Node, NodeType, WindowChange, WorkspaceChange};
use tokio::sync::{Mutex, RwLock, watch};
use tokio::task::JoinHandle;

use crate::criteria::format_workspace_target;
use crate::error::ipc_ctx;
use crate::snapshot::{
    self, Snapshot, WindowChangeKind, apply_window_event, apply_workspace_focus,
};
use crate::{
    FocusedWindowContext, Layout, OutputInfo, ResizeDir, WindowId, WindowManager, WmError,
    WmResult, Window, WorkspaceId, WorkspaceInfo,
};

// PR 4: the Snapshot struct + apply-event race rules now live in
// `crate::snapshot` so the i3 and Sway backends share them. Sway's
// `focused_class` semantics (prefer `app_id` for Wayland-native
// windows, fall back to the X11 class for XWayland views) live in
// `handle_window_event` below, where we project the native event into
// the shared `(id, class, title)` tuple.

/// Cast a sway `Node.id` (`i64`) to a [`WindowId`]. Sway never emits
/// non-positive ids in practice, but the bounds check keeps the
/// conversion total — non-positive ids are silently dropped (the
/// caller treats them as "no id available").
fn sway_id(raw: i64) -> Option<WindowId> {
    if raw <= 0 {
        return None;
    }
    WindowId::new(raw as u64)
}

/// `WindowManager` impl wrapping a single Sway IPC command socket.
/// Held inside `Arc<dyn WindowManager>` by the daemon's `AppState`.
pub struct SwayBackend {
    cmd: Arc<Mutex<Connection>>,
    snapshot: Arc<RwLock<Snapshot>>,
}

/// Returned by [`SwayBackend::start`] alongside the backend itself. The
/// daemon awaits [`SwayHandle::shutdown`] in its graceful-shutdown
/// block — same pattern as `I3Handle`.
pub struct SwayHandle {
    pub backend: Arc<SwayBackend>,
    event_task: JoinHandle<()>,
}

impl SwayHandle {
    pub async fn shutdown(self) {
        // The daemon flips `shutdown_tx` first; the event task selects
        // on `shutdown.changed()` and exits. Awaiting here just drains.
        let _ = self.event_task.await;
    }
}

impl SwayBackend {
    /// Connect to Sway's IPC sockets, seed the focused-window snapshot,
    /// and spawn the background event task.
    ///
    /// Returns `Err` when the Sway socket isn't reachable (Sway not
    /// running, `$SWAYSOCK` unset, X11/i3 session, …). The daemon
    /// catches that and substitutes `NoWindowManager` so the rest of
    /// startup proceeds.
    pub async fn start(mut shutdown: watch::Receiver<bool>) -> WmResult<SwayHandle> {
        // Two connections: `subscribe()` consumes its connection by
        // value, so a single socket can't multiplex command + event
        // traffic. Same constraint as i3.
        let mut cmd = Connection::new()
            .await
            .map_err(|e| ipc_ctx(e, "connect to sway IPC (cmd socket)"))?;
        let events_conn = Connection::new()
            .await
            .map_err(|e| ipc_ctx(e, "connect to sway IPC (events socket)"))?;

        // Seed snapshot before wrapping cmd in the Mutex — get_tree()
        // and get_workspaces() need `&mut Connection`.
        let initial = match seed_snapshot(&mut cmd).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("sway seed_snapshot failed: {e:#}; starting with empty focus state");
                Snapshot::default()
            }
        };
        let snapshot = Arc::new(RwLock::new(initial));

        let mut stream = events_conn
            .subscribe([EventType::Window, EventType::Workspace])
            .await
            .map_err(|e| ipc_ctx(e, "subscribe to sway window+workspace events"))?;

        let snap_for_task = snapshot.clone();
        let event_task = tokio::spawn(async move {
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
                                tracing::warn!("sway event stream error: {e}");
                                break;
                            }
                            None => break, // socket closed
                        }
                    }
                }
            }
            tracing::info!("sway event task exited");
        });

        let backend = Arc::new(Self {
            cmd: Arc::new(Mutex::new(cmd)),
            snapshot,
        });
        Ok(SwayHandle {
            backend,
            event_task,
        })
    }

    async fn run(&self, payload: &str) -> WmResult<()> {
        let outcomes = self
            .cmd
            .lock()
            .await
            .run_command(payload)
            .await
            .map_err(|e| ipc_ctx(e, "sway RUN_COMMAND"))?;
        for r in outcomes {
            r.map_err(|e| WmError::Rejected(format!("{payload}: {e}")))?;
        }
        Ok(())
    }
}

#[async_trait]
impl WindowManager for SwayBackend {
    async fn focus(&self, window: &WindowId) -> WmResult<()> {
        let cmd = format!(r#"[con_id="{}"] focus"#, window.get());
        self.run(&cmd).await
    }

    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        let target = format_workspace_target(workspace);
        let cmd = format!(r#"[con_id="{}"] move container to {target}"#, window.get());
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
            .map_err(|e| ipc_ctx(e, "sway GET_TREE"))?;
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
            .map_err(|e| ipc_ctx(e, "sway GET_WORKSPACES"))?;
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
        self.run(&sway_resize_payload(window, direction, pixels))
            .await
    }

    async fn set_layout(&self, layout: Layout) -> WmResult<()> {
        self.run(&sway_layout_payload(layout)).await
    }

    async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
        let outputs = self
            .cmd
            .lock()
            .await
            .get_outputs()
            .await
            .map_err(|e| ipc_ctx(e, "sway GET_OUTPUTS"))?;
        Ok(outputs
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
}

/// Format the Sway RUN_COMMAND payload for `resize_width`. PR 3b
/// drops the `app_id|class` composite — Sway accepts `[con_id=N]`
/// criteria for any window regardless of XWayland-vs-Wayland origin.
fn sway_resize_payload(window: &WindowId, direction: ResizeDir, pixels: u32) -> String {
    format!(
        r#"[con_id="{}"] resize {} width {} px or 0 ppt"#,
        window.get(),
        direction.as_str(),
        pixels,
    )
}

/// Format the Sway RUN_COMMAND payload for `set_layout`. Acts on the
/// focused container — same form i3 uses, since Sway speaks the i3
/// IPC dialect for `layout`.
fn sway_layout_payload(layout: Layout) -> String {
    format!("layout {}", layout.as_str())
}

/// Walk Sway's tree recursively, emitting one [`Window`] per leaf view.
/// Sway leaves are nodes with `app_id` (Wayland-native) or
/// `window_properties.class` (XWayland) set; we prefer `app_id` for the
/// returned `Window.id` so the caller sees the natural identifier and
/// can pass it back unchanged. Tracks the most recent
/// `NodeType::Workspace` ancestor in `current_ws` so each window is
/// tagged with its workspace.
fn collect_windows(node: &Node, current_ws: Option<&str>, out: &mut Vec<Window>) {
    let next_ws = if matches!(node.node_type, NodeType::Workspace) {
        node.name.as_deref()
    } else {
        current_ws
    };

    if matches!(node.node_type, NodeType::Con | NodeType::FloatingCon)
        && let Some(id) = sway_id(node.id)
    {
        // Prefer Wayland-native app_id; fall back to the X11 class for
        // XWayland views. A leaf with neither falls through with
        // app: None — the LLM can still target it by con_id.
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

async fn seed_snapshot(cmd: &mut Connection) -> Result<Snapshot> {
    let tree = cmd
        .get_tree()
        .await
        .map_err(|e| anyhow::anyhow!("sway GET_TREE: {e}"))?;
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
    // Active workspace seeded from a separate query: walking the tree
    // to the focused leaf's workspace ancestor would work, but
    // GET_WORKSPACES is one round-trip and gives an authoritative
    // `focused == true` row that already accounts for multi-output.
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

/// Project a Sway `WindowEvent` into the shared snapshot's
/// `(id, class, title, kind)` tuple and dispatch to
/// [`apply_window_event`]. The Sway-specific bits — `app_id` lookup,
/// XWayland fallback, name-vs-window_properties.title preference —
/// live here; the race rules live in `crate::snapshot`.
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

    fn id(n: u64) -> WindowId {
        WindowId::new(n).expect("test ids are non-zero")
    }

    #[test]
    fn resize_payload_uses_con_id_criteria() {
        // PR 3b: Sway's `[con_id=N]` is a single-clause match — no
        // more `app_id|class` composite, since con_id covers Wayland-
        // native and XWayland views uniformly.
        let p = sway_resize_payload(&id(42), ResizeDir::Grow, 50);
        assert_eq!(p, r#"[con_id="42"] resize grow width 50 px or 0 ppt"#);
    }

    #[test]
    fn resize_payload_renders_id_in_decimal() {
        let p = sway_resize_payload(&id(1234567890), ResizeDir::Shrink, 5);
        assert_eq!(p, r#"[con_id="1234567890"] resize shrink width 5 px or 0 ppt"#);
    }

    #[test]
    fn sway_id_rejects_non_positive() {
        assert!(sway_id(0).is_none());
        assert!(sway_id(-1).is_none());
        assert!(sway_id(-12345).is_none());
        assert_eq!(sway_id(42), WindowId::new(42));
    }

    #[test]
    fn layout_payload_emits_bare_form() {
        // No criteria prefix — Sway speaks the i3 dialect for layout
        // and acts on the focused container.
        for (l, expected) in [
            (Layout::Default, "layout default"),
            (Layout::Tabbed, "layout tabbed"),
            (Layout::SplitH, "layout splith"),
        ] {
            assert_eq!(sway_layout_payload(l), expected);
        }
    }
}

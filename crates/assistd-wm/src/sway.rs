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

use crate::criteria::{escape_for_criteria, format_workspace_target};
use crate::error::ipc_ctx;
use crate::{
    FocusedWindowContext, Layout, OutputInfo, ResizeDir, WindowId, WindowManager, WmError,
    WmResult, Window, WorkspaceId, WorkspaceInfo,
};

/// Cached focus state. Seeded from `get_tree()` + `get_workspaces()` at
/// startup, then updated on each Sway `Window::Focus`, `Window::Title`,
/// `Window::Close`, and `Workspace::Focus` event so synchronous reads
/// don't round-trip the IPC socket.
///
/// The field is named `focused_class` to match the i3 backend, but on
/// Sway it stores `app_id` for Wayland-native windows and falls back to
/// the X11 class only for XWayland views — see the module-level doc.
/// Stored as raw `String` (rather than [`WindowId`]) so it doubles as
/// the source of [`FocusedWindowContext::class`] for the prompt-injection
/// path; conversion to [`WindowId`] happens at the trait method boundary.
#[derive(Default, Clone)]
struct Snapshot {
    focused_class: Option<String>,
    focused_title: Option<String>,
    active_workspace: Option<String>,
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
                                    snap_for_task.write().await.active_workspace = ws;
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
        let cmd = composite_criteria_command(window, "focus");
        self.run(&cmd).await
    }

    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        let target = format_workspace_target(workspace);
        let action = format!("move container to {target}");
        let cmd = composite_criteria_command(window, &action);
        self.run(&cmd).await
    }

    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        Ok(self
            .snapshot
            .read()
            .await
            .focused_class
            .clone()
            .map(WindowId::from))
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

/// Emit a Sway command that runs `action` against either `[app_id="X"]`
/// or `[class="X"]` — Wayland-native windows expose `app_id`, XWayland
/// views expose `class`. Sway processes the two clauses sequentially;
/// each clause that matches 0 windows is a silent success, so the
/// composite is safe even when only one form applies.
fn composite_criteria_command(window: &WindowId, action: &str) -> String {
    let escaped = escape_for_criteria(window.as_str());
    format!(r#"[app_id="{escaped}"] {action}; [class="{escaped}"] {action}"#)
}

/// Format the Sway RUN_COMMAND payload for `resize_width`. Uses the
/// same `app_id|class` composite as `focus` / `move_to_workspace` so a
/// caller-provided id matches whichever form Sway has registered.
fn sway_resize_payload(window: &WindowId, direction: ResizeDir, pixels: u32) -> String {
    let action = format!("resize {} width {} px or 0 ppt", direction.as_str(), pixels);
    composite_criteria_command(window, &action)
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

    if matches!(node.node_type, NodeType::Con | NodeType::FloatingCon) {
        let class = node
            .window_properties
            .as_ref()
            .and_then(|p| p.class.clone());
        let id = node.app_id.clone().or(class);
        if let Some(id) = id {
            let title = node.name.clone().or_else(|| {
                node.window_properties
                    .as_ref()
                    .and_then(|p| p.title.clone())
            });
            out.push(Window {
                id: WindowId::from(id),
                title,
                workspace: next_ws.map(|s| s.to_string()),
            });
        }
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
        focused_class,
        focused_title,
        active_workspace,
    })
}

/// Apply one Sway `Window` event to the cached focus snapshot.
///
/// Race rules (mirror i3 backend):
/// - `Focus` overwrites both class/app_id and title from the new
///   container.
/// - `Title` only takes effect when the event's id matches the
///   currently-focused id — title events from background windows must
///   not overwrite the foreground title.
/// - `Close` clears focus state only when the closed container is the
///   focused one. Workspace is left intact.
async fn handle_window_event(w: &swayipc_async::WindowEvent, snap: &Arc<RwLock<Snapshot>>) {
    let id = w.container.app_id.clone().or_else(|| {
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
    match w.change {
        WindowChange::Focus => {
            let mut s = snap.write().await;
            s.focused_class = id;
            s.focused_title = title;
        }
        WindowChange::Title => {
            let mut s = snap.write().await;
            // Only honor title updates for the currently-focused id.
            if s.focused_class == id && id.is_some() {
                s.focused_title = title;
            }
        }
        WindowChange::Close => {
            let mut s = snap.write().await;
            if s.focused_class == id && id.is_some() {
                s.focused_class = None;
                s.focused_title = None;
            }
        }
        _ => {}
    }
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
    fn composite_criteria_command_emits_app_id_and_class_clauses() {
        let s = composite_criteria_command(&WindowId::from("firefox"), "focus");
        assert_eq!(s, r#"[app_id="firefox"] focus; [class="firefox"] focus"#);
    }

    #[test]
    fn composite_criteria_command_escapes_special_chars() {
        let s = composite_criteria_command(&WindowId::from(r#"a"b\c"#), "focus");
        assert_eq!(s, r#"[app_id="a\"b\\c"] focus; [class="a\"b\\c"] focus"#);
    }

    #[test]
    fn composite_criteria_command_uses_full_action() {
        let s = composite_criteria_command(
            &WindowId::from("kitty"),
            "move container to workspace number 3",
        );
        assert_eq!(
            s,
            r#"[app_id="kitty"] move container to workspace number 3; [class="kitty"] move container to workspace number 3"#
        );
    }

    #[test]
    fn resize_payload_shape_and_escape() {
        // Sway mirrors the focus dispatch's `app_id|class` composite so
        // the resize lands regardless of which identifier the
        // user-supplied id matches.
        let p = sway_resize_payload(&WindowId::from("firefox"), ResizeDir::Grow, 50);
        assert_eq!(
            p,
            r#"[app_id="firefox"] resize grow width 50 px or 0 ppt; [class="firefox"] resize grow width 50 px or 0 ppt"#
        );
    }

    #[test]
    fn resize_payload_escapes_quote_and_backslash() {
        let p = sway_resize_payload(&WindowId::from(r#"a"b\c"#), ResizeDir::Shrink, 5);
        assert_eq!(
            p,
            r#"[app_id="a\"b\\c"] resize shrink width 5 px or 0 ppt; [class="a\"b\\c"] resize shrink width 5 px or 0 ppt"#
        );
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

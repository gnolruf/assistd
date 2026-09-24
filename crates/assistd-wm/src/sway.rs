//! Sway backend for [`crate::WindowManager`], over `swayipc-async`.
//! Same shape as the i3 backend: one command socket, one event socket.
//! `swayipc-async` runs on `async-io`, which costs one extra reactor
//! thread alongside tokio. Views carry either `app_id` (Wayland-native)
//! or `window_properties.class` (XWayland); whichever is present is
//! surfaced as the window's app.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use swayipc_async::{
    Connection, Event, EventStream, EventType, Node, NodeType, WindowChange, WorkspaceChange,
};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::criteria::{format_focus, format_layout, format_move_to_workspace, format_resize_width};
use crate::ipc_backend::{IpcBackend, IpcEvent, IpcProtocol, NodeIdentity, OpLabels, Workspace};
use crate::snapshot::WindowChangeKind;
use crate::{
    FocusedWindowContext, Layout, OutputInfo, PlacementAnchor, PlacementCriteria, Rect, ResizeDir,
    TransportError, Window, WindowEvent, WindowId, WindowManager, WmError, WmResult, WorkspaceId,
    WorkspaceInfo,
};

fn sway_id(raw: i64) -> Option<WindowId> {
    if raw <= 0 {
        return None;
    }
    WindowId::new(raw as u64)
}

/// [`WindowManager`] over a single Sway IPC command socket.
pub struct SwayBackend {
    ipc: Arc<IpcBackend<SwayIpc>>,
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
        let (ipc, supervisor_task) = IpcBackend::start(SwayIpc, shutdown).await?;
        Ok(SwayHandle {
            backend: Arc::new(Self { ipc }),
            supervisor_task,
        })
    }

    async fn outputs(&self, ctx: &'static str) -> WmResult<Vec<swayipc_async::Output>> {
        self.ipc
            .with_conn(ctx, async |conn| conn.get_outputs().await)
            .await
    }
}

#[async_trait]
impl WindowManager for SwayBackend {
    async fn focus(&self, window: &WindowId) -> WmResult<()> {
        self.ipc.run_command(&format_focus(window)).await
    }

    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        self.ipc
            .run_command(&format_move_to_workspace(window, workspace))
            .await
    }

    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        Ok(self.ipc.focused_window().await)
    }

    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        Ok(self.ipc.focused_context().await)
    }

    async fn list_windows(&self) -> WmResult<Vec<Window>> {
        self.ipc.list_windows().await
    }

    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        self.ipc.list_workspaces().await
    }

    async fn resize_width(
        &self,
        window: &WindowId,
        direction: ResizeDir,
        pixels: u32,
    ) -> WmResult<()> {
        self.ipc
            .run_command(&format_resize_width(window, direction, pixels))
            .await
    }

    async fn set_layout(&self, layout: Layout) -> WmResult<()> {
        self.ipc.run_command(&format_layout(layout)).await
    }

    async fn focused_workspace_rect(&self) -> WmResult<Rect> {
        self.ipc.focused_workspace_rect().await
    }

    async fn place_floating(
        &self,
        criteria: &PlacementCriteria,
        anchor: PlacementAnchor,
    ) -> WmResult<()> {
        self.ipc.place_floating(criteria, anchor).await
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

    fn is_connected(&self) -> bool {
        self.ipc.is_connected()
    }
}

struct SwayIpc;

impl IpcProtocol for SwayIpc {
    type Cmd = Connection;
    type Events = EventStream;
    type Node = Node;

    const NAME: &'static str = "sway";
    const OPS: OpLabels = OpLabels {
        run_command: "sway RUN_COMMAND",
        get_tree: "sway GET_TREE",
        get_tree_window_rect: "sway GET_TREE (window rect)",
        get_workspaces: "sway GET_WORKSPACES",
        get_workspaces_focused_rect: "sway GET_WORKSPACES (focused rect)",
    };

    async fn connect(&self) -> WmResult<(Connection, EventStream)> {
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

    async fn next_event(events: &mut EventStream) -> Option<Result<IpcEvent, TransportError>> {
        Some(events.next().await?.map(ipc_event).map_err(Into::into))
    }

    async fn run_command(
        cmd: &mut Connection,
        payload: &str,
    ) -> Result<Result<(), String>, TransportError> {
        let outcomes = cmd.run_command(payload).await?;
        Ok(outcomes
            .into_iter()
            .collect::<Result<(), _>>()
            .map_err(|e| e.to_string()))
    }

    async fn get_tree(cmd: &mut Connection) -> Result<Node, TransportError> {
        Ok(cmd.get_tree().await?)
    }

    async fn get_workspaces(cmd: &mut Connection) -> Result<Vec<Workspace>, TransportError> {
        Ok(cmd
            .get_workspaces()
            .await?
            .into_iter()
            .map(|w| Workspace {
                rect: Rect {
                    x: w.rect.x,
                    y: w.rect.y,
                    width: w.rect.width.max(0) as u32,
                    height: w.rect.height.max(0) as u32,
                },
                info: WorkspaceInfo {
                    num: w.num,
                    name: w.name,
                    focused: w.focused,
                    output: w.output,
                },
            })
            .collect())
    }

    fn children(node: &Node) -> impl Iterator<Item = &Node> {
        node.nodes.iter().chain(node.floating_nodes.iter())
    }

    fn is_focused(node: &Node) -> bool {
        node.focused
    }

    fn identity(node: &Node) -> NodeIdentity {
        let props = node.window_properties.as_ref();
        NodeIdentity {
            id: sway_id(node.id),
            class: node
                .app_id
                .clone()
                .or_else(|| props.and_then(|p| p.class.clone())),
            title: node
                .name
                .clone()
                .or_else(|| props.and_then(|p| p.title.clone())),
        }
    }

    fn window_rect(node: &Node, criteria: &PlacementCriteria) -> Option<Rect> {
        (matches!(node.node_type, NodeType::Con | NodeType::FloatingCon)
            && sway_node_matches(node, criteria))
        .then(|| Rect {
            x: node.rect.x,
            y: node.rect.y,
            width: node.rect.width.max(0) as u32,
            height: node.rect.height.max(0) as u32,
        })
    }

    fn collect_windows(tree: &Node) -> Vec<Window> {
        let mut out = Vec::new();
        collect_windows(tree, None, &mut out);
        out
    }
}

fn ipc_event(event: Event) -> IpcEvent {
    match event {
        Event::Window(w) => IpcEvent::Window {
            focus: focus_change(&w),
            event: window_event_from_sway(&w),
        },
        Event::Workspace(data) if matches!(data.change, WorkspaceChange::Focus) => {
            IpcEvent::WorkspaceFocused(data.current.and_then(|n| n.name))
        }
        _ => IpcEvent::Ignored,
    }
}

fn focus_change(w: &swayipc_async::WindowEvent) -> Option<(WindowChangeKind, NodeIdentity)> {
    let kind = match w.change {
        WindowChange::Focus => WindowChangeKind::Focus,
        WindowChange::Title => WindowChangeKind::Title,
        WindowChange::Close => WindowChangeKind::Close,
        _ => return None,
    };
    Some((kind, SwayIpc::identity(&w.container)))
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

    for child in SwayIpc::children(node) {
        collect_windows(child, next_ws, out);
    }
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

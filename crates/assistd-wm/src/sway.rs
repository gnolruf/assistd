//! Sway IPC protocol over `swayipc-async`. A
//! view's app is its `app_id`, or its XWayland `class` when it has none.

use futures_util::StreamExt;
use swayipc_async::{
    Connection, Event, EventStream, EventType, Node, NodeType, WindowChange, WindowEvent,
    WorkspaceChange,
};

use crate::ipc_backend::{IpcEvent, IpcProtocol, NodeIdentity, OpLabels};
use crate::snapshot::WindowChangeKind;
use crate::{OutputInfo, TransportError, Window, WindowId, WmError, WmResult, WorkspaceInfo};

pub(crate) struct SwayIpc;

impl IpcProtocol for SwayIpc {
    type Cmd = Connection;
    type Events = EventStream;
    type Node = Node;

    const NAME: &'static str = "sway";
    const OPS: OpLabels = OpLabels {
        run_command: "sway RUN_COMMAND",
        get_tree: "sway GET_TREE",
        get_workspaces: "sway GET_WORKSPACES",
        get_outputs: "sway GET_OUTPUTS",
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
            .map_err(|rejected| rejected.to_string()))
    }

    async fn get_tree(cmd: &mut Connection) -> Result<Node, TransportError> {
        Ok(cmd.get_tree().await?)
    }

    async fn get_workspaces(cmd: &mut Connection) -> Result<Vec<WorkspaceInfo>, TransportError> {
        Ok(cmd
            .get_workspaces()
            .await?
            .into_iter()
            .map(|workspace| WorkspaceInfo {
                num: workspace.num,
                name: workspace.name,
                focused: workspace.focused,
                output: workspace.output,
            })
            .collect())
    }

    async fn get_outputs(cmd: &mut Connection) -> Result<Vec<OutputInfo>, TransportError> {
        Ok(cmd
            .get_outputs()
            .await?
            .into_iter()
            .map(|output| OutputInfo {
                name: output.name,
                active: output.active,
                primary: output.primary,
                current_mode: output.current_mode.map(|mode| {
                    (
                        u32::try_from(mode.width).unwrap_or(0),
                        u32::try_from(mode.height).unwrap_or(0),
                        u32::try_from(mode.refresh).unwrap_or(0),
                    )
                }),
                scale: output.scale,
                focused_workspace: output.current_workspace,
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
                .or_else(|| props.and_then(|props| props.class.clone())),
            title: node
                .name
                .clone()
                .or_else(|| props.and_then(|props| props.title.clone())),
        }
    }

    fn collect_windows(tree: &Node) -> Vec<Window> {
        let mut windows = Vec::new();
        collect_windows(tree, None, &mut windows);
        windows
    }
}

fn ipc_event(event: Event) -> IpcEvent {
    match event {
        Event::Window(window) => focus_change(&window).unwrap_or(IpcEvent::Ignored),
        Event::Workspace(data) if matches!(data.change, WorkspaceChange::Focus) => {
            IpcEvent::WorkspaceFocused(data.current.and_then(|node| node.name))
        }
        _ => IpcEvent::Ignored,
    }
}

fn focus_change(window: &WindowEvent) -> Option<IpcEvent> {
    let kind = match window.change {
        WindowChange::Focus => WindowChangeKind::Focus,
        WindowChange::Title => WindowChangeKind::Title,
        WindowChange::Close => WindowChangeKind::Close,
        _ => return None,
    };
    Some(IpcEvent::WindowChanged(
        kind,
        SwayIpc::identity(&window.container),
    ))
}

fn collect_windows(node: &Node, parent_workspace: Option<&str>, out: &mut Vec<Window>) {
    let workspace = if matches!(node.node_type, NodeType::Workspace) {
        node.name.as_deref()
    } else {
        parent_workspace
    };

    if is_view(node)
        && let Some(id) = sway_id(node.id)
    {
        let props = node.window_properties.as_ref();
        let class = props.and_then(|props| props.class.clone());
        let app = node.app_id.clone().or(class);
        let title = node
            .name
            .clone()
            .or_else(|| props.and_then(|props| props.title.clone()));
        out.push(Window {
            id,
            app,
            title,
            workspace: workspace.map(str::to_string),
        });
    }

    for child in SwayIpc::children(node) {
        collect_windows(child, workspace, out);
    }
}

/// A `Con` holding a client, as opposed to a split container of other cons.
fn is_view(node: &Node) -> bool {
    matches!(node.node_type, NodeType::Con | NodeType::FloatingCon)
        && (node.pid.is_some() || SwayIpc::children(node).next().is_none())
}

fn sway_id(raw: i64) -> Option<WindowId> {
    u64::try_from(raw).ok().and_then(WindowId::new)
}

#[cfg(test)]
mod tests;

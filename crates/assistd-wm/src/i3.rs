//! i3 backend for [`crate::WindowManager`], over `tokio-i3ipc`. One
//! socket serves commands; a second feeds the event stream that keeps
//! the focus snapshot current.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_i3ipc::{
    I3,
    event::{Event, Subscribe, WindowChange, WindowData, WorkspaceChange},
    reply,
};

use crate::criteria::{format_focus, format_layout, format_move_to_workspace, format_resize_width};
use crate::ipc_backend::{IpcBackend, IpcEvent, IpcProtocol, NodeIdentity, OpLabels};
use crate::snapshot::WindowChangeKind;
use crate::{
    FocusedWindowContext, Layout, ResizeDir, TransportError, Window, WindowId, WindowManager,
    WmError, WmResult, WorkspaceId, WorkspaceInfo,
};

/// [`WindowManager`] over a single i3 IPC command socket.
#[derive(Debug)]
pub struct I3Backend {
    ipc: Arc<IpcBackend<I3Ipc>>,
}

/// The backend plus its supervisor task, returned by [`I3Backend::start`].
#[derive(Debug)]
pub struct I3Handle {
    pub backend: Arc<I3Backend>,
    pub(crate) supervisor_task: JoinHandle<()>,
}

impl I3Handle {
    /// Awaits the supervisor task. Flip the shutdown watch first or
    /// this blocks until the socket drops.
    pub async fn shutdown(self) {
        let _ = self.supervisor_task.await;
    }
}

impl I3Backend {
    /// Connect to i3 and spawn the reconnecting supervisor. Errors only
    /// when the initial connect fails.
    pub async fn start(shutdown: watch::Receiver<bool>) -> WmResult<I3Handle> {
        let (ipc, supervisor_task) = IpcBackend::start(I3Ipc, shutdown).await?;
        Ok(I3Handle {
            backend: Arc::new(Self { ipc }),
            supervisor_task,
        })
    }
}

#[async_trait]
impl WindowManager for I3Backend {
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

    fn is_connected(&self) -> bool {
        self.ipc.is_connected()
    }
}

struct I3Ipc;

impl IpcProtocol for I3Ipc {
    type Cmd = I3;
    type Events = BoxStream<'static, std::io::Result<Event>>;
    type Node = reply::Node;

    const NAME: &'static str = "i3";
    const OPS: OpLabels = OpLabels {
        run_command: "i3 RUN_COMMAND",
        get_tree: "i3 GET_TREE",
        get_workspaces: "i3 GET_WORKSPACES",
    };

    async fn connect(&self) -> WmResult<(I3, Self::Events)> {
        let cmd = I3::connect()
            .await
            .map_err(|e| WmError::ipc("connect to i3 IPC (cmd socket)", e))?;
        let mut events_conn = I3::connect()
            .await
            .map_err(|e| WmError::ipc("connect to i3 IPC (events socket)", e))?;
        events_conn
            .subscribe([Subscribe::Window, Subscribe::Workspace])
            .await
            .map_err(|e| WmError::ipc("subscribe to i3 window+workspace events", e))?;
        Ok((cmd, events_conn.listen().boxed()))
    }

    async fn next_event(events: &mut Self::Events) -> Option<Result<IpcEvent, TransportError>> {
        Some(events.next().await?.map(ipc_event).map_err(Into::into))
    }

    async fn run_command(
        cmd: &mut I3,
        payload: &str,
    ) -> Result<Result<(), String>, TransportError> {
        let results = cmd.run_command(payload).await?;
        Ok(match results.into_iter().find(|result| !result.success) {
            Some(failed) => Err(failed.error.unwrap_or_else(|| "unknown error".into())),
            None => Ok(()),
        })
    }

    async fn get_tree(cmd: &mut I3) -> Result<reply::Node, TransportError> {
        Ok(cmd.get_tree().await?)
    }

    async fn get_workspaces(cmd: &mut I3) -> Result<Vec<WorkspaceInfo>, TransportError> {
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

    fn children(node: &reply::Node) -> impl Iterator<Item = &reply::Node> {
        node.nodes.iter().chain(node.floating_nodes.iter())
    }

    fn is_focused(node: &reply::Node) -> bool {
        node.focused
    }

    fn identity(node: &reply::Node) -> NodeIdentity {
        NodeIdentity {
            id: WindowId::new(node.id as u64),
            class: node
                .window_properties
                .as_ref()
                .and_then(|props| props.class.clone()),
            title: node.name.clone(),
        }
    }

    fn collect_windows(tree: &reply::Node) -> Vec<Window> {
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

fn focus_change(window: &WindowData) -> Option<IpcEvent> {
    let kind = match window.change {
        WindowChange::Focus => WindowChangeKind::Focus,
        WindowChange::Title => WindowChangeKind::Title,
        WindowChange::Close => WindowChangeKind::Close,
        _ => return None,
    };
    Some(IpcEvent::WindowChanged(
        kind,
        I3Ipc::identity(&window.container),
    ))
}

fn collect_windows(node: &reply::Node, parent_workspace: Option<&str>, out: &mut Vec<Window>) {
    let workspace = if matches!(node.node_type, reply::NodeType::Workspace) {
        node.name.as_deref()
    } else {
        parent_workspace
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
            workspace: workspace.map(str::to_string),
        });
    }

    for child in I3Ipc::children(node) {
        collect_windows(child, workspace, out);
    }
}

//! Connection machinery shared by the i3 and Sway backends: a timed command socket, an event
//! socket feeding the focus snapshot, and a reconnecting supervisor over [`IpcProtocol`].

use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use assistd_utils::backoff::backoff_delay;
use async_trait::async_trait;
use tokio::sync::{Mutex, Notify, RwLock, watch};
use tokio::task::JoinHandle;

use crate::criteria::{
    format_exec, format_focus, format_layout, format_move_to_workspace, format_resize_width,
};
use crate::snapshot::{self, Snapshot, WindowChangeKind};
use crate::{
    ExecCommand, FocusedWindowContext, Layout, NoWindowManager, OutputInfo, ResizeDir,
    TransportError, WM_IPC_TIMEOUT, Window, WindowId, WindowManager, WmError, WmHandle, WmResult,
    WorkspaceId, WorkspaceInfo,
};

/// Labels for the IPC calls the shared machinery makes, reported as the
/// `op` of [`WmError::Ipc`].
pub(crate) struct OpLabels {
    pub run_command: &'static str,
    pub get_tree: &'static str,
    pub get_workspaces: &'static str,
    pub get_outputs: &'static str,
}

/// The focus-relevant identity of a tree node.
#[derive(Default)]
pub(crate) struct NodeIdentity {
    pub id: Option<WindowId>,
    pub class: Option<String>,
    pub title: Option<String>,
}

/// A compositor event projected onto what the shared machinery acts on.
pub(crate) enum IpcEvent {
    /// A focus, title, or close change to apply to the snapshot.
    WindowChanged(WindowChangeKind, NodeIdentity),
    WorkspaceFocused(Option<String>),
    Ignored,
}

/// What differs between the i3 and Sway IPC client crates.
pub(crate) trait IpcProtocol: Send + Sync + Sized + 'static {
    type Cmd: Send + 'static;
    type Events: Send + 'static;
    type Node: Send;

    /// Compositor name used in log messages.
    const NAME: &'static str;
    const OPS: OpLabels;

    /// Open a command connection and an event stream subscribed to
    /// window and workspace events.
    fn connect(&self) -> impl Future<Output = WmResult<(Self::Cmd, Self::Events)>> + Send;

    /// `None` when the stream has ended. Must be cancel-safe.
    fn next_event(
        events: &mut Self::Events,
    ) -> impl Future<Output = Option<Result<IpcEvent, TransportError>>> + Send;

    /// The inner error is the compositor's message for the first
    /// command it rejected.
    fn run_command(
        cmd: &mut Self::Cmd,
        payload: &str,
    ) -> impl Future<Output = Result<Result<(), String>, TransportError>> + Send;

    fn get_tree(
        cmd: &mut Self::Cmd,
    ) -> impl Future<Output = Result<Self::Node, TransportError>> + Send;

    fn get_workspaces(
        cmd: &mut Self::Cmd,
    ) -> impl Future<Output = Result<Vec<WorkspaceInfo>, TransportError>> + Send;

    fn get_outputs(
        cmd: &mut Self::Cmd,
    ) -> impl Future<Output = Result<Vec<OutputInfo>, TransportError>> + Send;

    /// Tiled children, then floating children.
    fn children(node: &Self::Node) -> impl Iterator<Item = &Self::Node>;

    fn is_focused(node: &Self::Node) -> bool;

    fn identity(node: &Self::Node) -> NodeIdentity;

    fn collect_windows(tree: &Self::Node) -> Vec<Window>;
}

/// State shared between a backend's IPC calls and its supervisor task.
pub(crate) struct IpcBackend<P: IpcProtocol> {
    protocol: P,
    cmd: Mutex<Option<P::Cmd>>,
    /// Mirrors whether `cmd` holds a connection, readable without the lock.
    connected: AtomicBool,
    snapshot: RwLock<Snapshot>,
    reconnect: Notify,
}

impl<P: IpcProtocol> fmt::Debug for IpcBackend<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IpcBackend")
            .field("connected", &self.connected)
            .finish_non_exhaustive()
    }
}

impl<P: IpcProtocol> IpcBackend<P> {
    /// Connect, seed the focus snapshot, and spawn the reconnecting
    /// supervisor. Errors when the initial connect fails; a connect that
    /// exceeds [`WM_IPC_TIMEOUT`] starts disconnected and the supervisor retries.
    async fn start(
        protocol: P,
        shutdown: watch::Receiver<bool>,
    ) -> WmResult<(Arc<Self>, JoinHandle<()>)> {
        let (cmd, events, initial) = match connect_and_seed(&protocol).await {
            Ok(Connection {
                cmd,
                events,
                seeded,
            }) => (Some(cmd), Some(events), seeded_or_empty::<P>(seeded)),
            Err(WmError::Timeout(after)) => {
                tracing::warn!(
                    target: "assistd::wm",
                    "{} did not answer within {after:?}; starting disconnected",
                    P::NAME
                );
                (None, None, Snapshot::default())
            }
            Err(e) => return Err(e),
        };
        let backend = Arc::new(Self {
            protocol,
            connected: AtomicBool::new(cmd.is_some()),
            cmd: Mutex::new(cmd),
            snapshot: RwLock::new(initial),
            reconnect: Notify::new(),
        });
        let supervisor_task = tokio::spawn(supervise(backend.clone(), events, shutdown));
        Ok((backend, supervisor_task))
    }

    fn set_conn(&self, slot: &mut Option<P::Cmd>, conn: Option<P::Cmd>) {
        self.connected.store(conn.is_some(), Ordering::Relaxed);
        *slot = conn;
    }

    /// Run one IPC call on the command socket under [`WM_IPC_TIMEOUT`].
    /// A timeout or transport error drops the connection and wakes the
    /// supervisor to reconnect.
    async fn with_conn<T, E: Into<TransportError>>(
        &self,
        op_label: &'static str,
        op: impl AsyncFnOnce(&mut P::Cmd) -> Result<T, E>,
    ) -> WmResult<T> {
        let mut guard = self.cmd.lock().await;
        let conn = guard.as_mut().ok_or(WmError::Disconnected)?;
        let outcome = tokio::time::timeout(WM_IPC_TIMEOUT, op(conn)).await;
        let err = match outcome {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(e)) => WmError::ipc(op_label, e),
            Err(_) => WmError::Timeout(WM_IPC_TIMEOUT),
        };
        self.set_conn(&mut guard, None);
        self.reconnect.notify_one();
        Err(err)
    }

    async fn run_command(&self, payload: &str) -> WmResult<()> {
        self.with_conn(P::OPS.run_command, async |conn| {
            P::run_command(conn, payload).await
        })
        .await?
        .map_err(|e| WmError::Rejected(format!("{payload}: {e}")))
    }

    /// Drive one event stream. Returns `true` when the caller should
    /// reconnect, `false` on shutdown.
    async fn drive_events(
        &self,
        mut events: P::Events,
        shutdown: &mut watch::Receiver<bool>,
    ) -> bool {
        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() { return false; }
                }
                () = self.reconnect.notified() => {
                    return true;
                }
                next = P::next_event(&mut events) => {
                    match next {
                        Some(Ok(event)) => self.apply_event(event).await,
                        Some(Err(e)) => {
                            tracing::warn!(
                                target: "assistd::wm",
                                "{} event stream error: {e}",
                                P::NAME
                            );
                            return true;
                        }
                        None => return true,
                    }
                }
            }
        }
    }

    async fn apply_event(&self, event: IpcEvent) {
        match event {
            IpcEvent::WindowChanged(kind, NodeIdentity { id, class, title }) => {
                snapshot::apply_window_event(&self.snapshot, kind, id, class, title).await;
            }
            IpcEvent::WorkspaceFocused(name) => {
                snapshot::apply_workspace_focus(&self.snapshot, name).await;
            }
            IpcEvent::Ignored => {}
        }
    }
}

#[async_trait]
impl<P: IpcProtocol> WindowManager for IpcBackend<P> {
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

    async fn list_windows(&self) -> WmResult<Vec<Window>> {
        let tree = self
            .with_conn(P::OPS.get_tree, async |conn| P::get_tree(conn).await)
            .await?;
        Ok(P::collect_windows(&tree))
    }

    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        self.with_conn(P::OPS.get_workspaces, async |conn| {
            P::get_workspaces(conn).await
        })
        .await
    }

    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        Ok(snapshot::read_focused_context(&self.snapshot).await)
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

    async fn exec(&self, workspace: Option<&WorkspaceId>, command: &ExecCommand) -> WmResult<()> {
        self.run_command(&format_exec(workspace, command)).await
    }

    async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
        self.with_conn(P::OPS.get_outputs, async |conn| P::get_outputs(conn).await)
            .await
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }
}

/// A fresh command connection, its event stream, and the result of seeding
/// the focus snapshot over it.
struct Connection<P: IpcProtocol> {
    cmd: P::Cmd,
    events: P::Events,
    seeded: WmResult<Snapshot>,
}

/// Start `protocol`'s backend and supervisor; [`NoWindowManager`] and no
/// handle when the first connect is refused.
pub(crate) async fn start_or_disconnected<P: IpcProtocol>(
    protocol: P,
    shutdown: watch::Receiver<bool>,
) -> (Arc<dyn WindowManager>, Option<WmHandle>) {
    match IpcBackend::start(protocol, shutdown).await {
        Ok((backend, supervisor_task)) => {
            if backend.is_connected() {
                tracing::info!(target: "assistd::wm", "{} backend connected", P::NAME);
            }
            (backend, Some(WmHandle { supervisor_task }))
        }
        Err(e) => {
            tracing::warn!(
                target: "assistd::wm",
                "{} backend unavailable ({e:#}); window operations disabled",
                P::NAME
            );
            (Arc::new(NoWindowManager), None)
        }
    }
}

async fn supervise<P: IpcProtocol>(
    backend: Arc<IpcBackend<P>>,
    initial_events: Option<P::Events>,
    mut shutdown: watch::Receiver<bool>,
) {
    let name = P::NAME;
    if let Some(events) = initial_events
        && !backend.drive_events(events, &mut shutdown).await
    {
        tracing::info!(
            target: "assistd::wm",
            "{name} supervisor exited (shutdown during initial events stream)"
        );
        return;
    }

    let mut attempt: u32 = 0;
    loop {
        backend.set_conn(&mut *backend.cmd.lock().await, None);
        tracing::warn!(
            target: "assistd::wm",
            "{name} disconnected; reconnecting (attempt {})",
            attempt + 1
        );

        let reconnected = tokio::select! {
            _ = shutdown.wait_for(|stop| *stop) => {
                tracing::info!(
                    target: "assistd::wm",
                    "{name} supervisor exited (shutdown while reconnecting)"
                );
                return;
            }
            reconnected = reconnect_after_backoff(&backend.protocol, attempt) => reconnected,
        };

        match reconnected {
            Ok(Connection {
                cmd,
                events,
                seeded,
            }) => {
                if let Ok(seeded) = seeded {
                    *backend.snapshot.write().await = seeded;
                }
                backend.set_conn(&mut *backend.cmd.lock().await, Some(cmd));
                attempt = 0;
                tracing::info!(target: "assistd::wm", "{name} backend reconnected");
                if !backend.drive_events(events, &mut shutdown).await {
                    tracing::info!(
                        target: "assistd::wm",
                        "{name} supervisor exited (shutdown during events stream)"
                    );
                    return;
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "assistd::wm",
                    "{name} reconnect failed: {e}; will retry after backoff"
                );
                attempt = attempt.saturating_add(1);
            }
        }
    }
}

async fn reconnect_after_backoff<P: IpcProtocol>(
    protocol: &P,
    attempt: u32,
) -> WmResult<Connection<P>> {
    tokio::time::sleep(backoff_delay(attempt)).await;
    connect_and_seed(protocol).await
}

/// Connect and seed under one [`WM_IPC_TIMEOUT`] deadline, so a compositor
/// that accepts the socket but never answers fails like a refused connect.
async fn connect_and_seed<P: IpcProtocol>(protocol: &P) -> WmResult<Connection<P>> {
    let attempt = async {
        let (mut cmd, events) = protocol.connect().await?;
        let seeded = seed_snapshot::<P>(&mut cmd).await;
        Ok(Connection {
            cmd,
            events,
            seeded,
        })
    };
    tokio::time::timeout(WM_IPC_TIMEOUT, attempt)
        .await
        .unwrap_or(Err(WmError::Timeout(WM_IPC_TIMEOUT)))
}

fn seeded_or_empty<P: IpcProtocol>(seeded: WmResult<Snapshot>) -> Snapshot {
    seeded.unwrap_or_else(|e| {
        tracing::warn!(
            target: "assistd::wm",
            "{} seed_snapshot failed: {e:#}; starting with empty focus state",
            P::NAME
        );
        Snapshot::default()
    })
}

async fn seed_snapshot<P: IpcProtocol>(cmd: &mut P::Cmd) -> WmResult<Snapshot> {
    let tree = P::get_tree(cmd)
        .await
        .map_err(|e| WmError::ipc(P::OPS.get_tree, e))?;
    let NodeIdentity { id, class, title } = find_map_node::<P, _>(&tree, &|node| {
        P::is_focused(node).then(|| P::identity(node))
    })
    .unwrap_or_default();

    let active_workspace = match P::get_workspaces(cmd).await {
        Ok(workspaces) => workspaces
            .into_iter()
            .find(|workspace| workspace.focused)
            .map(|workspace| workspace.name),
        Err(e) => {
            tracing::warn!(
                target: "assistd::wm",
                "{} GET_WORKSPACES on seed failed: {e:#}",
                P::NAME
            );
            None
        }
    };
    Ok(Snapshot {
        focused_id: id,
        focused_class: class,
        focused_title: title,
        active_workspace,
    })
}

/// Pre-order search of the tree for the first node `visit` maps to `Some`.
fn find_map_node<P: IpcProtocol, T>(
    node: &P::Node,
    visit: &impl Fn(&P::Node) -> Option<T>,
) -> Option<T> {
    visit(node).or_else(|| P::children(node).find_map(|child| find_map_node::<P, T>(child, visit)))
}

#[cfg(test)]
mod tests;

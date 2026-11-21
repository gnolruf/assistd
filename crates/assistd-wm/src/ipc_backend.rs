//! Connection machinery shared by the i3 and Sway backends: a timed command socket, an event
//! socket feeding the focus snapshot, and a reconnecting supervisor over [`IpcProtocol`].

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use assistd_utils::backoff::backoff_delay;
use async_trait::async_trait;
use tokio::sync::{Mutex, Notify, RwLock, watch};
use tokio::task::JoinHandle;

use crate::criteria::{
    format_exec, format_focus, format_layout, format_move_to_workspace, format_resize_width,
};
use crate::snapshot::{self, Snapshot, WindowChangeKind};
use crate::{
    ExecCommand, FocusedWindowContext, Layout, OutputInfo, ResizeDir, TransportError,
    WM_IPC_TIMEOUT, Window, WindowId, WindowManager, WmError, WmHandle, WmResult, WorkspaceId,
    WorkspaceInfo,
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
    /// The live command connection; calls on it take turns.
    cmd: parking_lot::RwLock<Option<Arc<Mutex<P::Cmd>>>>,
    snapshot: RwLock<Snapshot>,
    reconnect: Notify,
}

impl<P: IpcProtocol> fmt::Debug for IpcBackend<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IpcBackend")
            .field("connected", &self.is_connected())
            .finish_non_exhaustive()
    }
}

impl<P: IpcProtocol> IpcBackend<P> {
    /// Connect, seed the focus snapshot, and spawn the reconnecting
    /// supervisor. A failed initial connect starts disconnected and the
    /// supervisor retries.
    async fn start(protocol: P, shutdown: watch::Receiver<bool>) -> (Arc<Self>, JoinHandle<()>) {
        let (cmd, events, initial) = match connect_and_seed(&protocol).await {
            Ok(Connection {
                cmd,
                events,
                seeded,
            }) => (Some(cmd), Some(events), seeded_or_empty::<P>(seeded)),
            Err(e) => {
                tracing::warn!(
                    target: "assistd::wm",
                    "{} backend unavailable ({e:#}); starting disconnected",
                    P::NAME
                );
                (None, None, Snapshot::default())
            }
        };
        let backend = Arc::new(Self {
            protocol,
            cmd: parking_lot::RwLock::new(cmd.map(share_conn)),
            snapshot: RwLock::new(initial),
            reconnect: Notify::new(),
        });
        let supervisor_task = tokio::spawn(supervise(backend.clone(), events, shutdown));
        (backend, supervisor_task)
    }

    fn set_conn(&self, conn: Option<P::Cmd>) {
        *self.cmd.write() = conn.map(share_conn);
    }

    fn is_current(&self, conn: &Arc<Mutex<P::Cmd>>) -> bool {
        self.cmd
            .read()
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, conn))
    }

    /// Drop `failed` and wake the supervisor, unless it was already replaced.
    fn drop_failed_conn(&self, failed: &Arc<Mutex<P::Cmd>>) {
        let mut current = self.cmd.write();
        if current
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, failed))
        {
            *current = None;
            self.reconnect.notify_one();
        }
    }

    /// Run one IPC call on the command socket under [`WM_IPC_TIMEOUT`].
    /// Anything but a completed exchange (a timeout, a transport error, a
    /// panic, or this future being dropped) drops the connection and wakes
    /// the supervisor to reconnect.
    async fn with_conn<T, E: Into<TransportError>>(
        &self,
        op_label: &'static str,
        op: impl AsyncFnOnce(&mut P::Cmd) -> Result<T, E>,
    ) -> WmResult<T> {
        let conn = self.cmd.read().clone().ok_or(WmError::Disconnected)?;
        let mut cmd = conn.lock().await;
        if !self.is_current(&conn) {
            return Err(WmError::Disconnected);
        }
        let mut exchange = ExchangeGuard {
            backend: self,
            conn: &conn,
            completed: false,
        };
        match tokio::time::timeout(WM_IPC_TIMEOUT, op(&mut cmd)).await {
            Ok(Ok(value)) => {
                exchange.completed = true;
                Ok(value)
            }
            Ok(Err(e)) => Err(WmError::ipc(op_label, e)),
            Err(_) => Err(WmError::Timeout(WM_IPC_TIMEOUT)),
        }
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
        self.cmd.read().is_some()
    }
}

/// Drops the connection it watches unless the exchange on it completed,
/// so a request left half-written or a reply left half-read is never reused.
struct ExchangeGuard<'a, P: IpcProtocol> {
    backend: &'a IpcBackend<P>,
    conn: &'a Arc<Mutex<P::Cmd>>,
    completed: bool,
}

impl<P: IpcProtocol> Drop for ExchangeGuard<'_, P> {
    fn drop(&mut self) {
        if !self.completed {
            self.backend.drop_failed_conn(self.conn);
        }
    }
}

/// A fresh command connection, its event stream, and the result of seeding
/// the focus snapshot over it.
struct Connection<P: IpcProtocol> {
    cmd: P::Cmd,
    events: P::Events,
    seeded: WmResult<Snapshot>,
}

/// Start `protocol`'s backend and supervisor, disconnected when the first
/// connect fails.
pub(crate) async fn start_or_disconnected<P: IpcProtocol>(
    protocol: P,
    shutdown: watch::Receiver<bool>,
) -> (Arc<dyn WindowManager>, Option<WmHandle>) {
    let (backend, supervisor_task) = IpcBackend::start(protocol, shutdown).await;
    if backend.is_connected() {
        tracing::info!(target: "assistd::wm", "{} backend connected", P::NAME);
    }
    (backend, Some(WmHandle { supervisor_task }))
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
        backend.set_conn(None);
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
                backend.set_conn(Some(cmd));
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

fn share_conn<C>(cmd: C) -> Arc<Mutex<C>> {
    Arc::new(Mutex::new(cmd))
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
mod tests {
    use std::time::Duration;

    use tokio::sync::mpsc;

    use super::*;

    type FakeEvents = mpsc::UnboundedReceiver<IpcEvent>;

    /// Each `connect` takes the next event stream the test hands over,
    /// failing once the test drops its sender.
    struct FakeProtocol {
        connections: Mutex<mpsc::UnboundedReceiver<FakeEvents>>,
    }

    impl IpcProtocol for FakeProtocol {
        type Cmd = ();
        type Events = FakeEvents;
        type Node = ();

        const NAME: &'static str = "fake";
        const OPS: OpLabels = OpLabels {
            run_command: "fake RUN_COMMAND",
            get_tree: "fake GET_TREE",
            get_workspaces: "fake GET_WORKSPACES",
            get_outputs: "fake GET_OUTPUTS",
        };

        async fn connect(&self) -> WmResult<((), FakeEvents)> {
            let events = self.connections.lock().await.recv().await;
            events
                .map(|events| ((), events))
                .ok_or(WmError::Disconnected)
        }

        async fn next_event(events: &mut FakeEvents) -> Option<Result<IpcEvent, TransportError>> {
            events.recv().await.map(Ok)
        }

        async fn run_command(
            (): &mut (),
            payload: &str,
        ) -> Result<Result<(), String>, TransportError> {
            match payload {
                "break" => Err(std::io::Error::other("socket closed").into()),
                "hang" => std::future::pending().await,
                _ => Ok(Ok(())),
            }
        }

        async fn get_tree((): &mut ()) -> Result<(), TransportError> {
            Ok(())
        }

        async fn get_workspaces((): &mut ()) -> Result<Vec<WorkspaceInfo>, TransportError> {
            Ok(Vec::new())
        }

        async fn get_outputs((): &mut ()) -> Result<Vec<OutputInfo>, TransportError> {
            Ok(Vec::new())
        }

        fn children((): &()) -> impl Iterator<Item = &()> {
            std::iter::empty()
        }

        fn is_focused((): &()) -> bool {
            false
        }

        fn identity((): &()) -> NodeIdentity {
            NodeIdentity::default()
        }

        fn collect_windows((): &()) -> Vec<Window> {
            Vec::new()
        }
    }

    async fn wait_until(what: &str, condition: impl Fn() -> bool) {
        tokio::time::timeout(Duration::from_secs(120), async {
            while !condition() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting until {what}"));
    }

    #[tokio::test(start_paused = true)]
    async fn is_connected_tracks_disconnect_and_reconnect() {
        let (connections_tx, connections_rx) = mpsc::unbounded_channel();
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        connections_tx.send(events_rx).unwrap();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let protocol = FakeProtocol {
            connections: Mutex::new(connections_rx),
        };
        let (backend, supervisor) = IpcBackend::start(protocol, shutdown_rx).await;
        assert!(backend.is_connected());

        drop(events_tx);
        wait_until("the event stream drop is noticed", || {
            !backend.is_connected()
        })
        .await;
        assert!(matches!(
            backend.run_command("nop").await,
            Err(WmError::Disconnected)
        ));

        let (_second_stream, events_rx) = mpsc::unbounded_channel();
        connections_tx.send(events_rx).unwrap();
        wait_until("the supervisor reconnects", || backend.is_connected()).await;
        backend.run_command("nop").await.unwrap();

        assert!(matches!(
            backend.run_command("break").await,
            Err(WmError::Ipc { .. })
        ));
        assert!(!backend.is_connected());

        let (_third_stream, events_rx) = mpsc::unbounded_channel();
        connections_tx.send(events_rx).unwrap();
        wait_until("the supervisor reconnects again", || backend.is_connected()).await;

        shutdown_tx.send(true).unwrap();
        supervisor.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn cancelled_call_drops_the_connection() {
        let (connections_tx, connections_rx) = mpsc::unbounded_channel();
        let (_events_tx, events_rx) = mpsc::unbounded_channel();
        connections_tx.send(events_rx).unwrap();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let protocol = FakeProtocol {
            connections: Mutex::new(connections_rx),
        };
        let (backend, supervisor) = IpcBackend::start(protocol, shutdown_rx).await;

        let cancelled =
            tokio::time::timeout(Duration::from_millis(1), backend.run_command("hang")).await;
        assert!(cancelled.is_err());
        assert!(!backend.is_connected());

        let (_second_stream, events_rx) = mpsc::unbounded_channel();
        connections_tx.send(events_rx).unwrap();
        wait_until("the supervisor reconnects", || backend.is_connected()).await;
        backend.run_command("nop").await.unwrap();

        shutdown_tx.send(true).unwrap();
        supervisor.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn unanswered_connect_starts_disconnected_and_shutdown_interrupts_reconnect() {
        let (connections_tx, connections_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let protocol = FakeProtocol {
            connections: Mutex::new(connections_rx),
        };
        let (backend, supervisor) = IpcBackend::start(protocol, shutdown_rx).await;
        assert!(!backend.is_connected());

        let (events_tx, events_rx) = mpsc::unbounded_channel();
        connections_tx.send(events_rx).unwrap();
        wait_until("the supervisor connects", || backend.is_connected()).await;

        drop(events_tx);
        wait_until("the event stream drop is noticed", || {
            !backend.is_connected()
        })
        .await;
        tokio::time::sleep(Duration::from_secs(3)).await;

        shutdown_tx.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(1), supervisor)
            .await
            .expect("shutdown interrupts a hung reconnect")
            .unwrap();
    }
}

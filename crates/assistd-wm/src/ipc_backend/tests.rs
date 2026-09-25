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
        get_tree_window_rect: "fake GET_TREE (window rect)",
        get_workspaces: "fake GET_WORKSPACES",
        get_workspaces_focused_rect: "fake GET_WORKSPACES (focused rect)",
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

    async fn run_command(_: &mut (), payload: &str) -> Result<Result<(), String>, TransportError> {
        match payload {
            "break" => Err(std::io::Error::other("socket closed").into()),
            _ => Ok(Ok(())),
        }
    }

    async fn get_tree(_: &mut ()) -> Result<(), TransportError> {
        Ok(())
    }

    async fn get_workspaces(_: &mut ()) -> Result<Vec<Workspace>, TransportError> {
        Ok(Vec::new())
    }

    fn children(_: &()) -> impl Iterator<Item = &()> {
        std::iter::empty()
    }

    fn is_focused(_: &()) -> bool {
        false
    }

    fn identity(_: &()) -> NodeIdentity {
        NodeIdentity::default()
    }

    fn window_rect(_: &(), _: &PlacementCriteria) -> Option<Rect> {
        None
    }

    fn collect_windows(_: &()) -> Vec<Window> {
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
    let (backend, supervisor) = IpcBackend::start(protocol, shutdown_rx).await.unwrap();
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

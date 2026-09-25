use serde_json::json;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::*;

struct Harness {
    input: mpsc::UnboundedSender<DriverInput>,
    state: watch::Receiver<PopupState>,
    place: mpsc::UnboundedReceiver<PlaceRequest>,
    driver: JoinHandle<()>,
}

impl Harness {
    fn spawn(cfg: TrayPopupConfig) -> Self {
        let (state_tx, state) = watch::channel(PopupState::default());
        let (input, input_rx) = mpsc::unbounded_channel();
        let (place_tx, place) = mpsc::unbounded_channel();
        let driver = tokio::spawn(drive_visibility(
            state_tx,
            input_rx,
            place_tx,
            cfg,
            IpcClient::with_path("/tmp/assistd-popup-tests-nonexistent.sock"),
        ));
        Self {
            input,
            state,
            place,
            driver,
        }
    }

    fn send(&self, input: DriverInput) {
        self.input.send(input).expect("send");
    }

    fn send_event(&self, event: Event) {
        self.send(DriverInput::Event(Box::new(event)));
    }

    async fn next_state(&mut self) -> PopupState {
        self.state.changed().await.expect("watch sender alive");
        self.state.borrow_and_update().clone()
    }

    fn visible(&self) -> bool {
        self.state.borrow().visible
    }

    async fn shutdown(self) {
        self.send(DriverInput::Shutdown);
        self.driver.await.expect("driver join");
    }
}

fn cfg(auto_hide_ms: u64) -> TrayPopupConfig {
    TrayPopupConfig {
        auto_hide_ms,
        ..TrayPopupConfig::default()
    }
}

fn last_delta(id: &str, text: &str) -> Event {
    Event::LastDelta {
        id: id.into(),
        text: text.into(),
    }
}

fn speaking(id: &str, speaking: bool) -> Event {
    Event::SpeakingState {
        id: id.into(),
        speaking,
    }
}

#[tokio::test]
async fn show_makes_state_visible_and_requests_placement() {
    let mut h = Harness::spawn(cfg(60_000));
    h.send(DriverInput::Show);
    assert!(h.next_state().await.visible);
    assert_eq!(
        h.place.recv().await,
        Some(PlaceRequest),
        "show triggers placement"
    );
    h.shutdown().await;
}

#[tokio::test]
async fn body_updates_while_visible_reset_the_auto_hide_timer() {
    let mut h = Harness::spawn(cfg(1_000));
    h.send(DriverInput::Show);
    h.next_state().await;

    for n in 0..5 {
        tokio::time::sleep(Duration::from_millis(300)).await;
        h.send_event(last_delta("a", &format!("chunk {n}")));
        h.next_state().await;
    }

    assert!(h.visible(), "events should keep the popup open");
    h.shutdown().await;
}

#[tokio::test]
async fn in_flight_turn_pins_popup_open_past_auto_hide() {
    let mut h = Harness::spawn(cfg(500));
    h.send(DriverInput::Show);
    h.next_state().await;
    h.send_event(Event::ToolCall {
        id: "a".into(),
        name: "bash".into(),
        args: json!({"command": "sleep 2"}),
    });
    h.next_state().await;

    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(
        h.visible(),
        "popup should stay visible while turn is in flight"
    );

    h.send_event(Event::Done { id: "a".into() });
    h.next_state().await;

    tokio::time::sleep(Duration::from_millis(900)).await;
    assert!(
        !h.visible(),
        "popup should auto-hide after the turn finishes"
    );
    h.shutdown().await;
}

#[tokio::test]
async fn listening_swaps_in_the_longer_auto_hide_window() {
    let mut h = Harness::spawn(cfg(500));
    h.send_event(Event::ListenState {
        id: "ls".into(),
        active: true,
    });
    h.send(DriverInput::Show);
    h.next_state().await;

    tokio::time::sleep(Duration::from_millis(900)).await;
    assert!(h.visible(), "listening should extend the auto-hide window");
    h.shutdown().await;
}

#[tokio::test]
async fn speaking_pins_popup_past_done_then_auto_hides_after_silence() {
    let mut h = Harness::spawn(cfg(500));
    h.send(DriverInput::Show);
    h.next_state().await;

    h.send_event(last_delta("a", "hi"));
    h.next_state().await;
    h.send_event(speaking("a", true));
    h.send_event(Event::Done { id: "a".into() });
    h.next_state().await;

    tokio::time::sleep(Duration::from_millis(900)).await;
    assert!(
        h.visible(),
        "popup should stay visible while TTS is playing"
    );

    h.send_event(speaking("a", false));
    h.next_state().await;
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert!(!h.visible(), "popup should auto-hide after TTS finishes");
    h.shutdown().await;
}

#[test]
fn push_with_visibility_skips_unchanged_snapshots() {
    let (tx, mut rx) = watch::channel(PopupState::default());
    push_with_visibility(&tx, PopupState::default(), false);
    assert!(!rx.has_changed().expect("sender alive"));
    push_with_visibility(&tx, PopupState::default(), true);
    assert!(rx.has_changed().expect("sender alive"));
    assert!(rx.borrow_and_update().visible);
    push_with_visibility(&tx, PopupState::default(), true);
    assert!(!rx.has_changed().expect("sender alive"));
}

#[tokio::test]
async fn tool_call_event_pushes_the_tracker_snapshot() {
    let mut h = Harness::spawn(cfg(60_000));
    h.send_event(Event::ToolCall {
        id: "a".into(),
        name: "bash".into(),
        args: json!({"command": "ls"}),
    });
    let footer = h.next_state().await.footer.expect("footer present");
    assert_eq!(footer.name, "bash");
    h.shutdown().await;
}

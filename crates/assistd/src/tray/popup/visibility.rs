//! Popup driver task: visibility state machine plus event ingestion.

use std::time::{Duration, Instant};

use assistd_config::TrayPopupConfig;
use assistd_ipc::{Event, IpcClient, Request};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::watch;
use tokio::time::interval;
use uuid::Uuid;

use super::state::{PopupState, PopupTracker};

#[derive(Debug)]
pub enum DriverInput {
    Event(Box<Event>),
    Disconnected,
    Show,
    Dismiss,
    Mapped,
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaceRequest;

pub async fn run(
    state_tx: watch::Sender<PopupState>,
    mut rx: UnboundedReceiver<DriverInput>,
    place_tx: UnboundedSender<PlaceRequest>,
    cfg: TrayPopupConfig,
    ipc: IpcClient,
) {
    let mut tracker = PopupTracker::default();
    let mut visible = false;
    let mut last_activity = Instant::now();
    let mut was_busy = false;
    let mut was_speaking = false;
    let auto_hide_default = Duration::from_millis(cfg.auto_hide_ms);
    let auto_hide_listening = Duration::from_millis(cfg.listen_auto_hide_ms);
    let mut ticker = interval(Duration::from_millis(250));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            biased;
            msg = rx.recv() => {
                let Some(msg) = msg else { break };
                match msg {
                    DriverInput::Shutdown => break,
                    DriverInput::Disconnected => {
                        tracker.set_disconnected();
                        was_busy = false;
                        was_speaking = false;
                        push_with_visibility(&state_tx, tracker.snapshot(&cfg), visible);
                    }
                    DriverInput::Event(ev) => {
                        let snap = tracker.ingest(&ev, &cfg);
                        if visible {
                            last_activity = Instant::now();
                        }
                        // Restart the auto-hide countdown on each
                        // held-open → idle edge (busy and speaking)
                        // so the popup lingers for the full window
                        // after the agent actually stops working.
                        let is_busy = tracker.is_busy();
                        let is_speaking = tracker.is_speaking();
                        if (was_busy && !is_busy) || (was_speaking && !is_speaking) {
                            last_activity = Instant::now();
                        }
                        was_busy = is_busy;
                        was_speaking = is_speaking;
                        push_with_visibility(&state_tx, snap, visible);
                    }
                    DriverInput::Show => {
                        last_activity = Instant::now();
                        if !visible {
                            visible = true;
                            push_with_visibility(&state_tx, tracker.snapshot(&cfg), visible);
                            let _ = place_tx.send(PlaceRequest);
                        }
                    }
                    DriverInput::Dismiss => {
                        if visible {
                            visible = false;
                            push_with_visibility(&state_tx, tracker.snapshot(&cfg), visible);
                        }
                        spawn_interrupt(ipc.clone());
                    }
                    DriverInput::Mapped => {}
                }
            }
            _ = ticker.tick() => {
                if !visible {
                    continue;
                }
                if tracker.is_busy() || tracker.is_speaking() {
                    continue;
                }
                let auto_hide = if tracker.is_listening() {
                    auto_hide_listening
                } else {
                    auto_hide_default
                };
                if last_activity.elapsed() >= auto_hide {
                    visible = false;
                    push_with_visibility(&state_tx, tracker.snapshot(&cfg), visible);
                }
            }
        }
    }
}

fn push_with_visibility(tx: &watch::Sender<PopupState>, mut snap: PopupState, visible: bool) {
    snap.visible = visible;
    tx.send_if_modified(|cur| {
        if *cur != snap {
            *cur = snap;
            true
        } else {
            false
        }
    });
}

fn spawn_interrupt(ipc: IpcClient) {
    tokio::spawn(async move {
        let req = Request::InterruptTurn {
            id: Uuid::new_v4().to_string(),
        };
        match ipc.one_shot(req).await {
            Ok(stream) => {
                if let Err(e) = stream.collect().await {
                    tracing::warn!(target: "tray", "popup dismiss: interrupt_turn stream: {e}");
                }
            }
            Err(e) => {
                tracing::warn!(target: "tray", "popup dismiss: interrupt_turn send failed: {e}");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::sync::mpsc;

    fn cfg(auto_hide_ms: u64) -> TrayPopupConfig {
        TrayPopupConfig {
            auto_hide_ms,
            ..TrayPopupConfig::default()
        }
    }

    fn dummy_ipc() -> IpcClient {
        IpcClient::with_path("/tmp/assistd-popup-tests-nonexistent.sock")
    }

    async fn drain_watch(rx: &mut watch::Receiver<PopupState>) -> PopupState {
        rx.changed().await.expect("watch sender alive");
        rx.borrow_and_update().clone()
    }

    #[tokio::test]
    async fn show_makes_state_visible_and_requests_placement() {
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, mut place_rx) = mpsc::unbounded_channel();

        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(60_000), dummy_ipc()));
        in_tx.send(DriverInput::Show).expect("send");
        let s = drain_watch(&mut state_rx).await;
        assert!(s.visible);
        assert_eq!(
            place_rx.recv().await,
            Some(PlaceRequest),
            "show triggers placement"
        );
        in_tx.send(DriverInput::Shutdown).expect("send");
        handle.await.expect("driver join");
    }

    #[tokio::test]
    async fn body_updates_while_visible_reset_the_auto_hide_timer() {
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(1_000), dummy_ipc()));

        in_tx.send(DriverInput::Show).expect("send");
        let _ = drain_watch(&mut state_rx).await;

        for n in 0..5 {
            tokio::time::sleep(Duration::from_millis(300)).await;
            in_tx
                .send(DriverInput::Event(Box::new(Event::LastDelta {
                    id: "a".into(),
                    text: format!("chunk {n}"),
                })))
                .expect("send");
            let _ = drain_watch(&mut state_rx).await;
        }

        let still_visible = state_rx.borrow().visible;
        assert!(still_visible, "events should keep the popup open");

        in_tx.send(DriverInput::Shutdown).expect("send");
        handle.await.expect("driver join");
    }

    #[tokio::test]
    async fn in_flight_turn_pins_popup_open_past_auto_hide() {
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(500), dummy_ipc()));

        in_tx.send(DriverInput::Show).expect("send");
        let _ = drain_watch(&mut state_rx).await;
        in_tx
            .send(DriverInput::Event(Box::new(Event::ToolCall {
                id: "a".into(),
                name: "bash".into(),
                args: json!({"command": "sleep 2"}),
            })))
            .expect("send");
        let _ = drain_watch(&mut state_rx).await;

        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert!(
            state_rx.borrow().visible,
            "popup should stay visible while turn is in flight"
        );

        in_tx
            .send(DriverInput::Event(Box::new(Event::Done { id: "a".into() })))
            .expect("send");
        let _ = drain_watch(&mut state_rx).await;

        tokio::time::sleep(Duration::from_millis(900)).await;
        assert!(
            !state_rx.borrow().visible,
            "popup should auto-hide after the turn finishes"
        );

        in_tx.send(DriverInput::Shutdown).expect("send");
        handle.await.expect("driver join");
    }

    #[tokio::test]
    async fn listening_swaps_in_the_longer_auto_hide_window() {
        let popup_cfg = TrayPopupConfig {
            auto_hide_ms: 500,
            listen_auto_hide_ms: 2000,
            ..TrayPopupConfig::default()
        };
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, popup_cfg, dummy_ipc()));

        in_tx
            .send(DriverInput::Event(Box::new(Event::ListenState {
                id: "ls".into(),
                active: true,
            })))
            .expect("send");
        in_tx.send(DriverInput::Show).expect("send");
        let _ = drain_watch(&mut state_rx).await;

        tokio::time::sleep(Duration::from_millis(900)).await;
        assert!(
            state_rx.borrow().visible,
            "listening should extend the auto-hide window"
        );

        in_tx.send(DriverInput::Shutdown).expect("send");
        handle.await.expect("driver join");
    }

    #[tokio::test]
    async fn speaking_pins_popup_past_done_then_auto_hides_after_silence() {
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(500), dummy_ipc()));

        in_tx.send(DriverInput::Show).expect("send");
        let _ = drain_watch(&mut state_rx).await;

        in_tx
            .send(DriverInput::Event(Box::new(Event::LastDelta {
                id: "a".into(),
                text: "hi".into(),
            })))
            .expect("send");
        let _ = drain_watch(&mut state_rx).await;
        in_tx
            .send(DriverInput::Event(Box::new(Event::SpeakingState {
                id: "a".into(),
                speaking: true,
            })))
            .expect("send");
        in_tx
            .send(DriverInput::Event(Box::new(Event::Done { id: "a".into() })))
            .expect("send");
        let _ = drain_watch(&mut state_rx).await;

        tokio::time::sleep(Duration::from_millis(900)).await;
        assert!(
            state_rx.borrow().visible,
            "popup should stay visible while TTS is playing"
        );

        in_tx
            .send(DriverInput::Event(Box::new(Event::SpeakingState {
                id: "a".into(),
                speaking: false,
            })))
            .expect("send");
        let _ = drain_watch(&mut state_rx).await;
        tokio::time::sleep(Duration::from_millis(900)).await;
        assert!(
            !state_rx.borrow().visible,
            "popup should auto-hide after TTS finishes"
        );

        in_tx.send(DriverInput::Shutdown).expect("send");
        handle.await.expect("driver join");
    }

    #[tokio::test]
    async fn tool_call_event_only_pushes_when_state_changed() {
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(60_000), dummy_ipc()));

        in_tx
            .send(DriverInput::Event(Box::new(Event::ToolCall {
                id: "a".into(),
                name: "bash".into(),
                args: json!({"command": "ls"}),
            })))
            .expect("send");
        let s = drain_watch(&mut state_rx).await;
        let footer = s.footer.expect("footer present");
        assert_eq!(footer.name, "bash");

        in_tx.send(DriverInput::Shutdown).expect("send");
        handle.await.expect("driver join");
    }
}

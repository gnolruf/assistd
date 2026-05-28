//! Popup driver: a single tokio task that owns the [`PopupTracker`],
//! the visibility state machine, and the `watch::Sender` the GUI
//! thread reads from.
//!
//! Inputs:
//! - daemon events (from the tray's subscribe loop), and a Disconnect
//!   signal that re-uses the same mpsc for compactness;
//! - explicit `Show` commands (tray-icon left-click + wake-on events);
//! - GUI events (focus lost, first paint after a show);
//! - a 250 ms timer that drives the auto-hide check.

use std::time::{Duration, Instant};

use assistd_config::TrayPopupConfig;
use assistd_ipc::Event;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::watch;
use tokio::time::interval;

use super::state::{PopupState, PopupTracker};

/// One input to the driver loop. Bundled into a single enum so the
/// driver can select! over a single mpsc and a single ticker without
/// fanning to many receivers. The `Event` variant is boxed because
/// `assistd_ipc::Event` is ~230 bytes and dominates the enum size
/// otherwise.
#[derive(Debug)]
pub enum DriverInput {
    /// New IPC event from the daemon's broadcast bus.
    Event(Box<Event>),
    /// IPC connection down — subscribe loop is in backoff or restart.
    Disconnected,
    /// Tray-icon left-click or a wake-on event matched. The popup
    /// should become visible (resetting the auto-hide timer).
    Show,
    /// GUI thread reports focus lost — popup should hide.
    FocusLost,
    /// GUI thread finished its first paint after a Show. Kept as a
    /// signal in case the driver ever needs to act on the
    /// compositor-aware moment; `place_floating` handles the
    /// map-time race against the compositor internally, so for now
    /// the driver doesn't need to react.
    Mapped,
    /// Cooperative shutdown.
    Shutdown,
}

/// Request sent to the placement worker. The worker holds the
/// criteria + anchor; the request itself is a unit value because all
/// the popup ever asks for is "place me now."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaceRequest;

/// Run the driver loop until a [`DriverInput::Shutdown`] arrives or
/// all senders are dropped. Pushes a new [`PopupState`] onto the
/// `watch` whenever the visible content changes; sends a single
/// [`PlaceRequest`] onto `place_tx` when the popup becomes visible.
/// One placement per wake is sufficient because
/// [`assistd_wm::WindowManager::place_floating`] waits for and
/// retries against the matching window-event internally.
pub async fn run(
    state_tx: watch::Sender<PopupState>,
    mut rx: UnboundedReceiver<DriverInput>,
    place_tx: UnboundedSender<PlaceRequest>,
    cfg: TrayPopupConfig,
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
                        // Held-open → idle transition: restart the
                        // auto-hide countdown from this instant so the
                        // popup lingers for the full auto_hide_ms after
                        // the agent finishes, regardless of how long
                        // the busy stretch was. Same treatment for the
                        // speaking → silent edge: TTS playback can
                        // extend well past the turn's `Done`, and we
                        // want the auto-hide window to start fresh
                        // when the voice actually stops.
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
                    DriverInput::FocusLost => {
                        if visible {
                            visible = false;
                            push_with_visibility(&state_tx, tracker.snapshot(&cfg), visible);
                        }
                    }
                    DriverInput::Mapped => {}
                }
            }
            _ = ticker.tick() => {
                if !visible {
                    continue;
                }
                // Pin the popup open while a turn is in flight — the
                // gap between a ToolCall and its ToolResult is silent
                // on the wire but the agent is still working, so an
                // auto-hide here would dismiss the popup mid-tool-call.
                // Same treatment while TTS is playing back: the user is
                // listening to the response and shouldn't lose the
                // activity surface mid-sentence.
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
    // send_if_modified avoids waking the GUI thread when nothing the
    // user can see actually changed.
    tx.send_if_modified(|cur| {
        if *cur != snap {
            *cur = snap;
            true
        } else {
            false
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

    async fn drain_watch(rx: &mut watch::Receiver<PopupState>) -> PopupState {
        rx.changed().await.expect("watch sender alive");
        rx.borrow_and_update().clone()
    }

    #[tokio::test]
    async fn show_makes_state_visible_and_requests_placement() {
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, mut place_rx) = mpsc::unbounded_channel();

        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(60_000)));
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
    async fn focus_lost_hides_the_popup() {
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(60_000)));

        in_tx.send(DriverInput::Show).expect("send");
        let visible_state = drain_watch(&mut state_rx).await;
        assert!(visible_state.visible);
        in_tx.send(DriverInput::FocusLost).expect("send");
        let hidden_state = drain_watch(&mut state_rx).await;
        assert!(!hidden_state.visible);

        in_tx.send(DriverInput::Shutdown).expect("send");
        handle.await.expect("driver join");
    }

    #[tokio::test]
    async fn body_updates_while_visible_reset_the_auto_hide_timer() {
        // Sanity check: incoming Events refresh last_activity, so a
        // 1s auto-hide doesn't fire while events keep arriving.
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(1_000)));

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
        // While a turn is in flight, the popup must stay visible past
        // auto_hide_ms — even when no events arrive — until Done.
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(500)));

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

        // Quiet stretch comfortably longer than auto_hide_ms (500ms);
        // the in-flight turn must hold the popup open.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert!(
            state_rx.borrow().visible,
            "popup should stay visible while turn is in flight"
        );

        in_tx
            .send(DriverInput::Event(Box::new(Event::Done { id: "a".into() })))
            .expect("send");
        let _ = drain_watch(&mut state_rx).await;

        // After Done, the auto-hide window starts fresh from this
        // moment; wait it out and confirm the popup closes.
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
        // With listening active, auto-hide uses listen_auto_hide_ms
        // instead of the shorter default — so the popup outlives the
        // regular timeout but still dismisses eventually.
        let popup_cfg = TrayPopupConfig {
            auto_hide_ms: 500,
            listen_auto_hide_ms: 2000,
            ..TrayPopupConfig::default()
        };
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, popup_cfg));

        in_tx
            .send(DriverInput::Event(Box::new(Event::ListenState {
                id: "ls".into(),
                active: true,
            })))
            .expect("send");
        in_tx.send(DriverInput::Show).expect("send");
        let _ = drain_watch(&mut state_rx).await;

        // Past the short timeout, well under the listen-extended one.
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
        // After Done, TTS playback can run for seconds. While
        // SpeakingState{true} is in flight, the popup must stay open
        // past auto_hide_ms; once SpeakingState{false} arrives, the
        // auto-hide countdown starts fresh from that instant.
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(500)));

        in_tx.send(DriverInput::Show).expect("send");
        let _ = drain_watch(&mut state_rx).await;

        // Turn starts, finishes, but TTS is mid-playback.
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

        // Past the short auto-hide; speaking should hold the popup.
        tokio::time::sleep(Duration::from_millis(900)).await;
        assert!(
            state_rx.borrow().visible,
            "popup should stay visible while TTS is playing"
        );

        // Playback ends — auto-hide starts fresh from this moment.
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
        // The driver's send_if_modified means watch only wakes when the
        // visible snapshot actually differs.
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(60_000)));

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

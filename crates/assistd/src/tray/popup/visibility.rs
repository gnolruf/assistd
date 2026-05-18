//! Popup driver: a single tokio task that owns the [`PopupTracker`],
//! the visibility state machine, and the `watch::Sender` the GUI
//! thread reads from.
//!
//! Inputs:
//! - daemon events (from the tray's subscribe loop), and a Connect /
//!   Disconnect signal that re-uses the same mpsc for compactness;
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
    /// IPC connection up — the daemon is reachable.
    Connected,
    /// IPC connection down — subscribe loop is in backoff or restart.
    Disconnected,
    /// Tray-icon left-click or a wake-on event matched. The popup
    /// should become visible (resetting the auto-hide timer).
    Show,
    /// GUI thread reports focus lost — popup should hide.
    FocusLost,
    /// GUI thread finished its first paint after a Show — the
    /// compositor knows about the window and `place_floating` can
    /// match it. Triggers a follow-up retry to handle the rare
    /// compositor-side race.
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
/// Earlier versions also re-fired on the GUI's `Mapped` event and on
/// a 250 ms ticker as a workaround for the MapNotify race, but
/// `WindowManager::place_floating` now retries internally, so a
/// single trigger is sufficient and avoids the 3× duplicate
/// placement IPC traffic per wake.
pub async fn run(
    state_tx: watch::Sender<PopupState>,
    mut rx: UnboundedReceiver<DriverInput>,
    place_tx: UnboundedSender<PlaceRequest>,
    cfg: TrayPopupConfig,
) {
    let mut tracker = PopupTracker::default();
    let mut visible = false;
    let mut last_activity = Instant::now();
    let auto_hide = Duration::from_millis(cfg.auto_hide_ms);
    let mut ticker = interval(Duration::from_millis(250));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            biased;
            msg = rx.recv() => {
                let Some(msg) = msg else { break };
                match msg {
                    DriverInput::Shutdown => break,
                    DriverInput::Connected => {
                        tracker.set_connected();
                        push_with_visibility(&state_tx, tracker.snapshot(&cfg), visible);
                    }
                    DriverInput::Disconnected => {
                        tracker.set_disconnected();
                        push_with_visibility(&state_tx, tracker.snapshot(&cfg), visible);
                    }
                    DriverInput::Event(ev) => {
                        let snap = tracker.ingest(&ev, &cfg);
                        if visible {
                            last_activity = Instant::now();
                        }
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
                    // GUI thread sends Mapped on first paint after Show;
                    // we used to re-fire placement here but the internal
                    // retry in place_floating makes it redundant.
                    DriverInput::Mapped => {}
                }
            }
            _ = ticker.tick() => {
                if visible && last_activity.elapsed() >= auto_hide {
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
        in_tx.send(DriverInput::Connected).expect("send");
        let _connected_state = drain_watch(&mut state_rx).await;
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

        in_tx.send(DriverInput::Connected).expect("send");
        let _ = drain_watch(&mut state_rx).await;
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

        in_tx.send(DriverInput::Connected).expect("send");
        let _ = drain_watch(&mut state_rx).await;
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
    async fn tool_call_event_only_pushes_when_state_changed() {
        // The driver's send_if_modified means watch only wakes when the
        // visible snapshot actually differs.
        let (state_tx, mut state_rx) = watch::channel(PopupState::default());
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let (place_tx, _place_rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(run(state_tx, in_rx, place_tx, cfg(60_000)));

        in_tx.send(DriverInput::Connected).expect("send");
        let _ = drain_watch(&mut state_rx).await;
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
        assert_eq!(s.title, "Generating");

        in_tx.send(DriverInput::Shutdown).expect("send");
        handle.await.expect("driver join");
    }
}

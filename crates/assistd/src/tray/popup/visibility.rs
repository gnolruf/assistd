//! Popup driver task: visibility state machine plus event ingestion.

use std::time::{Duration, Instant};

use assistd_config::TrayPopupConfig;
use assistd_ipc::{Event, IpcClient, Request};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::watch;
use tokio::task::JoinSet;
use tokio::time::interval;
use uuid::Uuid;

use super::state::{PopupState, PopupTracker};

const TICK_INTERVAL: Duration = Duration::from_millis(250);

/// Messages consumed by [`drive_visibility`].
#[derive(Debug)]
pub enum DriverInput {
    Event(Box<Event>),
    Disconnected,
    Show,
    /// User closed the popup; also interrupts the current turn.
    Dismiss,
    /// The window finished its first paint after being shown.
    Mapped,
    Shutdown,
}

/// Ask the window manager to place the popup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaceRequest;

/// Own the popup's visibility: show on request, hide on dismiss (also
/// interrupting the turn) or once idle past the auto-hide window.
pub async fn drive_visibility(
    state_tx: watch::Sender<PopupState>,
    mut rx: UnboundedReceiver<DriverInput>,
    place_tx: UnboundedSender<PlaceRequest>,
    cfg: TrayPopupConfig,
    ipc: IpcClient,
) {
    let mut driver = Driver::new(state_tx, &cfg);
    let mut ticker = interval(TICK_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut interrupts = JoinSet::new();

    loop {
        tokio::select! {
            biased;
            msg = rx.recv() => {
                let Some(msg) = msg else { break };
                match msg {
                    DriverInput::Shutdown => break,
                    DriverInput::Disconnected => driver.disconnect(),
                    DriverInput::Event(ev) => driver.ingest(&ev),
                    DriverInput::Show => {
                        if driver.show() {
                            let _ = place_tx.send(PlaceRequest);
                        }
                    }
                    DriverInput::Dismiss => {
                        driver.hide();
                        interrupts.spawn(interrupt_turn(ipc.clone()));
                    }
                    DriverInput::Mapped => {}
                }
            }
            Some(res) = interrupts.join_next() => {
                if let Err(e) = res {
                    tracing::warn!(target: "tray", "popup dismiss: interrupt_turn task failed: {e}");
                }
            }
            _ = ticker.tick() => driver.hide_if_idle(),
        }
    }
}

struct Driver {
    state_tx: watch::Sender<PopupState>,
    tracker: PopupTracker,
    visible: bool,
    last_activity: Instant,
    was_busy: bool,
    was_speaking: bool,
    auto_hide: Duration,
    listen_auto_hide: Duration,
}

impl Driver {
    fn new(state_tx: watch::Sender<PopupState>, cfg: &TrayPopupConfig) -> Self {
        Self {
            state_tx,
            tracker: PopupTracker::default(),
            visible: false,
            last_activity: Instant::now(),
            was_busy: false,
            was_speaking: false,
            auto_hide: Duration::from_millis(cfg.auto_hide_ms),
            listen_auto_hide: Duration::from_millis(cfg.listen_auto_hide_ms()),
        }
    }

    fn publish(&self) {
        push_with_visibility(&self.state_tx, self.tracker.snapshot(), self.visible);
    }

    fn disconnect(&mut self) {
        self.tracker.set_disconnected();
        self.was_busy = false;
        self.was_speaking = false;
        self.publish();
    }

    /// Any event while visible, or a turn or speech finishing, restarts the
    /// auto-hide timer.
    fn ingest(&mut self, ev: &Event) {
        self.tracker.ingest(ev);
        if self.visible {
            self.last_activity = Instant::now();
        }
        let is_busy = self.tracker.is_busy();
        let is_speaking = self.tracker.is_speaking();
        if (self.was_busy && !is_busy) || (self.was_speaking && !is_speaking) {
            self.last_activity = Instant::now();
        }
        self.was_busy = is_busy;
        self.was_speaking = is_speaking;
        self.publish();
    }

    /// Returns `true` when the popup was hidden and is now shown.
    fn show(&mut self) -> bool {
        self.last_activity = Instant::now();
        if self.visible {
            return false;
        }
        self.visible = true;
        self.publish();
        true
    }

    fn hide(&mut self) {
        if self.visible {
            self.visible = false;
            self.publish();
        }
    }

    fn hide_if_idle(&mut self) {
        if !self.visible || self.tracker.is_busy() || self.tracker.is_speaking() {
            return;
        }
        let timeout = if self.tracker.is_listening() {
            self.listen_auto_hide
        } else {
            self.auto_hide
        };
        if self.last_activity.elapsed() >= timeout {
            self.hide();
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

async fn interrupt_turn(ipc: IpcClient) {
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
}

#[cfg(test)]
mod tests;

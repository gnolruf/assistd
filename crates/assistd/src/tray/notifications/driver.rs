//! Notification driver: decides when the turn's notification is shown,
//! updated, hidden, or turned into an interrupt. Nothing is shown while
//! the chat has keyboard focus.

use std::time::Duration;

use assistd_config::TrayNotificationsConfig;
use assistd_ipc::{Event, IpcClient, Request};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::task::JoinSet;
use tokio::time::{Instant, MissedTickBehavior, interval};
use uuid::Uuid;

use super::activity::ActivityTracker;
use super::render::Notification;

/// Also the update throttle: changes are flushed at most once a tick.
const TICK_INTERVAL: Duration = Duration::from_millis(250);

/// Messages consumed by [`drive_notifications`].
#[derive(Debug)]
pub(crate) enum DriverInput {
    Event(Box<Event>),
    Disconnected,
    /// An event matched a configured wake rule.
    Wake,
    /// The tray icon was clicked: hide the notification, or bring it back.
    TrayActivated,
    /// The user dismissed the notification; interrupts a busy turn.
    Dismissed,
    /// The notification's Hide action ran.
    HideInvoked,
    /// The service closed the notification on its own, such as on expiry.
    Expired,
    Shutdown,
}

/// What the notification service should do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum NotifierCommand {
    /// Show or update the single assistd notification.
    Show(Notification),
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Display {
    Closed,
    /// `stale` means the tracker changed since the last `Show`.
    Shown {
        stale: bool,
    },
}

/// Run until `Shutdown` arrives or every sender drops, closing the
/// notification on the way out.
pub(super) async fn drive_notifications(
    mut rx: UnboundedReceiver<DriverInput>,
    commands: UnboundedSender<NotifierCommand>,
    cfg: TrayNotificationsConfig,
    ipc: IpcClient,
) {
    let mut driver = Driver::new(commands, &cfg);
    let mut ticker = interval(TICK_INTERVAL);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut interrupts = JoinSet::new();

    loop {
        tokio::select! {
            biased;
            msg = rx.recv() => {
                let Some(msg) = msg else { break };
                if matches!(msg, DriverInput::Shutdown) {
                    break;
                }
                if driver.apply(msg) {
                    interrupts.spawn(interrupt_turn(ipc.clone()));
                }
            }
            Some(res) = interrupts.join_next() => {
                if let Err(e) = res {
                    tracing::warn!(target: "tray", "notification dismiss: interrupt task failed: {e}");
                }
            }
            _ = ticker.tick() => driver.tick(),
        }
    }
    driver.close();
}

struct Driver {
    commands: UnboundedSender<NotifierCommand>,
    tracker: ActivityTracker,
    /// `None` until the daemon has answered, which holds notifications.
    chat_focused: Option<bool>,
    display: Display,
    /// The turn the user hid; wake rules leave it hidden.
    hidden_turn: Option<String>,
    last_activity: Instant,
    was_busy: bool,
    was_speaking: bool,
    auto_hide: Duration,
    listen_auto_hide: Duration,
}

impl Driver {
    fn new(commands: UnboundedSender<NotifierCommand>, cfg: &TrayNotificationsConfig) -> Self {
        Self {
            commands,
            tracker: ActivityTracker::default(),
            chat_focused: None,
            display: Display::Closed,
            hidden_turn: None,
            last_activity: Instant::now(),
            was_busy: false,
            was_speaking: false,
            auto_hide: Duration::from_millis(cfg.auto_hide_ms),
            listen_auto_hide: Duration::from_millis(cfg.listen_auto_hide_ms()),
        }
    }

    /// Apply one input; `true` means the in-flight turn must be interrupted.
    fn apply(&mut self, msg: DriverInput) -> bool {
        match msg {
            DriverInput::Event(ev) => self.ingest(&ev),
            DriverInput::Disconnected => self.disconnect(),
            DriverInput::Wake => self.wake(),
            DriverInput::TrayActivated => self.toggle(),
            DriverInput::Dismissed => return self.dismissed(),
            DriverInput::HideInvoked => self.hide_for_turn(),
            DriverInput::Expired => self.display = Display::Closed,
            DriverInput::Shutdown => self.close(),
        }
        false
    }

    /// Any event while shown, or a turn or speech finishing, restarts the
    /// auto-hide timer.
    fn ingest(&mut self, ev: &Event) {
        if let Event::ChatFocus { focused, .. } = ev {
            self.set_chat_focus(*focused);
            return;
        }
        self.tracker.ingest(ev);
        if let Display::Shown { stale } = &mut self.display {
            *stale = true;
            self.last_activity = Instant::now();
        }
        let is_busy = self.tracker.is_busy();
        let is_speaking = self.tracker.is_speaking();
        if (self.was_busy && !is_busy) || (self.was_speaking && !is_speaking) {
            self.last_activity = Instant::now();
        }
        self.was_busy = is_busy;
        self.was_speaking = is_speaking;
    }

    /// Gaining focus clears the screen; losing it mid-turn shows the
    /// turn's current state.
    fn set_chat_focus(&mut self, focused: bool) {
        self.chat_focused = Some(focused);
        if focused {
            self.close();
        } else if self.tracker.is_busy() {
            self.wake();
        }
    }

    fn disconnect(&mut self) {
        self.tracker.set_disconnected();
        self.chat_focused = None;
        self.was_busy = false;
        self.was_speaking = false;
        self.close();
    }

    fn wake(&mut self) {
        self.last_activity = Instant::now();
        let turn_hidden = self.hidden_turn.is_some()
            && self.hidden_turn.as_deref() == self.tracker.current_turn();
        if self.display == Display::Closed && !turn_hidden {
            self.open();
        }
    }

    fn toggle(&mut self) {
        match self.display {
            Display::Shown { .. } => self.hide_for_turn(),
            Display::Closed => {
                self.hidden_turn = None;
                self.open();
            }
        }
    }

    /// The notification is already gone; interrupt if a turn or its
    /// speech is still running.
    fn dismissed(&mut self) -> bool {
        self.display = Display::Closed;
        let running = self.tracker.is_busy() || self.tracker.is_speaking();
        if running {
            self.hidden_turn = self.tracker.current_turn().map(str::to_string);
        }
        running
    }

    fn hide_for_turn(&mut self) {
        self.hidden_turn = self.tracker.current_turn().map(str::to_string);
        self.close();
    }

    /// The single check every `Show` passes: never while the chat is
    /// focused or its focus is still unknown.
    fn open(&mut self) {
        if self.chat_focused != Some(false) {
            return;
        }
        self.last_activity = Instant::now();
        self.display = Display::Shown { stale: false };
        self.send_show();
    }

    fn close(&mut self) {
        if self.display != Display::Closed {
            self.display = Display::Closed;
            let _ = self.commands.send(NotifierCommand::Close);
        }
    }

    fn tick(&mut self) {
        if self.display == (Display::Shown { stale: true }) {
            self.display = Display::Shown { stale: false };
            self.send_show();
        }
        self.close_if_idle();
    }

    fn send_show(&self) {
        let pinned =
            self.tracker.is_busy() || self.tracker.is_speaking() || self.tracker.is_listening();
        let expire_ms = if pinned {
            0
        } else {
            u64::try_from(self.idle_timeout().as_millis()).unwrap_or(u64::MAX)
        };
        let _ = self.commands.send(NotifierCommand::Show(Notification {
            view: self.tracker.snapshot(),
            hideable: self.tracker.is_busy(),
            expire_ms,
        }));
    }

    fn idle_timeout(&self) -> Duration {
        if self.tracker.is_listening() {
            self.listen_auto_hide
        } else {
            self.auto_hide
        }
    }

    fn close_if_idle(&mut self) {
        if self.display == Display::Closed || self.tracker.is_busy() || self.tracker.is_speaking() {
            return;
        }
        if self.last_activity.elapsed() >= self.idle_timeout() {
            self.close();
        }
    }
}

async fn interrupt_turn(ipc: IpcClient) {
    let req = Request::InterruptTurn {
        id: Uuid::new_v4().to_string(),
    };
    match ipc.one_shot(req).await {
        Ok(stream) => {
            if let Err(e) = stream.collect().await {
                tracing::warn!(target: "tray", "notification dismiss: interrupt_turn stream: {e}");
            }
        }
        Err(e) => {
            tracing::warn!(target: "tray", "notification dismiss: interrupt_turn send failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests;

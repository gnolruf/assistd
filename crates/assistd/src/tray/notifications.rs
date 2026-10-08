//! Desktop notifications showing each turn's activity, spawned alongside
//! the tray icon.

use assistd_config::Config;
use assistd_ipc::{Event, IpcClient};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::task::JoinHandle;

use dbus::run_notifier;
use driver::{DriverInput, drive_notifications};

mod activity;
mod dbus;
mod driver;
mod render;

/// Cloneable handle that feeds daemon events to the notification driver.
#[derive(Clone)]
pub(crate) struct NotificationSink {
    driver_tx: UnboundedSender<DriverInput>,
    wake_tool_call: bool,
    wake_delta: bool,
    wake_error: bool,
}

impl NotificationSink {
    /// Forward an event into the driver, then a `Wake` when the event
    /// matches a configured wake rule.
    pub(crate) fn ingest(&self, ev: &Event) {
        let wakes = self.matches_wake_rule(ev);
        let _ = self
            .driver_tx
            .send(DriverInput::Event(Box::new(ev.clone())));
        if wakes {
            let _ = self.driver_tx.send(DriverInput::Wake);
        }
    }

    pub(crate) fn set_disconnected(&self) {
        let _ = self.driver_tx.send(DriverInput::Disconnected);
    }

    /// Report a click on the tray icon.
    pub(crate) fn tray_activated(&self) {
        let _ = self.driver_tx.send(DriverInput::TrayActivated);
    }

    fn matches_wake_rule(&self, ev: &Event) -> bool {
        match ev {
            Event::ToolCall { .. } => self.wake_tool_call,
            Event::Delta { .. } | Event::LastDelta { .. } | Event::ReasoningDelta { .. } => {
                self.wake_delta
            }
            Event::Error { .. } => self.wake_error,
            _ => false,
        }
    }
}

/// Owns the driver and notifier tasks.
pub(crate) struct NotificationsHandle {
    pub sink: NotificationSink,
    driver_task: JoinHandle<()>,
    notifier_task: JoinHandle<()>,
}

impl NotificationsHandle {
    /// Close the notification and stop both tasks.
    pub(crate) async fn shutdown(self) {
        let _ = self.sink.driver_tx.send(DriverInput::Shutdown);
        let _ = self.driver_task.await;
        let _ = self.notifier_task.await;
    }
}

/// Spawn the notification tasks; `None` when disabled by config.
pub(crate) fn spawn_notifications(cfg: &Config, ipc: IpcClient) -> Option<NotificationsHandle> {
    let notifications_cfg = &cfg.tray.notifications;
    if !notifications_cfg.enabled {
        tracing::info!(target: "tray", "notifications: disabled by config");
        return None;
    }
    let (driver_tx, driver_rx) = mpsc::unbounded_channel();
    let (commands_tx, commands_rx) = mpsc::unbounded_channel();
    let notifier_task = tokio::spawn(run_notifier(commands_rx, driver_tx.clone()));
    let driver_task = tokio::spawn(drive_notifications(
        driver_rx,
        commands_tx,
        notifications_cfg.clone(),
        ipc,
    ));
    Some(NotificationsHandle {
        sink: NotificationSink {
            driver_tx,
            wake_tool_call: notifications_cfg.wake_on.tool_call,
            wake_delta: notifications_cfg.wake_on.delta,
            wake_error: notifications_cfg.wake_on.error,
        },
        driver_task,
        notifier_task,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn sink(
        wake_tool_call: bool,
        wake_delta: bool,
        wake_error: bool,
    ) -> (NotificationSink, mpsc::UnboundedReceiver<DriverInput>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            NotificationSink {
                driver_tx: tx,
                wake_tool_call,
                wake_delta,
                wake_error,
            },
            rx,
        )
    }

    #[test]
    fn sink_wake_rules_respect_config() {
        let (s, _rx) = sink(true, false, true);
        assert!(s.matches_wake_rule(&Event::ToolCall {
            id: "a".into(),
            name: "x".into(),
            args: json!({}),
        }));
        assert!(!s.matches_wake_rule(&Event::Delta {
            id: "a".into(),
            text: "x".into(),
        }));
        assert!(s.matches_wake_rule(&Event::Error {
            id: "a".into(),
            message: "x".into(),
        }));
        assert!(!s.matches_wake_rule(&Event::Done { id: "a".into() }));
    }

    #[test]
    fn sink_sends_the_event_before_its_wake() {
        let (s, mut rx) = sink(true, true, true);
        s.ingest(&Event::ToolCall {
            id: "a".into(),
            name: "x".into(),
            args: json!({}),
        });
        let first = rx.try_recv().expect("event queued");
        assert!(
            matches!(&first, DriverInput::Event(ev) if matches!(**ev, Event::ToolCall { .. })),
            "{first:?}"
        );
        let second = rx.try_recv().expect("wake queued");
        assert!(matches!(second, DriverInput::Wake), "{second:?}");
    }

    #[test]
    fn sink_sends_only_the_event_when_no_rule_matches() {
        let (s, mut rx) = sink(false, false, false);
        s.ingest(&Event::Done { id: "a".into() });
        let only = rx.try_recv().expect("event queued");
        assert!(
            matches!(&only, DriverInput::Event(ev) if matches!(**ev, Event::Done { .. })),
            "{only:?}"
        );
        assert!(rx.try_recv().is_err(), "no Wake should have been queued");
    }
}

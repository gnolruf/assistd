//! The freedesktop notification service on the session bus, driven by
//! [`NotifierCommand`]s and reporting closes and actions back as
//! [`DriverInput`]s.

use std::collections::HashMap;

use futures_util::StreamExt;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use zbus::message::Message;
use zbus::zvariant::Value;
use zbus::{Connection, Proxy};

use super::driver::{DriverInput, NotifierCommand};
use super::render::{Capabilities, HIDE_ACTION, Notification, render};

const SERVICE: &str = "org.freedesktop.Notifications";
const PATH: &str = "/org/freedesktop/Notifications";
const APP_NAME: &str = "assistd";
/// dunst groups and replaces notifications sharing this tag.
const STACK_TAG: &str = "assistd";
/// `NotificationClosed` reason for a close by the user.
const REASON_DISMISSED: u32 = 2;

/// Apply `commands` until the driver drops its sender. Without a
/// notification service, logs once and discards every command.
pub(super) async fn run_notifier(
    mut commands: UnboundedReceiver<NotifierCommand>,
    driver: UnboundedSender<DriverInput>,
) {
    let mut service = match NotificationService::connect().await {
        Ok(service) => service,
        Err(e) => {
            tracing::warn!(
                target: "tray",
                "no notification service on the session bus ({e}); notifications are off"
            );
            while commands.recv().await.is_some() {}
            return;
        }
    };
    let mut signals = match service.proxy.receive_all_signals().await {
        Ok(signals) => signals,
        Err(e) => {
            tracing::warn!(target: "tray", "cannot follow notification signals: {e}");
            while commands.recv().await.is_some() {}
            return;
        }
    };
    loop {
        tokio::select! {
            command = commands.recv() => match command {
                Some(NotifierCommand::Show(notification)) => service.show(&notification).await,
                Some(NotifierCommand::Close) => service.close().await,
                None => break,
            },
            Some(message) = signals.next() => {
                if let Some(input) = service.translate_signal(&message) {
                    let _ = driver.send(input);
                }
            }
        }
    }
}

/// The service connection and the one notification assistd keeps on screen.
struct NotificationService {
    proxy: Proxy<'static>,
    caps: Capabilities,
    ids: NotificationIds,
    /// Logs only the first failure of a run.
    failing: bool,
}

impl NotificationService {
    async fn connect() -> zbus::Result<Self> {
        let connection = Connection::session().await?;
        let proxy = Proxy::new(&connection, SERVICE, PATH, SERVICE).await?;
        let capabilities: Vec<String> = proxy.call("GetCapabilities", &()).await?;
        let caps = Capabilities {
            markup: capabilities.iter().any(|c| c == "body-markup"),
            actions: capabilities.iter().any(|c| c == "actions"),
        };
        tracing::info!(target: "tray", "notification service capabilities: {caps:?}");
        Ok(Self {
            proxy,
            caps,
            ids: NotificationIds::default(),
            failing: false,
        })
    }

    async fn show(&mut self, notification: &Notification) {
        let rendered = render(notification, self.caps);
        let hints = HashMap::from([
            ("urgency", Value::U8(rendered.urgency)),
            ("x-dunst-stack-tag", Value::from(STACK_TAG)),
        ]);
        let args = (
            APP_NAME,
            self.ids.current.unwrap_or(0),
            rendered.icon,
            rendered.summary.as_str(),
            rendered.body.as_str(),
            rendered.actions,
            hints,
            rendered.expire_timeout,
        );
        let result: zbus::Result<u32> = self.proxy.call("Notify", &args).await;
        match result {
            Ok(id) => {
                self.ids.current = Some(id);
                self.failing = false;
            }
            Err(e) => self.log_failure("Notify", &e),
        }
    }

    async fn close(&mut self) {
        let Some(id) = self.ids.current.take() else {
            return;
        };
        let result: zbus::Result<()> = self.proxy.call("CloseNotification", &(id,)).await;
        if let Err(e) = result {
            self.log_failure("CloseNotification", &e);
        }
    }

    /// Driver input for a signal about assistd's notification, if any.
    /// `close` forgets the id first, so closes assistd asked for are dropped.
    fn translate_signal(&mut self, message: &Message) -> Option<DriverInput> {
        let header = message.header();
        match header.member()?.as_str() {
            "NotificationClosed" => {
                let (id, reason): (u32, u32) = message.body().deserialize().ok()?;
                self.ids.closed(id, reason)
            }
            "ActionInvoked" => {
                let (id, key): (u32, String) = message.body().deserialize().ok()?;
                self.ids.action_invoked(id, &key)
            }
            _ => None,
        }
    }

    fn log_failure(&mut self, method: &str, error: &zbus::Error) {
        if !self.failing {
            tracing::warn!(target: "tray", "notification {method} failed: {error}");
            self.failing = true;
        }
    }
}

/// Service ids of assistd's notification, mapping its signals to driver
/// input.
#[derive(Debug, Default)]
struct NotificationIds {
    current: Option<u32>,
    /// The notification whose Hide action ran; its next close is ours.
    hidden: Option<u32>,
}

impl NotificationIds {
    fn action_invoked(&mut self, id: u32, key: &str) -> Option<DriverInput> {
        (self.current == Some(id) && key == HIDE_ACTION).then(|| {
            self.hidden = Some(id);
            DriverInput::HideInvoked
        })
    }

    /// Daemons that close a notification after running its action report
    /// that as a dismissal, which must not interrupt.
    fn closed(&mut self, id: u32, reason: u32) -> Option<DriverInput> {
        if self.hidden == Some(id) {
            self.hidden = None;
            if self.current == Some(id) {
                self.current = None;
            }
            return None;
        }
        if self.current != Some(id) {
            return None;
        }
        self.current = None;
        Some(if reason == REASON_DISMISSED {
            DriverInput::Dismissed
        } else {
            DriverInput::Expired
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPIRED: u32 = 1;

    fn showing(id: u32) -> NotificationIds {
        NotificationIds {
            current: Some(id),
            hidden: None,
        }
    }

    #[test]
    fn user_dismissal_and_expiry_map_to_driver_input() {
        let mut ids = showing(7);
        assert!(matches!(
            ids.closed(7, REASON_DISMISSED),
            Some(DriverInput::Dismissed)
        ));
        assert_eq!(ids.current, None);
        let mut ids = showing(7);
        assert!(matches!(ids.closed(7, EXPIRED), Some(DriverInput::Expired)));
    }

    #[test]
    fn signals_about_other_notifications_are_ignored() {
        let mut ids = showing(7);
        assert!(ids.closed(8, REASON_DISMISSED).is_none());
        assert!(ids.action_invoked(8, HIDE_ACTION).is_none());
        assert!(ids.action_invoked(7, "other").is_none());
        assert_eq!(ids.current, Some(7));
    }

    #[test]
    fn the_close_after_a_hide_action_is_not_a_dismissal() {
        let mut ids = showing(7);
        assert!(matches!(
            ids.action_invoked(7, HIDE_ACTION),
            Some(DriverInput::HideInvoked)
        ));
        assert!(ids.closed(7, REASON_DISMISSED).is_none());
        assert_eq!(ids.current, None);
        assert_eq!(ids.hidden, None);
    }
}

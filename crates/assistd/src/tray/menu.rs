//! ksni `Tray` implementation and the small task that turns menu
//! clicks into IPC calls.
//!
//! The activate callbacks ksni invokes on a menu click must not block
//! (the panel freezes if they do), so they only push a [`MenuAction`]
//! onto an mpsc channel. [`run_actions`] drains that channel from a
//! tokio task and issues fresh `one_shot` IPC calls per action — never
//! multiplexed on the long-lived Subscribe connection that
//! `subscribe.rs` owns.

use anyhow::Result;
use assistd_config::TrayConfig;
use assistd_ipc::{Event, IpcClient, PresenceState, Request};
use ksni::{
    Category, Status, ToolTip, Tray,
    menu::{MenuItem, StandardItem},
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use uuid::Uuid;

use super::state::{TrayState, TrayTracker, icon_name_for, tooltip_for};

/// Menu click → background task.
#[derive(Debug, Clone, Copy)]
pub enum MenuAction {
    /// Issue `SetPresence(target)` to the daemon.
    SetPresence(PresenceState),
    /// Tear down the tray and exit cleanly.
    Quit,
}

/// State owned by the ksni service. Property reads and menu rendering
/// happen on ksni's task; `subscribe.rs` and `run_actions` reach in via
/// `Handle::update` and `Handle::shutdown` respectively.
pub struct TrayItem {
    tracker: TrayTracker,
    cfg: TrayConfig,
    actions: UnboundedSender<MenuAction>,
}

impl TrayItem {
    pub fn new(cfg: TrayConfig, actions: UnboundedSender<MenuAction>) -> Self {
        Self {
            tracker: TrayTracker::default(),
            cfg,
            actions,
        }
    }

    /// Apply an incoming daemon event. Returns `true` when the visible
    /// tray state changed.
    pub fn ingest(&mut self, event: &Event) -> bool {
        self.tracker.ingest(event)
    }

    /// Mark the IPC connection up. Returns `true` when the visible
    /// tray state changed.
    pub fn set_connected(&mut self) -> bool {
        self.tracker.set_connected()
    }

    /// Mark the IPC connection down. Returns `true` when the visible
    /// tray state changed.
    pub fn set_disconnected(&mut self) -> bool {
        self.tracker.set_disconnected()
    }
}

impl Tray for TrayItem {
    fn id(&self) -> String {
        "org.assistd.Tray".into()
    }

    fn title(&self) -> String {
        "assistd".into()
    }

    fn category(&self) -> Category {
        Category::ApplicationStatus
    }

    fn status(&self) -> Status {
        Status::Active
    }

    fn icon_name(&self) -> String {
        icon_name_for(self.tracker.current(), &self.cfg).to_string()
    }

    fn tool_tip(&self) -> ToolTip {
        ToolTip {
            icon_name: String::new(),
            icon_pixmap: Vec::new(),
            title: tooltip_for(self.tracker.current()).into(),
            description: String::new(),
        }
    }

    fn menu(&self) -> Vec<MenuItem<Self>> {
        let state = self.tracker.current();
        let toggle_enabled = !matches!(state, TrayState::Disconnected);
        let toggle_label = toggle_label_for(self.tracker.presence()).to_string();
        vec![
            StandardItem {
                label: toggle_label,
                enabled: toggle_enabled,
                activate: Box::new(|item: &mut Self| {
                    let target = toggle_target(item.tracker.presence());
                    let _ = item.actions.send(MenuAction::SetPresence(target));
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Quit tray".into(),
                activate: Box::new(|item: &mut Self| {
                    let _ = item.actions.send(MenuAction::Quit);
                }),
                ..Default::default()
            }
            .into(),
        ]
    }
}

fn toggle_label_for(presence: PresenceState) -> &'static str {
    match presence {
        PresenceState::Sleeping => "Wake",
        PresenceState::Active | PresenceState::Drowsy => "Sleep",
    }
}

fn toggle_target(presence: PresenceState) -> PresenceState {
    match presence {
        PresenceState::Sleeping => PresenceState::Active,
        PresenceState::Active | PresenceState::Drowsy => PresenceState::Sleeping,
    }
}

/// Drain the menu-action channel until a [`MenuAction::Quit`] is
/// received or the sender is dropped. Returns `Ok(())` on a clean exit
/// triggered by Quit; bubbles up only fatal channel errors (none today).
pub async fn run_actions(mut rx: UnboundedReceiver<MenuAction>, ipc: IpcClient) -> Result<()> {
    while let Some(action) = rx.recv().await {
        match action {
            MenuAction::SetPresence(target) => {
                if let Err(e) = send_set_presence(&ipc, target).await {
                    tracing::warn!(target: "tray", "set_presence({target:?}) failed: {e:#}");
                }
            }
            MenuAction::Quit => return Ok(()),
        }
    }
    Ok(())
}

async fn send_set_presence(ipc: &IpcClient, target: PresenceState) -> Result<()> {
    let req = Request::SetPresence {
        id: Uuid::new_v4().to_string(),
        target,
    };
    let mut stream = ipc.one_shot(req).await?;
    loop {
        match stream.next_event().await? {
            Some(Event::Done { .. }) => return Ok(()),
            Some(Event::Error { message, .. }) => {
                anyhow::bail!("daemon error: {message}");
            }
            Some(_) => continue,
            None => anyhow::bail!("daemon closed before Done"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_label_inverts_with_presence() {
        assert_eq!(toggle_label_for(PresenceState::Active), "Sleep");
        assert_eq!(toggle_label_for(PresenceState::Drowsy), "Sleep");
        assert_eq!(toggle_label_for(PresenceState::Sleeping), "Wake");
    }

    #[test]
    fn toggle_target_inverts_with_presence() {
        assert_eq!(
            toggle_target(PresenceState::Active),
            PresenceState::Sleeping
        );
        assert_eq!(
            toggle_target(PresenceState::Drowsy),
            PresenceState::Sleeping
        );
        assert_eq!(
            toggle_target(PresenceState::Sleeping),
            PresenceState::Active
        );
    }
}

//! ksni `Tray` implementation. Menu callbacks must not block (the panel
//! freezes), so they only queue a [`MenuAction`] for [`run_actions`].

use anyhow::Result;
use assistd_config::TrayIconsConfig;
use assistd_ipc::{Event, IpcClient, PresenceState, PresenceTarget, Request};
use ksni::{
    Category, Status, ToolTip, Tray,
    menu::{MenuItem, StandardItem},
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use uuid::Uuid;

use super::state::{TrayTracker, icon_name_for, tooltip_for};

#[derive(Debug, Clone, Copy)]
pub(super) enum MenuAction {
    /// Issue `SetPresence(target)` to the daemon.
    SetPresence(PresenceTarget),
    /// Tear down the tray and exit cleanly.
    Quit,
}

/// Invoked on left-click.
pub(super) type ActivateCallback = Box<dyn Fn() + Send + Sync>;

pub(super) struct TrayItem {
    tracker: TrayTracker,
    icons: TrayIconsConfig,
    actions: UnboundedSender<MenuAction>,
    on_activate: Option<ActivateCallback>,
}

impl TrayItem {
    pub(super) fn new(
        actions: UnboundedSender<MenuAction>,
        on_activate: Option<ActivateCallback>,
        icons: TrayIconsConfig,
        config_error: Option<String>,
    ) -> Self {
        Self {
            tracker: TrayTracker::new(config_error),
            icons,
            actions,
            on_activate,
        }
    }

    /// Returns `true` when the visible tray state changed.
    pub(super) fn ingest(&mut self, event: &Event) -> bool {
        self.tracker.ingest(event)
    }

    /// Returns `true` when the visible tray state changed.
    pub(super) fn set_connected(&mut self) -> bool {
        self.tracker.set_connected()
    }

    /// Returns `true` when the visible tray state changed.
    pub(super) fn set_disconnected(&mut self) -> bool {
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

    fn activate(&mut self, _x: i32, _y: i32) {
        if let Some(cb) = self.on_activate.as_ref() {
            cb();
        }
    }

    fn icon_name(&self) -> String {
        icon_name_for(self.tracker.current(), &self.icons).to_string()
    }

    fn tool_tip(&self) -> ToolTip {
        ToolTip {
            icon_name: String::new(),
            icon_pixmap: Vec::new(),
            title: tooltip_for(self.tracker.current()).into(),
            description: self
                .tracker
                .config_error()
                .map(str::to_string)
                .or_else(|| self.tracker.startup_summary())
                .unwrap_or_default(),
        }
    }

    fn menu(&self) -> Vec<MenuItem<Self>> {
        let toggle_enabled = self.tracker.connected();
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
    match toggle_target(presence) {
        PresenceTarget::Active => "Wake",
        PresenceTarget::Drowsy | PresenceTarget::Sleeping => "Sleep",
    }
}

/// A wake in progress toggles to `Sleeping`, which waits for the load.
fn toggle_target(presence: PresenceState) -> PresenceTarget {
    match presence {
        PresenceState::Sleeping => PresenceTarget::Active,
        PresenceState::Active | PresenceState::Drowsy | PresenceState::Waking => {
            PresenceTarget::Sleeping
        }
    }
}

/// Drain menu actions until [`MenuAction::Quit`] or the sender drops.
pub(super) async fn run_actions(
    mut rx: UnboundedReceiver<MenuAction>,
    ipc: IpcClient,
) -> Result<()> {
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

async fn send_set_presence(ipc: &IpcClient, target: PresenceTarget) -> Result<()> {
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
    fn toggle_label_and_target_invert_presence() {
        for (presence, label, target) in [
            (PresenceState::Active, "Sleep", PresenceTarget::Sleeping),
            (PresenceState::Drowsy, "Sleep", PresenceTarget::Sleeping),
            (PresenceState::Sleeping, "Wake", PresenceTarget::Active),
            (PresenceState::Waking, "Sleep", PresenceTarget::Sleeping),
        ] {
            assert_eq!(toggle_label_for(presence), label, "{presence:?}");
            assert_eq!(toggle_target(presence), target, "{presence:?}");
        }
    }
}

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_TRAY_ICON_ACTIVE, DEFAULT_TRAY_ICON_DISCONNECTED, DEFAULT_TRAY_ICON_GENERATING,
    DEFAULT_TRAY_ICON_LISTENING, DEFAULT_TRAY_ICON_SLEEPING,
    DEFAULT_TRAY_NOTIFICATIONS_AUTO_HIDE_MS, DEFAULT_TRAY_NOTIFICATIONS_BRIEF_WHEN_AWAY,
    DEFAULT_TRAY_NOTIFICATIONS_ENABLED, DEFAULT_TRAY_NOTIFICATIONS_WAKE_DELTA,
    DEFAULT_TRAY_NOTIFICATIONS_WAKE_ERROR, DEFAULT_TRAY_NOTIFICATIONS_WAKE_TOOL_CALL,
};

/// System-tray settings for `assistd tray`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TrayConfig {
    pub icons: TrayIconsConfig,
    pub notifications: TrayNotificationsConfig,
}

/// freedesktop icon-theme names the tray icon shows per state; none may be
/// empty. Custom artwork under `~/.local/share/icons/<theme>/` is found by
/// its file name without the extension.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TrayIconsConfig {
    /// The daemon is awake and idle.
    pub active: String,
    /// The daemon is drowsy, asleep, or loading the model.
    pub sleeping: String,
    /// Continuous listening is on.
    pub listening: String,
    /// A turn is in flight.
    pub generating: String,
    /// The daemon socket is unreachable.
    pub disconnected: String,
}

impl Default for TrayIconsConfig {
    fn default() -> Self {
        Self {
            active: DEFAULT_TRAY_ICON_ACTIVE.into(),
            sleeping: DEFAULT_TRAY_ICON_SLEEPING.into(),
            listening: DEFAULT_TRAY_ICON_LISTENING.into(),
            generating: DEFAULT_TRAY_ICON_GENERATING.into(),
            disconnected: DEFAULT_TRAY_ICON_DISCONNECTED.into(),
        }
    }
}

/// Desktop notifications showing a turn's activity, tool calls and reply.
/// None are sent while the chat has keyboard focus.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TrayNotificationsConfig {
    pub enabled: bool,
    /// Idle ms before the notification closes, `500..=60000`. Reset by
    /// each event and paused while a turn is in flight.
    pub auto_hide_ms: u64,
    /// Ask for a short plain-text reply to a voice turn while the chat is
    /// unfocused or closed.
    pub brief_when_away: bool,
    /// Events that raise a notification; tray-icon left-click always does.
    pub wake_on: TrayNotificationsWakeConfig,
}

impl TrayNotificationsConfig {
    /// Idle timeout while continuous listening is active: `3 × auto_hide_ms`
    /// (saturating), leaving time to hear the reply and answer verbally.
    pub fn listen_auto_hide_ms(&self) -> u64 {
        self.auto_hide_ms.saturating_mul(3)
    }
}

impl Default for TrayNotificationsConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_TRAY_NOTIFICATIONS_ENABLED,
            auto_hide_ms: DEFAULT_TRAY_NOTIFICATIONS_AUTO_HIDE_MS,
            brief_when_away: DEFAULT_TRAY_NOTIFICATIONS_BRIEF_WHEN_AWAY,
            wake_on: TrayNotificationsWakeConfig::default(),
        }
    }
}

/// Events that raise a notification, each independent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TrayNotificationsWakeConfig {
    /// Every tool call.
    pub tool_call: bool,
    /// The first reply text of a turn.
    pub delta: bool,
    /// A failed turn.
    pub error: bool,
}

impl Default for TrayNotificationsWakeConfig {
    fn default() -> Self {
        Self {
            tool_call: DEFAULT_TRAY_NOTIFICATIONS_WAKE_TOOL_CALL,
            delta: DEFAULT_TRAY_NOTIFICATIONS_WAKE_DELTA,
            error: DEFAULT_TRAY_NOTIFICATIONS_WAKE_ERROR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listen_auto_hide_is_triple_the_idle_timeout_and_saturates() {
        for (auto_hide_ms, expected) in [(3000, 9000), (u64::MAX, u64::MAX)] {
            let notifications = TrayNotificationsConfig {
                auto_hide_ms,
                ..TrayNotificationsConfig::default()
            };
            assert_eq!(
                notifications.listen_auto_hide_ms(),
                expected,
                "auto_hide_ms={auto_hide_ms}"
            );
        }
    }
}

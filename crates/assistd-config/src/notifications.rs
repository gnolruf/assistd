use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_NOTIFICATIONS_AUTO_HIDE_MS, DEFAULT_NOTIFICATIONS_BRIEF_WHEN_AWAY,
    DEFAULT_NOTIFICATIONS_ENABLED, DEFAULT_TRAY_ICON_ACTIVE, DEFAULT_TRAY_ICON_DISCONNECTED,
    DEFAULT_TRAY_ICON_GENERATING, DEFAULT_TRAY_ICON_LISTENING, DEFAULT_TRAY_ICON_SLEEPING,
};

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
    /// The wake phrase was heard, or a push-to-talk capture is underway.
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

/// Desktop notifications from `assistd tray` showing a turn's activity, tool
/// calls and reply, plus the tray icon's names. No notification is sent while
/// the chat has keyboard focus.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct NotificationsConfig {
    pub enabled: bool,
    /// Idle ms before the notification closes, `500..=60000`. Reset by
    /// each event and paused while a turn is in flight.
    pub auto_hide_ms: u64,
    /// Ask for a short plain-text reply to a voice turn while the chat is
    /// unfocused or closed.
    pub brief_when_away: bool,
    /// Used by the tray icon whether or not notifications are enabled.
    pub icons: TrayIconsConfig,
}

impl NotificationsConfig {
    /// Idle timeout while continuous listening is active: `3 × auto_hide_ms`
    /// (saturating), leaving time to hear the reply and answer verbally.
    pub fn listen_auto_hide_ms(&self) -> u64 {
        self.auto_hide_ms.saturating_mul(3)
    }
}

impl Default for NotificationsConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_NOTIFICATIONS_ENABLED,
            auto_hide_ms: DEFAULT_NOTIFICATIONS_AUTO_HIDE_MS,
            brief_when_away: DEFAULT_NOTIFICATIONS_BRIEF_WHEN_AWAY,
            icons: TrayIconsConfig::default(),
        }
    }
}

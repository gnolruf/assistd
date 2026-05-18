use crate::defaults::{
    DEFAULT_TRAY_ICON_ACTIVE, DEFAULT_TRAY_ICON_DISCONNECTED, DEFAULT_TRAY_ICON_DROWSY,
    DEFAULT_TRAY_ICON_GENERATING, DEFAULT_TRAY_ICON_LISTENING, DEFAULT_TRAY_ICON_SLEEPING,
};
use serde::{Deserialize, Serialize};

/// System-tray icon settings for `assistd tray`.
///
/// Each field is the name of an icon in the user's freedesktop icon
/// theme. The defaults pick names that ship in every major theme
/// (Adwaita, Breeze, Papirus); users who want assistd-branded artwork
/// install custom icons under `~/.local/share/icons/<theme>/` and
/// reference them by name here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrayConfig {
    /// Daemon presence is `Active` and no query is in flight.
    #[serde(default = "default_tray_icon_active")]
    pub icon_active: String,

    /// Daemon presence is `Drowsy`.
    #[serde(default = "default_tray_icon_drowsy")]
    pub icon_drowsy: String,

    /// Daemon presence is `Sleeping`.
    #[serde(default = "default_tray_icon_sleeping")]
    pub icon_sleeping: String,

    /// Continuous listening is active.
    #[serde(default = "default_tray_icon_listening")]
    pub icon_listening: String,

    /// At least one query is currently streaming.
    #[serde(default = "default_tray_icon_generating")]
    pub icon_generating: String,

    /// Daemon socket is unreachable (not running, or connection dropped).
    #[serde(default = "default_tray_icon_disconnected")]
    pub icon_disconnected: String,
}

impl Default for TrayConfig {
    fn default() -> Self {
        Self {
            icon_active: default_tray_icon_active(),
            icon_drowsy: default_tray_icon_drowsy(),
            icon_sleeping: default_tray_icon_sleeping(),
            icon_listening: default_tray_icon_listening(),
            icon_generating: default_tray_icon_generating(),
            icon_disconnected: default_tray_icon_disconnected(),
        }
    }
}

fn default_tray_icon_active() -> String {
    DEFAULT_TRAY_ICON_ACTIVE.to_string()
}

fn default_tray_icon_drowsy() -> String {
    DEFAULT_TRAY_ICON_DROWSY.to_string()
}

fn default_tray_icon_sleeping() -> String {
    DEFAULT_TRAY_ICON_SLEEPING.to_string()
}

fn default_tray_icon_listening() -> String {
    DEFAULT_TRAY_ICON_LISTENING.to_string()
}

fn default_tray_icon_generating() -> String {
    DEFAULT_TRAY_ICON_GENERATING.to_string()
}

fn default_tray_icon_disconnected() -> String {
    DEFAULT_TRAY_ICON_DISCONNECTED.to_string()
}

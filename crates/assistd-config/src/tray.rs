use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_TRAY_POPUP_AUTO_HIDE_MS, DEFAULT_TRAY_POPUP_ENABLED, DEFAULT_TRAY_POPUP_HEIGHT,
    DEFAULT_TRAY_POPUP_OFFSET_X, DEFAULT_TRAY_POPUP_OFFSET_Y, DEFAULT_TRAY_POPUP_WAKE_DELTA,
    DEFAULT_TRAY_POPUP_WAKE_ERROR, DEFAULT_TRAY_POPUP_WAKE_TOOL_CALL, DEFAULT_TRAY_POPUP_WIDTH,
};

/// System-tray settings for `assistd tray`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TrayConfig {
    /// Floating activity popup. Parsed on every build; used only by
    /// `tray-popup` builds.
    pub popup: TrayPopupConfig,
}

/// Floating popup showing the latest reply and tool call. Geometry is in
/// logical pixels; ranges are validated only when `enabled`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TrayPopupConfig {
    pub enabled: bool,
    /// Screen corner (or centre) the offsets are measured from.
    pub anchor: PopupAnchor,
    /// Horizontal offset; positive moves right, so negative moves a
    /// right-anchored popup inward.
    pub offset_x: i32,
    /// Vertical offset; positive moves down.
    pub offset_y: i32,
    /// `100..=1200`.
    pub width: u32,
    /// `60..=800`.
    pub height: u32,
    /// Idle ms before auto-hide, `500..=60000`. Reset by each event and
    /// paused while a turn is in flight.
    pub auto_hide_ms: u64,
    /// Events that open the popup; tray-icon left-click always does.
    pub wake_on: TrayPopupWakeConfig,
}

impl TrayPopupConfig {
    /// Idle timeout while continuous listening is active: `3 × auto_hide_ms`
    /// (saturating), leaving time to hear the reply and answer verbally.
    pub fn listen_auto_hide_ms(&self) -> u64 {
        self.auto_hide_ms.saturating_mul(3)
    }
}

impl Default for TrayPopupConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_TRAY_POPUP_ENABLED,
            anchor: PopupAnchor::default(),
            offset_x: DEFAULT_TRAY_POPUP_OFFSET_X,
            offset_y: DEFAULT_TRAY_POPUP_OFFSET_Y,
            width: DEFAULT_TRAY_POPUP_WIDTH,
            height: DEFAULT_TRAY_POPUP_HEIGHT,
            auto_hide_ms: DEFAULT_TRAY_POPUP_AUTO_HIDE_MS,
            wake_on: TrayPopupWakeConfig::default(),
        }
    }
}

/// Events that open the popup, each independent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TrayPopupWakeConfig {
    /// Every tool call.
    pub tool_call: bool,
    /// The first reply text of a turn.
    pub delta: bool,
    /// A failed turn.
    pub error: bool,
}

impl Default for TrayPopupWakeConfig {
    fn default() -> Self {
        Self {
            tool_call: DEFAULT_TRAY_POPUP_WAKE_TOOL_CALL,
            delta: DEFAULT_TRAY_POPUP_WAKE_DELTA,
            error: DEFAULT_TRAY_POPUP_WAKE_ERROR,
        }
    }
}

/// Position on the focused output the popup offsets apply from.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PopupAnchor {
    TopLeft,
    #[default]
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listen_auto_hide_is_triple_the_idle_timeout_and_saturates() {
        for (auto_hide_ms, expected) in [(3000, 9000), (u64::MAX, u64::MAX)] {
            let popup = TrayPopupConfig {
                auto_hide_ms,
                ..TrayPopupConfig::default()
            };
            assert_eq!(
                popup.listen_auto_hide_ms(),
                expected,
                "auto_hide_ms={auto_hide_ms}"
            );
        }
    }

    #[test]
    fn popup_anchor_parses_every_variant() {
        for (raw, want) in [
            ("top_left", PopupAnchor::TopLeft),
            ("top_right", PopupAnchor::TopRight),
            ("bottom_left", PopupAnchor::BottomLeft),
            ("bottom_right", PopupAnchor::BottomRight),
            ("center", PopupAnchor::Center),
        ] {
            let toml_src = format!("anchor = \"{raw}\"\n");
            let popup: TrayPopupConfig = toml::from_str(&toml_src).expect("deserialize");
            assert_eq!(popup.anchor, want, "raw {raw}");
        }
    }
}

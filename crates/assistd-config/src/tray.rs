use crate::defaults::{
    DEFAULT_TRAY_POPUP_AUTO_HIDE_MS, DEFAULT_TRAY_POPUP_ENABLED, DEFAULT_TRAY_POPUP_HEIGHT,
    DEFAULT_TRAY_POPUP_OFFSET_X, DEFAULT_TRAY_POPUP_OFFSET_Y, DEFAULT_TRAY_POPUP_WAKE_DELTA,
    DEFAULT_TRAY_POPUP_WAKE_ERROR, DEFAULT_TRAY_POPUP_WAKE_TOOL_CALL, DEFAULT_TRAY_POPUP_WIDTH,
};
use serde::{Deserialize, Serialize};

/// System-tray settings for `assistd tray`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TrayConfig {
    /// Floating activity popup shown alongside the tray icon on
    /// `tray-popup` builds. Parsed on every build, so one config file
    /// works regardless of features.
    pub popup: TrayPopupConfig,
}

/// Geometry and wake-up policy for the floating activity popup, which
/// shows the latest assistant reply and tool call. Geometry is in
/// logical pixels.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct TrayPopupConfig {
    /// Enable the popup. When `false`, no popup is shown even on a
    /// `tray-popup` build.
    pub enabled: bool,

    /// Screen corner the popup anchors to. Offsets are measured from
    /// the anchored corner inward; see [`PopupAnchor`].
    pub anchor: PopupAnchor,

    /// Horizontal offset from the anchor, in pixels. Positive moves
    /// right; negative moves left, so a negative value moves a
    /// right-anchored popup inward.
    pub offset_x: i32,

    /// Vertical offset from the anchor, in pixels. Positive moves down;
    /// negative moves up.
    pub offset_y: i32,

    /// Popup width in pixels. Validated to 100..=1200.
    pub width: u32,

    /// Popup height in pixels. Validated to 60..=800.
    pub height: u32,

    /// Idle timeout before the popup auto-hides. Reset by every new
    /// event while visible, and held off entirely while a turn is
    /// in flight (so a long tool call doesn't blink the popup out
    /// mid-response). Validated to 500..=60000.
    pub auto_hide_ms: u64,

    /// Which events automatically open the popup. The tray-icon
    /// left-click always shows it regardless of these flags.
    pub wake_on: TrayPopupWakeConfig,
}

impl TrayPopupConfig {
    /// Idle timeout while continuous listening is active: three times
    /// `auto_hide_ms`, since the user may reply verbally and the popup
    /// must outlast the time it takes to hear it and start speaking.
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

/// Events that automatically open the popup. Each flag is independent;
/// any combination is valid. Tray-icon left-click is not gated by
/// these (it always opens).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct TrayPopupWakeConfig {
    /// Open on every tool call. Default: `true`.
    pub tool_call: bool,

    /// Open as soon as the model starts replying in a turn. Default:
    /// `true`.
    pub delta: bool,

    /// Open when a turn fails. Default: `true`.
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

/// Screen position the popup anchors to; the configured offsets are
/// applied from it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PopupAnchor {
    /// Top-left of the focused output.
    TopLeft,
    /// Top-right of the focused output. The default, as the most common
    /// tray location.
    #[default]
    TopRight,
    /// Bottom-left of the focused output.
    BottomLeft,
    /// Bottom-right of the focused output.
    BottomRight,
    /// Centred horizontally and vertically on the focused output.
    Center,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listen_auto_hide_is_triple_the_idle_timeout_and_saturates() {
        for (auto_hide_ms, expected) in [(3000, 9000), (u64::MAX, u64::MAX)] {
            let p = TrayPopupConfig {
                auto_hide_ms,
                ..TrayPopupConfig::default()
            };
            assert_eq!(
                p.listen_auto_hide_ms(),
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
            let p: TrayPopupConfig = toml::from_str(&toml_src).expect("deserialize");
            assert_eq!(p.anchor, want, "raw {raw}");
        }
    }
}

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
    /// Floating activity popup spawned alongside the tray icon (feature
    /// `tray-popup`). Configuration is parsed regardless of the build
    /// feature so a config file authored once works on every build.
    pub popup: TrayPopupConfig,
}

/// Geometry and wake-up policy for the floating activity popup.
///
/// The popup is a borderless ~360×120 window anchored near the
/// system-tray icon. It surfaces the daemon's most-recent assistant
/// reply text plus the last tool call so the user can glance at
/// activity without alt-tabbing to the chat TUI. Placement is delegated
/// to the compositor through `assistd-wm`; see
/// [`PopupAnchor`] for the supported corners.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct TrayPopupConfig {
    /// Globally enable or disable the popup. When `false`, the popup
    /// task is not spawned even on a build with `--features tray-popup`.
    pub enabled: bool,

    /// Screen corner the popup anchors to. Offsets are measured from
    /// the anchored corner inward; see [`PopupAnchor`].
    pub anchor: PopupAnchor,

    /// Horizontal offset from the anchor, in pixels. Positive moves
    /// right; negative moves left. For a right-anchored popup, the
    /// default `-10` nudges the popup inward by 10 px so it doesn't
    /// kiss the screen edge.
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
    /// Idle timeout while the daemon's continuous listener is active. The
    /// user can reply verbally without touching a key, so the popup has to
    /// outlast the time it takes to hear it and start speaking.
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
    /// Open on `Event::ToolCall` — catches every MCP / bash / web
    /// invocation. Default: `true`.
    pub tool_call: bool,

    /// Open on the first `Event::LastDelta` of a turn — i.e. as soon as
    /// the model starts replying. Default: `true`. Flip to `false` if
    /// you live in the chat TUI and don't want a popup on every reply.
    pub delta: bool,

    /// Open on `Event::Error`. Useful for noticing failures that would
    /// otherwise only land in the tracing log. Default: `true`.
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

/// Screen corner the popup anchors to. The compositor (i3 / sway) is
/// responsible for the actual placement; the tray sends a `floating
/// enable, resize set W H, move position …` IPC sequence templated
/// from this variant plus the configured offsets.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PopupAnchor {
    /// Top-left of the focused output.
    TopLeft,
    /// Top-right of the focused output. Default — matches the most
    /// common tray location on Waybar / xfce-panel / KDE.
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
    fn default_tray_includes_default_popup() {
        let t = TrayConfig::default();
        assert_eq!(t.popup, TrayPopupConfig::default());
        assert!(t.popup.enabled);
        assert_eq!(t.popup.anchor, PopupAnchor::TopRight);
    }

    #[test]
    fn popup_defaults_match_constants() {
        let p = TrayPopupConfig::default();
        assert_eq!(p.enabled, DEFAULT_TRAY_POPUP_ENABLED);
        assert_eq!(p.offset_x, DEFAULT_TRAY_POPUP_OFFSET_X);
        assert_eq!(p.offset_y, DEFAULT_TRAY_POPUP_OFFSET_Y);
        assert_eq!(p.width, DEFAULT_TRAY_POPUP_WIDTH);
        assert_eq!(p.height, DEFAULT_TRAY_POPUP_HEIGHT);
        assert_eq!(p.auto_hide_ms, DEFAULT_TRAY_POPUP_AUTO_HIDE_MS);
        assert!(p.listen_auto_hide_ms() > p.auto_hide_ms);
        assert_eq!(p.wake_on.tool_call, DEFAULT_TRAY_POPUP_WAKE_TOOL_CALL);
        assert_eq!(p.wake_on.delta, DEFAULT_TRAY_POPUP_WAKE_DELTA);
        assert_eq!(p.wake_on.error, DEFAULT_TRAY_POPUP_WAKE_ERROR);
    }

    #[test]
    fn popup_roundtrips_through_toml() {
        let p = TrayPopupConfig {
            anchor: PopupAnchor::BottomLeft,
            offset_x: 5,
            offset_y: -5,
            width: 500,
            height: 200,
            auto_hide_ms: 7000,
            enabled: false,
            wake_on: TrayPopupWakeConfig {
                tool_call: false,
                delta: true,
                error: false,
            },
        };
        let s = toml::to_string(&p).expect("serialize");
        let back: TrayPopupConfig = toml::from_str(&s).expect("deserialize");
        assert_eq!(p, back);
    }

    #[test]
    fn popup_anchor_serializes_as_snake_case() {
        let s = toml::to_string(&TrayPopupConfig {
            anchor: PopupAnchor::TopLeft,
            ..TrayPopupConfig::default()
        })
        .expect("serialize");
        assert!(
            s.contains(r#"anchor = "top_left""#),
            "anchor should serialize as snake_case: {s}"
        );
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

    #[test]
    fn missing_popup_section_uses_defaults() {
        let t: TrayConfig = toml::from_str("").expect("deserialize");
        assert_eq!(t.popup, TrayPopupConfig::default());
    }

    #[test]
    fn missing_wake_subkey_uses_defaults() {
        // wake_on default is all-true; the popup section may omit it.
        let toml_src = r#"
            [popup]
            width = 400
        "#;
        let t: TrayConfig = toml::from_str(toml_src).expect("deserialize");
        assert_eq!(t.popup.width, 400);
        assert_eq!(t.popup.wake_on, TrayPopupWakeConfig::default());
    }
}

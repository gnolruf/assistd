use crate::defaults::{
    DEFAULT_TRAY_ICON_ACTIVE, DEFAULT_TRAY_ICON_DISCONNECTED, DEFAULT_TRAY_ICON_DROWSY,
    DEFAULT_TRAY_ICON_GENERATING, DEFAULT_TRAY_ICON_LISTENING, DEFAULT_TRAY_ICON_SLEEPING,
    DEFAULT_TRAY_POPUP_AUTO_HIDE_MS, DEFAULT_TRAY_POPUP_ENABLED, DEFAULT_TRAY_POPUP_HEIGHT,
    DEFAULT_TRAY_POPUP_OFFSET_X, DEFAULT_TRAY_POPUP_OFFSET_Y, DEFAULT_TRAY_POPUP_TRUNCATE_CHARS,
    DEFAULT_TRAY_POPUP_WAKE_DELTA, DEFAULT_TRAY_POPUP_WAKE_ERROR,
    DEFAULT_TRAY_POPUP_WAKE_TOOL_CALL, DEFAULT_TRAY_POPUP_WIDTH,
};
use serde::{Deserialize, Serialize};

/// System-tray icon settings for `assistd tray`.
///
/// Each field is the name of an icon in the user's freedesktop icon
/// theme. The defaults pick names that ship in every major theme
/// (Adwaita, Breeze, Papirus); users who want assistd-branded artwork
/// install custom icons under `~/.local/share/icons/<theme>/` and
/// reference them by name here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

    /// Floating activity popup spawned alongside the tray icon (feature
    /// `tray-popup`). Configuration is parsed regardless of the build
    /// feature so a config file authored once works on every build.
    #[serde(default)]
    pub popup: TrayPopupConfig,
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
            popup: TrayPopupConfig::default(),
        }
    }
}

/// Geometry and wake-up policy for the floating activity popup.
///
/// The popup is a borderless ~360×120 window anchored near the
/// system-tray icon. It surfaces the daemon's most-recent assistant
/// reply text plus the last tool call so the user can glance at
/// activity without alt-tabbing to the chat TUI. Placement is delegated
/// to the compositor through `assistd-wm`; see
/// [`PopupAnchor`] for the supported corners.
///
/// Only `PartialEq` is derived (not `Eq`) because `scale_factor` is
/// an `Option<f32>` and `f32` has NaN, which breaks total equality.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrayPopupConfig {
    /// Globally enable or disable the popup. When `false`, the popup
    /// task is not spawned even on a build with `--features tray-popup`.
    #[serde(default = "default_popup_enabled")]
    pub enabled: bool,

    /// Screen corner the popup anchors to. Offsets are measured from
    /// the anchored corner inward; see [`PopupAnchor`].
    #[serde(default)]
    pub anchor: PopupAnchor,

    /// Horizontal offset from the anchor, in pixels. Positive moves
    /// right; negative moves left. For a right-anchored popup, the
    /// default `-10` nudges the popup inward by 10 px so it doesn't
    /// kiss the screen edge.
    #[serde(default = "default_popup_offset_x")]
    pub offset_x: i32,

    /// Vertical offset from the anchor, in pixels. Positive moves down;
    /// negative moves up.
    #[serde(default = "default_popup_offset_y")]
    pub offset_y: i32,

    /// Popup width in pixels. Validated to 100..=1200.
    #[serde(default = "default_popup_width")]
    pub width: u32,

    /// Popup height in pixels. Validated to 60..=800.
    #[serde(default = "default_popup_height")]
    pub height: u32,

    /// Idle timeout before the popup auto-hides. Reset by every new
    /// event while visible. Validated to 500..=60000.
    #[serde(default = "default_popup_auto_hide_ms")]
    pub auto_hide_ms: u64,

    /// Cap on the body text rendered in the popup. The displayed text
    /// is the **last** `truncate_chars` codepoints of the running
    /// reply, so partial Unicode is never split. Validated to
    /// 1..=10000.
    #[serde(default = "default_popup_truncate_chars")]
    pub truncate_chars: usize,

    /// Which events automatically open the popup. The tray-icon
    /// left-click always shows it regardless of these flags.
    #[serde(default)]
    pub wake_on: TrayPopupWakeConfig,

    /// Manual HiDPI scale factor for pre-positioning. When set, the
    /// popup's pre-position is computed using `width * scale_factor`
    /// and `height * scale_factor` so the window lands at the precise
    /// corner from its very first map on a HiDPI display. When
    /// unset (the default), the tray reads `Xft.dpi` from `xrdb` at
    /// spawn time and computes the scale automatically — that works
    /// for most i3 setups but misses XSETTINGS-only configurations
    /// (where `xrdb -query` returns no `Xft.dpi` line). Set this to
    /// your display's actual scale (e.g. `1.5`, `1.6667`, `2.0`)
    /// when the auto-detect fails. A value of `1.0` opts out of any
    /// scaling.
    #[serde(default)]
    pub scale_factor: Option<f32>,
}

impl Default for TrayPopupConfig {
    fn default() -> Self {
        Self {
            enabled: default_popup_enabled(),
            anchor: PopupAnchor::default(),
            offset_x: default_popup_offset_x(),
            offset_y: default_popup_offset_y(),
            width: default_popup_width(),
            height: default_popup_height(),
            auto_hide_ms: default_popup_auto_hide_ms(),
            truncate_chars: default_popup_truncate_chars(),
            wake_on: TrayPopupWakeConfig::default(),
            scale_factor: None,
        }
    }
}

/// Events that automatically open the popup. Each flag is independent;
/// any combination is valid. Tray-icon left-click is not gated by
/// these (it always opens).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrayPopupWakeConfig {
    /// Open on `Event::ToolCall` — catches every MCP / bash / web
    /// invocation. Default: `true`.
    #[serde(default = "default_popup_wake_tool_call")]
    pub tool_call: bool,

    /// Open on the first `Event::LastDelta` of a turn — i.e. as soon as
    /// the model starts replying. Default: `true`. Flip to `false` if
    /// you live in the chat TUI and don't want a popup on every reply.
    #[serde(default = "default_popup_wake_delta")]
    pub delta: bool,

    /// Open on `Event::Error`. Useful for noticing failures that would
    /// otherwise only land in the tracing log. Default: `true`.
    #[serde(default = "default_popup_wake_error")]
    pub error: bool,
}

impl Default for TrayPopupWakeConfig {
    fn default() -> Self {
        Self {
            tool_call: default_popup_wake_tool_call(),
            delta: default_popup_wake_delta(),
            error: default_popup_wake_error(),
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

fn default_popup_enabled() -> bool {
    DEFAULT_TRAY_POPUP_ENABLED
}

fn default_popup_offset_x() -> i32 {
    DEFAULT_TRAY_POPUP_OFFSET_X
}

fn default_popup_offset_y() -> i32 {
    DEFAULT_TRAY_POPUP_OFFSET_Y
}

fn default_popup_width() -> u32 {
    DEFAULT_TRAY_POPUP_WIDTH
}

fn default_popup_height() -> u32 {
    DEFAULT_TRAY_POPUP_HEIGHT
}

fn default_popup_auto_hide_ms() -> u64 {
    DEFAULT_TRAY_POPUP_AUTO_HIDE_MS
}

fn default_popup_truncate_chars() -> usize {
    DEFAULT_TRAY_POPUP_TRUNCATE_CHARS
}

fn default_popup_wake_tool_call() -> bool {
    DEFAULT_TRAY_POPUP_WAKE_TOOL_CALL
}

fn default_popup_wake_delta() -> bool {
    DEFAULT_TRAY_POPUP_WAKE_DELTA
}

fn default_popup_wake_error() -> bool {
    DEFAULT_TRAY_POPUP_WAKE_ERROR
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
        assert_eq!(p.truncate_chars, DEFAULT_TRAY_POPUP_TRUNCATE_CHARS);
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
            truncate_chars: 1024,
            enabled: false,
            wake_on: TrayPopupWakeConfig {
                tool_call: false,
                delta: true,
                error: false,
            },
            scale_factor: Some(1.5),
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
        // Mimic an upgrade: existing config has [tray] without
        // [tray.popup]. Defaults must fill in.
        let toml_src = r#"
            icon_active = "user-available"
            icon_drowsy = "user-away"
            icon_sleeping = "user-offline"
            icon_listening = "audio-input-microphone"
            icon_generating = "system-run"
            icon_disconnected = "network-offline"
        "#;
        let t: TrayConfig = toml::from_str(toml_src).expect("deserialize");
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

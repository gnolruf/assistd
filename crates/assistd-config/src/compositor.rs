use serde::{Deserialize, Serialize};

/// Supported tiling compositors.
///
/// `Auto` (the default) defers the choice to runtime detection; see
/// [`detect_from_env`]. Existing configs with an explicit
/// `type = "i3"` / `"sway"` / `"hyprland"` continue to override.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CompositorType {
    Auto,
    I3,
    Sway,
    Hyprland,
}

/// Compositor integration settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositorConfig {
    /// Which compositor to integrate with.
    #[serde(rename = "type")]
    pub compositor_type: CompositorType,
}

impl Default for CompositorConfig {
    fn default() -> Self {
        Self {
            compositor_type: CompositorType::Auto,
        }
    }
}

/// Resolve a compositor from the running session's environment.
///
/// Priority:
/// 1. `$SWAYSOCK` set → Sway (Sway exports this for child processes)
/// 2. `$I3SOCK` set → i3
/// 3. `$HYPRLAND_INSTANCE_SIGNATURE` set → Hyprland
/// 4. `$XDG_CURRENT_DESKTOP` matched case-insensitively
/// 5. otherwise `None` (caller falls back to `NoWindowManager`)
///
/// Pure / arg-driven so it's testable without mutating the process
/// environment. Mirrors the style of `detect_wayland_compositor_from_env`
/// in `assistd-tools/src/commands/screenshot.rs`.
pub fn detect_from_env(
    has_swaysock: bool,
    has_i3sock: bool,
    has_hypr_signature: bool,
    xdg_current_desktop: Option<&str>,
) -> Option<CompositorType> {
    if has_swaysock {
        return Some(CompositorType::Sway);
    }
    if has_i3sock {
        return Some(CompositorType::I3);
    }
    if has_hypr_signature {
        return Some(CompositorType::Hyprland);
    }
    match xdg_current_desktop
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "sway" => Some(CompositorType::Sway),
        "i3" => Some(CompositorType::I3),
        "hyprland" => Some(CompositorType::Hyprland),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_is_default() {
        assert_eq!(
            CompositorConfig::default().compositor_type,
            CompositorType::Auto
        );
    }

    #[test]
    fn auto_serde_roundtrip() {
        let toml_in = "type = \"auto\"\n";
        let parsed: CompositorConfig = toml::from_str(toml_in).unwrap();
        assert_eq!(parsed.compositor_type, CompositorType::Auto);
        let toml_out = toml::to_string(&parsed).unwrap();
        assert!(toml_out.contains("\"auto\""), "{toml_out}");
    }

    #[test]
    fn explicit_sway_overrides_default() {
        let parsed: CompositorConfig = toml::from_str("type = \"sway\"\n").unwrap();
        assert_eq!(parsed.compositor_type, CompositorType::Sway);
    }

    #[test]
    fn detect_swaysock_wins_over_i3sock() {
        assert_eq!(
            detect_from_env(true, true, false, None),
            Some(CompositorType::Sway)
        );
    }

    #[test]
    fn detect_i3sock_only() {
        assert_eq!(
            detect_from_env(false, true, false, None),
            Some(CompositorType::I3)
        );
    }

    #[test]
    fn detect_hypr_signature() {
        assert_eq!(
            detect_from_env(false, false, true, None),
            Some(CompositorType::Hyprland)
        );
    }

    #[test]
    fn detect_falls_back_to_xdg_current_desktop() {
        assert_eq!(
            detect_from_env(false, false, false, Some("sway")),
            Some(CompositorType::Sway)
        );
        assert_eq!(
            detect_from_env(false, false, false, Some("Hyprland")),
            Some(CompositorType::Hyprland)
        );
        assert_eq!(
            detect_from_env(false, false, false, Some("i3")),
            Some(CompositorType::I3)
        );
    }

    #[test]
    fn detect_returns_none_when_nothing_matches() {
        assert_eq!(detect_from_env(false, false, false, None), None);
        assert_eq!(detect_from_env(false, false, false, Some("")), None);
        assert_eq!(detect_from_env(false, false, false, Some("KDE")), None);
        assert_eq!(detect_from_env(false, false, false, Some("GNOME")), None);
    }

    #[test]
    fn detect_xdg_match_is_case_insensitive() {
        // XDG_CURRENT_DESKTOP capitalization varies (`sway` vs `Sway`).
        assert_eq!(
            detect_from_env(false, false, false, Some("SWAY")),
            Some(CompositorType::Sway)
        );
    }
}

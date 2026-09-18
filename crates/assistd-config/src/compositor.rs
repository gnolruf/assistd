use serde::{Deserialize, Serialize};

/// Supported tiling compositors. `Auto` defers to [`detect_from_env`].
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
#[serde(default, deny_unknown_fields)]
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

/// The session variables compositor auto-detection reads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionEnv {
    pub swaysock: bool,
    pub i3sock: bool,
    pub hyprland_signature: bool,
    pub xdg_current_desktop: Option<String>,
}

impl SessionEnv {
    /// Read `$SWAYSOCK`, `$I3SOCK`, `$HYPRLAND_INSTANCE_SIGNATURE` and
    /// `$XDG_CURRENT_DESKTOP` from the current process.
    pub fn from_process() -> Self {
        Self {
            swaysock: std::env::var_os("SWAYSOCK").is_some(),
            i3sock: std::env::var_os("I3SOCK").is_some(),
            hyprland_signature: std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some(),
            xdg_current_desktop: std::env::var("XDG_CURRENT_DESKTOP").ok(),
        }
    }
}

/// Resolve a compositor from the session environment, in priority
/// order: `$SWAYSOCK`, `$I3SOCK`, `$HYPRLAND_INSTANCE_SIGNATURE`, then
/// `$XDG_CURRENT_DESKTOP` case-insensitively.
pub fn detect_from_env(env: &SessionEnv) -> Option<CompositorType> {
    if env.swaysock {
        return Some(CompositorType::Sway);
    }
    if env.i3sock {
        return Some(CompositorType::I3);
    }
    if env.hyprland_signature {
        return Some(CompositorType::Hyprland);
    }
    match env
        .xdg_current_desktop
        .as_deref()
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
            detect_from_env(&SessionEnv {
                swaysock: true,
                i3sock: true,
                ..SessionEnv::default()
            }),
            Some(CompositorType::Sway)
        );
    }

    #[test]
    fn detect_i3sock_only() {
        assert_eq!(
            detect_from_env(&SessionEnv {
                i3sock: true,
                ..SessionEnv::default()
            }),
            Some(CompositorType::I3)
        );
    }

    #[test]
    fn detect_hypr_signature() {
        assert_eq!(
            detect_from_env(&SessionEnv {
                hyprland_signature: true,
                ..SessionEnv::default()
            }),
            Some(CompositorType::Hyprland)
        );
    }

    #[test]
    fn detect_falls_back_to_xdg_current_desktop() {
        assert_eq!(
            detect_from_env(&SessionEnv {
                xdg_current_desktop: Some("sway".into()),
                ..SessionEnv::default()
            }),
            Some(CompositorType::Sway)
        );
        assert_eq!(
            detect_from_env(&SessionEnv {
                xdg_current_desktop: Some("Hyprland".into()),
                ..SessionEnv::default()
            }),
            Some(CompositorType::Hyprland)
        );
        assert_eq!(
            detect_from_env(&SessionEnv {
                xdg_current_desktop: Some("i3".into()),
                ..SessionEnv::default()
            }),
            Some(CompositorType::I3)
        );
    }

    #[test]
    fn detect_returns_none_when_nothing_matches() {
        assert_eq!(detect_from_env(&SessionEnv::default()), None);
        assert_eq!(
            detect_from_env(&SessionEnv {
                xdg_current_desktop: Some(String::new()),
                ..SessionEnv::default()
            }),
            None
        );
        assert_eq!(
            detect_from_env(&SessionEnv {
                xdg_current_desktop: Some("KDE".into()),
                ..SessionEnv::default()
            }),
            None
        );
        assert_eq!(
            detect_from_env(&SessionEnv {
                xdg_current_desktop: Some("GNOME".into()),
                ..SessionEnv::default()
            }),
            None
        );
    }

    #[test]
    fn detect_xdg_match_is_case_insensitive() {
        // XDG_CURRENT_DESKTOP capitalization varies (`sway` vs `Sway`).
        assert_eq!(
            detect_from_env(&SessionEnv {
                xdg_current_desktop: Some("SWAY".into()),
                ..SessionEnv::default()
            }),
            Some(CompositorType::Sway)
        );
    }
}

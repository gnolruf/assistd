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
    fn compositor_type_parses_lowercase_names() {
        for (raw, want) in [
            ("auto", CompositorType::Auto),
            ("i3", CompositorType::I3),
            ("sway", CompositorType::Sway),
            ("hyprland", CompositorType::Hyprland),
        ] {
            let parsed: CompositorConfig =
                toml::from_str(&format!("type = \"{raw}\"\n")).expect("deserialize");
            assert_eq!(parsed.compositor_type, want, "raw {raw}");
        }
    }

    #[test]
    fn detect_from_env_follows_priority_order() {
        let xdg = |d: &str| SessionEnv {
            xdg_current_desktop: Some(d.into()),
            ..SessionEnv::default()
        };
        let cases = [
            (
                "swaysock beats i3sock",
                SessionEnv {
                    swaysock: true,
                    i3sock: true,
                    ..SessionEnv::default()
                },
                Some(CompositorType::Sway),
            ),
            (
                "i3sock only",
                SessionEnv {
                    i3sock: true,
                    ..SessionEnv::default()
                },
                Some(CompositorType::I3),
            ),
            (
                "hyprland signature",
                SessionEnv {
                    hyprland_signature: true,
                    ..SessionEnv::default()
                },
                Some(CompositorType::Hyprland),
            ),
            (
                "socket beats xdg",
                SessionEnv {
                    i3sock: true,
                    xdg_current_desktop: Some("sway".into()),
                    ..SessionEnv::default()
                },
                Some(CompositorType::I3),
            ),
            ("xdg sway", xdg("sway"), Some(CompositorType::Sway)),
            ("xdg i3", xdg("i3"), Some(CompositorType::I3)),
            (
                "xdg Hyprland",
                xdg("Hyprland"),
                Some(CompositorType::Hyprland),
            ),
            ("xdg SWAY", xdg("SWAY"), Some(CompositorType::Sway)),
            ("nothing set", SessionEnv::default(), None),
            ("xdg empty", xdg(""), None),
            ("xdg KDE", xdg("KDE"), None),
            ("xdg GNOME", xdg("GNOME"), None),
        ];
        for (label, env, want) in cases {
            assert_eq!(detect_from_env(&env), want, "{label}");
        }
    }
}

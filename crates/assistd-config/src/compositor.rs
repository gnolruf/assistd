use serde::{Deserialize, Serialize};

use crate::settings::{DescribeSettings, describe_fields};

/// The `compositor.type` values. `Auto` defers to [`detect_from_env`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CompositorType {
    Auto,
    I3,
    Sway,
    Hyprland,
}

/// A supported compositor, once `Auto` is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compositor {
    I3,
    Sway,
    Hyprland,
}

/// The compositor that window management and screenshots target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CompositorConfig {
    /// The `type` key.
    #[serde(rename = "type")]
    pub compositor_type: CompositorType,
}

impl DescribeSettings for CompositorConfig {
    describe_fields! {
        compositor_type as "type" => (
            "compositor",
            "Compositor to manage windows and take screenshots on; auto detects"
        ),
    }
}

impl CompositorConfig {
    /// The configured compositor, else the one the process environment
    /// names; `None` when `Auto` detects none.
    pub fn resolve(&self) -> Option<Compositor> {
        match self.compositor_type {
            CompositorType::Auto => detect_from_env(&SessionEnv::from_process()),
            CompositorType::I3 => Some(Compositor::I3),
            CompositorType::Sway => Some(Compositor::Sway),
            CompositorType::Hyprland => Some(Compositor::Hyprland),
        }
    }
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
pub fn detect_from_env(env: &SessionEnv) -> Option<Compositor> {
    if env.swaysock {
        return Some(Compositor::Sway);
    }
    if env.i3sock {
        return Some(Compositor::I3);
    }
    if env.hyprland_signature {
        return Some(Compositor::Hyprland);
    }
    match env
        .xdg_current_desktop
        .as_deref()
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "sway" => Some(Compositor::Sway),
        "i3" => Some(Compositor::I3),
        "hyprland" => Some(Compositor::Hyprland),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                Some(Compositor::Sway),
            ),
            (
                "hyprland signature",
                SessionEnv {
                    hyprland_signature: true,
                    ..SessionEnv::default()
                },
                Some(Compositor::Hyprland),
            ),
            (
                "socket beats xdg",
                SessionEnv {
                    i3sock: true,
                    xdg_current_desktop: Some("sway".into()),
                    ..SessionEnv::default()
                },
                Some(Compositor::I3),
            ),
            ("xdg SWAY", xdg("SWAY"), Some(Compositor::Sway)),
            ("nothing set", SessionEnv::default(), None),
            ("xdg KDE", xdg("KDE"), None),
        ];
        for (label, env, want) in cases {
            assert_eq!(detect_from_env(&env), want, "{label}");
        }
    }
}

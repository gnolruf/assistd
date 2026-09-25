//! Display-server and Wayland-compositor detection from the environment.

use super::Backend;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum WaylandCompositor {
    Sway,
    Hyprland,
    Unknown(String),
}

/// The environment variables a display server advertises itself with.
struct DisplayEnv<'a> {
    session_type: Option<&'a str>,
    wayland_display: bool,
    x_display: bool,
}

/// The environment variables a Wayland compositor advertises itself with.
struct WaylandEnv<'a> {
    swaysock: bool,
    hyprland_signature: bool,
    current_desktop: Option<&'a str>,
}

pub(super) fn detect_backend() -> Result<Backend, &'static str> {
    let session_type = std::env::var("XDG_SESSION_TYPE").ok();
    detect_backend_in(DisplayEnv {
        session_type: session_type.as_deref(),
        wayland_display: std::env::var_os("WAYLAND_DISPLAY").is_some(),
        x_display: std::env::var_os("DISPLAY").is_some(),
    })
}

/// Trusts `XDG_SESSION_TYPE`, then prefers Wayland when both displays are
/// set: under XWayland grim still captures X clients, but not the inverse.
fn detect_backend_in(env: DisplayEnv<'_>) -> Result<Backend, &'static str> {
    match env.session_type {
        Some("wayland") => return Ok(Backend::Wayland),
        Some("x11") => return Ok(Backend::X11),
        _ => {}
    }
    match (env.wayland_display, env.x_display) {
        (true, _) => Ok(Backend::Wayland),
        (false, true) => Ok(Backend::X11),
        (false, false) => Err("no display server detected (no WAYLAND_DISPLAY or DISPLAY)"),
    }
}

pub(super) fn detect_wayland_compositor() -> WaylandCompositor {
    let current_desktop = std::env::var("XDG_CURRENT_DESKTOP").ok();
    detect_wayland_compositor_in(WaylandEnv {
        swaysock: std::env::var_os("SWAYSOCK").is_some(),
        hyprland_signature: std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some(),
        current_desktop: current_desktop.as_deref(),
    })
}

fn detect_wayland_compositor_in(env: WaylandEnv<'_>) -> WaylandCompositor {
    if env.swaysock {
        return WaylandCompositor::Sway;
    }
    if env.hyprland_signature {
        return WaylandCompositor::Hyprland;
    }
    let desktop = env.current_desktop.unwrap_or("");
    match desktop.to_ascii_lowercase().as_str() {
        "sway" => WaylandCompositor::Sway,
        "hyprland" => WaylandCompositor::Hyprland,
        _ => WaylandCompositor::Unknown(desktop.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_backend_prefers_session_type_then_wayland() {
        let none = Err("no display server detected (no WAYLAND_DISPLAY or DISPLAY)");
        let cases = [
            (Some("wayland"), false, false, Ok(Backend::Wayland)),
            (Some("x11"), false, false, Ok(Backend::X11)),
            (None, false, false, none),
            (Some("tty"), false, false, none),
            (Some("tty"), false, true, Ok(Backend::X11)),
            (None, true, true, Ok(Backend::Wayland)),
            (None, false, true, Ok(Backend::X11)),
        ];
        for (session_type, wayland_display, x_display, expected) in cases {
            assert_eq!(
                detect_backend_in(DisplayEnv {
                    session_type,
                    wayland_display,
                    x_display,
                }),
                expected,
                "session={session_type:?} wayland={wayland_display} x={x_display}"
            );
        }
    }

    #[test]
    fn detect_wayland_compositor_from_env() {
        let cases = [
            (true, true, Some("KDE"), WaylandCompositor::Sway),
            (false, true, None, WaylandCompositor::Hyprland),
            (false, false, Some("sway"), WaylandCompositor::Sway),
            (false, false, Some("Hyprland"), WaylandCompositor::Hyprland),
            (
                false,
                false,
                Some("KDE"),
                WaylandCompositor::Unknown("KDE".into()),
            ),
            (
                false,
                false,
                None,
                WaylandCompositor::Unknown(String::new()),
            ),
        ];
        for (swaysock, hyprland_signature, current_desktop, expected) in cases {
            assert_eq!(
                detect_wayland_compositor_in(WaylandEnv {
                    swaysock,
                    hyprland_signature,
                    current_desktop,
                }),
                expected,
                "sway={swaysock} hypr={hyprland_signature} desktop={current_desktop:?}"
            );
        }
    }
}

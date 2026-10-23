//! Display-server detection from the environment.

use super::Backend;

/// The environment variables a display server advertises itself with.
#[derive(Clone, Copy)]
struct DisplayEnv<'a> {
    session_type: Option<&'a str>,
    wayland_display: bool,
    x_display: bool,
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
}

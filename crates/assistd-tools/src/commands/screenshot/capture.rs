//! Running the capture binaries and mapping their failures to output.

use std::io;
use std::time::Duration;

use assistd_config::Compositor;
use assistd_utils::text::human_size;
use serde_json::Value;
use tokio::process::Command as ProcCommand;

use super::geometry::{find_focused_sway_rect, parse_hyprland_geom, parse_xrandr_monitor_geom};
use super::{Backend, Target};
use crate::attachment::MAX_IMAGE_BYTES;
use crate::command::{CommandOutput, Hint, error_line};
use crate::exec::{SPAWN_FAILED_EXIT, TIMEOUT_EXIT, WaitOutcome, capture, exit_code};

const STDERR_TAIL_LINES: usize = 20;

#[derive(Debug)]
pub(super) enum CaptureError {
    BinaryMissing {
        binary: String,
    },
    Spawn {
        binary: String,
        msg: String,
    },
    Timeout,
    NonZero {
        binary: String,
        status: i32,
        stderr_tail: String,
    },
    EmptyOutput {
        binary: String,
    },
    TooLarge {
        size: usize,
    },
    FocusedUnsupportedOnWayland {
        compositor: String,
    },
    Parse {
        what: String,
    },
}

impl CaptureError {
    /// The failed `screenshot` output reporting this error.
    pub(super) fn to_output(&self) -> CommandOutput {
        CommandOutput::failed(self.exit_code(), self.error_line().into_bytes())
    }

    fn exit_code(&self) -> i32 {
        match self {
            Self::BinaryMissing { .. } => SPAWN_FAILED_EXIT,
            Self::Timeout => TIMEOUT_EXIT,
            Self::FocusedUnsupportedOnWayland { .. } => 2,
            Self::Spawn { .. }
            | Self::NonZero { .. }
            | Self::EmptyOutput { .. }
            | Self::TooLarge { .. }
            | Self::Parse { .. } => 1,
        }
    }

    fn error_line(&self) -> String {
        let (what, hint, recovery): (String, Hint, String) = match self {
            Self::BinaryMissing { binary } => (
                format!("backend binary not found: {binary}"),
                Hint::Install,
                install_hint(binary).into(),
            ),
            Self::Spawn { binary, msg } => (
                format!("spawn failed: {binary}: {msg}"),
                Hint::Check,
                format!("{binary} runs from your shell"),
            ),
            Self::Timeout => (
                "capture timed out".into(),
                Hint::Try,
                "screenshot again or check the compositor is responsive".into(),
            ),
            Self::NonZero {
                binary,
                status,
                stderr_tail,
            } => {
                let what = if stderr_tail.is_empty() {
                    format!("{binary} exited {status}")
                } else {
                    format!("{binary} exited {status}: {stderr_tail}")
                };
                (what, Hint::Try, "a different target or backend".into())
            }
            Self::EmptyOutput { binary } => (
                format!("{binary} produced no image bytes"),
                Hint::Try,
                "screenshot --full".into(),
            ),
            Self::TooLarge { size } => (
                format!(
                    "captured PNG too large ({} > {} max)",
                    human_size(*size as u64),
                    human_size(MAX_IMAGE_BYTES),
                ),
                Hint::Try,
                "--focused, or capture a single monitor".into(),
            ),
            Self::FocusedUnsupportedOnWayland { compositor } => (
                format!("--focused not supported on Wayland compositor: {compositor}"),
                Hint::Use,
                "screenshot --full (supported compositors for --focused: sway, Hyprland)".into(),
            ),
            Self::Parse { what } => (
                format!("failed to parse: {what}"),
                Hint::Try,
                "screenshot --full".into(),
            ),
        };
        error_line("screenshot", what, hint, recovery)
    }
}

/// Capture `target` as PNG bytes with the binaries `backend` uses.
pub(super) async fn capture_target(
    backend: Backend,
    compositor: Option<Compositor>,
    target: &Target,
    deadline: Duration,
) -> Result<Vec<u8>, CaptureError> {
    match (backend, target) {
        (Backend::X11, Target::Full) => run_capture("maim", &[], deadline).await,
        (Backend::X11, Target::Focused) => capture_x11_focused(deadline).await,
        (Backend::X11, Target::Monitor(name)) => capture_x11_monitor(name, deadline).await,
        (Backend::Wayland, Target::Full) => run_capture("grim", &["-"], deadline).await,
        (Backend::Wayland, Target::Focused) => capture_wayland_focused(compositor, deadline).await,
        (Backend::Wayland, Target::Monitor(name)) => {
            run_capture("grim", &["-o", name, "-"], deadline).await
        }
    }
}

async fn capture_x11_monitor(monitor: &str, deadline: Duration) -> Result<Vec<u8>, CaptureError> {
    let raw = run_capture("xrandr", &["--listmonitors"], deadline).await?;
    let listing = String::from_utf8_lossy(&raw);
    let geom = parse_xrandr_monitor_geom(&listing, monitor).ok_or_else(|| CaptureError::Parse {
        what: format!("monitor `{monitor}` not found in xrandr output"),
    })?;
    run_capture("maim", &["-g", &geom], deadline).await
}

async fn capture_x11_focused(deadline: Duration) -> Result<Vec<u8>, CaptureError> {
    let id_bytes = run_capture("xdotool", &["getactivewindow"], deadline).await?;
    let window_id = String::from_utf8_lossy(&id_bytes).trim().to_string();
    if window_id.is_empty() || window_id.parse::<u64>().is_err() {
        return Err(CaptureError::Parse {
            what: format!("xdotool active window id: {window_id:?}"),
        });
    }
    run_capture("maim", &["-i", &window_id], deadline).await
}

async fn capture_wayland_focused(
    compositor: Option<Compositor>,
    deadline: Duration,
) -> Result<Vec<u8>, CaptureError> {
    let geom = match compositor {
        Some(Compositor::Sway) => focused_geom_sway(deadline).await?,
        Some(Compositor::Hyprland) => focused_geom_hyprland(deadline).await?,
        Some(Compositor::I3) | None => {
            return Err(CaptureError::FocusedUnsupportedOnWayland {
                compositor: std::env::var("XDG_CURRENT_DESKTOP")
                    .ok()
                    .filter(|desktop| !desktop.is_empty())
                    .unwrap_or_else(|| "unknown".into()),
            });
        }
    };
    run_capture("grim", &["-g", &geom, "-"], deadline).await
}

async fn focused_geom_sway(deadline: Duration) -> Result<String, CaptureError> {
    let json = run_capture("swaymsg", &["-t", "get_tree", "-r"], deadline).await?;
    let tree: Value = serde_json::from_slice(&json).map_err(|e| CaptureError::Parse {
        what: format!("swaymsg JSON: {e}"),
    })?;
    find_focused_sway_rect(&tree).ok_or_else(|| CaptureError::Parse {
        what: "no focused node in swaymsg tree".to_string(),
    })
}

async fn focused_geom_hyprland(deadline: Duration) -> Result<String, CaptureError> {
    let json = run_capture("hyprctl", &["activewindow", "-j"], deadline).await?;
    let window: Value = serde_json::from_slice(&json).map_err(|e| CaptureError::Parse {
        what: format!("hyprctl JSON: {e}"),
    })?;
    parse_hyprland_geom(&window).ok_or_else(|| CaptureError::Parse {
        what: "missing at/size in hyprctl activewindow".to_string(),
    })
}

/// Run `binary` and return its stdout, which must be non-empty and within
/// [`MAX_IMAGE_BYTES`].
async fn run_capture(
    binary: &str,
    args: &[&str],
    deadline: Duration,
) -> Result<Vec<u8>, CaptureError> {
    let mut cmd = ProcCommand::new(binary);
    cmd.args(args);
    let max_output = usize::try_from(MAX_IMAGE_BYTES).unwrap_or(usize::MAX);
    let captured = capture(cmd, None, &[], deadline, max_output)
        .await
        .map_err(|e| spawn_error(binary, &e))?;
    match captured.outcome {
        WaitOutcome::Exited(status) if status.success() => {}
        WaitOutcome::Exited(status) => {
            return Err(CaptureError::NonZero {
                binary: binary.to_string(),
                status: exit_code(status),
                stderr_tail: stderr_tail(&captured.stderr),
            });
        }
        WaitOutcome::WaitErr(e) => {
            return Err(CaptureError::Spawn {
                binary: binary.to_string(),
                msg: format!("wait: {e}"),
            });
        }
        WaitOutcome::Timeout => return Err(CaptureError::Timeout),
        WaitOutcome::Overflow => {
            return Err(CaptureError::TooLarge {
                size: captured.stdout.len(),
            });
        }
    }
    if captured.stdout.is_empty() {
        return Err(CaptureError::EmptyOutput {
            binary: binary.to_string(),
        });
    }
    Ok(captured.stdout)
}

fn spawn_error(binary: &str, err: &io::Error) -> CaptureError {
    if err.kind() == io::ErrorKind::NotFound {
        CaptureError::BinaryMissing {
            binary: binary.to_string(),
        }
    } else {
        CaptureError::Spawn {
            binary: binary.to_string(),
            msg: err.to_string(),
        }
    }
}

fn stderr_tail(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(STDERR_TAIL_LINES)..].join("\n")
}

fn install_hint(binary: &str) -> &'static str {
    match binary {
        "maim" => "pacman -S maim or apt install maim",
        "grim" => "pacman -S grim or apt install grim",
        "xdotool" => "pacman -S xdotool or apt install xdotool",
        "swaymsg" => "install sway (the swaymsg binary ships with it)",
        "hyprctl" => "install Hyprland (the hyprctl binary ships with it)",
        _ => "the appropriate package for your distro",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn missing_binary_returns_127_with_install_hint() {
        let err = run_capture(
            "assistd-screenshot-not-a-real-bin-xyz",
            &[],
            Duration::from_secs(2),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, CaptureError::BinaryMissing { .. }), "{err:?}");
        let out = err.to_output();
        assert_eq!(out.exit_code, 127);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] screenshot: backend binary not found: assistd-screenshot-not-a-real-bin-xyz. \
             Install: the appropriate package for your distro\n"
        );
    }
}

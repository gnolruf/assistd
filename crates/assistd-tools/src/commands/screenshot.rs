//! `screenshot [--full|--focused|--monitor=NAME]`: capture the screen
//! through `maim` (X11) or `grim` (Wayland) and attach the PNG as a
//! vision input. `--focused` resolves the window geometry through
//! xdotool, swaymsg, or hyprctl. The PNG bytes never touch disk.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command as ProcCommand;

use crate::attachment::MAX_IMAGE_BYTES;
use crate::command::{Attachment, Command, CommandInput, CommandOutput, Hint, error_line};
use crate::commands::cat::human_size;
use crate::exec::{SPAWN_FAILED_EXIT, TIMEOUT_EXIT, WaitOutcome, capture, exit_code};
use crate::vision::VisionGate;

const STDERR_TAIL_LINES: usize = 20;

/// Configuration for the screenshot command.
#[derive(Debug, Clone)]
pub struct ScreenshotPolicyCfg {
    /// Force a specific backend; `None` auto-detects on every call.
    pub backend: Option<Backend>,
    pub timeout: Duration,
}

impl Default for ScreenshotPolicyCfg {
    fn default() -> Self {
        Self {
            backend: None,
            timeout: Duration::from_secs(5),
        }
    }
}

/// Display-server backend, which selects the capture binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    X11,
    Wayland,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    Full,
    Focused,
    Monitor(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum WaylandCompositor {
    Sway,
    Hyprland,
    Unknown(String),
}

/// `screenshot [--full|--focused|--monitor=NAME]`: capture the screen as a PNG
/// and attach it as a vision input for the next LLM turn.
pub struct ScreenshotCommand {
    cfg: Arc<ScreenshotPolicyCfg>,
    gate: Arc<VisionGate>,
}

impl ScreenshotCommand {
    /// A `screenshot` command configured by `cfg` that refuses to run
    /// while `gate` reports no vision support.
    pub fn new(cfg: Arc<ScreenshotPolicyCfg>, gate: Arc<VisionGate>) -> Self {
        Self { cfg, gate }
    }
}

#[cfg(test)]
impl Default for ScreenshotCommand {
    fn default() -> Self {
        Self::new(
            Arc::new(ScreenshotPolicyCfg::default()),
            VisionGate::new(true),
        )
    }
}

#[async_trait]
impl Command for ScreenshotCommand {
    fn name(&self) -> &str {
        "screenshot"
    }

    fn summary(&self) -> &'static str {
        if self.gate.supported() {
            "capture the screen as a PNG and attach it for the next LLM turn"
        } else {
            "(unavailable: model has no vision encoder)"
        }
    }

    fn help(&self) -> String {
        "usage: screenshot [--full|--focused|--monitor=<name>]\n\
         \n\
         Capture the screen and attach it as a vision input for the next \
         LLM turn (kept in memory; never written to disk by default).\n\
         \n\
         Without arguments, captures the full screen (which on multi-head \
         setups means the bounding box across all monitors; pass \
         --monitor=<name> to capture a single output). Pass --focused to \
         capture only the currently focused window.\n\
         \n\
         The display server is auto-detected: maim is used on X11, grim \
         on Wayland. --focused requires:\n  \
           X11: xdotool to find the active window\n  \
           Wayland (sway): swaymsg to read the focused node geometry\n  \
           Wayland (Hyprland): hyprctl to read the active-window geometry\n\
         \n\
         --monitor=<name> requires:\n  \
           X11: xrandr to resolve the connector name to a geometry\n  \
           Wayland: grim's -o flag (no extra binary)\n\
         List monitors with `xrandr --listmonitors` (X11) or \
         `swaymsg -t get_outputs` / `hyprctl monitors` (Wayland).\n\
         \n\
         Exit codes:\n  \
           0   success (image attached)\n  \
           1   capture backend exited non-zero\n  \
           2   no display server, unsupported compositor, or bad args\n  \
           127 capture binary not found (Install: hint follows)\n  \
           137 capture timed out\n\
         \n\
         Privacy: the captured pixels are sent to the LLM as part of its \
         next turn. Only enable this command in environments where that \
         is acceptable.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if !self.gate.supported() {
            return CommandOutput::failed(
                1,
                error_line(
                    "screenshot",
                    "vision not available: model does not support images",
                    Hint::Use,
                    "a model with mmproj loaded",
                )
                .into_bytes(),
            );
        }
        let target = match parse_target(&input.args) {
            Ok(t) => t,
            Err(msg) => {
                return CommandOutput::usage_error(
                    "screenshot",
                    msg,
                    "screenshot --full or screenshot --focused",
                );
            }
        };

        let backend = match self.cfg.backend {
            Some(b) => b,
            None => match detect_backend() {
                Ok(b) => b,
                Err(msg) => {
                    return CommandOutput::failed(
                        2,
                        error_line(
                            "screenshot",
                            msg,
                            Hint::Try,
                            "running this from a graphical session",
                        )
                        .into_bytes(),
                    );
                }
            },
        };

        match capture_target(backend, &target, self.cfg.timeout).await {
            Ok(png) => {
                let stdout = format!(
                    "captured PNG ({}, {}, backend={}); attached to next turn\n",
                    human_size(png.len()),
                    target_label(&target),
                    backend_label(backend),
                );
                CommandOutput {
                    stdout: stdout.into_bytes(),
                    stderr: Vec::new(),
                    exit_code: 0,
                    attachments: vec![Attachment::Image {
                        mime: "image/png".to_string(),
                        bytes: png,
                    }],
                }
            }
            Err(e) => capture_error_to_output(e),
        }
    }
}

fn parse_target(args: &[String]) -> Result<Target, String> {
    match args.len() {
        0 => Ok(Target::Full),
        1 => match args[0].as_str() {
            "--full" => Ok(Target::Full),
            "--focused" => Ok(Target::Focused),
            s if s.starts_with("--monitor=") => {
                let name = &s["--monitor=".len()..];
                if name.is_empty() {
                    Err("--monitor requires a name (try `xrandr --listmonitors` or `swaymsg -t get_outputs`)".into())
                } else {
                    Ok(Target::Monitor(name.to_string()))
                }
            }
            "--monitor" => {
                Err("--monitor requires a value: --monitor=<name> (e.g. --monitor=DP-1)".into())
            }
            other => Err(format!("unknown flag: {other}")),
        },
        2 if args[0] == "--monitor" => {
            if args[1].is_empty() {
                Err("--monitor requires a non-empty name".into())
            } else {
                Ok(Target::Monitor(args[1].clone()))
            }
        }
        _ => Err("expects at most one flag (--full, --focused, or --monitor=<name>)".into()),
    }
}

fn target_label(t: &Target) -> &'static str {
    match t {
        Target::Full => "full-screen",
        Target::Focused => "focused-window",
        Target::Monitor(_) => "monitor",
    }
}

fn backend_label(b: Backend) -> &'static str {
    match b {
        Backend::X11 => "x11",
        Backend::Wayland => "wayland",
    }
}

/// The environment variables a display server advertises itself with.
struct DisplayEnv<'a> {
    session_type: Option<&'a str>,
    wayland_display: bool,
    x_display: bool,
}

fn detect_backend() -> Result<Backend, &'static str> {
    let session_type = std::env::var("XDG_SESSION_TYPE").ok();
    detect_backend_in(DisplayEnv {
        session_type: session_type.as_deref(),
        wayland_display: std::env::var_os("WAYLAND_DISPLAY").is_some(),
        x_display: std::env::var_os("DISPLAY").is_some(),
    })
}

fn detect_backend_in(env: DisplayEnv<'_>) -> Result<Backend, &'static str> {
    match env.session_type {
        Some("wayland") => return Ok(Backend::Wayland),
        Some("x11") => return Ok(Backend::X11),
        _ => {}
    }
    match (env.wayland_display, env.x_display) {
        // In a Wayland + XWayland session grim still captures X clients;
        // the inverse is not true.
        (true, _) => Ok(Backend::Wayland),
        (false, true) => Ok(Backend::X11),
        (false, false) => Err("no display server detected (no WAYLAND_DISPLAY or DISPLAY)"),
    }
}

/// The environment variables a Wayland compositor advertises itself with.
struct WaylandEnv<'a> {
    swaysock: bool,
    hyprland_signature: bool,
    current_desktop: Option<&'a str>,
}

fn detect_wayland_compositor() -> WaylandCompositor {
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
    let xdg = env.current_desktop.unwrap_or("");
    match xdg.to_ascii_lowercase().as_str() {
        "sway" => WaylandCompositor::Sway,
        "hyprland" => WaylandCompositor::Hyprland,
        _ => WaylandCompositor::Unknown(xdg.to_string()),
    }
}

#[derive(Debug)]
enum CaptureError {
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

async fn capture_target(
    backend: Backend,
    target: &Target,
    deadline: Duration,
) -> Result<Vec<u8>, CaptureError> {
    match (backend, target) {
        (Backend::X11, Target::Full) => run_capture("maim", &[], deadline).await,
        (Backend::X11, Target::Focused) => capture_x11_focused(deadline).await,
        (Backend::X11, Target::Monitor(name)) => capture_x11_monitor(name, deadline).await,
        (Backend::Wayland, Target::Full) => run_capture("grim", &["-"], deadline).await,
        (Backend::Wayland, Target::Focused) => capture_wayland_focused(deadline).await,
        (Backend::Wayland, Target::Monitor(name)) => {
            run_capture("grim", &["-o", name, "-"], deadline).await
        }
    }
}

/// Find `monitor` in `xrandr --listmonitors` output and return its
/// geometry as `WxH+X+Y` for maim's `-g` flag. Lines look like
/// ` 0: +*HDMI-1 1920/598x1200/336+0+0  HDMI-1`; the trailing
/// connector name is matched because the flags on the leading one vary.
fn parse_xrandr_monitor_geom(listing: &str, monitor: &str) -> Option<String> {
    for line in listing.lines() {
        if !line.starts_with(|c: char| c.is_whitespace() || c.is_ascii_digit()) {
            continue;
        }
        let trimmed = line.trim();
        let name = trimmed.split_whitespace().next_back()?;
        if name != monitor {
            continue;
        }
        for tok in trimmed.split_whitespace() {
            if let Some(geom) = strip_xrandr_geom_token(tok) {
                return Some(geom);
            }
        }
    }
    None
}

/// Reduce an xrandr geometry token `<w>/<wmm>x<h>/<hmm>±<x>±<y>` to
/// maim's `<w>x<h>±<x>±<y>`.
fn strip_xrandr_geom_token(tok: &str) -> Option<String> {
    let (lhs, after_x) = tok.split_once('x')?;
    let (w_with_mm, _) = lhs.split_once('/')?;
    let w: u32 = w_with_mm.parse().ok()?;
    let h_end = after_x.find(['+', '-']).filter(|i| *i > 0)?;
    let (h_with_mm, _) = after_x[..h_end].split_once('/')?;
    let h: u32 = h_with_mm.parse().ok()?;
    let offsets = &after_x[h_end..];
    let y_start = offsets[1..].find(['+', '-'])? + 1;
    let (x_part, y_part) = offsets.split_at(y_start);
    x_part.parse::<i32>().ok()?;
    y_part.parse::<i32>().ok()?;
    Some(format!("{w}x{h}{x_part}{y_part}"))
}

async fn capture_x11_monitor(monitor: &str, deadline: Duration) -> Result<Vec<u8>, CaptureError> {
    let raw = run_capture("xrandr", &["--listmonitors"], deadline).await?;
    let listing = String::from_utf8_lossy(&raw);
    let geom = parse_xrandr_monitor_geom(&listing, monitor).ok_or_else(|| CaptureError::Parse {
        what: format!("monitor `{monitor}` not found in xrandr output"),
    })?;
    run_capture("maim", &["-g", &geom], deadline).await
}

async fn run_capture(
    binary: &str,
    args: &[&str],
    deadline: Duration,
) -> Result<Vec<u8>, CaptureError> {
    let mut cmd = ProcCommand::new(binary);
    cmd.args(args);
    let max_output = MAX_IMAGE_BYTES as usize;
    let captured = capture(cmd, &[], deadline, max_output).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            CaptureError::BinaryMissing {
                binary: binary.to_string(),
            }
        } else {
            CaptureError::Spawn {
                binary: binary.to_string(),
                msg: e.to_string(),
            }
        }
    })?;
    match captured.outcome {
        WaitOutcome::Exited(status) if status.success() => {}
        WaitOutcome::Exited(status) => {
            return Err(CaptureError::NonZero {
                binary: binary.to_string(),
                status: exit_code(&status),
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

fn stderr_tail(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(STDERR_TAIL_LINES)..].join("\n")
}

async fn capture_x11_focused(deadline: Duration) -> Result<Vec<u8>, CaptureError> {
    let id_bytes = run_capture("xdotool", &["getactivewindow"], deadline).await?;
    let id_str = String::from_utf8_lossy(&id_bytes).trim().to_string();
    if id_str.is_empty() || id_str.parse::<u64>().is_err() {
        return Err(CaptureError::Parse {
            what: format!("xdotool active window id: {id_str:?}"),
        });
    }
    run_capture("maim", &["-i", &id_str], deadline).await
}

async fn capture_wayland_focused(deadline: Duration) -> Result<Vec<u8>, CaptureError> {
    let geom = match detect_wayland_compositor() {
        WaylandCompositor::Sway => focused_geom_sway(deadline).await?,
        WaylandCompositor::Hyprland => focused_geom_hyprland(deadline).await?,
        WaylandCompositor::Unknown(name) => {
            return Err(CaptureError::FocusedUnsupportedOnWayland {
                compositor: if name.is_empty() {
                    "unknown".into()
                } else {
                    name
                },
            });
        }
    };
    run_capture("grim", &["-g", &geom, "-"], deadline).await
}

async fn focused_geom_sway(deadline: Duration) -> Result<String, CaptureError> {
    let json = run_capture("swaymsg", &["-t", "get_tree", "-r"], deadline).await?;
    let v: Value = serde_json::from_slice(&json).map_err(|e| CaptureError::Parse {
        what: format!("swaymsg JSON: {e}"),
    })?;
    find_focused_sway_rect(&v).ok_or_else(|| CaptureError::Parse {
        what: "no focused node in swaymsg tree".to_string(),
    })
}

fn find_focused_sway_rect(v: &Value) -> Option<String> {
    if v.get("focused").and_then(|f| f.as_bool()) == Some(true) {
        let r = v.get("rect")?;
        let x = r.get("x")?.as_i64()?;
        let y = r.get("y")?.as_i64()?;
        let w = r.get("width")?.as_i64()?;
        let h = r.get("height")?.as_i64()?;
        return Some(format!("{x},{y} {w}x{h}"));
    }
    for key in ["nodes", "floating_nodes"] {
        if let Some(arr) = v.get(key).and_then(|n| n.as_array()) {
            for child in arr {
                if let Some(rect) = find_focused_sway_rect(child) {
                    return Some(rect);
                }
            }
        }
    }
    None
}

async fn focused_geom_hyprland(deadline: Duration) -> Result<String, CaptureError> {
    let json = run_capture("hyprctl", &["activewindow", "-j"], deadline).await?;
    let v: Value = serde_json::from_slice(&json).map_err(|e| CaptureError::Parse {
        what: format!("hyprctl JSON: {e}"),
    })?;
    parse_hyprland_geom(&v).ok_or_else(|| CaptureError::Parse {
        what: "missing at/size in hyprctl activewindow".to_string(),
    })
}

fn parse_hyprland_geom(v: &Value) -> Option<String> {
    let at = v.get("at")?.as_array()?;
    let size = v.get("size")?.as_array()?;
    let x = at.first()?.as_i64()?;
    let y = at.get(1)?.as_i64()?;
    let w = size.first()?.as_i64()?;
    let h = size.get(1)?.as_i64()?;
    Some(format!("{x},{y} {w}x{h}"))
}

fn capture_error_to_output(err: CaptureError) -> CommandOutput {
    match err {
        CaptureError::BinaryMissing { binary } => CommandOutput::failed(
            SPAWN_FAILED_EXIT,
            error_line(
                "screenshot",
                format_args!("backend binary not found: {binary}"),
                Hint::Install,
                install_hint(&binary),
            )
            .into_bytes(),
        ),
        CaptureError::Spawn { binary, msg } => CommandOutput::failed(
            1,
            error_line(
                "screenshot",
                format_args!("spawn failed: {binary}: {msg}"),
                Hint::Check,
                format_args!("{binary} runs from your shell"),
            )
            .into_bytes(),
        ),
        CaptureError::Timeout => CommandOutput::failed(
            TIMEOUT_EXIT,
            error_line(
                "screenshot",
                "capture timed out",
                Hint::Try,
                "screenshot again or check the compositor is responsive",
            )
            .into_bytes(),
        ),
        CaptureError::NonZero {
            binary,
            status,
            stderr_tail,
        } => {
            let what = if stderr_tail.is_empty() {
                format!("{binary} exited {status}")
            } else {
                format!("{binary} exited {status}: {stderr_tail}")
            };
            CommandOutput::failed(
                1,
                error_line(
                    "screenshot",
                    what,
                    Hint::Try,
                    "a different target or backend",
                )
                .into_bytes(),
            )
        }
        CaptureError::EmptyOutput { binary } => CommandOutput::failed(
            1,
            error_line(
                "screenshot",
                format_args!("{binary} produced no image bytes"),
                Hint::Try,
                "screenshot --full",
            )
            .into_bytes(),
        ),
        CaptureError::TooLarge { size } => CommandOutput::failed(
            1,
            error_line(
                "screenshot",
                format_args!(
                    "captured PNG too large ({} > {} max)",
                    human_size(size),
                    human_size(MAX_IMAGE_BYTES as usize),
                ),
                Hint::Try,
                "--focused, or capture a single monitor",
            )
            .into_bytes(),
        ),
        CaptureError::FocusedUnsupportedOnWayland { compositor } => CommandOutput::failed(
            2,
            error_line(
                "screenshot",
                format_args!("--focused not supported on Wayland compositor: {compositor}"),
                Hint::Use,
                "screenshot --full (supported compositors for --focused: sway, Hyprland)",
            )
            .into_bytes(),
        ),
        CaptureError::Parse { what } => CommandOutput::failed(
            1,
            error_line(
                "screenshot",
                format_args!("failed to parse: {what}"),
                Hint::Try,
                "screenshot --full",
            )
            .into_bytes(),
        ),
    }
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

    fn args(raw: &[&str]) -> Vec<String> {
        raw.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_target_accepts_each_form() {
        let cases: [(&[&str], Target); 5] = [
            (&[], Target::Full),
            (&["--full"], Target::Full),
            (&["--focused"], Target::Focused),
            (&["--monitor=DP-1"], Target::Monitor("DP-1".into())),
            (&["--monitor", "HDMI-1"], Target::Monitor("HDMI-1".into())),
        ];
        for (raw, expected) in cases {
            assert_eq!(parse_target(&args(raw)), Ok(expected), "{raw:?}");
        }
    }

    #[test]
    fn parse_target_rejects_malformed_args() {
        let cases: [(&[&str], &str); 5] = [
            (
                &["--full", "--focused"],
                "expects at most one flag (--full, --focused, or --monitor=<name>)",
            ),
            (
                &["--monitor"],
                "--monitor requires a value: --monitor=<name> (e.g. --monitor=DP-1)",
            ),
            (
                &["--monitor="],
                "--monitor requires a name (try `xrandr --listmonitors` or `swaymsg -t get_outputs`)",
            ),
            (&["--monitor", ""], "--monitor requires a non-empty name"),
            (&["--bogus"], "unknown flag: --bogus"),
        ];
        for (raw, expected) in cases {
            assert_eq!(parse_target(&args(raw)), Err(expected.into()), "{raw:?}");
        }
    }

    #[tokio::test]
    async fn bogus_flag_emits_usage_error() {
        let out = ScreenshotCommand::default()
            .run(CommandInput {
                args: args(&["--bogus-flag"]),
                stdin: None,
            })
            .await;
        assert_eq!(out.exit_code, 2);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] screenshot: unknown flag: --bogus-flag. \
             Use: screenshot --full or screenshot --focused\n"
        );
        assert!(out.attachments.is_empty());
    }

    #[test]
    fn detect_backend_prefers_session_type_then_wayland() {
        let none = Err("no display server detected (no WAYLAND_DISPLAY or DISPLAY)");
        let cases = [
            (Some("wayland"), false, false, Ok(Backend::Wayland)),
            (Some("x11"), false, false, Ok(Backend::X11)),
            (None, false, false, none),
            // A TTY login falls back to the display variables.
            (Some("tty"), false, false, none),
            (Some("tty"), false, true, Ok(Backend::X11)),
            // XWayland sets both; grim still captures X clients.
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

    #[test]
    fn hyprland_geom_from_json() {
        let cases = [
            (
                serde_json::json!({"at": [100, 200], "size": [800, 600]}),
                Some("100,200 800x600"),
            ),
            (serde_json::json!({"at": [0, 0]}), None),
            (
                serde_json::json!({"at": ["100", "200"], "size": ["800", "600"]}),
                None,
            ),
            (serde_json::json!({"at": [100], "size": [800, 600]}), None),
        ];
        for (v, expected) in cases {
            assert_eq!(parse_hyprland_geom(&v).as_deref(), expected, "{v}");
        }
    }

    #[test]
    fn sway_tree_walk_finds_focused_rect() {
        let cases = [
            (
                serde_json::json!({
                    "focused": true,
                    "rect": {"x": 10, "y": 20, "width": 300, "height": 400}
                }),
                Some("10,20 300x400"),
            ),
            (
                serde_json::json!({
                    "focused": false,
                    "nodes": [
                        {"focused": false, "nodes": [
                            {"focused": true, "rect": {"x": 5, "y": 6, "width": 7, "height": 8}}
                        ]}
                    ]
                }),
                Some("5,6 7x8"),
            ),
            (
                serde_json::json!({
                    "focused": false,
                    "floating_nodes": [
                        {"focused": true, "rect": {"x": 1, "y": 2, "width": 3, "height": 4}}
                    ]
                }),
                Some("1,2 3x4"),
            ),
            (serde_json::json!({"focused": false, "nodes": []}), None),
            (serde_json::json!({"focused": true}), None),
        ];
        for (v, expected) in cases {
            assert_eq!(find_focused_sway_rect(&v).as_deref(), expected, "{v}");
        }
    }

    /// Two-head laptop+external: the second monitor sits past 1920 on x,
    /// and only the primary carries the `*` flag.
    const XRANDR_DUAL: &str = "Monitors: 2\n \
        0: +*eDP-1 1920/300x1080/180+0+0  eDP-1\n \
        1: +DP-2 2560/600x1440/340+1920+0  DP-2\n";

    #[test]
    fn parse_xrandr_picks_the_named_monitor() {
        let cases = [
            ("eDP-1", Some("1920x1080+0+0")),
            ("DP-2", Some("2560x1440+1920+0")),
            ("VGA-1", None),
        ];
        for (monitor, expected) in cases {
            assert_eq!(
                parse_xrandr_monitor_geom(XRANDR_DUAL, monitor).as_deref(),
                expected,
                "{monitor}"
            );
        }
    }

    #[test]
    fn strip_xrandr_geom_token_handles_negative_offsets() {
        assert_eq!(
            strip_xrandr_geom_token("1920/598x1080/336-1920+0").as_deref(),
            Some("1920x1080-1920+0")
        );
    }

    #[test]
    fn strip_xrandr_geom_token_rejects_unrelated_tokens() {
        assert_eq!(strip_xrandr_geom_token("HDMI-1"), None);
        assert_eq!(strip_xrandr_geom_token("1920/598x1200/336"), None);
    }

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
        let out = capture_error_to_output(err);
        assert_eq!(out.exit_code, 127);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] screenshot: backend binary not found: assistd-screenshot-not-a-real-bin-xyz. \
             Install: the appropriate package for your distro\n"
        );
    }

    #[test]
    fn capture_errors_map_to_exit_code_and_message() {
        let cases = [
            (
                CaptureError::FocusedUnsupportedOnWayland {
                    compositor: "KDE".into(),
                },
                2,
                "[error] screenshot: --focused not supported on Wayland compositor: KDE. \
                 Use: screenshot --full (supported compositors for --focused: sway, Hyprland)\n",
            ),
            (
                CaptureError::Timeout,
                137,
                "[error] screenshot: capture timed out. \
                 Try: screenshot again or check the compositor is responsive\n",
            ),
            (
                CaptureError::NonZero {
                    binary: "grim".into(),
                    status: 1,
                    stderr_tail: "compositor not running".into(),
                },
                1,
                "[error] screenshot: grim exited 1: compositor not running. \
                 Try: a different target or backend\n",
            ),
        ];
        for (err, exit_code, stderr) in cases {
            let label = format!("{err:?}");
            let out = capture_error_to_output(err);
            assert_eq!(out.exit_code, exit_code, "{label}");
            assert_eq!(String::from_utf8_lossy(&out.stderr), stderr, "{label}");
        }
    }

    #[tokio::test]
    async fn vision_disabled_returns_exact_error() {
        let cmd = ScreenshotCommand::new(
            Arc::new(ScreenshotPolicyCfg::default()),
            VisionGate::new(false),
        );
        let out = cmd
            .run(CommandInput {
                args: args(&["--full"]),
                stdin: None,
            })
            .await;
        assert_eq!(out.exit_code, 1);
        assert!(out.stdout.is_empty());
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] screenshot: vision not available: model does not support images. \
             Use: a model with mmproj loaded\n"
        );
        assert!(out.attachments.is_empty());
    }

    #[test]
    fn summary_changes_when_vision_disabled() {
        let summary = |supported| {
            ScreenshotCommand::new(
                Arc::new(ScreenshotPolicyCfg::default()),
                VisionGate::new(supported),
            )
            .summary()
        };
        assert!(summary(true).contains("capture the screen"));
        assert!(summary(false).contains("unavailable"));
    }
}

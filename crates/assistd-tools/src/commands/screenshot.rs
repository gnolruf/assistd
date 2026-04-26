//! `screenshot [--full|--focused]` — capture the screen as a PNG and
//! attach it as a vision input for the next LLM turn.
//!
//! Mirrors `SeeCommand` in shape: returns a [`crate::Attachment::Image`]
//! on success. The bytes come from `maim` (X11) or `grim` (Wayland) instead
//! of `tokio::fs::read`. Backend is auto-detected from `XDG_SESSION_TYPE`
//! / `WAYLAND_DISPLAY` / `DISPLAY` unless the policy overrides it.
//!
//! For `--focused` we delegate window/region resolution to the platform's
//! native tools (xdotool on X11; swaymsg / hyprctl on Wayland) and feed
//! the resulting geometry to grim. Compositors we don't recognise return
//! a descriptive error pointing at `--full`.
//!
//! In-memory only: PNG bytes never touch disk on the way out.

use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command as ProcCommand;
use tokio::time::timeout;

use crate::command::{Attachment, Command, CommandInput, CommandOutput, error_line};
use crate::commands::cat::human_size;

const SPAWN_FAILED_EXIT: i32 = 127;
const TIMEOUT_EXIT: i32 = 137;
/// Cap on stderr captured per backend invocation. Prevents a chatty
/// child from filling memory if something goes badly wrong.
const STDERR_TAIL_LINES: usize = 20;

/// Configuration for the screenshot command. Built from `[tools.screenshot]`
/// in the user's TOML config; see `assistd_config::ToolsScreenshotConfig`.
#[derive(Debug, Clone)]
pub struct ScreenshotPolicyCfg {
    /// Force a specific backend. `None` = auto-detect on every call.
    pub backend: Option<Backend>,
    /// Subprocess timeout. Capture is fast on a healthy compositor; the
    /// timeout exists to prevent a wedged child from locking the agent.
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

/// Display-server backend. The capture binary depends on this:
/// `maim` for X11, `grim` for Wayland.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    X11,
    Wayland,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Full,
    Focused,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum WaylandCompositor {
    Sway,
    Hyprland,
    Unknown(String),
}

pub struct ScreenshotCommand {
    cfg: Arc<ScreenshotPolicyCfg>,
}

impl ScreenshotCommand {
    pub fn new(cfg: Arc<ScreenshotPolicyCfg>) -> Self {
        Self { cfg }
    }
}

#[cfg(test)]
impl Default for ScreenshotCommand {
    /// Test-only default: auto-detect backend, 5-second timeout. Lets the
    /// convention-compliance harness in `command.rs` construct an instance
    /// without config plumbing.
    fn default() -> Self {
        Self::new(Arc::new(ScreenshotPolicyCfg::default()))
    }
}

#[async_trait]
impl Command for ScreenshotCommand {
    fn name(&self) -> &str {
        "screenshot"
    }

    fn summary(&self) -> &'static str {
        "capture the screen as a PNG and attach it for the next LLM turn"
    }

    fn help(&self) -> String {
        "usage: screenshot [--full|--focused]\n\
         \n\
         Capture the screen and attach it as a vision input for the next \
         LLM turn (kept in memory; never written to disk by default).\n\
         \n\
         Without arguments, captures the full screen. Pass --focused to \
         capture only the currently focused window.\n\
         \n\
         The display server is auto-detected: maim is used on X11, grim \
         on Wayland. --focused requires:\n  \
           X11: xdotool to find the active window\n  \
           Wayland (sway): swaymsg to read the focused node geometry\n  \
           Wayland (Hyprland): hyprctl to read the active-window geometry\n\
         \n\
         Exit codes:\n  \
           0   success — image attached\n  \
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

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        let target = match parse_args(&input.args) {
            Ok(t) => t,
            Err(msg) => {
                return Ok(CommandOutput::failed(
                    2,
                    error_line(
                        "screenshot",
                        msg,
                        "Use",
                        "screenshot --full or screenshot --focused",
                    )
                    .into_bytes(),
                ));
            }
        };

        let backend = match self.cfg.backend {
            Some(b) => b,
            None => match detect_backend() {
                Ok(b) => b,
                Err(msg) => {
                    return Ok(CommandOutput::failed(
                        2,
                        error_line(
                            "screenshot",
                            msg,
                            "Try",
                            "running this from a graphical session",
                        )
                        .into_bytes(),
                    ));
                }
            },
        };

        match capture(backend, target, self.cfg.timeout).await {
            Ok(png) => {
                let stdout = format!(
                    "captured PNG ({}, {}, backend={}) — attached to next turn\n",
                    human_size(png.len()),
                    target_label(target),
                    backend_label(backend),
                );
                Ok(CommandOutput {
                    stdout: stdout.into_bytes(),
                    stderr: Vec::new(),
                    exit_code: 0,
                    attachments: vec![Attachment::Image {
                        mime: "image/png".to_string(),
                        bytes: png,
                    }],
                })
            }
            Err(e) => Ok(capture_error_to_output(e)),
        }
    }
}

fn parse_args(args: &[String]) -> Result<Target, String> {
    match args.len() {
        0 => Ok(Target::Full),
        1 => match args[0].as_str() {
            "--full" => Ok(Target::Full),
            "--focused" => Ok(Target::Focused),
            other => Err(format!("unknown flag: {other}")),
        },
        _ => Err("expects at most one flag (--full or --focused)".into()),
    }
}

fn target_label(t: Target) -> &'static str {
    match t {
        Target::Full => "full-screen",
        Target::Focused => "focused-window",
    }
}

fn backend_label(b: Backend) -> &'static str {
    match b {
        Backend::X11 => "x11",
        Backend::Wayland => "wayland",
    }
}

// ---------------------------------------------------------------- detection

fn detect_backend() -> Result<Backend, &'static str> {
    detect_backend_from_env(
        std::env::var("XDG_SESSION_TYPE").ok().as_deref(),
        std::env::var_os("WAYLAND_DISPLAY").is_some(),
        std::env::var_os("DISPLAY").is_some(),
    )
}

fn detect_backend_from_env(
    xdg_session_type: Option<&str>,
    has_wayland: bool,
    has_x: bool,
) -> Result<Backend, &'static str> {
    if let Some(s) = xdg_session_type {
        match s {
            "wayland" => return Ok(Backend::Wayland),
            "x11" => return Ok(Backend::X11),
            _ => {} // unrecognised — fall through to env-var check
        }
    }
    match (has_wayland, has_x) {
        // Hybrid (Wayland + XWayland) sessions: prefer Wayland tooling.
        // X-only programs still get caught by grim if they're on a
        // Wayland output; the inverse is not true.
        (true, _) => Ok(Backend::Wayland),
        (false, true) => Ok(Backend::X11),
        (false, false) => Err("no display server detected (no WAYLAND_DISPLAY or DISPLAY)"),
    }
}

fn detect_wayland_compositor() -> WaylandCompositor {
    detect_wayland_compositor_from_env(
        std::env::var_os("SWAYSOCK").is_some(),
        std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some(),
        std::env::var("XDG_CURRENT_DESKTOP").ok().as_deref(),
    )
}

fn detect_wayland_compositor_from_env(
    has_swaysock: bool,
    has_hypr_signature: bool,
    xdg_current_desktop: Option<&str>,
) -> WaylandCompositor {
    if has_swaysock {
        return WaylandCompositor::Sway;
    }
    if has_hypr_signature {
        return WaylandCompositor::Hyprland;
    }
    let xdg = xdg_current_desktop.unwrap_or("");
    match xdg.to_ascii_lowercase().as_str() {
        "sway" => WaylandCompositor::Sway,
        "hyprland" => WaylandCompositor::Hyprland,
        _ => WaylandCompositor::Unknown(xdg.to_string()),
    }
}

// ------------------------------------------------------------------ capture

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
    FocusedUnsupportedOnWayland {
        compositor: String,
    },
    Parse {
        what: String,
    },
}

async fn capture(
    backend: Backend,
    target: Target,
    deadline: Duration,
) -> Result<Vec<u8>, CaptureError> {
    match (backend, target) {
        (Backend::X11, Target::Full) => spawn_subprocess("maim", &[], deadline).await,
        (Backend::X11, Target::Focused) => capture_x11_focused(deadline).await,
        (Backend::Wayland, Target::Full) => spawn_subprocess("grim", &["-"], deadline).await,
        (Backend::Wayland, Target::Focused) => capture_wayland_focused(deadline).await,
    }
}

/// Spawn `binary` with `args`, drain stdout to `Vec<u8>`, drain stderr to
/// a tail, wait for exit (or timeout), and return the bytes. The subprocess
/// pattern (`kill_on_drop` + `process_group(0)` on Unix) mirrors
/// `assistd-voice/src/piper/synth.rs` — a timeout drops the `Child`,
/// which sends SIGKILL to the process group via `kill_on_drop`.
async fn spawn_subprocess(
    binary: &str,
    args: &[&str],
    deadline: Duration,
) -> Result<Vec<u8>, CaptureError> {
    let mut cmd = ProcCommand::new(binary);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return Err(if e.kind() == std::io::ErrorKind::NotFound {
                CaptureError::BinaryMissing {
                    binary: binary.to_string(),
                }
            } else {
                CaptureError::Spawn {
                    binary: binary.to_string(),
                    msg: e.to_string(),
                }
            });
        }
    };

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let drain_stdout = async {
        let mut buf = Vec::new();
        let mut reader = stdout;
        let _ = reader.read_to_end(&mut buf).await;
        buf
    };
    let drain_stderr = async {
        let mut tail: VecDeque<String> = VecDeque::with_capacity(STDERR_TAIL_LINES);
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if tail.len() == STDERR_TAIL_LINES {
                tail.pop_front();
            }
            tail.push_back(line);
        }
        tail.into_iter().collect::<Vec<_>>().join("\n")
    };

    let work = async {
        let (stdout_bytes, stderr_tail, status) =
            tokio::join!(drain_stdout, drain_stderr, child.wait());
        let status = status.map_err(|e| CaptureError::Spawn {
            binary: binary.to_string(),
            msg: format!("wait: {e}"),
        })?;
        Ok::<_, CaptureError>((stdout_bytes, stderr_tail, status))
    };

    let (stdout_bytes, stderr_tail, status) = match timeout(deadline, work).await {
        Ok(Ok(triple)) => triple,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(CaptureError::Timeout),
    };

    if !status.success() {
        return Err(CaptureError::NonZero {
            binary: binary.to_string(),
            status: status.code().unwrap_or(TIMEOUT_EXIT),
            stderr_tail,
        });
    }
    if stdout_bytes.is_empty() {
        return Err(CaptureError::EmptyOutput {
            binary: binary.to_string(),
        });
    }
    Ok(stdout_bytes)
}

async fn capture_x11_focused(deadline: Duration) -> Result<Vec<u8>, CaptureError> {
    let id_bytes = spawn_subprocess("xdotool", &["getactivewindow"], deadline).await?;
    let id_str = String::from_utf8_lossy(&id_bytes).trim().to_string();
    if id_str.is_empty() || id_str.parse::<u64>().is_err() {
        return Err(CaptureError::Parse {
            what: format!("xdotool active window id: {id_str:?}"),
        });
    }
    spawn_subprocess("maim", &["-i", &id_str], deadline).await
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
    spawn_subprocess("grim", &["-g", &geom, "-"], deadline).await
}

async fn focused_geom_sway(deadline: Duration) -> Result<String, CaptureError> {
    let json = spawn_subprocess("swaymsg", &["-t", "get_tree", "-r"], deadline).await?;
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
    let json = spawn_subprocess("hyprctl", &["activewindow", "-j"], deadline).await?;
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

// ----------------------------------------------------- error -> CommandOutput

fn capture_error_to_output(err: CaptureError) -> CommandOutput {
    match err {
        CaptureError::BinaryMissing { binary } => CommandOutput::failed(
            SPAWN_FAILED_EXIT,
            error_line(
                "screenshot",
                format_args!("backend binary not found: {binary}"),
                "Install",
                install_hint(&binary),
            )
            .into_bytes(),
        ),
        CaptureError::Spawn { binary, msg } => CommandOutput::failed(
            1,
            error_line(
                "screenshot",
                format_args!("spawn failed: {binary}: {msg}"),
                "Check",
                format_args!("{binary} runs from your shell"),
            )
            .into_bytes(),
        ),
        CaptureError::Timeout => CommandOutput::failed(
            TIMEOUT_EXIT,
            error_line(
                "screenshot",
                "capture timed out",
                "Try",
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
                error_line("screenshot", what, "Try", "a different target or backend").into_bytes(),
            )
        }
        CaptureError::EmptyOutput { binary } => CommandOutput::failed(
            1,
            error_line(
                "screenshot",
                format_args!("{binary} produced no image bytes"),
                "Try",
                "screenshot --full",
            )
            .into_bytes(),
        ),
        CaptureError::FocusedUnsupportedOnWayland { compositor } => CommandOutput::failed(
            2,
            error_line(
                "screenshot",
                format_args!("--focused not supported on Wayland compositor: {compositor}"),
                "Use",
                "screenshot --full (supported compositors for --focused: sway, Hyprland)",
            )
            .into_bytes(),
        ),
        CaptureError::Parse { what } => CommandOutput::failed(
            1,
            error_line(
                "screenshot",
                format_args!("failed to parse: {what}"),
                "Try",
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

    // ---- parse_args ------------------------------------------------------

    #[test]
    fn parse_full_default() {
        assert_eq!(parse_args(&[]), Ok(Target::Full));
    }

    #[test]
    fn parse_full_explicit() {
        assert_eq!(parse_args(&["--full".into()]), Ok(Target::Full));
    }

    #[test]
    fn parse_focused() {
        assert_eq!(parse_args(&["--focused".into()]), Ok(Target::Focused));
    }

    #[test]
    fn parse_too_many_args() {
        let err = parse_args(&["--full".into(), "--focused".into()]).unwrap_err();
        assert!(err.contains("at most one"), "{err}");
    }

    /// The convention-compliance test in `command.rs` drives this exact path
    /// (bogus flag) to verify our error format. Re-asserting here keeps the
    /// failure message in this file when the format changes.
    #[tokio::test]
    async fn bogus_flag_emits_convention_compliant_error() {
        let cmd = ScreenshotCommand::default();
        let out = cmd
            .run(CommandInput {
                args: vec!["--bogus-flag".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.starts_with("[error] screenshot: unknown flag: --bogus-flag"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: screenshot --full"), "{stderr}");
        assert!(out.attachments.is_empty());
    }

    // ---- detect_backend_from_env -----------------------------------------

    #[test]
    fn detect_backend_xdg_wayland() {
        assert_eq!(
            detect_backend_from_env(Some("wayland"), false, false),
            Ok(Backend::Wayland)
        );
    }

    #[test]
    fn detect_backend_xdg_x11() {
        assert_eq!(
            detect_backend_from_env(Some("x11"), false, false),
            Ok(Backend::X11)
        );
    }

    #[test]
    fn detect_backend_no_display() {
        assert!(detect_backend_from_env(None, false, false).is_err());
    }

    #[test]
    fn detect_backend_xdg_tty_falls_back_to_env() {
        // Login on TTY without WAYLAND_DISPLAY/DISPLAY → no display server.
        assert!(detect_backend_from_env(Some("tty"), false, false).is_err());
        // TTY but DISPLAY is forwarded → X11.
        assert_eq!(
            detect_backend_from_env(Some("tty"), false, true),
            Ok(Backend::X11)
        );
    }

    #[test]
    fn detect_backend_hybrid_prefers_wayland() {
        // XWayland-enabled Wayland session: both env vars set, prefer Wayland.
        assert_eq!(
            detect_backend_from_env(None, true, true),
            Ok(Backend::Wayland)
        );
    }

    #[test]
    fn detect_backend_x11_via_display_only() {
        assert_eq!(detect_backend_from_env(None, false, true), Ok(Backend::X11));
    }

    // ---- detect_wayland_compositor_from_env ------------------------------

    #[test]
    fn compositor_swaysock_wins_over_other_signals() {
        assert_eq!(
            detect_wayland_compositor_from_env(true, true, Some("KDE")),
            WaylandCompositor::Sway
        );
    }

    #[test]
    fn compositor_hypr_signature() {
        assert_eq!(
            detect_wayland_compositor_from_env(false, true, None),
            WaylandCompositor::Hyprland
        );
    }

    #[test]
    fn compositor_xdg_sway() {
        assert_eq!(
            detect_wayland_compositor_from_env(false, false, Some("sway")),
            WaylandCompositor::Sway
        );
    }

    #[test]
    fn compositor_xdg_hyprland_capitalized() {
        assert_eq!(
            detect_wayland_compositor_from_env(false, false, Some("Hyprland")),
            WaylandCompositor::Hyprland
        );
    }

    #[test]
    fn compositor_unknown_carries_xdg_value() {
        assert_eq!(
            detect_wayland_compositor_from_env(false, false, Some("KDE")),
            WaylandCompositor::Unknown("KDE".into())
        );
    }

    #[test]
    fn compositor_unknown_when_xdg_missing() {
        assert_eq!(
            detect_wayland_compositor_from_env(false, false, None),
            WaylandCompositor::Unknown(String::new())
        );
    }

    // ---- Hyprland geometry -----------------------------------------------

    #[test]
    fn hyprland_geom_from_json() {
        let v = serde_json::json!({
            "at": [100, 200],
            "size": [800, 600],
        });
        assert_eq!(parse_hyprland_geom(&v), Some("100,200 800x600".to_string()));
    }

    #[test]
    fn hyprland_geom_missing_size_returns_none() {
        let v = serde_json::json!({"at": [0, 0]});
        assert!(parse_hyprland_geom(&v).is_none());
    }

    // ---- Sway tree walking -----------------------------------------------

    #[test]
    fn sway_focused_at_root() {
        let v = serde_json::json!({
            "focused": true,
            "rect": {"x": 10, "y": 20, "width": 300, "height": 400}
        });
        assert_eq!(
            find_focused_sway_rect(&v),
            Some("10,20 300x400".to_string())
        );
    }

    #[test]
    fn sway_focused_in_nested_node() {
        let v = serde_json::json!({
            "focused": false,
            "nodes": [
                {"focused": false, "nodes": [
                    {"focused": true, "rect": {"x": 5, "y": 6, "width": 7, "height": 8}}
                ]}
            ]
        });
        assert_eq!(find_focused_sway_rect(&v), Some("5,6 7x8".to_string()));
    }

    #[test]
    fn sway_focused_in_floating_node() {
        let v = serde_json::json!({
            "focused": false,
            "floating_nodes": [
                {"focused": true, "rect": {"x": 1, "y": 2, "width": 3, "height": 4}}
            ]
        });
        assert_eq!(find_focused_sway_rect(&v), Some("1,2 3x4".to_string()));
    }

    #[test]
    fn sway_no_focused_node_returns_none() {
        let v = serde_json::json!({"focused": false, "nodes": []});
        assert!(find_focused_sway_rect(&v).is_none());
    }

    // ---- missing-binary path ---------------------------------------------

    /// AC #4: a missing capture binary returns a descriptive error. Drives
    /// the BinaryMissing branch directly via `spawn_subprocess` with a
    /// guaranteed-missing executable name (no env mutation needed).
    #[tokio::test]
    async fn missing_binary_returns_127_with_install_hint() {
        let err = spawn_subprocess(
            "assistd-screenshot-not-a-real-bin-xyz",
            &[],
            Duration::from_secs(2),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, CaptureError::BinaryMissing { .. }), "{err:?}");
        let out = capture_error_to_output(err);
        assert_eq!(out.exit_code, 127);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.starts_with("[error] screenshot: backend binary not found:"),
            "{stderr}"
        );
        assert!(stderr.contains("Install:"), "{stderr}");
    }

    #[test]
    fn install_hint_known_binaries() {
        assert!(install_hint("maim").contains("maim"));
        assert!(install_hint("grim").contains("grim"));
        assert!(install_hint("xdotool").contains("xdotool"));
        assert!(install_hint("swaymsg").contains("sway"));
        assert!(install_hint("hyprctl").contains("Hyprland"));
        // Unknown binaries fall back to a generic but useful message.
        assert!(!install_hint("something-else").is_empty());
    }

    // ---- error -> output formatting --------------------------------------

    #[test]
    fn focused_unsupported_on_wayland_message_format() {
        let out = capture_error_to_output(CaptureError::FocusedUnsupportedOnWayland {
            compositor: "KDE".into(),
        });
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("--focused not supported on Wayland compositor: KDE"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: screenshot --full"), "{stderr}");
    }

    #[test]
    fn timeout_error_format() {
        let out = capture_error_to_output(CaptureError::Timeout);
        assert_eq!(out.exit_code, 137);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] screenshot: capture timed out"),
            "{stderr}"
        );
        assert!(stderr.contains("Try:"), "{stderr}");
    }

    #[test]
    fn nonzero_error_includes_stderr_tail() {
        let out = capture_error_to_output(CaptureError::NonZero {
            binary: "grim".into(),
            status: 1,
            stderr_tail: "compositor not running".into(),
        });
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("grim exited 1"), "{stderr}");
        assert!(stderr.contains("compositor not running"), "{stderr}");
    }

    // ---- summary / help shape --------------------------------------------

    #[test]
    fn summary_under_80_chars() {
        let s = ScreenshotCommand::default().summary();
        assert!(s.len() <= 80, "summary is {} chars: {s:?}", s.len());
    }

    #[test]
    fn help_starts_with_usage() {
        let h = ScreenshotCommand::default().help();
        assert!(h.starts_with("usage: screenshot"), "{h}");
    }
}

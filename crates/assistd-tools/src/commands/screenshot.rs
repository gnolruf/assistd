//! `screenshot [--full|--focused|--monitor=NAME]`: capture the screen
//! through `maim` (X11) or `grim` (Wayland) and attach the PNG in memory.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::command::{Attachment, Command, CommandInput, CommandOutput, Hint, error_line};
use crate::commands::cat::human_size;
use crate::vision::VisionGate;

use capture::capture_target;
use detect::detect_backend;

mod capture;
mod detect;
mod geometry;

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

impl Backend {
    fn label(self) -> &'static str {
        match self {
            Self::X11 => "x11",
            Self::Wayland => "wayland",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    Full,
    Focused,
    Monitor(String),
}

impl Target {
    fn label(&self) -> &'static str {
        match self {
            Self::Full => "full-screen",
            Self::Focused => "focused-window",
            Self::Monitor(_) => "monitor",
        }
    }
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

    fn resolve_backend(&self) -> Result<Backend, CommandOutput> {
        match self.cfg.backend {
            Some(backend) => Ok(backend),
            None => detect_backend().map_err(|msg| {
                CommandOutput::failed(
                    2,
                    error_line(
                        "screenshot",
                        msg,
                        Hint::Try,
                        "running this from a graphical session",
                    )
                    .into_bytes(),
                )
            }),
        }
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
        let backend = match self.resolve_backend() {
            Ok(backend) => backend,
            Err(failure) => return failure,
        };
        match capture_target(backend, &target, self.cfg.timeout).await {
            Ok(png) => attach_png(png, &target, backend),
            Err(e) => e.to_output(),
        }
    }
}

fn attach_png(png: Vec<u8>, target: &Target, backend: Backend) -> CommandOutput {
    let stdout = format!(
        "captured PNG ({}, {}, backend={}); attached to next turn\n",
        human_size(png.len()),
        target.label(),
        backend.label(),
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

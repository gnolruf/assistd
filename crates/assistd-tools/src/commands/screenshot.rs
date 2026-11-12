//! `screenshot [--full|--focused|--monitor=NAME]`: capture the screen
//! through `maim` (X11) or `grim` (Wayland) and attach the PNG in memory.

use std::sync::Arc;
use std::time::Duration;

use assistd_config::Compositor;
use assistd_ipc::ImageMime;
use assistd_utils::text::human_size;
use async_trait::async_trait;

use crate::chain::Word;
use crate::command::{Attachment, Command, CommandError, CommandInput, CommandOutput, Hint, Usage};
use crate::vision::VisionGate;

use capture::capture_target;
use detect::detect_backend;

mod capture;
mod detect;
mod geometry;

/// Configuration for the screenshot command.
#[derive(Debug, Clone)]
pub struct ScreenshotPolicyCfg {
    /// Picks the display server and the `--focused` method; `None` detects
    /// the display server on every call and supports `--focused` on X11 only.
    pub compositor: Option<Compositor>,
    pub timeout: Duration,
}

impl Default for ScreenshotPolicyCfg {
    fn default() -> Self {
        Self {
            compositor: None,
            timeout: Duration::from_secs(5),
        }
    }
}

/// Display-server backend, which selects the capture binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
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
#[derive(Debug)]
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

    fn resolve_backend(&self) -> Result<Backend, CommandError> {
        match self.cfg.compositor {
            Some(Compositor::I3) => Ok(Backend::X11),
            Some(Compositor::Sway | Compositor::Hyprland) => Ok(Backend::Wayland),
            None => detect_backend().map_err(|what| {
                CommandError::new(2, what, Hint::Try, "running this from a graphical session")
            }),
        }
    }
}

#[async_trait]
impl Command for ScreenshotCommand {
    fn name(&self) -> &'static str {
        "screenshot"
    }

    fn summary(&self) -> &'static str {
        if self.gate.supported() {
            "capture the screen as a PNG and attach it for the next LLM turn"
        } else {
            "(unavailable: model has no vision encoder)"
        }
    }

    fn usage(&self) -> Usage {
        Usage::new(
            "[--full|--focused|--monitor=<name>]",
            "Capture the screen and attach it as a vision input for the next \
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
         is acceptable.\n",
        )
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
        if !self.gate.supported() {
            return Err(CommandError::failed(
                "vision not available: model does not support images",
                Hint::Use,
                "a model with mmproj loaded",
            ));
        }
        let target = parse_target(&input.args).map_err(|what| {
            CommandError::usage(what, "screenshot --full or screenshot --focused")
        })?;
        let backend = self.resolve_backend()?;
        let png = capture_target(backend, self.cfg.compositor, &target, self.cfg.timeout).await?;
        Ok(attach_png(png, &target, backend))
    }
}

fn attach_png(png: Vec<u8>, target: &Target, backend: Backend) -> CommandOutput {
    let stdout = format!(
        "captured PNG ({}, {}, backend={}); attached to next turn\n",
        human_size(png.len() as u64),
        target.label(),
        backend.label(),
    );
    CommandOutput {
        stdout: stdout.into_bytes(),
        stderr: Vec::new(),
        exit_code: 0,
        attachments: vec![Attachment::Image {
            mime: ImageMime::Png,
            bytes: png,
        }],
    }
}

fn parse_target(args: &[Word]) -> Result<Target, String> {
    let args: Vec<&str> = args.iter().map(Word::as_str).collect();
    match args.as_slice() {
        [] | ["--full"] => Ok(Target::Full),
        ["--focused"] => Ok(Target::Focused),
        ["--monitor"] => {
            Err("--monitor requires a value: --monitor=<name> (e.g. --monitor=DP-1)".into())
        }
        [flag] => match flag.strip_prefix("--monitor=") {
            Some("") => Err("--monitor requires a name (try `xrandr --listmonitors` or `swaymsg -t get_outputs`)".into()),
            Some(name) => Ok(Target::Monitor(name.to_string())),
            None => Err(format!("unknown flag: {flag}")),
        },
        ["--monitor", ""] => Err("--monitor requires a non-empty name".into()),
        ["--monitor", name] => Ok(Target::Monitor((*name).to_string())),
        _ => Err("expects at most one flag (--full, --focused, or --monitor=<name>)".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TurnContext;
    use crate::command::run_rendered;

    fn args(raw: &[&str]) -> Vec<Word> {
        raw.iter().copied().map(Word::bare).collect()
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

    #[tokio::test]
    async fn vision_disabled_returns_exact_error() {
        let cmd = ScreenshotCommand::new(
            Arc::new(ScreenshotPolicyCfg::default()),
            VisionGate::new(false),
        );
        let input = CommandInput {
            args: args(&["--full"]),
            stdin: None,
            scope: crate::FileScope::Host,
            turn: TurnContext::default(),
        };
        let out = run_rendered(&cmd, input).await;
        assert_eq!(out.exit_code, 1);
        assert!(out.stdout.is_empty());
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] screenshot: vision not available: model does not support images. \
             Use: a model with mmproj loaded\n"
        );
        assert!(out.attachments.is_empty());
    }
}

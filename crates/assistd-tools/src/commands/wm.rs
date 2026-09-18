//! `wm <subcommand> [args]`: drive the active [`WindowManager`] from the
//! LLM's `run` tool. `wm open` spawns model-chosen argv, so it runs
//! under the same policy as `bash`. When no compositor is connected
//! every subcommand fails with one uniform error.

use std::sync::Arc;

use std::fmt::Display;

use anyhow::Result;
use async_trait::async_trait;

use assistd_wm::{Layout, ResizeDir, WindowId, WindowManager, WmError, WorkspaceId};

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line};
use crate::exec::{SPAWN_FAILED_EXIT, spawn_detached};
use crate::policy::{
    BashPolicyCfg, ConfirmationGate, SandboxAccess, SandboxInfo, SubprocessPolicy,
    matches_destructive,
};

/// The `[error] wm: <op> failed: …` line for a backend error, with the
/// recovery hint chosen by the error variant.
fn wm_error(op: impl Display, err: &WmError) -> CommandOutput {
    let (label, hint) = hint_for(err);
    CommandOutput::failed(
        1,
        error_line(NAME, format_args!("{op} failed: {err}"), label, hint).into_bytes(),
    )
}

fn hint_for(err: &WmError) -> (Hint, &'static str) {
    match err {
        WmError::Disconnected => (
            Hint::Check,
            "[compositor] in config.toml and that i3/sway/hyprland is running",
        ),
        WmError::NotFound(_) => (Hint::Use, "wm list to find the right window"),
        WmError::Rejected(_) => (Hint::Try, "wm list to verify the window/workspace exists"),
        WmError::Timeout(_) => (
            Hint::Note,
            "compositor unresponsive; retry once before assuming it crashed",
        ),
        WmError::Unsupported(_) => (
            Hint::Note,
            "the active backend may not support this operation (i3 does not list outputs)",
        ),
        WmError::Ipc(_) => (Hint::Check, "compositor connection (see daemon logs)"),
    }
}

const NAME: &str = "wm";
const SUMMARY: &str = "manage windows and workspaces (focus, move, open, list, workspaces, etc.)";

/// `wm <subcommand> [args]`: drive the active window manager from the LLM's `run` tool.
pub struct WmCommand {
    wm: Arc<dyn WindowManager>,
    policy: SubprocessPolicy,
}

impl WmCommand {
    /// `cfg`, `sandbox`, and `gate` are the same policy `bash` runs
    /// under; `wm open` is gated identically rather than by a parallel
    /// policy that could drift.
    pub fn new(
        wm: Arc<dyn WindowManager>,
        cfg: Arc<BashPolicyCfg>,
        sandbox: Arc<SandboxInfo>,
        gate: Arc<dyn ConfirmationGate>,
    ) -> Self {
        Self {
            wm,
            policy: SubprocessPolicy { cfg, sandbox, gate },
        }
    }
}

#[cfg(test)]
impl WmCommand {
    pub(crate) fn for_test(wm: Arc<dyn WindowManager>) -> Self {
        use crate::policy::AlwaysAllowGate;
        Self::new(
            wm,
            Arc::new(BashPolicyCfg::default()),
            SandboxInfo::none(),
            Arc::new(AlwaysAllowGate),
        )
    }
}

#[async_trait]
impl Command for WmCommand {
    fn name(&self) -> &str {
        NAME
    }

    fn summary(&self) -> &'static str {
        SUMMARY
    }

    fn help(&self) -> String {
        "usage: wm <subcommand> [args...]\n\
         \n\
         Manage windows and workspaces via the active compositor backend \
         (i3, sway, or hyprland). Window identifiers are decimal con_ids \
         (e.g. \"94567128432192\"); run `wm list` first to find the id \
         for the window you want to act on; the second column is the \
         application label.\n\
         \n\
         Subcommands:\n  \
           focus <id>                             focus the window with this con_id\n  \
           move <id> <workspace>                  move window to workspace\n  \
           open <app> [args...]                   launch an application (policy-gated)\n  \
           active                                 TSV: id, app of the focused window\n  \
           resize <id> <grow|shrink> <px>         width-only resize\n  \
           list                                   TSV: id, app, workspace, title\n  \
           workspaces                             TSV: num, name, focused, output\n  \
           outputs                                TSV: name, active, primary, mode, scale, focused_workspace\n  \
           layout <default|tabbed|stacking|splith|splitv>\n                                          \
         set the focused container's layout\n\
         \n\
         Call any subcommand with no arguments to see its parameter \
         help. When no compositor is connected, every subcommand emits \
         `[error] wm: compositor not connected. …` instead.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput::usage(self.help()));
        }
        if !self.wm.is_connected() {
            return Ok(CommandOutput::failed(
                1,
                error_line(
                    NAME,
                    "compositor not connected",
                    Hint::Check,
                    "[compositor] in config.toml and that i3/sway/hyprland is running",
                )
                .into_bytes(),
            ));
        }

        let sub = input.args[0].as_str();
        let rest = &input.args[1..];
        match sub {
            "focus" => focus(self.wm.as_ref(), rest).await,
            "move" => move_window(self.wm.as_ref(), rest).await,
            "open" => self.open(rest).await,
            "active" => active(self.wm.as_ref()).await,
            "resize" => resize(self.wm.as_ref(), rest).await,
            "list" => list(self.wm.as_ref()).await,
            "workspaces" => workspaces(self.wm.as_ref()).await,
            "outputs" => outputs(self.wm.as_ref()).await,
            "layout" => layout(self.wm.as_ref(), rest).await,
            other => Ok(CommandOutput::failed(
                2,
                error_line(
                    NAME,
                    format_args!("unknown subcommand '{other}'"),
                    Hint::Available,
                    "focus, move, open, active, resize, list, workspaces, outputs, layout",
                )
                .into_bytes(),
            )),
        }
    }
}

const FOCUS_HELP: &str = "usage: wm focus <id>\n\
    \n\
    Focus the window with the given decimal con_id. Run `wm list` \
    first to find ids; the first column is the id, the second is \
    the application label.\n";

async fn focus(wm: &dyn WindowManager, args: &[String]) -> Result<CommandOutput> {
    if args.is_empty() {
        return Ok(CommandOutput::usage(FOCUS_HELP.to_string()));
    }
    let id_arg = &args[0];
    let id: WindowId = match id_arg.parse() {
        Ok(i) => i,
        Err(_) => return Ok(parse_id_error("focus", id_arg)),
    };
    match wm.focus(&id).await {
        Ok(()) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) => Ok(wm_error(format_args!("focus {id_arg}"), &e)),
    }
}

fn parse_id_error(op: &'static str, raw: &str) -> CommandOutput {
    CommandOutput::usage_error(
        NAME,
        format_args!("{op}: '{raw}' is not a valid window id (positive decimal con_id)"),
        "wm list to see ids (first TSV column)",
    )
}

const MOVE_HELP: &str = "usage: wm move <id> <workspace>\n\
    \n\
    Move the window with the given con_id to the named workspace. \
    Numeric workspace identifiers (e.g. `3`) match by number; \
    non-numeric identifiers match by exact name.\n";

async fn move_window(wm: &dyn WindowManager, args: &[String]) -> Result<CommandOutput> {
    if args.len() < 2 {
        return Ok(CommandOutput::usage(MOVE_HELP.to_string()));
    }
    let id_arg = &args[0];
    let workspace_arg = &args[1];
    let id: WindowId = match id_arg.parse() {
        Ok(i) => i,
        Err(_) => return Ok(parse_id_error("move", id_arg)),
    };
    let workspace: WorkspaceId = workspace_arg
        .parse()
        .expect("WorkspaceId parser is infallible");
    match wm.move_to_workspace(&id, &workspace).await {
        Ok(()) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) => Ok(wm_error(
            format_args!("move {id_arg} to '{workspace_arg}'"),
            &e,
        )),
    }
}

const OPEN_HELP: &str = "usage: wm open <app> [args...]\n\
    \n\
    Launch an application. <app> is resolved through PATH; remaining \
    arguments are forwarded to the spawned process.\n\
    \n\
    Runs under the same policy as `bash`: denylist, destructive-pattern \
    confirmation, and the bubblewrap sandbox (widened only to reach the \
    compositor and D-Bus session sockets).\n\
    \n\
    The application is briefly watched, then left running. If it exits \
    during that window its exit code and output are returned, which is \
    how a failed launch surfaces; if it is still alive, exit 0 with no \
    output means the launch succeeded. Stdin is not forwarded, and no \
    timeout applies once it is running.\n";

impl WmCommand {
    async fn open(&self, args: &[String]) -> Result<CommandOutput> {
        let Some((app, extra)) = args.split_first() else {
            return Ok(CommandOutput::usage(OPEN_HELP.to_string()));
        };
        let argv = args.join(" ");
        let destructive = matches_destructive_argv(args, &self.policy.cfg.destructive_patterns);
        if let Err(denied) = self
            .policy
            .authorize(NAME, "open", &argv, destructive)
            .await
        {
            return Ok(denied);
        }

        let cmd = self
            .policy
            .sandbox
            .command(SandboxAccess::Session, app, extra);
        spawn_detached(NAME, cmd).await.or_else(|e| {
            let line = if e.kind() == std::io::ErrorKind::NotFound {
                error_line(
                    NAME,
                    format_args!("open: binary '{app}' not found on PATH"),
                    Hint::Check,
                    format_args!("which {app}"),
                )
            } else {
                error_line(
                    NAME,
                    format_args!("open '{app}' failed: {e}"),
                    Hint::Try,
                    "a different binary or absolute path",
                )
            };
            Ok(CommandOutput::failed(SPAWN_FAILED_EXIT, line.into_bytes()))
        })
    }
}

/// Destructive-pattern match over `wm open`'s argv.
///
/// Two passes, because argv is not a shell script. The joined form
/// anchors ordinary invocations (`wm open rm -rf ~`); re-checking each
/// argument alone catches a script smuggled into one word
/// (`wm open bash -c "rm -rf ~"`), whose tokens sit mid-line with no
/// command anchor for the first pass to find.
fn matches_destructive_argv<'a>(
    argv: &[String],
    prefixes: &'a [Vec<String>],
) -> Option<&'a [String]> {
    matches_destructive(&argv.join(" "), prefixes).or_else(|| {
        argv.iter()
            .find_map(|arg| matches_destructive(arg, prefixes))
    })
}

async fn active(wm: &dyn WindowManager) -> Result<CommandOutput> {
    match wm.focused_context().await {
        Ok(Some(ctx)) => {
            let id_str = match ctx.id {
                Some(i) => i.to_string(),
                None => "-".into(),
            };
            let app = ctx.class.as_deref().unwrap_or("-");
            Ok(CommandOutput::ok(format!("{id_str}\t{app}\n").into_bytes()))
        }
        Ok(None) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) => Ok(wm_error("active", &e)),
    }
}

const RESIZE_HELP: &str = "usage: wm resize <id> <grow|shrink> <px>\n\
    \n\
    Resize the named window's width by the given pixel amount. \
    Direction is one of `grow` or `shrink`; <px> is a non-negative \
    integer count of pixels.\n";

async fn resize(wm: &dyn WindowManager, args: &[String]) -> Result<CommandOutput> {
    if args.len() < 3 {
        return Ok(CommandOutput::usage(RESIZE_HELP.to_string()));
    }
    let id_arg = &args[0];
    let id: WindowId = match id_arg.parse() {
        Ok(i) => i,
        Err(_) => return Ok(parse_id_error("resize", id_arg)),
    };
    let direction: ResizeDir = match args[1].parse() {
        Ok(d) => d,
        Err(_) => {
            return Ok(CommandOutput::usage_error(
                NAME,
                format_args!(
                    "resize: direction must be 'grow' or 'shrink', got '{}'",
                    args[1]
                ),
                "wm resize <id> <grow|shrink> <px>",
            ));
        }
    };
    let amount: u32 = match args[2].parse() {
        Ok(n) => n,
        Err(_) => {
            return Ok(CommandOutput::usage_error(
                NAME,
                format_args!(
                    "resize: pixel amount must be a non-negative integer, got '{}'",
                    args[2]
                ),
                "wm resize <id> <grow|shrink> <px>",
            ));
        }
    };
    match wm.resize_width(&id, direction, amount).await {
        Ok(()) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) => Ok(wm_error(format_args!("resize {id_arg}"), &e)),
    }
}

async fn list(wm: &dyn WindowManager) -> Result<CommandOutput> {
    use std::fmt::Write;
    match wm.list_windows().await {
        Ok(mut windows) => {
            windows.sort_by(|a, b| {
                a.workspace
                    .as_deref()
                    .unwrap_or("")
                    .cmp(b.workspace.as_deref().unwrap_or(""))
                    .then_with(|| {
                        a.app
                            .as_deref()
                            .unwrap_or("")
                            .cmp(b.app.as_deref().unwrap_or(""))
                    })
                    .then_with(|| a.id.cmp(&b.id))
            });
            let mut out = String::new();
            for w in windows {
                let _ = write!(&mut out, "{}", w.id);
                out.push('\t');
                out.push_str(w.app.as_deref().unwrap_or("-"));
                out.push('\t');
                out.push_str(w.workspace.as_deref().unwrap_or("-"));
                out.push('\t');
                out.push_str(w.title.as_deref().unwrap_or(""));
                out.push('\n');
            }
            Ok(CommandOutput::ok(out.into_bytes()))
        }
        Err(e) => Ok(wm_error("list", &e)),
    }
}

async fn outputs(wm: &dyn WindowManager) -> Result<CommandOutput> {
    match wm.list_outputs().await {
        Ok(mut outputs) => {
            outputs.sort_by(|a, b| a.name.cmp(&b.name));
            let mut out = String::new();
            for o in outputs {
                out.push_str(&o.name);
                out.push('\t');
                out.push(if o.active { '*' } else { '-' });
                out.push('\t');
                out.push(if o.primary { '*' } else { '-' });
                out.push('\t');
                match o.current_mode {
                    Some((w, h, hz)) => {
                        // Sway reports refresh in mHz.
                        let hz_int = hz / 1000;
                        let hz_frac = hz % 1000;
                        if hz_frac == 0 {
                            out.push_str(&format!("{w}x{h}@{hz_int}Hz"));
                        } else {
                            out.push_str(&format!("{w}x{h}@{hz_int}.{:03}Hz", hz_frac));
                        }
                    }
                    None => out.push('-'),
                }
                out.push('\t');
                match o.scale {
                    Some(s) => out.push_str(&format!("{s}")),
                    None => out.push('-'),
                }
                out.push('\t');
                out.push_str(o.focused_workspace.as_deref().unwrap_or("-"));
                out.push('\n');
            }
            Ok(CommandOutput::ok(out.into_bytes()))
        }
        Err(e) => Ok(wm_error("outputs", &e)),
    }
}

async fn workspaces(wm: &dyn WindowManager) -> Result<CommandOutput> {
    match wm.list_workspaces().await {
        Ok(mut workspaces) => {
            workspaces.sort_by_key(|w| w.num);
            let mut out = String::new();
            for w in workspaces {
                out.push_str(&w.num.to_string());
                out.push('\t');
                out.push_str(&w.name);
                out.push('\t');
                out.push(if w.focused { '*' } else { '-' });
                out.push('\t');
                out.push_str(&w.output);
                out.push('\n');
            }
            Ok(CommandOutput::ok(out.into_bytes()))
        }
        Err(e) => Ok(wm_error("workspaces", &e)),
    }
}

const LAYOUT_HELP: &str = "usage: wm layout <default|tabbed|stacking|splith|splitv>\n\
    \n\
    Set the layout of the currently focused container. `default` \
    toggles between split, tabbed, and stacking based on the \
    container's previous layout.\n";

async fn layout(wm: &dyn WindowManager, args: &[String]) -> Result<CommandOutput> {
    if args.is_empty() {
        return Ok(CommandOutput::usage(LAYOUT_HELP.to_string()));
    }
    let raw = args[0].as_str();
    let layout: Layout = match raw.parse() {
        Ok(l) => l,
        Err(_) => {
            return Ok(CommandOutput::usage_error(
                NAME,
                format_args!("layout: '{raw}' is not a known layout"),
                "default | tabbed | stacking | splith | splitv",
            ));
        }
    };
    match wm.set_layout(layout).await {
        Ok(()) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) => Ok(wm_error(format_args!("layout '{layout}'"), &e)),
    }
}

#[cfg(test)]
mod tests;

//! `wm <subcommand> [args]`: drive the active [`WindowManager`] from the
//! LLM's `run` tool.

use std::fmt::{Display, Write};
use std::sync::Arc;

use assistd_wm::{Layout, OutputInfo, ResizeDir, WindowId, WindowManager, WmError, WorkspaceId};
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line};
use crate::exec::POLICY_DENIED_EXIT;
use crate::policy::{
    Approval, ApprovalGate, Approvals, ConfirmationGate, ConfirmationRequest, Risk,
    on_blocking_pool,
};

mod desktop_entry;
mod open;

use open::{OpenArgs, OpenError, PreparedLaunch};

const NAME: &str = "wm";
const SUMMARY: &str = "manage windows and workspaces (focus, move, open, list, workspaces, etc.)";

const FOCUS_HELP: &str = "usage: wm focus <id>\n\
    \n\
    Focus the window with the given decimal con_id. Run `wm list` \
    first to find ids; the first column is the id, the second is \
    the application label.\n";

const MOVE_HELP: &str = "usage: wm move <id> <workspace>\n\
    \n\
    Move the window with the given con_id to the named workspace. \
    Numeric workspace identifiers (e.g. `3`) match by number; \
    non-numeric identifiers match by exact name.\n";

const OPEN_HELP: &str = "usage: wm open [--workspace <ws>] [--cwd <dir>] <app> [file|url...]\n\
    \n\
    Launch an installed application, as the user would from their app \
    menu. <app> is a desktop id: the name of a .desktop file in \
    /usr/share/applications or ~/.local/share/applications without the \
    extension (e.g. firefox, nvim, org.gnome.Nautilus). Applications that \
    run in a terminal, such as nvim, open in a terminal emulator.\n\
    \n\
    --workspace switches to that workspace (number or name) first, so the \
    window opens there. --cwd sets the application's working directory. \
    Files (absolute or ~/ paths) and URLs are accepted only by applications \
    whose desktop entry takes them.\n\
    \n\
    The application runs as the user, outside the sandbox, so the user \
    confirms the launch unless they chose \"always allow\" for that \
    application; a launch with files or URLs always asks. It starts in the \
    background: success means the compositor accepted the launch, not that a \
    window appeared, so check with `wm list`.\n";

const RESIZE_HELP: &str = "usage: wm resize <id> <grow|shrink> <px>\n\
    \n\
    Resize the named window's width by the given pixel amount. \
    Direction is one of `grow` or `shrink`; <px> is a non-negative \
    integer count of pixels.\n";

const LAYOUT_HELP: &str = "usage: wm layout <default|tabbed|stacking|splith|splitv>\n\
    \n\
    Set the layout of the currently focused container. `default` \
    toggles between split, tabbed, and stacking based on the \
    container's previous layout.\n";

/// `wm <subcommand> [args]`: drive the active window manager from the LLM's `run` tool.
#[derive(Debug)]
pub struct WmCommand {
    wm: Arc<dyn WindowManager>,
    gate: Arc<dyn ConfirmationGate>,
    /// Desktop entries, by path, that `wm open` launches without asking.
    apps: ApprovalGate,
}

impl WmCommand {
    /// A `wm` command driving `wm`; `wm open` asks `gate` before launching
    /// an application `approved_apps` lacks.
    pub fn new(
        wm: Arc<dyn WindowManager>,
        gate: Arc<dyn ConfirmationGate>,
        approved_apps: Arc<Approvals>,
    ) -> Self {
        let apps = ApprovalGate::new(Arc::clone(&gate), approved_apps);
        Self { wm, gate, apps }
    }

    async fn open(&self, args: &[String]) -> CommandOutput {
        if args.is_empty() {
            return CommandOutput::usage(OPEN_HELP.to_string());
        }
        let args = match OpenArgs::parse(args) {
            Ok(args) => args,
            Err(what) => {
                return CommandOutput::usage_error(
                    NAME,
                    format_args!("open: {what}"),
                    "wm open [--workspace <ws>] [--cwd <dir>] <app> [file|url...]",
                );
            }
        };
        let (args, prepared) = on_blocking_pool(move || {
            let prepared = PreparedLaunch::find(&args);
            (args, prepared)
        })
        .await;
        let launch = match prepared {
            Ok(launch) => launch,
            Err(e) => return open_failed(&e),
        };
        if !self.confirmed(&args, &launch).await {
            return open_cancelled();
        }
        match self.wm.exec(args.workspace.as_ref(), &launch.command).await {
            Ok(()) => opened(&launch, args.workspace.as_ref()),
            Err(e) => wm_error(format_args!("open {}", launch.entry.name), &e),
        }
    }

    /// Whether the user lets `launch` run. Approvals cover a launch with no
    /// files or URLs; one with them always asks.
    async fn confirmed(&self, args: &OpenArgs, launch: &PreparedLaunch) -> bool {
        let entry = &launch.entry;
        let key = entry.path.to_string_lossy().into_owned();
        let has_targets = args.has_targets();
        let request = || ConfirmationRequest {
            tool: NAME.to_string(),
            script: launch.command.as_str().to_string(),
            matched_pattern: format!(
                "opens {} ({key}) {} as you, outside the sandbox",
                entry.name,
                workspace_phrase(args.workspace.as_ref())
            ),
            always_allow: if has_targets {
                Vec::new()
            } else {
                vec![key.clone()]
            },
            risk: if has_targets {
                Risk::Elevated
            } else {
                Risk::Routine
            },
        };
        if has_targets {
            self.gate.confirm(request()).await != Approval::Deny
        } else {
            self.apps.confirm(&key, request).await
        }
    }
}

#[cfg(test)]
impl WmCommand {
    pub(crate) fn for_test(wm: Arc<dyn WindowManager>) -> Self {
        Self::new(
            wm,
            Arc::new(crate::policy::AlwaysAllowGate),
            Arc::new(Approvals::unsaved()),
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
           open <app> [file|url...]               launch an installed application, e.g. on\n                                          \
         a workspace: wm open --workspace 2 firefox\n  \
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

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if input.args.is_empty() {
            return CommandOutput::usage(self.help());
        }
        if !self.wm.is_connected() {
            return CommandOutput::failed(
                1,
                error_line(
                    NAME,
                    "compositor not connected",
                    Hint::Check,
                    "[compositor] in config.toml and that i3/sway/hyprland is running",
                )
                .into_bytes(),
            );
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
            other => CommandOutput::failed(
                2,
                error_line(
                    NAME,
                    format_args!("unknown subcommand '{other}'"),
                    Hint::Available,
                    "focus, move, open, active, resize, list, workspaces, outputs, layout",
                )
                .into_bytes(),
            ),
        }
    }
}

async fn focus(wm: &dyn WindowManager, args: &[String]) -> CommandOutput {
    if args.is_empty() {
        return CommandOutput::usage(FOCUS_HELP.to_string());
    }
    let id_arg = &args[0];
    let id: WindowId = match id_arg.parse() {
        Ok(id) => id,
        Err(_) => return parse_id_error("focus", id_arg),
    };
    match wm.focus(&id).await {
        Ok(()) => CommandOutput::ok(Vec::new()),
        Err(e) => wm_error(format_args!("focus {id_arg}"), &e),
    }
}

fn parse_id_error(op: &'static str, raw: &str) -> CommandOutput {
    CommandOutput::usage_error(
        NAME,
        format_args!("{op}: '{raw}' is not a valid window id (positive decimal con_id)"),
        "wm list to see ids (first TSV column)",
    )
}

async fn move_window(wm: &dyn WindowManager, args: &[String]) -> CommandOutput {
    if args.len() < 2 {
        return CommandOutput::usage(MOVE_HELP.to_string());
    }
    let id_arg = &args[0];
    let workspace_arg = &args[1];
    let id: WindowId = match id_arg.parse() {
        Ok(id) => id,
        Err(_) => return parse_id_error("move", id_arg),
    };
    let workspace: WorkspaceId = workspace_arg
        .parse()
        .expect("WorkspaceId parser is infallible");
    match wm.move_to_workspace(&id, &workspace).await {
        Ok(()) => CommandOutput::ok(Vec::new()),
        Err(e) => wm_error(format_args!("move {id_arg} to '{workspace_arg}'"), &e),
    }
}

async fn active(wm: &dyn WindowManager) -> CommandOutput {
    match wm.focused_context().await {
        Ok(Some(ctx)) => {
            let id_str = match ctx.id {
                Some(id) => id.to_string(),
                None => "-".into(),
            };
            let app = ctx.class.as_deref().unwrap_or("-");
            CommandOutput::ok(format!("{id_str}\t{app}\n").into_bytes())
        }
        Ok(None) => CommandOutput::ok(Vec::new()),
        Err(e) => wm_error("active", &e),
    }
}

async fn resize(wm: &dyn WindowManager, args: &[String]) -> CommandOutput {
    if args.len() < 3 {
        return CommandOutput::usage(RESIZE_HELP.to_string());
    }
    let id_arg = &args[0];
    let id: WindowId = match id_arg.parse() {
        Ok(id) => id,
        Err(_) => return parse_id_error("resize", id_arg),
    };
    let direction: ResizeDir = match args[1].parse() {
        Ok(direction) => direction,
        Err(_) => {
            return CommandOutput::usage_error(
                NAME,
                format_args!(
                    "resize: direction must be 'grow' or 'shrink', got '{}'",
                    args[1]
                ),
                "wm resize <id> <grow|shrink> <px>",
            );
        }
    };
    let amount: u32 = match args[2].parse() {
        Ok(amount) => amount,
        Err(_) => {
            return CommandOutput::usage_error(
                NAME,
                format_args!(
                    "resize: pixel amount must be a non-negative integer, got '{}'",
                    args[2]
                ),
                "wm resize <id> <grow|shrink> <px>",
            );
        }
    };
    match wm.resize_width(&id, direction, amount).await {
        Ok(()) => CommandOutput::ok(Vec::new()),
        Err(e) => wm_error(format_args!("resize {id_arg}"), &e),
    }
}

async fn list(wm: &dyn WindowManager) -> CommandOutput {
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
            for window in windows {
                let _ = write!(&mut out, "{}", window.id);
                out.push('\t');
                out.push_str(window.app.as_deref().unwrap_or("-"));
                out.push('\t');
                out.push_str(window.workspace.as_deref().unwrap_or("-"));
                out.push('\t');
                out.push_str(window.title.as_deref().unwrap_or(""));
                out.push('\n');
            }
            CommandOutput::ok(out.into_bytes())
        }
        Err(e) => wm_error("list", &e),
    }
}

async fn outputs(wm: &dyn WindowManager) -> CommandOutput {
    match wm.list_outputs().await {
        Ok(mut outputs) => {
            outputs.sort_by(|a, b| a.name.cmp(&b.name));
            let mut out = String::new();
            for output in &outputs {
                push_output_row(&mut out, output);
            }
            CommandOutput::ok(out.into_bytes())
        }
        Err(e) => wm_error("outputs", &e),
    }
}

fn push_output_row(out: &mut String, output: &OutputInfo) {
    out.push_str(&output.name);
    out.push('\t');
    out.push(if output.active { '*' } else { '-' });
    out.push('\t');
    out.push(if output.primary { '*' } else { '-' });
    out.push('\t');
    match output.current_mode {
        Some((width, height, millihertz)) => out.push_str(&format_mode(width, height, millihertz)),
        None => out.push('-'),
    }
    out.push('\t');
    match output.scale {
        Some(scale) => out.push_str(&format!("{scale}")),
        None => out.push('-'),
    }
    out.push('\t');
    out.push_str(output.focused_workspace.as_deref().unwrap_or("-"));
    out.push('\n');
}

/// `WxH@RHz`, from a refresh rate in mHz as Sway reports it.
fn format_mode(width: u32, height: u32, millihertz: u32) -> String {
    let hz_int = millihertz / 1000;
    let hz_frac = millihertz % 1000;
    if hz_frac == 0 {
        format!("{width}x{height}@{hz_int}Hz")
    } else {
        format!("{width}x{height}@{hz_int}.{hz_frac:03}Hz")
    }
}

async fn workspaces(wm: &dyn WindowManager) -> CommandOutput {
    match wm.list_workspaces().await {
        Ok(mut workspaces) => {
            workspaces.sort_by_key(|w| w.num);
            let mut out = String::new();
            for workspace in workspaces {
                out.push_str(&workspace.num.to_string());
                out.push('\t');
                out.push_str(&workspace.name);
                out.push('\t');
                out.push(if workspace.focused { '*' } else { '-' });
                out.push('\t');
                out.push_str(&workspace.output);
                out.push('\n');
            }
            CommandOutput::ok(out.into_bytes())
        }
        Err(e) => wm_error("workspaces", &e),
    }
}

async fn layout(wm: &dyn WindowManager, args: &[String]) -> CommandOutput {
    if args.is_empty() {
        return CommandOutput::usage(LAYOUT_HELP.to_string());
    }
    let raw = args[0].as_str();
    let layout: Layout = match raw.parse() {
        Ok(layout) => layout,
        Err(_) => {
            return CommandOutput::usage_error(
                NAME,
                format_args!("layout: '{raw}' is not a known layout"),
                "default | tabbed | stacking | splith | splitv",
            );
        }
    };
    match wm.set_layout(layout).await {
        Ok(()) => CommandOutput::ok(Vec::new()),
        Err(e) => wm_error(format_args!("layout '{layout}'"), &e),
    }
}

fn opened(launch: &PreparedLaunch, workspace: Option<&WorkspaceId>) -> CommandOutput {
    let line = format!(
        "opened {} {}; it starts in the background, so run `wm list` to see its window \
         once it appears\n",
        launch.entry.name,
        workspace_phrase(workspace)
    );
    CommandOutput::ok(line.into_bytes())
}

fn workspace_phrase(workspace: Option<&WorkspaceId>) -> String {
    match workspace {
        Some(workspace) => format!("on workspace {workspace}"),
        None => "on the focused workspace".to_string(),
    }
}

fn open_failed(err: &OpenError) -> CommandOutput {
    let (hint, recovery) = err.recovery();
    CommandOutput::failed(
        1,
        error_line(NAME, format_args!("open: {err}"), hint, recovery).into_bytes(),
    )
}

fn open_cancelled() -> CommandOutput {
    CommandOutput::failed(
        POLICY_DENIED_EXIT,
        error_line(
            NAME,
            "open cancelled by user",
            Hint::Try,
            "a different approach",
        )
        .into_bytes(),
    )
}

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
        WmError::Rejected(_) => (Hint::Try, "wm list to verify the window/workspace exists"),
        WmError::Timeout(_) => (
            Hint::Note,
            "compositor unresponsive; retry once before assuming it crashed",
        ),
        WmError::Unsupported(_) => (
            Hint::Note,
            "the active backend may not support this operation (i3 does not list outputs)",
        ),
        WmError::Ipc { .. } => (Hint::Check, "compositor connection (see daemon logs)"),
    }
}

#[cfg(test)]
mod tests;

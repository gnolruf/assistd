//! `wm <subcommand> [args]`: drive the active [`WindowManager`] from the
//! LLM's `run` tool.

use std::fmt::{Display, Write};
use std::path::Path;
use std::sync::Arc;

use assistd_ipc::{ConfirmSource, PlannedAction};
use assistd_wm::{Layout, OutputInfo, ResizeDir, WindowId, WindowManager, WmError, WorkspaceId};
use async_trait::async_trait;

use crate::TurnContext;
use crate::chain::Word;
use crate::command::{Command, CommandError, CommandInput, CommandOutput, Hint, Usage};
use crate::exec::POLICY_DENIED_EXIT;
use crate::policy::{
    Approval, ApprovalGate, Approvals, ConfirmationGate, ConfirmationRequest, JudgeContext,
    JudgedClass, Risk, Summary, Unattended, on_blocking_pool,
};

mod default_app;
mod desktop_entry;
mod open;

use desktop_entry::{DesktopEntry, Executable};
use open::{OpenArgs, OpenError, PreparedLaunch, Subject};

const NAME: &str = "wm";
const SUMMARY: &str = "manage windows and workspaces through the compositor";

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

const OPEN_USAGE: &str = "wm open [--workspace <ws>] [--cwd <dir>] <app|file> [file|url...]";

const OPEN_HELP: &str = "usage: wm open [--workspace <ws>] [--cwd <dir>] <app|file> [file|url...]\n\
    \n\
    Launch an installed application, as the user would from their app \
    menu. <app> is a desktop id: the name of a .desktop file in \
    /usr/share/applications or ~/.local/share/applications without the \
    extension (e.g. firefox, nvim, org.gnome.Nautilus). Applications that \
    run in a terminal, such as nvim, open in a terminal emulator.\n\
    \n\
    To put a file on the user's screen, name the file in place of the \
    application (wm open ~/notes/report.md): it opens in the user's default \
    application for its type, as a double click would.\n\
    \n\
    --workspace switches to that workspace (number or name) first, so the \
    window opens there. --cwd sets the application's working directory. \
    Files (absolute or ~/ paths) and URLs are accepted only by applications \
    whose desktop entry takes them.\n\
    \n\
    The application runs as the user, outside the sandbox, so the user \
    confirms the launch unless they chose \"always allow\" for that \
    application; a launch with files or URLs always asks, and so does one \
    whose program sandboxed commands can write. It starts in the \
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
    /// Desktop entries, each with the program it runs, that `wm open` launches without asking.
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

    async fn open(&self, turn: &TurnContext, args: &[Word]) -> Result<CommandOutput, CommandError> {
        if args.is_empty() {
            return Ok(CommandOutput::usage(OPEN_HELP.to_string()));
        }
        let args = OpenArgs::parse(args)
            .map_err(|what| CommandError::usage(format_args!("open: {what}"), OPEN_USAGE))?;
        let (args, prepared) = on_blocking_pool(move || {
            let prepared = PreparedLaunch::find(&args);
            (args, prepared)
        })
        .await;
        let launch = prepared.map_err(|e| open_failed(&e))?;
        if !self.confirmed(turn, &args, &launch).await {
            return Err(open_cancelled());
        }
        self.wm
            .exec(args.workspace.as_ref(), &launch.command)
            .await
            .map_err(|e| wm_error(format_args!("open {}", launch.entry.name), &e))?;
        Ok(opened(&launch, args.workspace.as_ref()))
    }

    /// Whether the user, or in auto mode the safety judge, lets `launch` run. Approvals cover a
    /// launch with no files, URLs, or `--cwd` of an entry while it runs the program it was
    /// approved with; any other launch is never approved for good, and one that opens a URL
    /// or runs what sandboxed commands can write always asks.
    async fn confirmed(
        &self,
        turn: &TurnContext,
        args: &OpenArgs,
        launch: &PreparedLaunch,
    ) -> bool {
        let entry = &launch.entry;
        let approval = approval_key(launch, args);
        let caution = launch.executable.caution();
        let request = || ConfirmationRequest {
            tool: NAME.to_string(),
            script: launch.command.as_str().to_string(),
            matched_pattern: format!(
                "opens {} ({}) {} as you, outside the sandbox{}",
                entry.name,
                entry.path.display(),
                workspace_phrase(args.workspace.as_ref()),
                caution
                    .as_deref()
                    .map(|caution| format!(". {caution}"))
                    .unwrap_or_default()
            ),
            summary: launch_summary(launch, args),
            always_allow: approval.iter().cloned().collect(),
            risk: if approval.is_some() {
                Risk::Routine
            } else {
                Risk::Elevated
            },
            unattended: unattended_launch(launch, args),
            source: ConfirmSource::Main,
        };
        match &approval {
            Some(key) => self.apps.confirm(turn, key, request).await,
            None => self.gate.confirm(turn, request()).await != Approval::Deny,
        }
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

    fn usage(&self) -> Usage {
        Usage::new(
            "<focus|move|open|active|resize|list|workspaces|outputs|layout> [args...]",
            "Manage windows and workspaces via the active compositor backend \
         (i3, sway, or hyprland). Window identifiers are decimal con_ids \
         (e.g. \"94567128432192\"); run `wm list` first to find the id \
         for the window you want to act on; the second column is the \
         application label.\n\
         \n\
         Subcommands:\n  \
           focus <id>                             focus the window with this con_id\n  \
           move <id> <workspace>                  move window to workspace\n  \
           open [--workspace <ws>] [--cwd <dir>] <app|file> [file|url...]\n                                          \
         launch an installed application by desktop id (e.g. firefox);\n                                          \
         terminal programs such as nvim open in a terminal by themselves:\n                                          \
         wm open --workspace 2 --cwd ~/project nvim\n                                          \
         or show the user a file in their default application for it:\n                                          \
         wm open ~/notes/report.md\n  \
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
         `[error] wm: compositor not connected. …` instead.\n",
        )
    }

    /// URLs and window criteria never name files, and `open` expands its own `~`.
    fn expands_args(&self) -> bool {
        false
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
        let Some((subcommand, rest)) = input.args.split_first() else {
            return Ok(CommandOutput::usage(self.help()));
        };
        if !self.wm.is_connected() {
            return Err(CommandError::failed(
                "compositor not connected",
                Hint::Check,
                "[compositor] in config.toml and that i3/sway/hyprland is running",
            ));
        }

        match subcommand.as_str() {
            "focus" => focus(self.wm.as_ref(), rest).await,
            "move" => move_window(self.wm.as_ref(), rest).await,
            "open" => self.open(&input.turn, rest).await,
            "active" => active(self.wm.as_ref()).await,
            "resize" => resize(self.wm.as_ref(), rest).await,
            "list" => list(self.wm.as_ref()).await,
            "workspaces" => workspaces(self.wm.as_ref()).await,
            "outputs" => outputs(self.wm.as_ref()).await,
            "layout" => layout(self.wm.as_ref(), rest).await,
            other => Err(CommandError::new(
                2,
                format_args!("unknown subcommand '{other}'"),
                Hint::Available,
                "focus, move, open, active, resize, list, workspaces, outputs, layout",
            )),
        }
    }
}

async fn focus(wm: &dyn WindowManager, args: &[Word]) -> Result<CommandOutput, CommandError> {
    let [id_arg, ..] = args else {
        return Ok(CommandOutput::usage(FOCUS_HELP.to_string()));
    };
    let id = parse_id("focus", id_arg)?;
    wm.focus(&id)
        .await
        .map_err(|e| wm_error(format_args!("focus {id_arg}"), &e))?;
    Ok(CommandOutput::ok(Vec::new()))
}

/// Why `wm open` could not launch the app with desktop entry `id`, if it could not. Blocks on
/// the filesystem.
pub(crate) fn unlaunchable_app(id: &str) -> Option<String> {
    DesktopEntry::find(id).err().map(|e| e.to_string())
}

fn parse_id(op: &'static str, raw: &Word) -> Result<WindowId, CommandError> {
    raw.as_str().parse().map_err(|_| {
        CommandError::usage(
            format_args!("{op}: '{raw}' is not a valid window id (positive decimal con_id)"),
            "wm list to see ids (first TSV column)",
        )
    })
}

async fn move_window(wm: &dyn WindowManager, args: &[Word]) -> Result<CommandOutput, CommandError> {
    let [id_arg, workspace_arg, ..] = args else {
        return Ok(CommandOutput::usage(MOVE_HELP.to_string()));
    };
    let id = parse_id("move", id_arg)?;
    let workspace = WorkspaceId::from(workspace_arg.as_str());
    wm.move_to_workspace(&id, &workspace)
        .await
        .map_err(|e| wm_error(format_args!("move {id_arg} to '{workspace_arg}'"), &e))?;
    Ok(CommandOutput::ok(Vec::new()))
}

async fn active(wm: &dyn WindowManager) -> Result<CommandOutput, CommandError> {
    let focused = wm
        .focused_context()
        .await
        .map_err(|e| wm_error("active", &e))?;
    let Some(ctx) = focused else {
        return Ok(CommandOutput::ok(Vec::new()));
    };
    let id_str = match ctx.id {
        Some(id) => id.to_string(),
        None => "-".into(),
    };
    let app = ctx.class.as_deref().unwrap_or("-");
    Ok(CommandOutput::ok(format!("{id_str}\t{app}\n").into_bytes()))
}

async fn resize(wm: &dyn WindowManager, args: &[Word]) -> Result<CommandOutput, CommandError> {
    let [id_arg, direction_arg, amount_arg, ..] = args else {
        return Ok(CommandOutput::usage(RESIZE_HELP.to_string()));
    };
    let id = parse_id("resize", id_arg)?;
    let direction: ResizeDir = direction_arg.as_str().parse().map_err(|_| {
        CommandError::usage(
            format_args!("resize: direction must be 'grow' or 'shrink', got '{direction_arg}'"),
            "wm resize <id> <grow|shrink> <px>",
        )
    })?;
    let amount: u32 = amount_arg.as_str().parse().map_err(|_| {
        CommandError::usage(
            format_args!("resize: pixel amount must be a non-negative integer, got '{amount_arg}'"),
            "wm resize <id> <grow|shrink> <px>",
        )
    })?;
    wm.resize_width(&id, direction, amount)
        .await
        .map_err(|e| wm_error(format_args!("resize {id_arg}"), &e))?;
    Ok(CommandOutput::ok(Vec::new()))
}

async fn list(wm: &dyn WindowManager) -> Result<CommandOutput, CommandError> {
    let mut windows = wm.list_windows().await.map_err(|e| wm_error("list", &e))?;
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
    Ok(CommandOutput::ok(out.into_bytes()))
}

async fn outputs(wm: &dyn WindowManager) -> Result<CommandOutput, CommandError> {
    let mut outputs = wm
        .list_outputs()
        .await
        .map_err(|e| wm_error("outputs", &e))?;
    outputs.sort_by(|a, b| a.name.cmp(&b.name));
    let mut out = String::new();
    for output in &outputs {
        push_output_row(&mut out, output);
    }
    Ok(CommandOutput::ok(out.into_bytes()))
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

async fn workspaces(wm: &dyn WindowManager) -> Result<CommandOutput, CommandError> {
    let mut workspaces = wm
        .list_workspaces()
        .await
        .map_err(|e| wm_error("workspaces", &e))?;
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
    Ok(CommandOutput::ok(out.into_bytes()))
}

async fn layout(wm: &dyn WindowManager, args: &[Word]) -> Result<CommandOutput, CommandError> {
    let [raw, ..] = args else {
        return Ok(CommandOutput::usage(LAYOUT_HELP.to_string()));
    };
    let layout: Layout = raw.as_str().parse().map_err(|_| {
        CommandError::usage(
            format_args!("layout: '{raw}' is not a known layout"),
            "default | tabbed | stacking | splith | splitv",
        )
    })?;
    wm.set_layout(layout)
        .await
        .map_err(|e| wm_error(format_args!("layout '{layout}'"), &e))?;
    Ok(CommandOutput::ok(Vec::new()))
}

/// The name `launch` is approved for good under: its entry and the program the entry runs.
/// None for a launch handed files, URLs or `--cwd`, or whose program cannot be pinned.
fn approval_key(launch: &PreparedLaunch, args: &OpenArgs) -> Option<String> {
    let program = launch.executable.pinned()?;
    (!args.has_targets_or_cwd())
        .then(|| format!("{} -> {}", launch.entry.path.display(), program.display()))
}

/// Opening a URL is egress, and a program sandboxed commands can write may have been replaced,
/// so no mode approves either; the safety judge may settle any other launch.
fn unattended_launch(launch: &PreparedLaunch, args: &OpenArgs) -> Unattended {
    if launch.opens_url || matches!(launch.executable, Executable::Writable(_)) {
        return Unattended::Never;
    }
    let grantable = match &args.subject {
        Subject::App(id) if approval_key(launch, args).is_some() => vec![PlannedAction::App {
            desktop_entry: id.clone(),
        }],
        _ => Vec::new(),
    };
    Unattended::Judged(JudgeContext {
        class: JudgedClass::Launch,
        cwd: None,
        why: format!(
            "it launches {} ({}) as the user, outside the sandbox, and the launch is not yet \
             approved",
            launch.entry.name,
            launch.entry.path.display()
        ),
        grantable,
    })
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

/// `launch` as `args` asks, in a sentence that names the file it is to show, with why its
/// program cannot be pinned, or else that it is handed files, URLs or a working directory.
fn launch_summary(launch: &PreparedLaunch, args: &OpenArgs) -> Summary {
    let name = &launch.entry.name;
    let shown = match &args.subject {
        Subject::File(file) => Path::new(file).file_name(),
        Subject::App(_) => None,
    };
    let what = match shown {
        Some(file) => format!("opening {} in {name}", file.to_string_lossy()),
        None => format!("opening {name}"),
    };
    let handed_targets = (shown.is_none() && args.has_targets_or_cwd())
        .then(|| "It is handed files, addresses or a folder to open.".to_string());
    Summary {
        what,
        why: launch.executable.caution().or(handed_targets),
    }
}

fn workspace_phrase(workspace: Option<&WorkspaceId>) -> String {
    match workspace {
        Some(workspace) => format!("on workspace {workspace}"),
        None => "on the focused workspace".to_string(),
    }
}

fn open_failed(err: &OpenError) -> CommandError {
    let (hint, recovery) = err.recovery();
    CommandError::failed(format_args!("open: {err}"), hint, recovery)
}

fn open_cancelled() -> CommandError {
    CommandError::new(
        POLICY_DENIED_EXIT,
        "open cancelled by user",
        Hint::Try,
        "a different approach",
    )
}

/// `<op> failed: …` for a backend error, with the recovery hint chosen by the error variant.
fn wm_error(op: impl Display, err: &WmError) -> CommandError {
    let (label, hint) = hint_for(err);
    CommandError::failed(format_args!("{op} failed: {err}"), label, hint)
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
        WmError::Ipc { .. } => (Hint::Check, "compositor connection (see daemon logs)"),
    }
}

#[cfg(test)]
mod tests;

//! `wm <subcommand> [args]` — drive the active [`WindowManager`] backend
//! from the LLM's `run` tool.
//!
//! Subcommand surface:
//!
//! - `wm focus <class>` — focus the named window
//! - `wm move <class> <workspace>` — move window to workspace
//! - `wm open <app> [args...]` — launch a process (does not go through
//!   the WindowManager — i3/sway/hyprland don't spawn processes, they
//!   only manage already-mapped windows)
//! - `wm active` — class of the focused window
//! - `wm resize <class> <grow|shrink> <px>` — width-only resize
//! - `wm list` — TSV `<class>\t<workspace>\t<title>`
//! - `wm workspaces` — TSV `<num>\t<name>\t<focused>\t<output>`
//! - `wm outputs` — TSV `<name>\t<active>\t<primary>\t<mode>\t<scale>\t<focused_workspace>`
//! - `wm layout <default|tabbed|stacking|splith|splitv>` — set the
//!   focused container's layout
//!
//! Discovery: `wm` (no args) returns the help block on stdout (exit 2).
//! Every subcommand with too few args returns its own usage block on
//! stdout (exit 2). All real failures emit a convention-compliant
//! `[error] wm: …. <Hint>: <recovery>` line on stderr.
//!
//! When the backend is [`assistd_wm::NoWindowManager`] (no compositor
//! configured / connect failure / Sway+Hyprland not yet implemented),
//! every subcommand short-circuits with `[error] wm: compositor not
//! connected. …` so the LLM gets one uniform error to recover from.

use std::process::Stdio;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::process::Command as ProcCommand;

use assistd_wm::{Layout, ResizeDir, WindowId, WindowManager, WmError, WorkspaceId};

use crate::command::{Command, CommandInput, CommandOutput, error_line};

/// Pick the `(label, hint)` pair attached to a [`WmError`] for the
/// `[error] wm: …. <label>: <hint>` line the LLM sees. The variant
/// determines the recovery action; the handler-specific operation is
/// already in the message body (`"focus 'Firefox' failed: …"`), so the
/// hint stays per-variant rather than per-handler.
fn hint_for(err: &WmError) -> (&'static str, &'static str) {
    match err {
        WmError::Disconnected => (
            "Check",
            "[compositor] in config.toml and that i3/sway/hyprland is running",
        ),
        WmError::NotFound(_) => ("Use", "wm list to find the right window"),
        WmError::Rejected(_) => ("Try", "wm list to verify the window/workspace exists"),
        WmError::Timeout(_) => (
            "Note",
            "compositor unresponsive; retry once before assuming it crashed",
        ),
        WmError::Unsupported(_) => (
            "Note",
            "the active backend may not support this operation (i3 does not list outputs)",
        ),
        WmError::Ipc(_) => ("Check", "compositor connection (see daemon logs)"),
    }
}

const NAME: &str = "wm";
const SUMMARY: &str = "manage windows and workspaces (focus, move, open, list, workspaces, etc.)";

pub struct WmCommand {
    wm: Arc<dyn WindowManager>,
}

impl WmCommand {
    pub fn new(wm: Arc<dyn WindowManager>) -> Self {
        Self { wm }
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
         (i3, sway, or hyprland). Window identifiers are X11 classes on \
         i3 (e.g. \"Firefox\", \"code\") — see `wm list` for the exact \
         strings to use.\n\
         \n\
         Subcommands:\n  \
           focus <class>                          focus the named window\n  \
           move <class> <workspace>               move window to workspace\n  \
           open <app> [args...]                   launch an application\n  \
           active                                 class of the focused window\n  \
           resize <class> <grow|shrink> <px>      width-only resize\n  \
           list                                   TSV: class, workspace, title\n  \
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
            return Ok(help_output(self.help()));
        }
        if !self.wm.is_connected() {
            return Ok(CommandOutput::failed(
                1,
                error_line(
                    NAME,
                    "compositor not connected",
                    "Check",
                    "[compositor] in config.toml and that i3/sway/hyprland is running",
                )
                .into_bytes(),
            ));
        }

        let sub = input.args[0].as_str();
        let rest = &input.args[1..];
        match sub {
            "focus" => handle_focus(self.wm.as_ref(), rest).await,
            "move" => handle_move(self.wm.as_ref(), rest).await,
            "open" => handle_open(rest).await,
            "active" => handle_active(self.wm.as_ref()).await,
            "resize" => handle_resize(self.wm.as_ref(), rest).await,
            "list" => handle_list(self.wm.as_ref()).await,
            "workspaces" => handle_workspaces(self.wm.as_ref()).await,
            "outputs" => handle_outputs(self.wm.as_ref()).await,
            "layout" => handle_layout(self.wm.as_ref(), rest).await,
            other => Ok(CommandOutput::failed(
                2,
                error_line(
                    NAME,
                    format_args!("unknown subcommand '{other}'"),
                    "Available",
                    "focus, move, open, active, resize, list, workspaces, outputs, layout",
                )
                .into_bytes(),
            )),
        }
    }
}

/// Stdout-help with exit 2, matching the convention used by other
/// commands that have argument-required modes (grep, write, …).
fn help_output(text: String) -> CommandOutput {
    CommandOutput {
        stdout: text.into_bytes(),
        stderr: Vec::new(),
        exit_code: 2,
        attachments: Vec::new(),
    }
}

// --------- subcommand handlers ---------

const FOCUS_HELP: &str = "usage: wm focus <class>\n\
    \n\
    Focus the window with the given class. Use `wm list` to see \
    available windows.\n";

async fn handle_focus(wm: &dyn WindowManager, args: &[String]) -> Result<CommandOutput> {
    if args.is_empty() {
        return Ok(help_output(FOCUS_HELP.to_string()));
    }
    // PR 3a: arg parsing into WindowId is currently infallible (any
    // string is a valid id). PR 3b will tighten this to a decimal
    // con_id parser and reject non-numeric input here.
    let class_arg = &args[0];
    let id: WindowId = class_arg.parse().expect("WindowId parser is infallible");
    match wm.focus(&id).await {
        Ok(()) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) => {
            let (label, hint) = hint_for(&e);
            Ok(CommandOutput::failed(
                1,
                error_line(
                    NAME,
                    format_args!("focus '{class_arg}' failed: {e}"),
                    label,
                    hint,
                )
                .into_bytes(),
            ))
        }
    }
}

const MOVE_HELP: &str = "usage: wm move <class> <workspace>\n\
    \n\
    Move the window with the given class to the named workspace. \
    Numeric workspace identifiers (e.g. `3`) match by number; \
    non-numeric identifiers match by exact name.\n";

async fn handle_move(wm: &dyn WindowManager, args: &[String]) -> Result<CommandOutput> {
    if args.len() < 2 {
        return Ok(help_output(MOVE_HELP.to_string()));
    }
    let class_arg = &args[0];
    let workspace_arg = &args[1];
    let id: WindowId = class_arg.parse().expect("WindowId parser is infallible");
    let workspace: WorkspaceId = workspace_arg
        .parse()
        .expect("WorkspaceId parser is infallible");
    match wm.move_to_workspace(&id, &workspace).await {
        Ok(()) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) => {
            let (label, hint) = hint_for(&e);
            Ok(CommandOutput::failed(
                1,
                error_line(
                    NAME,
                    format_args!("move '{class_arg}' to '{workspace_arg}' failed: {e}"),
                    label,
                    hint,
                )
                .into_bytes(),
            ))
        }
    }
}

const OPEN_HELP: &str = "usage: wm open <app> [args...]\n\
    \n\
    Launch an application. <app> is resolved through PATH; remaining \
    arguments are forwarded to the spawned process. The child is \
    detached — stdin/stdout/stderr are nulled and the daemon does not \
    wait for it to exit.\n";

async fn handle_open(args: &[String]) -> Result<CommandOutput> {
    if args.is_empty() {
        return Ok(help_output(OPEN_HELP.to_string()));
    }
    let app = &args[0];
    let extra = &args[1..];
    let mut cmd = ProcCommand::new(app);
    cmd.args(extra)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    match cmd.spawn() {
        Ok(_child) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(CommandOutput::failed(
            1,
            error_line(
                NAME,
                format_args!("open: binary '{app}' not found on PATH"),
                "Check",
                format_args!("which {app}"),
            )
            .into_bytes(),
        )),
        Err(e) => Ok(CommandOutput::failed(
            1,
            error_line(
                NAME,
                format_args!("open '{app}' failed: {e}"),
                "Try",
                "a different binary or absolute path",
            )
            .into_bytes(),
        )),
    }
}

async fn handle_active(wm: &dyn WindowManager) -> Result<CommandOutput> {
    match wm.focused_window().await {
        Ok(Some(id)) => {
            let mut out = String::from(id).into_bytes();
            out.push(b'\n');
            Ok(CommandOutput::ok(out))
        }
        Ok(None) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) => {
            let (label, hint) = hint_for(&e);
            Ok(CommandOutput::failed(
                1,
                error_line(NAME, format_args!("active failed: {e}"), label, hint).into_bytes(),
            ))
        }
    }
}

const RESIZE_HELP: &str = "usage: wm resize <class> <grow|shrink> <px>\n\
    \n\
    Resize the named window's width by the given pixel amount. \
    Direction is one of `grow` or `shrink`; <px> is a non-negative \
    integer count of pixels.\n";

async fn handle_resize(wm: &dyn WindowManager, args: &[String]) -> Result<CommandOutput> {
    if args.len() < 3 {
        return Ok(help_output(RESIZE_HELP.to_string()));
    }
    let class = &args[0];
    let direction: ResizeDir = match args[1].parse() {
        Ok(d) => d,
        Err(_) => {
            return Ok(CommandOutput::failed(
                2,
                error_line(
                    NAME,
                    format_args!(
                        "resize: direction must be 'grow' or 'shrink', got '{}'",
                        args[1]
                    ),
                    "Use",
                    "wm resize <class> <grow|shrink> <px>",
                )
                .into_bytes(),
            ));
        }
    };
    let amount: u32 = match args[2].parse() {
        Ok(n) => n,
        Err(_) => {
            return Ok(CommandOutput::failed(
                2,
                error_line(
                    NAME,
                    format_args!(
                        "resize: pixel amount must be a non-negative integer, got '{}'",
                        args[2]
                    ),
                    "Use",
                    "wm resize <class> <grow|shrink> <px>",
                )
                .into_bytes(),
            ));
        }
    };
    let id: WindowId = class.parse().expect("WindowId parser is infallible");
    match wm.resize_width(&id, direction, amount).await {
        Ok(()) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) => {
            let (label, hint) = hint_for(&e);
            Ok(CommandOutput::failed(
                1,
                error_line(
                    NAME,
                    format_args!("resize '{class}' failed: {e}"),
                    label,
                    hint,
                )
                .into_bytes(),
            ))
        }
    }
}

async fn handle_list(wm: &dyn WindowManager) -> Result<CommandOutput> {
    match wm.list_windows().await {
        Ok(mut windows) => {
            windows.sort_by(|a, b| {
                a.workspace
                    .as_deref()
                    .unwrap_or("")
                    .cmp(b.workspace.as_deref().unwrap_or(""))
                    .then_with(|| a.id.cmp(&b.id))
            });
            let mut out = String::new();
            for w in windows {
                out.push_str(w.id.as_str());
                out.push('\t');
                out.push_str(w.workspace.as_deref().unwrap_or("-"));
                out.push('\t');
                out.push_str(w.title.as_deref().unwrap_or(""));
                out.push('\n');
            }
            Ok(CommandOutput::ok(out.into_bytes()))
        }
        Err(e) => {
            let (label, hint) = hint_for(&e);
            Ok(CommandOutput::failed(
                1,
                error_line(NAME, format_args!("list failed: {e}"), label, hint).into_bytes(),
            ))
        }
    }
}

async fn handle_outputs(wm: &dyn WindowManager) -> Result<CommandOutput> {
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
                        // Sway reports refresh in mHz; emit a friendly Hz
                        // form rounded to the nearest integer when the
                        // mantissa is exactly zero, else 3-decimal Hz.
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
        Err(e) => {
            let (label, hint) = hint_for(&e);
            Ok(CommandOutput::failed(
                1,
                error_line(NAME, format_args!("outputs failed: {e}"), label, hint).into_bytes(),
            ))
        }
    }
}

async fn handle_workspaces(wm: &dyn WindowManager) -> Result<CommandOutput> {
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
        Err(e) => {
            let (label, hint) = hint_for(&e);
            Ok(CommandOutput::failed(
                1,
                error_line(NAME, format_args!("workspaces failed: {e}"), label, hint).into_bytes(),
            ))
        }
    }
}

const LAYOUT_HELP: &str = "usage: wm layout <default|tabbed|stacking|splith|splitv>\n\
    \n\
    Set the layout of the currently focused container. `default` \
    toggles between split, tabbed, and stacking based on the \
    container's previous layout.\n";

async fn handle_layout(wm: &dyn WindowManager, args: &[String]) -> Result<CommandOutput> {
    if args.is_empty() {
        return Ok(help_output(LAYOUT_HELP.to_string()));
    }
    let raw = args[0].as_str();
    let layout: Layout = match raw.parse() {
        Ok(l) => l,
        Err(_) => {
            return Ok(CommandOutput::failed(
                2,
                error_line(
                    NAME,
                    format_args!("layout: '{raw}' is not a known layout"),
                    "Use",
                    "default | tabbed | stacking | splith | splitv",
                )
                .into_bytes(),
            ));
        }
    };
    match wm.set_layout(layout).await {
        Ok(()) => Ok(CommandOutput::ok(Vec::new())),
        Err(e) => {
            let (label, hint) = hint_for(&e);
            Ok(CommandOutput::failed(
                1,
                error_line(
                    NAME,
                    format_args!("layout '{layout}' failed: {e}"),
                    label,
                    hint,
                )
                .into_bytes(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assistd_wm::{
        Layout, NoWindowManager, OutputInfo, ResizeDir, Window, WindowId, WmResult, WorkspaceId,
        WorkspaceInfo,
    };
    use std::sync::Mutex;

    /// Test fixture for [`WindowManager`]. Records every call so tests
    /// can assert on the typed argument tuples that would be dispatched
    /// to the backend, and lets each operation be wired to fail with a
    /// canned error message.
    #[derive(Default)]
    struct StubWm {
        connected: bool,
        windows: Vec<Window>,
        workspaces: Vec<WorkspaceInfo>,
        outputs: Vec<OutputInfo>,
        focused: Option<WindowId>,
        focus_calls: Mutex<Vec<WindowId>>,
        move_calls: Mutex<Vec<(WindowId, WorkspaceId)>>,
        resize_calls: Mutex<Vec<(WindowId, ResizeDir, u32)>>,
        layout_calls: Mutex<Vec<Layout>>,
        focus_err: Option<String>,
        move_err: Option<String>,
        resize_err: Option<String>,
        layout_err: Option<String>,
        list_windows_err: Option<String>,
        list_workspaces_err: Option<String>,
        list_outputs_err: Option<String>,
        focused_err: Option<String>,
        list_outputs_unsupported: bool,
    }

    impl StubWm {
        fn connected() -> Self {
            Self {
                connected: true,
                ..Self::default()
            }
        }
    }

    /// Wrap a `&Option<String>` as a `WmError::Ipc(anyhow!(msg))`. Tests
    /// that want to inject a backend failure write `focus_err: Some("…")`;
    /// without typed variants they used `anyhow::bail!`. The Ipc variant
    /// preserves the message body (which the tests assert on) and routes
    /// through the `Check: compositor connection` recovery hint.
    fn ipc_err(msg: &str) -> WmError {
        WmError::Ipc(anyhow::anyhow!("{msg}"))
    }

    #[async_trait]
    impl WindowManager for StubWm {
        async fn focus(&self, window: &WindowId) -> WmResult<()> {
            self.focus_calls.lock().unwrap().push(window.clone());
            if let Some(msg) = &self.focus_err {
                return Err(ipc_err(msg));
            }
            Ok(())
        }
        async fn move_to_workspace(
            &self,
            window: &WindowId,
            workspace: &WorkspaceId,
        ) -> WmResult<()> {
            self.move_calls
                .lock()
                .unwrap()
                .push((window.clone(), workspace.clone()));
            if let Some(msg) = &self.move_err {
                return Err(ipc_err(msg));
            }
            Ok(())
        }
        async fn focused_window(&self) -> WmResult<Option<WindowId>> {
            if let Some(msg) = &self.focused_err {
                return Err(ipc_err(msg));
            }
            Ok(self.focused.clone())
        }
        async fn list_windows(&self) -> WmResult<Vec<Window>> {
            if let Some(msg) = &self.list_windows_err {
                return Err(ipc_err(msg));
            }
            Ok(self.windows.clone())
        }
        async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
            if let Some(msg) = &self.list_workspaces_err {
                return Err(ipc_err(msg));
            }
            Ok(self.workspaces.clone())
        }
        async fn resize_width(
            &self,
            window: &WindowId,
            direction: ResizeDir,
            pixels: u32,
        ) -> WmResult<()> {
            self.resize_calls
                .lock()
                .unwrap()
                .push((window.clone(), direction, pixels));
            if let Some(msg) = &self.resize_err {
                return Err(ipc_err(msg));
            }
            Ok(())
        }
        async fn set_layout(&self, layout: Layout) -> WmResult<()> {
            self.layout_calls.lock().unwrap().push(layout);
            if let Some(msg) = &self.layout_err {
                return Err(ipc_err(msg));
            }
            Ok(())
        }
        async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
            if self.list_outputs_unsupported {
                // Mirror the trait default — backends that don't
                // implement outputs return Unsupported so the wm tool
                // can tell the LLM the difference between a connected
                // machine with zero monitors and an i3-class backend.
                return Err(WmError::Unsupported("output enumeration"));
            }
            if let Some(msg) = &self.list_outputs_err {
                return Err(ipc_err(msg));
            }
            Ok(self.outputs.clone())
        }
        fn is_connected(&self) -> bool {
            self.connected
        }
    }

    async fn run_wm(wm: Arc<dyn WindowManager>, args: &[&str]) -> CommandOutput {
        WmCommand::new(wm)
            .run(CommandInput {
                args: args.iter().map(|s| s.to_string()).collect(),
                stdin: Vec::new(),
            })
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn no_args_returns_help() {
        let out = run_wm(Arc::new(StubWm::connected()), &[]).await;
        assert_eq!(out.exit_code, 2);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.starts_with("usage: wm <subcommand>"), "{stdout}");
        assert!(stdout.contains("focus"), "{stdout}");
        assert!(stdout.contains("workspaces"), "{stdout}");
    }

    #[tokio::test]
    async fn unknown_subcommand_errors_with_available_list() {
        let out = run_wm(Arc::new(StubWm::connected()), &["bogus"]).await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] wm: unknown subcommand 'bogus'"),
            "{stderr}"
        );
        assert!(stderr.contains("Available:"), "{stderr}");
    }

    #[tokio::test]
    async fn disconnected_short_circuits() {
        // StubWm::default() has connected = false.
        let out = run_wm(Arc::new(StubWm::default()), &["focus", "Firefox"]).await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] wm: compositor not connected"),
            "{stderr}"
        );
        assert!(stderr.contains("Check:"), "{stderr}");
    }

    #[tokio::test]
    async fn no_window_manager_short_circuits_on_focus() {
        // Ensures the production `NoWindowManager` produces the same
        // behavior as the StubWm disconnected path — this is the gate
        // for the "mock mode" acceptance criterion.
        let out = run_wm(Arc::new(NoWindowManager), &["focus", "Firefox"]).await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] wm: compositor not connected"),
            "{stderr}"
        );
    }

    #[tokio::test]
    async fn focus_no_args_returns_subcommand_help() {
        let out = run_wm(Arc::new(StubWm::connected()), &["focus"]).await;
        assert_eq!(out.exit_code, 2);
        assert!(String::from_utf8_lossy(&out.stdout).contains("usage: wm focus"));
    }

    #[tokio::test]
    async fn focus_calls_backend_with_class() {
        let stub = Arc::new(StubWm::connected());
        let out = run_wm(stub.clone(), &["focus", "Firefox"]).await;
        assert_eq!(out.exit_code, 0);
        let calls = stub.focus_calls.lock().unwrap();
        assert_eq!(*calls, vec![WindowId::from("Firefox")]);
    }

    #[tokio::test]
    async fn focus_translates_backend_error() {
        let stub = Arc::new(StubWm {
            connected: true,
            focus_err: Some("i3 socket dropped".into()),
            ..Default::default()
        });
        let out = run_wm(stub, &["focus", "Firefox"]).await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] wm: focus 'Firefox' failed"),
            "{stderr}"
        );
        // Backend Ipc errors route through the "Check: compositor
        // connection" hint — different from the static "Use: wm list"
        // hint that the pre-WmError handler emitted unconditionally.
        assert!(stderr.contains("Check:"), "{stderr}");
    }

    #[test]
    fn hint_for_disconnected() {
        let (label, hint) = hint_for(&WmError::Disconnected);
        assert_eq!(label, "Check");
        assert!(hint.contains("config.toml"), "{hint}");
    }

    #[test]
    fn hint_for_not_found() {
        let (label, hint) = hint_for(&WmError::NotFound("Firefox".into()));
        assert_eq!(label, "Use");
        assert!(hint.contains("wm list"), "{hint}");
    }

    #[test]
    fn hint_for_rejected() {
        let (label, _) = hint_for(&WmError::Rejected("focus: bad criteria".into()));
        assert_eq!(label, "Try");
    }

    #[test]
    fn hint_for_timeout() {
        let (label, hint) = hint_for(&WmError::Timeout(std::time::Duration::from_secs(5)));
        assert_eq!(label, "Note");
        assert!(hint.contains("retry"), "{hint}");
    }

    #[test]
    fn hint_for_unsupported() {
        let (label, hint) = hint_for(&WmError::Unsupported("output enumeration"));
        assert_eq!(label, "Note");
        assert!(hint.contains("i3 does not"), "{hint}");
    }

    #[test]
    fn hint_for_ipc() {
        let (label, _) = hint_for(&WmError::Ipc(anyhow::anyhow!("socket dropped")));
        assert_eq!(label, "Check");
    }

    #[tokio::test]
    async fn move_needs_two_args() {
        let out = run_wm(Arc::new(StubWm::connected()), &["move", "Firefox"]).await;
        assert_eq!(out.exit_code, 2);
        assert!(String::from_utf8_lossy(&out.stdout).contains("usage: wm move"));
    }

    #[tokio::test]
    async fn move_calls_backend() {
        let stub = Arc::new(StubWm::connected());
        let out = run_wm(stub.clone(), &["move", "Firefox", "3"]).await;
        assert_eq!(out.exit_code, 0);
        let calls = stub.move_calls.lock().unwrap();
        // "3" parses as numeric → WorkspaceId::Num(3); the args are
        // typed all the way through to the backend now.
        assert_eq!(
            *calls,
            vec![(WindowId::from("Firefox"), WorkspaceId::Num(3))]
        );
    }

    #[tokio::test]
    async fn open_no_args_returns_help() {
        let out = run_wm(Arc::new(StubWm::connected()), &["open"]).await;
        assert_eq!(out.exit_code, 2);
        assert!(String::from_utf8_lossy(&out.stdout).contains("usage: wm open"));
    }

    #[tokio::test]
    async fn open_missing_binary_returns_path_error() {
        let out = run_wm(
            Arc::new(StubWm::connected()),
            &["open", "definitely-not-a-real-binary-xyzzy-12345"],
        )
        .await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] wm: open: binary 'definitely-not-a-real-binary-xyzzy-12345' not found on PATH"),
            "{stderr}"
        );
        assert!(stderr.contains("Check:"), "{stderr}");
    }

    #[tokio::test]
    async fn active_prints_focused_class() {
        let stub = Arc::new(StubWm {
            connected: true,
            focused: Some("Firefox".into()),
            ..Default::default()
        });
        let out = run_wm(stub, &["active"]).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"Firefox\n");
    }

    #[tokio::test]
    async fn active_with_no_focus_is_empty_stdout() {
        let stub = Arc::new(StubWm::connected());
        let out = run_wm(stub, &["active"]).await;
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout.is_empty());
    }

    #[tokio::test]
    async fn resize_too_few_args_returns_help() {
        let out = run_wm(
            Arc::new(StubWm::connected()),
            &["resize", "Firefox", "grow"],
        )
        .await;
        assert_eq!(out.exit_code, 2);
        assert!(String::from_utf8_lossy(&out.stdout).contains("usage: wm resize"));
    }

    #[tokio::test]
    async fn resize_bad_direction_errors() {
        let out = run_wm(
            Arc::new(StubWm::connected()),
            &["resize", "Firefox", "sideways", "10"],
        )
        .await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] wm: resize"), "{stderr}");
        assert!(stderr.contains("Use:"), "{stderr}");
    }

    #[tokio::test]
    async fn resize_bad_pixel_count_errors() {
        let out = run_wm(
            Arc::new(StubWm::connected()),
            &["resize", "Firefox", "grow", "lots"],
        )
        .await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] wm: resize"), "{stderr}");
    }

    #[tokio::test]
    async fn resize_dispatches_typed_args() {
        // wm.rs passes the parsed direction + pixel count to the
        // backend's typed `resize_width` method. The literal i3-syntax
        // payload (`[class="…"] resize …`) and its escape semantics are
        // tested in `assistd_wm::i3::tests` directly — wm.rs no longer
        // formats the payload itself.
        let stub = Arc::new(StubWm::connected());
        let out = run_wm(stub.clone(), &["resize", "Firefox", "grow", "50"]).await;
        assert_eq!(out.exit_code, 0);
        let calls = stub.resize_calls.lock().unwrap();
        assert_eq!(
            *calls,
            vec![(WindowId::from("Firefox"), ResizeDir::Grow, 50)]
        );
    }

    #[tokio::test]
    async fn resize_passes_special_chars_unaltered() {
        // The class arg flows through to the backend without
        // transformation; backends own the escape step. This guards
        // against a regression where wm.rs starts pre-escaping
        // (which would double-escape once the backend escapes again).
        let stub = Arc::new(StubWm::connected());
        let _ = run_wm(stub.clone(), &["resize", r#"a"b\c"#, "shrink", "5"]).await;
        let calls = stub.resize_calls.lock().unwrap();
        assert_eq!(
            *calls,
            vec![(WindowId::from(r#"a"b\c"#), ResizeDir::Shrink, 5)]
        );
    }

    #[tokio::test]
    async fn list_emits_tsv_sorted_by_workspace_then_class() {
        let stub = Arc::new(StubWm {
            connected: true,
            windows: vec![
                Window {
                    id: "Firefox".into(),
                    title: Some("GitHub".into()),
                    workspace: Some("3".into()),
                },
                Window {
                    id: "code".into(),
                    title: Some("wm.rs".into()),
                    workspace: Some("1".into()),
                },
                Window {
                    id: "Alacritty".into(),
                    title: None,
                    workspace: Some("1".into()),
                },
            ],
            ..Default::default()
        });
        let out = run_wm(stub, &["list"]).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "Alacritty\t1\t\ncode\t1\twm.rs\nFirefox\t3\tGitHub\n"
        );
    }

    #[tokio::test]
    async fn list_orphans_use_dash_for_workspace() {
        let stub = Arc::new(StubWm {
            connected: true,
            windows: vec![Window {
                id: "Scratch".into(),
                title: Some("notes".into()),
                workspace: None,
            }],
            ..Default::default()
        });
        let out = run_wm(stub, &["list"]).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"Scratch\t-\tnotes\n");
    }

    #[tokio::test]
    async fn workspaces_emits_tsv_with_focus_marker() {
        let stub = Arc::new(StubWm {
            connected: true,
            workspaces: vec![
                WorkspaceInfo {
                    num: 3,
                    name: "3".into(),
                    focused: false,
                    output: "DP-1".into(),
                },
                WorkspaceInfo {
                    num: 1,
                    name: "1:web".into(),
                    focused: true,
                    output: "DP-1".into(),
                },
            ],
            ..Default::default()
        });
        let out = run_wm(stub, &["workspaces"]).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "1\t1:web\t*\tDP-1\n3\t3\t-\tDP-1\n"
        );
    }

    #[tokio::test]
    async fn layout_no_args_returns_help() {
        let out = run_wm(Arc::new(StubWm::connected()), &["layout"]).await;
        assert_eq!(out.exit_code, 2);
        assert!(String::from_utf8_lossy(&out.stdout).contains("usage: wm layout"));
    }

    #[tokio::test]
    async fn layout_unknown_name_errors() {
        let out = run_wm(Arc::new(StubWm::connected()), &["layout", "spinning"]).await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] wm: layout"), "{stderr}");
        assert!(stderr.contains("Use:"), "{stderr}");
    }

    #[tokio::test]
    async fn layout_dispatches_typed_arg() {
        let stub = Arc::new(StubWm::connected());
        let out = run_wm(stub.clone(), &["layout", "tabbed"]).await;
        assert_eq!(out.exit_code, 0);
        let calls = stub.layout_calls.lock().unwrap();
        assert_eq!(*calls, vec![Layout::Tabbed]);
    }

    #[test]
    fn summary_fits_eighty_chars() {
        assert!(SUMMARY.len() <= 80, "{} chars: {SUMMARY}", SUMMARY.len());
    }

    #[tokio::test]
    async fn outputs_emits_tsv_sorted_by_name() {
        let stub = Arc::new(StubWm {
            connected: true,
            outputs: vec![
                OutputInfo {
                    name: "DP-2".into(),
                    active: true,
                    primary: false,
                    current_mode: Some((2560, 1440, 144_000)),
                    scale: Some(1.0),
                    focused_workspace: Some("3".into()),
                },
                OutputInfo {
                    name: "DP-1".into(),
                    active: true,
                    primary: true,
                    current_mode: Some((1920, 1080, 60_000)),
                    scale: Some(1.5),
                    focused_workspace: Some("1:web".into()),
                },
            ],
            ..Default::default()
        });
        let out = run_wm(stub, &["outputs"]).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "DP-1\t*\t*\t1920x1080@60Hz\t1.5\t1:web\nDP-2\t*\t-\t2560x1440@144Hz\t1\t3\n"
        );
    }

    #[tokio::test]
    async fn outputs_handles_missing_fields_with_dash() {
        let stub = Arc::new(StubWm {
            connected: true,
            outputs: vec![OutputInfo {
                name: "HDMI-A-1".into(),
                active: false,
                primary: false,
                current_mode: None,
                scale: None,
                focused_workspace: None,
            }],
            ..Default::default()
        });
        let out = run_wm(stub, &["outputs"]).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"HDMI-A-1\t-\t-\t-\t-\t-\n");
    }

    #[tokio::test]
    async fn outputs_unsupported_backend_emits_error_with_note() {
        // Mirrors the i3-class case: list_outputs returns
        // "does not support" so wm outputs surfaces a helpful error
        // rather than empty stdout.
        let stub = Arc::new(StubWm {
            connected: true,
            list_outputs_unsupported: true,
            ..Default::default()
        });
        let out = run_wm(stub, &["outputs"]).await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] wm: outputs failed"), "{stderr}");
        assert!(stderr.contains("Note:"), "{stderr}");
        assert!(stderr.contains("i3 does not"), "{stderr}");
    }

    #[tokio::test]
    async fn outputs_propagates_runtime_error() {
        let stub = Arc::new(StubWm {
            connected: true,
            list_outputs_err: Some("sway socket dropped".into()),
            ..Default::default()
        });
        let out = run_wm(stub, &["outputs"]).await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] wm: outputs failed"), "{stderr}");
    }

    #[tokio::test]
    async fn unknown_subcommand_lists_outputs_in_available() {
        // Regression check that the help hint includes the new subcommand.
        let out = run_wm(Arc::new(StubWm::connected()), &["bogus"]).await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("outputs"), "{stderr}");
    }

    #[tokio::test]
    async fn help_block_advertises_outputs_subcommand() {
        let out = run_wm(Arc::new(StubWm::connected()), &[]).await;
        assert_eq!(out.exit_code, 2);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("outputs"), "{stdout}");
    }
}

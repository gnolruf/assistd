use super::*;

use crate::command::Hint;

use crate::exec::POLICY_DENIED_EXIT;
use crate::policy::ConfirmationRequest;
use crate::policy::{AlwaysAllowGate, DenyAllGate};
use assistd_wm::{
    FocusedWindowContext, Layout, NoWindowManager, OutputInfo, ResizeDir, Window, WindowId,
    WmResult, WorkspaceId, WorkspaceInfo,
};
use parking_lot::Mutex;

/// Test-only id constructor; every fixture id is non-zero by
/// construction, so this `expect` is unreachable at runtime.
fn id(n: u64) -> WindowId {
    WindowId::new(n).expect("test ids are non-zero")
}

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
    /// Human-readable label of the focused window, surfaced via
    /// `focused_context().class`. Independent of `focused` so tests
    /// can exercise "id present but app unknown" code paths.
    focused_app: Option<String>,
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
        self.focus_calls.lock().push(*window);
        if let Some(msg) = &self.focus_err {
            return Err(ipc_err(msg));
        }
        Ok(())
    }
    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        self.move_calls.lock().push((*window, workspace.clone()));
        if let Some(msg) = &self.move_err {
            return Err(ipc_err(msg));
        }
        Ok(())
    }
    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        if let Some(msg) = &self.focused_err {
            return Err(ipc_err(msg));
        }
        Ok(self.focused)
    }
    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        if let Some(msg) = &self.focused_err {
            return Err(ipc_err(msg));
        }
        if self.focused.is_none() && self.focused_app.is_none() {
            return Ok(None);
        }
        Ok(Some(FocusedWindowContext {
            id: self.focused,
            class: self.focused_app.clone(),
            title: None,
            workspace: None,
        }))
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
        self.resize_calls.lock().push((*window, direction, pixels));
        if let Some(msg) = &self.resize_err {
            return Err(ipc_err(msg));
        }
        Ok(())
    }
    async fn set_layout(&self, layout: Layout) -> WmResult<()> {
        self.layout_calls.lock().push(layout);
        if let Some(msg) = &self.layout_err {
            return Err(ipc_err(msg));
        }
        Ok(())
    }
    async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
        if self.list_outputs_unsupported {
            // Mirror the trait default: backends that don't
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
    WmCommand::for_test(wm)
        .run(CommandInput {
            args: args.iter().map(|s| s.to_string()).collect(),
            stdin: None,
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
    // The production `NoWindowManager` must behave exactly as the
    // StubWm disconnected path does.
    let out = run_wm(Arc::new(NoWindowManager), &["focus", "42"]).await;
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
async fn focus_calls_backend_with_id() {
    let stub = Arc::new(StubWm::connected());
    let out = run_wm(stub.clone(), &["focus", "42"]).await;
    assert_eq!(out.exit_code, 0);
    let calls = stub.focus_calls.lock();
    assert_eq!(*calls, vec![id(42)]);
}

#[tokio::test]
async fn focus_rejects_non_numeric_arg() {
    let stub = Arc::new(StubWm::connected());
    let out = run_wm(stub.clone(), &["focus", "Firefox"]).await;
    assert_eq!(out.exit_code, 2);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not a valid window id"), "{stderr}");
    assert!(stderr.contains("wm list"), "{stderr}");
    assert!(stub.focus_calls.lock().is_empty());
}

#[tokio::test]
async fn focus_translates_backend_error() {
    let stub = Arc::new(StubWm {
        connected: true,
        focus_err: Some("i3 socket dropped".into()),
        ..Default::default()
    });
    let out = run_wm(stub, &["focus", "42"]).await;
    assert_eq!(out.exit_code, 1);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("[error] wm: focus 42 failed"), "{stderr}");
    assert!(stderr.contains("Check:"), "{stderr}");
}

#[test]
fn hint_for_disconnected() {
    let (label, hint) = hint_for(&WmError::Disconnected);
    assert_eq!(label, Hint::Check);
    assert!(hint.contains("config.toml"), "{hint}");
}

#[test]
fn hint_for_not_found() {
    let (label, hint) = hint_for(&WmError::NotFound(id(42)));
    assert_eq!(label, Hint::Use);
    assert!(hint.contains("wm list"), "{hint}");
}

#[test]
fn hint_for_rejected() {
    let (label, _) = hint_for(&WmError::Rejected("focus: bad criteria".into()));
    assert_eq!(label, Hint::Try);
}

#[test]
fn hint_for_timeout() {
    let (label, hint) = hint_for(&WmError::Timeout(std::time::Duration::from_secs(5)));
    assert_eq!(label, Hint::Note);
    assert!(hint.contains("retry"), "{hint}");
}

#[test]
fn hint_for_unsupported() {
    let (label, hint) = hint_for(&WmError::Unsupported("output enumeration"));
    assert_eq!(label, Hint::Note);
    assert!(hint.contains("i3 does not"), "{hint}");
}

#[test]
fn hint_for_ipc() {
    let (label, _) = hint_for(&WmError::Ipc(anyhow::anyhow!("socket dropped")));
    assert_eq!(label, Hint::Check);
}

#[tokio::test]
async fn move_needs_two_args() {
    let out = run_wm(Arc::new(StubWm::connected()), &["move", "42"]).await;
    assert_eq!(out.exit_code, 2);
    assert!(String::from_utf8_lossy(&out.stdout).contains("usage: wm move"));
}

#[tokio::test]
async fn move_calls_backend() {
    let stub = Arc::new(StubWm::connected());
    let out = run_wm(stub.clone(), &["move", "42", "3"]).await;
    assert_eq!(out.exit_code, 0);
    let calls = stub.move_calls.lock();
    // "3" parses as numeric → WorkspaceId::Num(3); the args are
    // typed all the way through to the backend now.
    assert_eq!(*calls, vec![(id(42), WorkspaceId::Num(3))]);
}

#[tokio::test]
async fn move_rejects_non_numeric_id() {
    let stub = Arc::new(StubWm::connected());
    let out = run_wm(stub.clone(), &["move", "Firefox", "3"]).await;
    assert_eq!(out.exit_code, 2);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not a valid window id"), "{stderr}");
    assert!(stub.move_calls.lock().is_empty());
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
    assert_eq!(out.exit_code, SPAWN_FAILED_EXIT);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(
            "[error] wm: open: binary 'definitely-not-a-real-binary-xyzzy-12345' not found on PATH"
        ),
        "{stderr}"
    );
    assert!(stderr.contains("Check:"), "{stderr}");
}

fn policed_wm(cfg: BashPolicyCfg, gate: Arc<dyn ConfirmationGate>) -> WmCommand {
    WmCommand::new(
        Arc::new(StubWm::connected()),
        Arc::new(cfg),
        SandboxInfo::none(),
        gate,
    )
}

async fn run_open(cmd: &WmCommand, args: &[&str]) -> CommandOutput {
    let mut argv = vec!["open".to_string()];
    argv.extend(args.iter().map(|s| s.to_string()));
    cmd.run(CommandInput {
        args: argv,
        stdin: None,
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn open_captures_child_output() {
    let out = run_wm(Arc::new(StubWm::connected()), &["open", "echo", "hi"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"hi\n");
}

#[tokio::test]
async fn open_surfaces_nonzero_child_exit() {
    let cmd = policed_wm(BashPolicyCfg::default(), Arc::new(AlwaysAllowGate));
    let out = run_open(&cmd, &["false"]).await;
    assert_eq!(out.exit_code, 1);
}

#[tokio::test]
async fn open_denylist_blocks_before_spawn() {
    let cmd = policed_wm(
        BashPolicyCfg {
            denylist: vec!["rm -rf /".into()],
            ..Default::default()
        },
        Arc::new(AlwaysAllowGate),
    );
    let out = run_open(&cmd, &["bash", "-c", "rm -rf /"]).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("denylist pattern: rm -rf /"), "{stderr}");
}

#[tokio::test]
async fn open_destructive_argv_consults_gate() {
    let cmd = policed_wm(
        BashPolicyCfg {
            destructive_patterns: vec![vec!["rm".into(), "-rf".into()]],
            ..Default::default()
        },
        Arc::new(DenyAllGate),
    );
    let out = run_open(&cmd, &["rm", "-rf", "/tmp/whatever"]).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Matched destructive pattern: rm -rf"),
        "{stderr}"
    );
}

/// A script passed as one argument still has to reach the gate; it
/// only matches once each argument is checked on its own.
#[tokio::test]
async fn open_destructive_inside_bash_c_argument_consults_gate() {
    let cmd = policed_wm(
        BashPolicyCfg {
            destructive_patterns: vec![vec!["rm".into(), "-rf".into()]],
            ..Default::default()
        },
        Arc::new(DenyAllGate),
    );
    let out = run_open(&cmd, &["bash", "-c", "rm -rf /tmp/whatever"]).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
}

#[tokio::test]
async fn open_gate_approval_lets_the_process_run() {
    let cmd = policed_wm(
        BashPolicyCfg {
            destructive_patterns: vec![vec!["true".into()]],
            ..Default::default()
        },
        Arc::new(AlwaysAllowGate),
    );
    let out = run_open(&cmd, &["true"]).await;
    assert_eq!(out.exit_code, 0);
}

/// An application outliving the startup probe keeps running, and the
/// launch reports success rather than blocking the turn.
#[tokio::test]
async fn open_leaves_a_surviving_process_running() {
    let marker = std::env::temp_dir().join(format!("assistd-wm-open-{}", std::process::id()));
    let _ = std::fs::remove_file(&marker);

    let cmd = policed_wm(BashPolicyCfg::default(), Arc::new(AlwaysAllowGate));
    let started = std::time::Instant::now();
    let out = run_open(
        &cmd,
        &[
            "bash",
            "-c",
            &format!("sleep 1; touch {}", marker.display()),
        ],
    )
    .await;

    assert_eq!(out.exit_code, 0, "surviving launch should report success");
    assert!(out.stdout.is_empty());
    assert!(
        started.elapsed() < std::time::Duration::from_millis(900),
        "launch should return on the probe, not on the child exiting"
    );
    assert!(!marker.exists(), "child should not have finished yet");

    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    assert!(
        marker.exists(),
        "detached child must keep running after the launch returned"
    );
    let _ = std::fs::remove_file(&marker);
}

#[tokio::test]
async fn open_reports_a_failed_startup_with_its_output() {
    let cmd = policed_wm(BashPolicyCfg::default(), Arc::new(AlwaysAllowGate));
    let out = run_open(&cmd, &["bash", "-c", "echo boom >&2; exit 3"]).await;
    assert_eq!(out.exit_code, 3);
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("boom"),
        "startup failure must surface the child's stderr"
    );
}

/// Policy is scoped to `open`; the compositor subcommands never
/// consult the gate or the denylist.
#[tokio::test]
async fn non_open_subcommands_skip_the_gate() {
    struct PanicGate;
    #[async_trait]
    impl ConfirmationGate for PanicGate {
        async fn confirm(&self, _r: ConfirmationRequest) -> bool {
            panic!("wm policy must only gate `open`");
        }
    }
    let cmd = policed_wm(
        BashPolicyCfg {
            denylist: vec!["focus".into()],
            destructive_patterns: vec![vec!["focus".into()]],
            ..Default::default()
        },
        Arc::new(PanicGate),
    );
    let out = cmd
        .run(CommandInput {
            args: vec!["focus".into(), "42".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
}

#[tokio::test]
async fn active_prints_id_tab_app() {
    // wm active emits `<id>\t<app>\n` so the LLM can pipe either
    // column: `wm focus $(wm active | cut -f1)` or read the app
    // label from column 2 to confirm what's focused.
    let stub = Arc::new(StubWm {
        connected: true,
        focused: Some(id(42)),
        focused_app: Some("Firefox".into()),
        ..Default::default()
    });
    let out = run_wm(stub, &["active"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"42\tFirefox\n");
}

#[tokio::test]
async fn active_renders_dash_when_app_missing() {
    let stub = Arc::new(StubWm {
        connected: true,
        focused: Some(id(7)),
        focused_app: None,
        ..Default::default()
    });
    let out = run_wm(stub, &["active"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"7\t-\n");
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
    let out = run_wm(Arc::new(StubWm::connected()), &["resize", "42", "grow"]).await;
    assert_eq!(out.exit_code, 2);
    assert!(String::from_utf8_lossy(&out.stdout).contains("usage: wm resize"));
}

#[tokio::test]
async fn resize_bad_direction_errors() {
    let out = run_wm(
        Arc::new(StubWm::connected()),
        &["resize", "42", "sideways", "10"],
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
        &["resize", "42", "grow", "lots"],
    )
    .await;
    assert_eq!(out.exit_code, 2);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("[error] wm: resize"), "{stderr}");
}

#[tokio::test]
async fn resize_dispatches_typed_args() {
    // wm.rs passes the parsed direction + pixel count to the
    // backend's typed `resize_width` method. The literal con_id
    // payload (`[con_id="…"] resize …`) is tested in
    // `assistd_wm::i3::tests` / `sway::tests` directly.
    let stub = Arc::new(StubWm::connected());
    let out = run_wm(stub.clone(), &["resize", "42", "grow", "50"]).await;
    assert_eq!(out.exit_code, 0);
    let calls = stub.resize_calls.lock();
    assert_eq!(*calls, vec![(id(42), ResizeDir::Grow, 50)]);
}

#[tokio::test]
async fn resize_rejects_non_numeric_id() {
    let stub = Arc::new(StubWm::connected());
    let out = run_wm(stub.clone(), &["resize", "Firefox", "grow", "5"]).await;
    assert_eq!(out.exit_code, 2);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not a valid window id"), "{stderr}");
    assert!(stub.resize_calls.lock().is_empty());
}

#[tokio::test]
async fn list_emits_tsv_sorted_by_workspace_then_app() {
    let stub = Arc::new(StubWm {
        connected: true,
        windows: vec![
            Window {
                id: id(1001),
                app: Some("Firefox".into()),
                title: Some("GitHub".into()),
                workspace: Some("3".into()),
            },
            Window {
                id: id(1002),
                app: Some("code".into()),
                title: Some("wm.rs".into()),
                workspace: Some("1".into()),
            },
            Window {
                id: id(1003),
                app: Some("Alacritty".into()),
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
        "1003\tAlacritty\t1\t\n1002\tcode\t1\twm.rs\n1001\tFirefox\t3\tGitHub\n"
    );
}

#[tokio::test]
async fn list_orphans_use_dash_for_missing_columns() {
    let stub = Arc::new(StubWm {
        connected: true,
        windows: vec![Window {
            id: id(7),
            app: None,
            title: Some("notes".into()),
            workspace: None,
        }],
        ..Default::default()
    });
    let out = run_wm(stub, &["list"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"7\t-\t-\tnotes\n");
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
    let calls = stub.layout_calls.lock();
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

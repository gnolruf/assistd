use super::*;

use crate::commands::RecordingGate;
use crate::exec::POLICY_DENIED_EXIT;
use crate::policy::{AlwaysAllowGate, DenyAllGate};
use assistd_wm::{
    FocusedWindowContext, Layout, NoWindowManager, OutputInfo, ResizeDir, Window, WindowId,
    WmResult, WorkspaceId, WorkspaceInfo,
};
use parking_lot::Mutex;

fn id(n: u64) -> WindowId {
    WindowId::new(n).expect("test ids are non-zero")
}

/// A transport failure whose message is `msg`.
fn ipc_err(msg: &str) -> WmError {
    WmError::Ipc {
        op: "stub",
        source: std::io::Error::other(msg.to_string()).into(),
    }
}

/// [`WindowManager`] fixture that records every mutating call with its
/// typed arguments. `error` makes every operation fail with that message
/// as a [`WmError::Ipc`].
#[derive(Default)]
struct StubWm {
    connected: bool,
    windows: Vec<Window>,
    workspaces: Vec<WorkspaceInfo>,
    outputs: Vec<OutputInfo>,
    focused: Option<WindowId>,
    /// Surfaced as `focused_context().class`, independently of `focused`.
    focused_app: Option<String>,
    focus_calls: Mutex<Vec<WindowId>>,
    move_calls: Mutex<Vec<(WindowId, WorkspaceId)>>,
    resize_calls: Mutex<Vec<(WindowId, ResizeDir, u32)>>,
    layout_calls: Mutex<Vec<Layout>>,
    error: Option<&'static str>,
    list_outputs_unsupported: bool,
}

impl StubWm {
    fn connected() -> Self {
        Self {
            connected: true,
            ..Self::default()
        }
    }

    fn fail(&self) -> WmResult<()> {
        self.error.map_or(Ok(()), |msg| Err(ipc_err(msg)))
    }

    fn no_backend_calls(&self) -> bool {
        self.focus_calls.lock().is_empty()
            && self.move_calls.lock().is_empty()
            && self.resize_calls.lock().is_empty()
            && self.layout_calls.lock().is_empty()
    }
}

#[async_trait]
impl WindowManager for StubWm {
    async fn focus(&self, window: &WindowId) -> WmResult<()> {
        self.focus_calls.lock().push(*window);
        self.fail()
    }
    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        self.move_calls.lock().push((*window, workspace.clone()));
        self.fail()
    }
    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        self.fail()?;
        Ok(self.focused)
    }
    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        self.fail()?;
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
        self.fail()?;
        Ok(self.windows.clone())
    }
    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        self.fail()?;
        Ok(self.workspaces.clone())
    }
    async fn resize_width(
        &self,
        window: &WindowId,
        direction: ResizeDir,
        pixels: u32,
    ) -> WmResult<()> {
        self.resize_calls.lock().push((*window, direction, pixels));
        self.fail()
    }
    async fn set_layout(&self, layout: Layout) -> WmResult<()> {
        self.layout_calls.lock().push(layout);
        self.fail()
    }
    async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
        if self.list_outputs_unsupported {
            return Err(WmError::Unsupported("output enumeration"));
        }
        self.fail()?;
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
}

#[tokio::test]
async fn no_args_returns_help() {
    let out = run_wm(Arc::new(StubWm::connected()), &[]).await;
    assert_eq!(out.exit_code, 2);
    assert!(out.stdout.starts_with(b"usage: wm <subcommand>"), "{out:?}");
}

#[tokio::test]
async fn unknown_subcommand_errors_with_available_list() {
    let out = run_wm(Arc::new(StubWm::connected()), &["bogus"]).await;
    assert_eq!(out.exit_code, 2);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] wm: unknown subcommand 'bogus'. \
         Available: focus, move, open, active, resize, list, workspaces, outputs, layout\n"
    );
}

/// The connection check comes before argument parsing, so even a
/// malformed id reports the missing compositor.
#[tokio::test]
async fn disconnected_backend_short_circuits() {
    let backends: [Arc<dyn WindowManager>; 2] =
        [Arc::new(StubWm::default()), Arc::new(NoWindowManager)];
    for wm in backends {
        let out = run_wm(wm, &["focus", "Firefox"]).await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] wm: compositor not connected. \
             Check: [compositor] in config.toml and that i3/sway/hyprland is running\n"
        );
    }
}

#[tokio::test]
async fn subcommand_with_missing_args_returns_its_help() {
    let cases: [(&[&str], &str); 5] = [
        (&["focus"], "usage: wm focus"),
        (&["move", "42"], "usage: wm move"),
        (&["open"], "usage: wm open"),
        (&["resize", "42", "grow"], "usage: wm resize"),
        (&["layout"], "usage: wm layout"),
    ];
    for (args, usage) in cases {
        let out = run_wm(Arc::new(StubWm::connected()), args).await;
        assert_eq!(out.exit_code, 2, "{args:?}");
        assert!(
            String::from_utf8_lossy(&out.stdout).starts_with(usage),
            "{args:?}: {out:?}"
        );
    }
}

#[tokio::test]
async fn malformed_arguments_are_rejected_before_the_backend() {
    let bad_id = |op: &str| {
        format!(
            "[error] wm: {op}: 'Firefox' is not a valid window id (positive decimal con_id). \
             Use: wm list to see ids (first TSV column)\n"
        )
    };
    let resize_use = "Use: wm resize <id> <grow|shrink> <px>\n";
    let cases: [(&[&str], String); 6] = [
        (&["focus", "Firefox"], bad_id("focus")),
        (&["move", "Firefox", "3"], bad_id("move")),
        (&["resize", "Firefox", "grow", "5"], bad_id("resize")),
        (
            &["resize", "42", "sideways", "10"],
            format!(
                "[error] wm: resize: direction must be 'grow' or 'shrink', got 'sideways'. {resize_use}"
            ),
        ),
        (
            &["resize", "42", "grow", "lots"],
            format!(
                "[error] wm: resize: pixel amount must be a non-negative integer, got 'lots'. {resize_use}"
            ),
        ),
        (
            &["layout", "spinning"],
            "[error] wm: layout: 'spinning' is not a known layout. \
             Use: default | tabbed | stacking | splith | splitv\n"
                .to_string(),
        ),
    ];
    for (args, stderr) in cases {
        let stub = Arc::new(StubWm::connected());
        let out = run_wm(stub.clone(), args).await;
        assert_eq!(out.exit_code, 2, "{args:?}");
        assert_eq!(String::from_utf8_lossy(&out.stderr), stderr, "{args:?}");
        assert!(stub.no_backend_calls(), "{args:?}");
    }
}

#[tokio::test]
async fn focus_calls_backend_with_id() {
    let stub = Arc::new(StubWm::connected());
    let out = run_wm(stub.clone(), &["focus", "42"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(*stub.focus_calls.lock(), [id(42)]);
}

#[tokio::test]
async fn focus_translates_backend_error() {
    let stub = Arc::new(StubWm {
        connected: true,
        error: Some("i3 socket dropped"),
        ..Default::default()
    });
    let out = run_wm(stub, &["focus", "42"]).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] wm: focus 42 failed: stub: i3 socket dropped. \
         Check: compositor connection (see daemon logs)\n"
    );
}

#[test]
fn hint_for_picks_label_per_error_variant() {
    let cases = [
        (WmError::Disconnected, Hint::Check, "config.toml"),
        (
            WmError::Rejected("focus: bad criteria".into()),
            Hint::Try,
            "wm list",
        ),
        (
            WmError::Timeout(std::time::Duration::from_secs(5)),
            Hint::Note,
            "retry",
        ),
        (
            WmError::Unsupported("output enumeration"),
            Hint::Note,
            "i3 does not",
        ),
        (
            ipc_err("socket dropped"),
            Hint::Check,
            "compositor connection",
        ),
    ];
    for (err, expected_label, fragment) in cases {
        let (label, hint) = hint_for(&err);
        assert_eq!(label, expected_label, "{err:?}");
        assert!(hint.contains(fragment), "{err:?}: {hint}");
    }
}

/// A numeric workspace argument reaches the backend as a number.
#[tokio::test]
async fn move_calls_backend() {
    let stub = Arc::new(StubWm::connected());
    let out = run_wm(stub.clone(), &["move", "42", "3"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(*stub.move_calls.lock(), [(id(42), WorkspaceId::Num(3))]);
}

#[tokio::test]
async fn resize_dispatches_typed_args() {
    let stub = Arc::new(StubWm::connected());
    let out = run_wm(stub.clone(), &["resize", "42", "grow", "50"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(*stub.resize_calls.lock(), [(id(42), ResizeDir::Grow, 50)]);
}

#[tokio::test]
async fn layout_dispatches_typed_arg() {
    let stub = Arc::new(StubWm::connected());
    let out = run_wm(stub.clone(), &["layout", "tabbed"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(*stub.layout_calls.lock(), [Layout::Tabbed]);
}

#[tokio::test]
async fn open_missing_binary_returns_path_error() {
    let out = run_wm(
        Arc::new(StubWm::connected()),
        &["open", "definitely-not-a-real-binary-xyzzy-12345"],
    )
    .await;
    assert_eq!(out.exit_code, SPAWN_FAILED_EXIT);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] wm: open: binary 'definitely-not-a-real-binary-xyzzy-12345' not found on PATH. \
         Check: which definitely-not-a-real-binary-xyzzy-12345\n"
    );
}

fn policed_wm(cfg: BashPolicyCfg, gate: Arc<dyn ConfirmationGate>) -> WmCommand {
    WmCommand::new(
        Arc::new(StubWm::connected()),
        Arc::new(cfg),
        SandboxInfo::none(),
        gate,
    )
}

fn rm_rf_is_destructive(gate: Arc<dyn ConfirmationGate>) -> WmCommand {
    policed_wm(
        BashPolicyCfg {
            destructive_patterns: vec![vec!["rm".into(), "-rf".into()]],
            ..Default::default()
        },
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
}

#[tokio::test]
async fn open_captures_child_output() {
    let out = run_wm(Arc::new(StubWm::connected()), &["open", "echo", "hi"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"hi\n");
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
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] wm: open denied by policy. Matched denylist pattern: rm -rf /. \
         Try: a non-destructive alternative\n"
    );
}

#[tokio::test]
async fn open_destructive_argv_consults_gate() {
    let out = run_open(
        &rm_rf_is_destructive(Arc::new(DenyAllGate)),
        &["rm", "-rf", "/tmp/whatever"],
    )
    .await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] wm: open cancelled by user. Matched destructive pattern: rm -rf. \
         Try: a different approach\n"
    );
}

/// A script passed as one argument still has to reach the gate; it
/// only matches once each argument is checked on its own.
#[tokio::test]
async fn open_destructive_inside_bash_c_argument_consults_gate() {
    let gate = RecordingGate::new(false);
    let out = run_open(
        &rm_rf_is_destructive(gate.clone()),
        &["bash", "-c", "rm -rf /tmp/whatever"],
    )
    .await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert_eq!(
        gate.prompts(),
        [(
            "wm".to_string(),
            "bash -c rm -rf /tmp/whatever".to_string(),
            "rm -rf".to_string()
        )]
    );
}

#[tokio::test]
async fn open_gate_approval_lets_the_process_run() {
    let gate = RecordingGate::new(true);
    let cmd = policed_wm(
        BashPolicyCfg {
            destructive_patterns: vec![vec!["true".into()]],
            ..Default::default()
        },
        gate.clone(),
    );
    let out = run_open(&cmd, &["true"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(
        gate.prompts(),
        [("wm".to_string(), "true".to_string(), "true".to_string())]
    );
}

/// An application outliving the startup probe keeps running, and the
/// launch reports success rather than blocking the turn.
#[tokio::test]
async fn open_leaves_a_surviving_process_running() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("finished");

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

    let mut finished = false;
    for _ in 0..100 {
        if marker.exists() {
            finished = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        finished,
        "detached child must keep running after the launch returned"
    );
}

#[tokio::test]
async fn open_reports_a_failed_startup_with_its_output() {
    let cmd = policed_wm(BashPolicyCfg::default(), Arc::new(AlwaysAllowGate));
    let out = run_open(&cmd, &["bash", "-c", "echo boom >&2; exit 3"]).await;
    assert_eq!(out.exit_code, 3);
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("boom"),
        "startup failure must surface the child's stderr: {out:?}"
    );
}

/// Policy is scoped to `open`; the compositor subcommands never
/// consult the gate or the denylist.
#[tokio::test]
async fn non_open_subcommands_skip_the_policy() {
    let gate = RecordingGate::new(false);
    let cmd = policed_wm(
        BashPolicyCfg {
            denylist: vec!["focus".into()],
            destructive_patterns: vec![vec!["focus".into()]],
            ..Default::default()
        },
        gate.clone(),
    );
    let out = cmd
        .run(CommandInput {
            args: vec!["focus".into(), "42".into()],
            stdin: None,
        })
        .await;
    assert_eq!(out.exit_code, 0);
    assert!(gate.prompts().is_empty(), "{:?}", gate.prompts());
}

/// `wm active` prints `<id>\t<app>` so either column can be piped on.
#[tokio::test]
async fn active_prints_id_tab_app() {
    let cases: [(Option<u64>, Option<&str>, &[u8]); 3] = [
        (Some(42), Some("Firefox"), b"42\tFirefox\n"),
        (Some(7), None, b"7\t-\n"),
        (None, None, b""),
    ];
    for (focused, app, expected) in cases {
        let stub = Arc::new(StubWm {
            connected: true,
            focused: focused.map(id),
            focused_app: app.map(str::to_string),
            ..Default::default()
        });
        let out = run_wm(stub, &["active"]).await;
        assert_eq!(out.exit_code, 0, "{focused:?} {app:?}");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(expected),
            "{focused:?} {app:?}"
        );
    }
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
                current_mode: Some((1920, 1080, 59_951)),
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
        "DP-1\t*\t*\t1920x1080@59.951Hz\t1.5\t1:web\nDP-2\t*\t-\t2560x1440@144Hz\t1\t3\n"
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

/// An i3-class backend that cannot list outputs gets an explanatory
/// error rather than empty stdout.
#[tokio::test]
async fn outputs_unsupported_backend_emits_error_with_note() {
    let stub = Arc::new(StubWm {
        connected: true,
        list_outputs_unsupported: true,
        ..Default::default()
    });
    let out = run_wm(stub, &["outputs"]).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] wm: outputs failed: backend does not support output enumeration. \
         Note: the active backend may not support this operation (i3 does not list outputs)\n"
    );
}

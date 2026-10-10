use std::time::{Duration, Instant};

use assistd_wm::{
    FocusedWindowContext, Layout, OutputInfo, ResizeDir, Window, WindowId, WmResult, WorkspaceId,
    WorkspaceInfo,
};
use parking_lot::Mutex;

use super::*;
use crate::commands::RecordingGate;
use crate::commands::test_patterns as patterns;
use crate::exec::{OUTPUT_OVERFLOW_EXIT, POLICY_DENIED_EXIT};
use crate::policy::{AlwaysAllowGate, DenyAllGate};

fn id(n: u64) -> WindowId {
    WindowId::new(n).expect("test ids are non-zero")
}

/// [`WindowManager`] fixture recording every mutating call.
#[derive(Debug, Default)]
struct StubWm {
    windows: Vec<Window>,
    outputs: Vec<OutputInfo>,
    focus_calls: Mutex<Vec<WindowId>>,
    move_calls: Mutex<Vec<(WindowId, WorkspaceId)>>,
    resize_calls: Mutex<Vec<(WindowId, ResizeDir, u32)>>,
    layout_calls: Mutex<Vec<Layout>>,
}

impl StubWm {
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
        Ok(())
    }
    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        self.move_calls.lock().push((*window, workspace.clone()));
        Ok(())
    }
    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        Ok(None)
    }
    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        Ok(None)
    }
    async fn list_windows(&self) -> WmResult<Vec<Window>> {
        Ok(self.windows.clone())
    }
    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        Ok(Vec::new())
    }
    async fn resize_width(
        &self,
        window: &WindowId,
        direction: ResizeDir,
        pixels: u32,
    ) -> WmResult<()> {
        self.resize_calls.lock().push((*window, direction, pixels));
        Ok(())
    }
    async fn set_layout(&self, layout: Layout) -> WmResult<()> {
        self.layout_calls.lock().push(layout);
        Ok(())
    }
    async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
        Ok(self.outputs.clone())
    }
    fn is_connected(&self) -> bool {
        true
    }
}

async fn run_wm(wm: Arc<dyn WindowManager>, args: &[&str]) -> CommandOutput {
    WmCommand::for_test(wm)
        .run(CommandInput {
            args: args.iter().map(ToString::to_string).collect(),
            stdin: None,
        })
        .await
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
    let cases: [(&[&str], String); 4] = [
        (&["move", "Firefox", "3"], bad_id("move")),
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
        let stub = Arc::new(StubWm::default());
        let out = run_wm(stub.clone(), args).await;
        assert_eq!(out.exit_code, 2, "{args:?}");
        assert_eq!(String::from_utf8_lossy(&out.stderr), stderr, "{args:?}");
        assert!(stub.no_backend_calls(), "{args:?}");
    }
}

#[tokio::test]
async fn resize_dispatches_typed_args() {
    let stub = Arc::new(StubWm::default());
    let out = run_wm(stub.clone(), &["resize", "42", "grow", "50"]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(*stub.resize_calls.lock(), [(id(42), ResizeDir::Grow, 50)]);
}

fn policed_wm(cfg: BashPolicyCfg, gate: Arc<dyn ConfirmationGate>) -> WmCommand {
    WmCommand::new(
        Arc::new(StubWm::default()),
        Arc::new(cfg),
        SandboxInfo::none(),
        gate,
    )
}

fn rm_rf_is_destructive(gate: Arc<dyn ConfirmationGate>) -> WmCommand {
    policed_wm(
        BashPolicyCfg {
            destructive_patterns: patterns(&["rm -rf"]),
            ..Default::default()
        },
        gate,
    )
}

async fn run_open(cmd: &WmCommand, args: &[&str]) -> CommandOutput {
    let mut argv = vec!["open".to_string()];
    argv.extend(args.iter().map(ToString::to_string));
    cmd.run(CommandInput {
        args: argv,
        stdin: None,
    })
    .await
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
            destructive_patterns: patterns(&["true"]),
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

#[tokio::test]
async fn open_leaves_a_surviving_process_running() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("finished");

    let cmd = policed_wm(BashPolicyCfg::default(), Arc::new(AlwaysAllowGate));
    let started = Instant::now();
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
        started.elapsed() < Duration::from_millis(900),
        "launch should return on the probe, not on the child exiting"
    );
    assert!(!marker.exists(), "child should not have finished yet");

    let mut finished = false;
    for _ in 0..100 {
        if marker.exists() {
            finished = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
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

#[tokio::test]
async fn open_asks_before_launching_a_program_with_no_desktop_entry() {
    let gate = RecordingGate::new(false);
    let cmd = policed_wm(BashPolicyCfg::default(), gate.clone());
    let out = run_open(&cmd, &["cat", "/dev/urandom"]).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert_eq!(
        gate.prompts(),
        [(
            "wm".to_string(),
            "cat /dev/urandom".to_string(),
            "cat is not a desktop application".to_string()
        )]
    );
}

#[tokio::test]
async fn open_closes_the_pipe_of_a_chatty_launch_at_the_cap() {
    let cmd = policed_wm(BashPolicyCfg::default(), Arc::new(AlwaysAllowGate));
    let out = run_open(&cmd, &["yes"]).await;
    assert_eq!(
        out.exit_code, OUTPUT_OVERFLOW_EXIT,
        "yes should die of SIGPIPE"
    );
    assert_eq!(out.stdout.len(), 64 * 1024);
}

#[tokio::test]
async fn open_refuses_once_the_launch_cap_is_reached() {
    let cmd = policed_wm(BashPolicyCfg::default(), Arc::new(AlwaysAllowGate));
    for _ in 0..MAX_LAUNCHED {
        assert_eq!(run_open(&cmd, &["sleep", "5"]).await.exit_code, 0);
    }
    let out = run_open(&cmd, &["sleep", "5"]).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
}

#[tokio::test]
async fn non_open_subcommands_skip_the_policy() {
    let gate = RecordingGate::new(false);
    let cmd = policed_wm(
        BashPolicyCfg {
            denylist: vec!["focus".into()],
            destructive_patterns: patterns(&["focus"]),
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

#[tokio::test]
async fn list_emits_tsv_sorted_by_workspace_then_app() {
    let stub = Arc::new(StubWm {
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
async fn outputs_emits_tsv_sorted_by_name() {
    let stub = Arc::new(StubWm {
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

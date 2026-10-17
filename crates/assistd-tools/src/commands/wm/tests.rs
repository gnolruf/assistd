use std::path::PathBuf;

use assistd_wm::{
    ExecCommand, FocusedWindowContext, Layout, OutputInfo, ResizeDir, Window, WindowId, WmResult,
    WorkspaceId, WorkspaceInfo,
};
use parking_lot::Mutex;

use super::desktop_entry::DesktopEntry;
use super::*;
use crate::commands::RecordingGate;

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
    exec_calls: Mutex<Vec<(Option<WorkspaceId>, ExecCommand)>>,
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
    async fn exec(&self, workspace: Option<&WorkspaceId>, command: &ExecCommand) -> WmResult<()> {
        self.exec_calls
            .lock()
            .push((workspace.cloned(), command.clone()));
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

fn viewer(exec: &str) -> DesktopEntry {
    let text = format!("[Desktop Entry]\nType=Application\nName=Viewer\nExec={exec}\n");
    DesktopEntry::parse(
        PathBuf::from("/usr/share/applications/viewer.desktop"),
        "viewer",
        &text,
    )
    .expect("launchable entry")
}

fn open_args(args: &[&str]) -> OpenArgs {
    let args: Vec<String> = args.iter().map(ToString::to_string).collect();
    OpenArgs::parse(&args).expect("parsable")
}

#[test]
fn launch_lines_keep_each_target_one_word_and_refuse_option_like_ones() {
    let launch = |exec: &str, args: &[&str]| {
        PreparedLaunch::new(viewer(exec), &open_args(args))
            .map(|launch| launch.command.as_str().to_string())
    };
    assert_eq!(
        launch(
            "viewer --url -- %U",
            &[
                "--cwd",
                "/",
                "viewer",
                "/tmp/a b;c",
                "/tmp/it's",
                "https://x.org/?q=1&r=$HOME"
            ],
        )
        .expect("launchable"),
        "cd / && exec viewer --url -- '/tmp/a b;c' '/tmp/it'\\''s' 'https://x.org/?q=1&r=$HOME'"
    );
    for (exec, args) in [
        ("viewer %U", &["viewer", "--help"][..]),
        ("viewer %U", &["viewer", "notes.txt"]),
        ("viewer %U", &["viewer", "-x:y"]),
        ("viewer %F", &["viewer", "https://x.org"]),
        ("viewer", &["viewer", "/tmp/a"]),
        ("viewer %u", &["viewer", "/tmp/a", "/tmp/b"]),
        ("viewer %f", &["viewer", "/tmp/\"; kill"]),
        ("viewer", &["--cwd", "relative", "viewer"]),
    ] {
        assert!(launch(exec, args).is_err(), "{exec} {args:?}");
    }
}

#[test]
fn open_options_must_precede_the_application() {
    let args = open_args(&["--workspace", "2", "--cwd", "~/src", "nvim", "--cwd"]);
    assert_eq!(args.workspace, Some(WorkspaceId::Num(2)));
    assert!(args.has_targets(), "options after the app are targets");
    for bad in [&["--wrkspace", "2", "nvim"][..], &["--workspace"], &[]] {
        let bad: Vec<String> = bad.iter().map(ToString::to_string).collect();
        assert!(OpenArgs::parse(&bad).is_err(), "{bad:?}");
    }
}

#[tokio::test]
async fn approved_apps_skip_the_prompt_only_when_launched_without_targets() {
    let gate = RecordingGate::new(false);
    let approved = Arc::new(Approvals::unsaved());
    approved
        .approve("/usr/share/applications/viewer.desktop")
        .await
        .expect("unsaved approvals never fail");
    let cmd = WmCommand::new(Arc::new(StubWm::default()), gate.clone(), approved);
    let confirmed = async |args: &[&str]| {
        let args = open_args(args);
        let launch = PreparedLaunch::new(viewer("viewer %F"), &args).expect("launchable");
        cmd.confirmed(&args, &launch).await
    };
    assert!(confirmed(&["viewer"]).await);
    assert!(gate.prompts().is_empty(), "{:?}", gate.prompts());
    assert!(!confirmed(&["--workspace", "2", "viewer", "/etc/passwd"]).await);
    assert_eq!(
        gate.prompts(),
        [(
            "wm".to_string(),
            "exec viewer /etc/passwd".to_string(),
            "opens Viewer (/usr/share/applications/viewer.desktop) on workspace 2 as you, \
             outside the sandbox"
                .to_string()
        )]
    );
}

#[tokio::test]
async fn an_unknown_app_reaches_neither_the_gate_nor_the_compositor() {
    let gate = RecordingGate::new(true);
    let stub = Arc::new(StubWm::default());
    let cmd = WmCommand::new(stub.clone(), gate.clone(), Arc::new(Approvals::unsaved()));
    let out = cmd
        .run(CommandInput {
            args: ["open", "--workspace", "2", "no-such-app-assistd"]
                .map(String::from)
                .to_vec(),
            stdin: None,
        })
        .await;
    assert_eq!(out.exit_code, 1);
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .starts_with("[error] wm: open: no installed application has the desktop id"),
        "{out:?}"
    );
    assert!(gate.prompts().is_empty());
    assert!(stub.exec_calls.lock().is_empty());
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

use super::*;
use crate::exec::{OUTPUT_BUF_MAX, OUTPUT_OVERFLOW_EXIT};
use crate::policy::{AlwaysAllowGate, DenyAllGate};

fn bash_with_cfg(cfg: BashPolicyCfg, gate: Arc<dyn ConfirmationGate>) -> BashCommand {
    BashCommand::new(Arc::new(cfg), SandboxInfo::none(), gate)
}

#[tokio::test]
async fn bash_runs_echo() {
    let out = BashCommand::default()
        .run(CommandInput {
            args: vec!["echo hi".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"hi\n");
}

#[tokio::test]
async fn bash_propagates_nonzero_exit() {
    let out = BashCommand::default()
        .run(CommandInput {
            args: vec!["exit 3".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 3);
}

#[tokio::test]
async fn bash_receives_stdin() {
    let out = BashCommand::default()
        .run(CommandInput {
            args: vec!["tr a-z A-Z".into()],
            stdin: Some(b"hello".to_vec()),
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"HELLO");
}

/// A timed-out script returns exit 137 with the byte-exact message
/// `[error] bash: timed out after 30s [exit:137 | 30.0s]`. The
/// timeout is 100ms here for speed; the format is unchanged.
#[tokio::test]
async fn bash_timeout_returns_137_with_ac_format() {
    let cfg = BashPolicyCfg {
        timeout: Duration::from_millis(100),
        ..Default::default()
    };
    let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
    let out = cmd
        .run(CommandInput {
            args: vec!["sleep 5".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 137);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.starts_with("[error] bash: timed out after 0s [exit:137 | "),
        "{stderr}"
    );
    assert!(stderr.ends_with("s]\n"), "{stderr}");
}

#[tokio::test]
async fn bash_missing_script_errors() {
    let out = BashCommand::default()
        .run(CommandInput {
            args: Vec::new(),
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 2);
}

/// Acceptance: `bash "<nonexistent-dep>"` must forward the subprocess's
/// own stderr ("command not found") so the LLM sees *which* dependency
/// is missing, not a bare exit:127.
#[tokio::test]
async fn bash_missing_dependency_forwards_subprocess_stderr() {
    let out = BashCommand::default()
        .run(CommandInput {
            args: vec!["assistd-definitely-not-a-real-binary-xyz".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 127);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("command not found"), "{stderr}");
    assert!(
        stderr.contains("assistd-definitely-not-a-real-binary-xyz"),
        "{stderr}"
    );
}

/// A script matching a denylist pattern is rejected before spawn,
/// with the byte-exact error message
/// `[error] bash: command denied by policy. Matched denylist pattern:
/// rm -rf /. Try: a non-destructive alternative\n`.
#[tokio::test]
async fn ac1_bash_rm_rf_root_rejected() {
    let cfg = BashPolicyCfg {
        denylist: vec!["rm -rf /".into()],
        ..Default::default()
    };
    let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
    let out = cmd
        .run(CommandInput {
            args: vec!["rm -rf /".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 126);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        stderr,
        "[error] bash: command denied by policy. Matched denylist pattern: rm -rf /. Try: a non-destructive alternative\n"
    );
}

#[tokio::test]
async fn bash_denylist_is_case_insensitive() {
    let cfg = BashPolicyCfg {
        denylist: vec!["mkfs".into()],
        ..Default::default()
    };
    let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
    let out = cmd
        .run(CommandInput {
            args: vec!["MKFS.ext4 /dev/sda1".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 126);
}

/// When a destructive pattern is matched and the gate approves, the
/// command runs normally. Uses `true` as a no-op script so the test
/// doesn't actually delete anything.
#[tokio::test]
async fn destructive_pattern_invokes_gate_and_proceeds_when_approved() {
    let cfg = BashPolicyCfg {
        destructive_patterns: vec![vec!["true".into()]],
        ..Default::default()
    };
    let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
    let out = cmd
        .run(CommandInput {
            args: vec!["true".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
}

/// When a destructive pattern is matched and the gate denies, the
/// command does NOT run and returns exit 126 with a cancellation
/// message.
#[tokio::test]
async fn destructive_pattern_invokes_gate_and_cancels_when_denied() {
    let cfg = BashPolicyCfg {
        destructive_patterns: vec![vec!["rm".into(), "-rf".into()]],
        ..Default::default()
    };
    // DenyAllGate is the production default for IPC-connected clients;
    // verify it blocks destructive commands as documented.
    let cmd = bash_with_cfg(cfg, Arc::new(DenyAllGate));
    let out = cmd
        .run(CommandInput {
            // Use /tmp/nonexistent so that even if the gate is buggy
            // and allows execution, no real data is lost.
            args: vec!["rm -rf /tmp/this-directory-does-not-exist-XYZ".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 126);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cancelled by user"),
        "expected cancellation message, got {stderr}"
    );
    assert!(
        stderr.contains("Matched destructive pattern: rm -rf"),
        "expected matched pattern in message, got {stderr}"
    );
}

/// A destructive pattern substring inside a quoted literal must not
/// trigger the gate; `echo "rm -rf"` is harmless and legitimate.
#[tokio::test]
async fn destructive_matcher_ignores_quoted_literals() {
    // Counter-gate that panics if called; asserts the gate is NOT
    // invoked for this script.
    struct PanicGate;
    #[async_trait]
    impl ConfirmationGate for PanicGate {
        async fn confirm(&self, _req: ConfirmationRequest) -> bool {
            panic!("gate should not be invoked for quoted literal");
        }
    }
    let cfg = BashPolicyCfg {
        destructive_patterns: vec![vec!["rm".into(), "-rf".into()]],
        ..Default::default()
    };
    let cmd = bash_with_cfg(cfg, Arc::new(PanicGate));
    let out = cmd
        .run(CommandInput {
            args: vec!["echo \"rm -rf\"".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"rm -rf\n");
}

/// A runaway script (`yes`) emits gigabytes per second. Without an
/// execution-time cap, `wait_with_output()` would buffer it all into
/// memory before the chain executor's PIPE_BUF_MAX check fires. With
/// the cap, the child is killed at OUTPUT_BUF_MAX and we return exit
/// 141 with a bounded `stdout`. We use a short timeout so the test
/// is fast even if the overflow path is broken — but the test only
/// passes if overflow (141) fires *before* timeout (137).
#[tokio::test]
async fn bash_output_overflow_kills_child_and_returns_141() {
    let cfg = BashPolicyCfg {
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
    let out = cmd
        .run(CommandInput {
            args: vec!["yes".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, OUTPUT_OVERFLOW_EXIT);
    assert!(
        out.stdout.len() <= OUTPUT_BUF_MAX,
        "stdout was {} bytes, expected <= {OUTPUT_BUF_MAX}",
        out.stdout.len()
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("output exceeded"),
        "expected overflow message in stderr, got {stderr}"
    );
}

/// Stderr is bounded too: a script that floods only stderr is killed
/// once it crosses the cap. Uses `bash -c` redirection inside the
/// script so the bytes land on fd 2.
#[tokio::test]
async fn bash_stderr_overflow_also_caps() {
    let cfg = BashPolicyCfg {
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
    let out = cmd
        .run(CommandInput {
            args: vec!["yes 1>&2".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, OUTPUT_OVERFLOW_EXIT);
    assert!(
        out.stderr.len() <= OUTPUT_BUF_MAX + 256,
        "stderr was {} bytes, expected <= ~{OUTPUT_BUF_MAX} + overflow message",
        out.stderr.len()
    );
}

/// A script that writes well under the cap and exits cleanly must
/// still return exit 0 with its full stdout — i.e. the streaming
/// path doesn't drop bytes or false-positive on overflow.
#[tokio::test]
async fn bash_below_cap_returns_full_output() {
    let out = BashCommand::default()
        .run(CommandInput {
            // ~50 KiB, comfortably below 10 MiB.
            args: vec!["printf '%.0sx' {1..51200}".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.len(), 51200);
}

/// A script killed by a signal (here SIGSEGV via `kill -SEGV $$`) must
/// report `128 + signum` (139), not the timeout sentinel 137. Before
/// the fix, `status.code()` returning `None` collapsed both the
/// signal-death and timeout cases to 137.
#[cfg(unix)]
#[tokio::test]
async fn bash_signal_death_reports_128_plus_signum_not_timeout() {
    let out = BashCommand::default()
        .run(CommandInput {
            args: vec!["kill -SEGV $$".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(
        out.exit_code, 139,
        "SIGSEGV should surface as 128+11=139, not the timeout sentinel"
    );
}

/// Poll `kill -0` until `pid` is gone, up to two seconds. Returns
/// whether it exited; a SIGKILL'd orphan reparents to init and is
/// reaped within a few milliseconds.
#[cfg(unix)]
async fn waited_for_exit(pid: &str) -> bool {
    for _ in 0..40 {
        let alive = std::process::Command::new("kill")
            .args(["-0", pid])
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if !alive {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

/// Run `script` (which must write a background child's PID to
/// `pidfile`) and return the output plus that PID. The whole call is
/// bounded: an un-killed grandchild holds the output pipe open, so
/// without the group kill `supervise` never reaches EOF and hangs for
/// as long as the orphan lives.
#[cfg(unix)]
async fn run_leaving_background_child(
    cfg: BashPolicyCfg,
    script: impl Fn(&std::path::Path) -> String,
) -> (CommandOutput, String) {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("child.pid");
    let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
    let out = tokio::time::timeout(
        Duration::from_secs(10),
        cmd.run(CommandInput {
            args: vec![script(&pidfile)],
            stdin: None,
        }),
    )
    .await
    .expect("bash did not return: a grandchild kept the output pipe open")
    .unwrap();
    let pid = std::fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .to_string();
    assert!(!pid.is_empty(), "script did not record a background PID");
    (out, pid)
}

/// A timed-out script must take its whole process group down, not
/// just the `bash` group leader. `sleep 300 &` outlives a
/// leader-only SIGKILL, reparents to init, and keeps the inherited
/// stdout pipe open.
#[cfg(unix)]
#[tokio::test]
async fn timeout_kills_backgrounded_grandchild() {
    let cfg = BashPolicyCfg {
        timeout: Duration::from_millis(200),
        ..Default::default()
    };
    let (out, pid) = run_leaving_background_child(cfg, |pidfile| {
        format!("sleep 300 & echo $! > {}; wait", pidfile.display())
    })
    .await;
    assert_eq!(out.exit_code, 137);
    assert!(
        waited_for_exit(&pid).await,
        "grandchild {pid} survived the timeout"
    );
}

/// Same requirement on the overflow path: the flood (`yes`) dies of
/// SIGPIPE once we drop the read end, but a quiet background child
/// only dies if the group is signalled.
#[cfg(unix)]
#[tokio::test]
async fn output_overflow_kills_backgrounded_grandchild() {
    let cfg = BashPolicyCfg {
        timeout: Duration::from_secs(30),
        ..Default::default()
    };
    let (out, pid) = run_leaving_background_child(cfg, |pidfile| {
        format!("sleep 300 & echo $! > {}; yes", pidfile.display())
    })
    .await;
    assert_eq!(out.exit_code, OUTPUT_OVERFLOW_EXIT);
    assert!(
        waited_for_exit(&pid).await,
        "grandchild {pid} survived the overflow kill"
    );
}

/// Sandbox mode `None` executes bash directly with no wrapper.
#[tokio::test]
async fn bash_sandbox_none_runs_unsandboxed() {
    let cfg = BashPolicyCfg::default();
    let cmd = BashCommand::new(
        Arc::new(cfg),
        SandboxInfo::none(),
        Arc::new(AlwaysAllowGate),
    );
    let out = cmd
        .run(CommandInput {
            args: vec!["echo sandboxed".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"sandboxed\n");
}

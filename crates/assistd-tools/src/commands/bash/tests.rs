use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use super::*;
use crate::chain::{execute, parse_chain};
use crate::command::CommandRegistry;
use crate::commands::RecordingGate;
use crate::commands::test_patterns as patterns;
use crate::exec::{OUTPUT_BUF_MAX, OUTPUT_OVERFLOW_EXIT};
use crate::policy::{AlwaysAllowGate, DenyAllGate};

fn bash_with_cfg(cfg: BashPolicyCfg, gate: Arc<dyn ConfirmationGate>) -> BashCommand {
    BashCommand::new(Arc::new(cfg), SandboxInfo::none(), gate)
}

fn with_timeout(timeout: Duration) -> BashCommand {
    bash_with_cfg(
        BashPolicyCfg {
            timeout,
            ..Default::default()
        },
        Arc::new(AlwaysAllowGate),
    )
}

fn rm_rf_is_destructive(gate: Arc<dyn ConfirmationGate>) -> BashCommand {
    bash_with_cfg(
        BashPolicyCfg {
            destructive_patterns: patterns(&["rm -rf"]),
            ..Default::default()
        },
        gate,
    )
}

async fn run(cmd: &BashCommand, script: &str, stdin: Option<Vec<u8>>) -> CommandOutput {
    cmd.run(CommandInput {
        args: vec![script.into()],
        stdin,
    })
    .await
}

#[tokio::test]
async fn glob_matches_reach_bash_as_file_names_not_code() {
    let dir = tempfile::tempdir().expect("tempdir");
    let planted = dir.path().join("a$(echo injected).txt");
    std::fs::write(&planted, b"").expect("planted file");
    let mut registry = CommandRegistry::new();
    registry.register(BashCommand::default());
    let line = format!("bash echo {}/*.txt", dir.path().display());

    let out = execute(&parse_chain(&line).expect("parses"), &registry, None).await;

    assert_eq!(out.exit_code, 0, "{out:?}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{}\n", planted.display())
    );
}

#[tokio::test]
async fn leading_dash_c_is_accepted_as_bash_would() {
    let mut registry = CommandRegistry::new();
    registry.register(BashCommand::default());
    let line = "bash -c 'n=$((1+4)); echo RESULT_$n | tr A-Z a-z'";

    let out = execute(&parse_chain(line).expect("parses"), &registry, None).await;

    assert_eq!(out.exit_code, 0, "{out:?}");
    assert_eq!(out.stdout, b"result_5\n");
}

#[tokio::test]
async fn bash_timeout_returns_137_with_timeout_message() {
    let out = run(&with_timeout(Duration::from_millis(100)), "sleep 5", None).await;
    assert_eq!(out.exit_code, 137);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.starts_with("[error] bash: timed out after 0s [exit:137 | "),
        "{stderr}"
    );
    assert!(stderr.ends_with("s]\n"), "{stderr}");
}

#[tokio::test]
async fn denylist_match_is_rejected_before_spawn() {
    let cmd = bash_with_cfg(
        BashPolicyCfg {
            denylist: vec!["rm -rf /".into()],
            ..Default::default()
        },
        Arc::new(AlwaysAllowGate),
    );
    let out = run(&cmd, "rm -rf /", None).await;
    assert_eq!(out.exit_code, 126);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] bash: command denied by policy. Matched denylist pattern: rm -rf /. Try: a non-destructive alternative\n"
    );
}

#[tokio::test]
async fn destructive_pattern_is_cancelled_when_denied() {
    let cmd = rm_rf_is_destructive(Arc::new(DenyAllGate));
    let out = run(&cmd, "rm -rf /tmp/this-directory-does-not-exist-XYZ", None).await;
    assert_eq!(out.exit_code, 126);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] bash: command cancelled by user. Matched destructive pattern: rm -rf. Try: a different approach\n"
    );
}

#[tokio::test]
async fn quoted_destructive_literal_does_not_prompt() {
    let gate = RecordingGate::new(false);
    let out = run(&rm_rf_is_destructive(gate.clone()), "echo \"rm -rf\"", None).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"rm -rf\n");
    assert!(gate.prompts().is_empty(), "{:?}", gate.prompts());
}

/// `yes` must be killed at the capture cap (141), not buffered until the
/// timeout (137).
#[tokio::test]
async fn bash_output_overflow_kills_child_and_returns_141() {
    let out = run(&with_timeout(Duration::from_secs(10)), "yes", None).await;
    assert_eq!(out.exit_code, OUTPUT_OVERFLOW_EXIT);
    assert!(
        out.stdout.len() <= OUTPUT_BUF_MAX,
        "stdout was {} bytes, expected <= {OUTPUT_BUF_MAX}",
        out.stdout.len()
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("output exceeded"), "{stderr}");
}

/// Poll `kill -0` for up to two seconds; returns whether `pid` exited.
#[cfg(unix)]
async fn waited_for_exit(pid: &str) -> bool {
    for _ in 0..40 {
        let alive = std::process::Command::new("kill")
            .args(["-0", pid])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if !alive {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

/// Run `script`, which must write a background child's PID to `pidfile`,
/// and return the output plus that PID. Bounded, since a surviving
/// grandchild would hold the output pipe open forever.
#[cfg(unix)]
async fn run_leaving_background_child(
    cfg: BashPolicyCfg,
    script: impl Fn(&Path) -> String,
) -> (CommandOutput, String) {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("child.pid");
    let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
    let out = tokio::time::timeout(Duration::from_secs(10), run(&cmd, &script(&pidfile), None))
        .await
        .expect("bash did not return: a grandchild kept the output pipe open");
    let pid = std::fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .to_string();
    assert!(!pid.is_empty(), "script did not record a background PID");
    (out, pid)
}

/// `sleep 300 &` outlives a leader-only SIGKILL, so this checks the whole
/// process group is killed.
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

#[cfg(unix)]
#[tokio::test]
async fn normal_exit_kills_backgrounded_grandchild() {
    let (out, pid) = run_leaving_background_child(BashPolicyCfg::default(), |pidfile| {
        format!("sleep 300 & echo $! > {}; echo done", pidfile.display())
    })
    .await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"done\n");
    assert!(
        waited_for_exit(&pid).await,
        "grandchild {pid} survived the script's exit"
    );
}

#[tokio::test]
async fn unread_stdin_does_not_block_the_timeout() {
    let cmd = with_timeout(Duration::from_millis(200));
    let out = tokio::time::timeout(
        Duration::from_secs(10),
        run(&cmd, "sleep 300", Some(vec![b'x'; 1024 * 1024])),
    )
    .await
    .expect("stdin write blocked before the timeout armed");
    assert_eq!(out.exit_code, 137);
}

#[tokio::test]
async fn stdin_larger_than_a_pipe_buffer_arrives_whole() {
    let out = run(
        &BashCommand::default(),
        "wc -c",
        Some(vec![b'x'; 1024 * 1024]),
    )
    .await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "1048576");
}

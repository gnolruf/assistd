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
async fn bash_runs_echo() {
    let out = run(&BashCommand::default(), "echo hi", None).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"hi\n");
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
async fn bash_propagates_nonzero_exit() {
    let out = run(&BashCommand::default(), "exit 3", None).await;
    assert_eq!(out.exit_code, 3);
}

#[tokio::test]
async fn bash_receives_stdin() {
    let out = run(
        &BashCommand::default(),
        "tr a-z A-Z",
        Some(b"hello".to_vec()),
    )
    .await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"HELLO");
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
async fn bash_without_script_emits_usage() {
    let out = BashCommand::default()
        .run(CommandInput {
            args: Vec::new(),
            stdin: None,
        })
        .await;
    assert_eq!(out.exit_code, 2);
    assert!(out.stdout.starts_with(b"usage: bash"), "{out:?}");
}

/// The model must see *which* dependency is missing, not a bare exit 127.
#[tokio::test]
async fn bash_missing_dependency_forwards_subprocess_stderr() {
    let out = run(
        &BashCommand::default(),
        "assistd-definitely-not-a-real-binary-xyz",
        None,
    )
    .await;
    assert_eq!(out.exit_code, 127);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("command not found"), "{stderr}");
    assert!(
        stderr.contains("assistd-definitely-not-a-real-binary-xyz"),
        "{stderr}"
    );
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
async fn destructive_pattern_prompts_and_runs_when_approved() {
    let gate = RecordingGate::new(true);
    let cmd = bash_with_cfg(
        BashPolicyCfg {
            destructive_patterns: patterns(&["true"]),
            ..Default::default()
        },
        gate.clone(),
    );
    let out = run(&cmd, "true", None).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(
        gate.prompts(),
        [("bash".to_string(), "true".to_string(), "true".to_string())]
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

#[tokio::test]
async fn bash_stderr_overflow_also_caps() {
    let out = run(&with_timeout(Duration::from_secs(10)), "yes 1>&2", None).await;
    assert_eq!(out.exit_code, OUTPUT_OVERFLOW_EXIT);
    assert!(
        out.stderr.len() <= OUTPUT_BUF_MAX + 256,
        "stderr was {} bytes, expected <= ~{OUTPUT_BUF_MAX} + overflow message",
        out.stderr.len()
    );
}

#[tokio::test]
async fn bash_below_cap_returns_full_output() {
    let out = run(&BashCommand::default(), "printf '%.0sx' {1..51200}", None).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, vec![b'x'; 51200]);
}

#[cfg(unix)]
#[tokio::test]
async fn bash_signal_death_reports_128_plus_signum_not_timeout() {
    let out = run(&BashCommand::default(), "kill -SEGV $$", None).await;
    assert_eq!(out.exit_code, 139);
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

/// `yes` dies of SIGPIPE on its own; the quiet background child only dies
/// if the group is signalled.
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

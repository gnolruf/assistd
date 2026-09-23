//! Integration tests for the bash command's policy and sandbox layers,
//! in the order they fire: the denylist (gate skipped, exit 126), the
//! destructive-pattern gate, and the bwrap sandbox. The matchers
//! themselves are unit-tested alongside the policy module.
//!
//! Bwrap-dependent tests return early when `bwrap` is not on PATH so
//! hosts without bubblewrap still pass.

use std::sync::Arc;

use assistd_tools::commands::{BashCommand, BashPolicyCfg};
use assistd_tools::policy::{ResolvedSandboxMode, probe_sandbox};
use assistd_tools::{
    AlwaysAllowGate, Command, CommandInput, ConfirmationGate, ConfirmationRequest, DenyAllGate,
    SandboxInfo, SandboxRequest,
};
use async_trait::async_trait;
use parking_lot::Mutex;

const POLICY_DENIED_EXIT: i32 = 126;

fn bash_with(
    denylist: Vec<&str>,
    destructive: Vec<Vec<&str>>,
    gate: Arc<dyn ConfirmationGate>,
    sandbox: Arc<SandboxInfo>,
) -> BashCommand {
    let cfg = BashPolicyCfg {
        timeout: std::time::Duration::from_secs(10),
        denylist: denylist.into_iter().map(|s| s.to_string()).collect(),
        destructive_patterns: destructive
            .into_iter()
            .map(|prefix| prefix.into_iter().map(|s| s.to_string()).collect())
            .collect(),
    };
    BashCommand::new(Arc::new(cfg), sandbox, gate)
}

fn no_sandbox() -> Arc<SandboxInfo> {
    SandboxInfo::none()
}

fn input(script: &str) -> CommandInput {
    CommandInput {
        args: vec![script.to_string()],
        stdin: None,
    }
}

fn bwrap_or_none() -> Option<Arc<SandboxInfo>> {
    let info = probe_sandbox(SandboxRequest::Bwrap, Vec::new()).ok()?;
    matches!(info.mode, ResolvedSandboxMode::Bwrap { .. }).then_some(info)
}

/// Fails the test if the policy ever consults it.
struct PanicGate;

#[async_trait]
impl ConfirmationGate for PanicGate {
    async fn confirm(&self, req: ConfirmationRequest) -> bool {
        panic!("gate must not be consulted for {:?}", req.script);
    }
}

/// Approves every request and records what it was asked.
#[derive(Default)]
struct RecordingGate {
    asked: Mutex<Vec<ConfirmationRequest>>,
}

#[async_trait]
impl ConfirmationGate for RecordingGate {
    async fn confirm(&self, req: ConfirmationRequest) -> bool {
        self.asked.lock().push(req);
        true
    }
}

#[tokio::test]
async fn denylist_blocks_rm_rf_root() {
    let cmd = bash_with(
        vec!["rm -rf /"],
        vec![],
        Arc::new(AlwaysAllowGate),
        no_sandbox(),
    );
    let out = cmd.run(input("rm -rf /")).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("denylist pattern: rm -rf /"),
        "stderr should name the matched pattern: {stderr}"
    );
}

#[tokio::test]
async fn denylist_bypasses_confirmation_gate() {
    let cmd = bash_with(
        vec!["rm -rf"],
        vec![vec!["rm", "-rf"]],
        Arc::new(PanicGate),
        no_sandbox(),
    );
    let out = cmd.run(input("rm -rf /tmp/whatever")).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
}

#[tokio::test]
async fn destructive_pattern_is_blocked_when_gate_denies() {
    let cmd = bash_with(
        vec![],
        vec![vec!["rm", "-rf"]],
        Arc::new(DenyAllGate),
        no_sandbox(),
    );
    let out = cmd.run(input("touch /tmp/x && rm -rf /tmp/x")).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Matched destructive pattern: rm -rf"),
        "expected matched pattern in cancellation message: {stderr}"
    );
}

#[tokio::test]
async fn destructive_pattern_asks_the_gate_and_runs_when_approved() {
    let gate = Arc::new(RecordingGate::default());
    let cmd = bash_with(vec![], vec![vec!["true"]], gate.clone(), no_sandbox());
    let out = cmd.run(input("true && echo ran")).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"ran\n");
    let asked = gate.asked.lock();
    let [req] = asked.as_slice() else {
        panic!("expected exactly one confirmation, got {asked:?}");
    };
    assert_eq!(req.tool, "bash");
    assert_eq!(req.script, "true && echo ran");
    assert_eq!(req.matched_pattern, "true");
}

#[tokio::test]
async fn quoted_literal_does_not_trigger_destructive_pattern() {
    let cmd = bash_with(
        vec![],
        vec![vec!["rm", "-rf"]],
        Arc::new(PanicGate),
        no_sandbox(),
    );
    let out = cmd.run(input("echo \"rm -rf /\"")).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"rm -rf /\n");
}

/// Documented limitation: the destructive matcher is syntactic, so a
/// separator glued to a word (`hi;rm`) or a command substitution
/// (`$(echo rm)`) hides the command from it. The denylist and the
/// sandbox are the real defense. Pinned so a tightening of the matcher
/// is a deliberate change.
#[tokio::test]
async fn syntactic_evasions_slip_past_the_destructive_gate() {
    for template in ["echo hi;rm -rf {}", "$(echo rm) -rf {}"] {
        let scratch = tempfile::tempdir().unwrap();
        let target = scratch.path().join("file");
        std::fs::write(&target, b"x").unwrap();
        let script = template.replace("{}", &target.to_string_lossy());

        let cmd = bash_with(
            vec![],
            vec![vec!["rm", "-rf"]],
            Arc::new(PanicGate),
            no_sandbox(),
        );
        let out = cmd.run(input(&script)).await;
        assert_eq!(
            out.exit_code,
            0,
            "{script}: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!target.exists(), "{script}: the script should have run");
    }
}

#[tokio::test]
async fn bwrap_allows_writes_to_tmp() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    let unique = format!("/tmp/assistd-sandbox-test-{}", std::process::id());
    let cmd = bash_with(vec![], vec![], Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd.run(input(&format!("touch {unique} && echo ok"))).await;
    let _ = std::fs::remove_file(&unique);
    assert_eq!(
        out.exit_code,
        0,
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"ok\n");
}

#[tokio::test]
async fn bwrap_blocks_writes_to_read_only_root() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    // /usr is part of the `--ro-bind / /` mount, so any write under it
    // fails with EROFS regardless of the host user's permissions.
    let cmd = bash_with(vec![], vec![], Arc::new(AlwaysAllowGate), sandbox);
    let unique = format!("/usr/assistd-sandbox-test-{}", std::process::id());
    let out = cmd.run(input(&format!("touch {unique}"))).await;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_ne!(out.exit_code, 0, "write under /usr succeeded: {combined}");
    let lower = combined.to_ascii_lowercase();
    assert!(
        lower.contains("read-only")
            || lower.contains("permission denied")
            || lower.contains("operation not permitted"),
        "expected EROFS/EACCES-shaped error, got: {combined}"
    );
}

#[tokio::test]
async fn bwrap_unshares_pid_namespace() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    // Inside a fresh PID namespace bwrap is PID 1 and bash a small
    // number after it; host PIDs are far larger.
    let cmd = bash_with(vec![], vec![], Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd.run(input("echo $$")).await;
    assert_eq!(out.exit_code, 0);
    let pid_str = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let pid: u32 = pid_str
        .parse()
        .unwrap_or_else(|_| panic!("expected numeric PID, got {pid_str:?}"));
    assert!(pid < 100, "expected low PID inside the sandbox, got {pid}");
}

#[test]
fn probe_sandbox_auto_with_bwrap_present_resolves_to_bwrap() {
    if bwrap_or_none().is_none() {
        return;
    }
    let info = probe_sandbox(SandboxRequest::Auto, Vec::new()).expect("auto probe");
    assert!(matches!(info.mode, ResolvedSandboxMode::Bwrap { .. }));
}

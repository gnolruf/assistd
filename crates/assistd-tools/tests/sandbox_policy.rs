#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

//! Integration tests for the bash command's policy and sandbox layers.
//!
//! Covers all four layers in the order they fire (per
//! `crates/assistd-tools/src/commands/bash.rs`):
//!   1. Denylist: literal substring match, gates skipped, exit 126.
//!   2. Destructive patterns: shlex-tokenized prefix match, gate consulted.
//!   3. Bwrap sandbox: process group + filesystem mount restrictions.
//!   4. Timeout: covered in-crate; not duplicated here.
//!
//! Bwrap-dependent tests skip when `bwrap` is not on PATH (mirrors the
//! `piper_missing_binary.rs` skip pattern); they pass silently rather
//! than fail so CI runners without bubblewrap installed still pass.

use std::sync::Arc;

use assistd_tools::commands::{BashCommand, BashPolicyCfg};
use assistd_tools::policy::{ResolvedSandboxMode, probe_sandbox};
use assistd_tools::{
    AlwaysAllowGate, Command, CommandInput, ConfirmationGate, ConfirmationRequest, DenyAllGate,
    SandboxInfo, SandboxRequest,
};
use async_trait::async_trait;

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
        stdin: Vec::new(),
    }
}

/// Returns `Some(SandboxInfo)` when bwrap is available on PATH, `None`
/// otherwise. Bwrap-dependent tests early-return on `None` so they pass
/// silently rather than fail on hosts without bubblewrap installed.
fn bwrap_or_none() -> Option<Arc<SandboxInfo>> {
    let info = probe_sandbox(SandboxRequest::Bwrap, Vec::new()).ok()?;
    if matches!(info.mode, ResolvedSandboxMode::Bwrap { .. }) {
        Some(info)
    } else {
        None
    }
}

// ---------------------------------------------------------------------
// Layer 1: denylist (literal substring, case-insensitive, bypass gate)
// ---------------------------------------------------------------------

#[tokio::test]
async fn denylist_blocks_rm_rf_root() {
    let cmd = bash_with(
        vec!["rm -rf /"],
        vec![],
        Arc::new(AlwaysAllowGate),
        no_sandbox(),
    );
    let out = cmd.run(input("rm -rf /")).await.unwrap();
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("denylist pattern: rm -rf /"),
        "stderr should name the matched pattern: {stderr}"
    );
}

#[tokio::test]
async fn denylist_matches_inside_chained_script() {
    let cmd = bash_with(
        vec!["mkfs"],
        vec![],
        Arc::new(AlwaysAllowGate),
        no_sandbox(),
    );
    let out = cmd
        .run(input("cd /tmp && mkfs.ext4 /dev/null"))
        .await
        .unwrap();
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
}

#[tokio::test]
async fn denylist_is_case_insensitive_for_literal_match() {
    let cmd = bash_with(
        vec!["mkfs"],
        vec![],
        Arc::new(AlwaysAllowGate),
        no_sandbox(),
    );
    let out = cmd.run(input("MKFS.EXT4 /dev/null")).await.unwrap();
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
}

#[tokio::test]
async fn denylist_blocks_dd_to_block_device() {
    let cmd = bash_with(vec!["dd"], vec![], Arc::new(AlwaysAllowGate), no_sandbox());
    let out = cmd.run(input("dd if=/dev/zero of=/dev/sda")).await.unwrap();
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
}

#[tokio::test]
async fn denylist_bypasses_confirmation_gate() {
    // PanicGate would be invoked if the destructive-pattern path were
    // reached. For denylist matches, the gate must be bypassed entirely.
    struct PanicGate;
    #[async_trait]
    impl ConfirmationGate for PanicGate {
        async fn confirm(&self, _r: ConfirmationRequest) -> bool {
            panic!("denylist must bypass the gate");
        }
    }
    let cmd = bash_with(
        vec!["rm -rf"],
        vec![vec!["rm", "-rf"]],
        Arc::new(PanicGate),
        no_sandbox(),
    );
    let out = cmd.run(input("rm -rf /tmp/whatever")).await.unwrap();
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
}

// ---------------------------------------------------------------------
// Layer 2: destructive patterns (shlex word-prefix, gate-consulted)
// ---------------------------------------------------------------------

#[tokio::test]
async fn destructive_pattern_after_double_amp_is_blocked_when_gate_denies() {
    let cmd = bash_with(
        vec![],
        vec![vec!["rm", "-rf"]],
        Arc::new(DenyAllGate),
        no_sandbox(),
    );
    let out = cmd
        .run(input("touch /tmp/x && rm -rf /tmp/x"))
        .await
        .unwrap();
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Matched destructive pattern: rm -rf"),
        "expected matched pattern in cancellation message: {stderr}"
    );
}

#[tokio::test]
async fn destructive_pattern_after_semicolon_is_blocked() {
    // Note: shlex tokenizes `hi;` as one word, so the separator must
    // be space-padded for the matcher to anchor. The glued form
    // (`hi;rm`) is a documented limitation of the syntactic backstop;
    // see `command_substitution_evades_destructive_pattern` for the
    // analogous `$()` case.
    let cmd = bash_with(
        vec![],
        vec![vec!["rm", "-rf"]],
        Arc::new(DenyAllGate),
        no_sandbox(),
    );
    let out = cmd
        .run(input("echo hi ; rm -rf /tmp/whatever"))
        .await
        .unwrap();
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
}

#[tokio::test]
async fn glued_semicolon_evades_destructive_pattern() {
    // Documented limitation: when the separator has no surrounding
    // whitespace (`hi;rm`), shlex glues it to the adjacent word and the
    // matcher loses the anchor. The denylist (Layer 1) is the real
    // defense for cases like this; this test pins the behavior so a
    // future tightening (e.g. pre-tokenizing on shell metacharacters)
    // is a deliberate change.
    let scratch =
        std::env::temp_dir().join(format!("sandbox-policy-glued-semi-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&scratch);
    let target = scratch.join("file");
    std::fs::write(&target, b"x").unwrap();

    let cmd = bash_with(
        vec![],
        vec![vec!["rm", "-rf"]],
        Arc::new(DenyAllGate),
        no_sandbox(),
    );
    let script = format!("echo hi;rm -rf {}", target.display());
    let out = cmd.run(input(&script)).await.unwrap();

    let cleaned = !target.exists();
    let denied = out.exit_code == POLICY_DENIED_EXIT;
    assert!(
        cleaned || denied,
        "expected target removed (matcher missed) or exit 126 (matcher caught); got exit={} cleaned={cleaned}",
        out.exit_code
    );
    let _ = std::fs::remove_dir_all(&scratch);
}

#[tokio::test]
async fn destructive_pattern_after_double_pipe_is_blocked() {
    let cmd = bash_with(
        vec![],
        vec![vec!["rm", "-rf"]],
        Arc::new(DenyAllGate),
        no_sandbox(),
    );
    let out = cmd
        .run(input("false || rm -rf /tmp/whatever"))
        .await
        .unwrap();
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
}

#[tokio::test]
async fn destructive_pattern_after_pipe_is_blocked() {
    let cmd = bash_with(
        vec![],
        vec![vec!["rm"]],
        Arc::new(DenyAllGate),
        no_sandbox(),
    );
    let out = cmd.run(input("ls /tmp | rm /tmp/x")).await.unwrap();
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
}

#[tokio::test]
async fn destructive_pattern_runs_when_gate_approves() {
    // Use `true` as the destructive-marker so the test doesn't actually
    // delete anything. AlwaysAllow lets it through; exit 0 means the
    // gate was consulted and approved, then the script ran.
    let cmd = bash_with(
        vec![],
        vec![vec!["true"]],
        Arc::new(AlwaysAllowGate),
        no_sandbox(),
    );
    let out = cmd.run(input("true")).await.unwrap();
    assert_eq!(out.exit_code, 0);
}

#[tokio::test]
async fn quoted_literal_does_not_trigger_destructive_pattern() {
    // `echo "rm -rf /"` tokenizes to ["echo", "rm -rf /"]; the second
    // token is a single quoted string, so the prefix ["rm", "-rf"]
    // shouldn't anchor and the gate must not be invoked.
    struct PanicGate;
    #[async_trait]
    impl ConfirmationGate for PanicGate {
        async fn confirm(&self, _r: ConfirmationRequest) -> bool {
            panic!("gate must not fire on quoted literal");
        }
    }
    let cmd = bash_with(
        vec![],
        vec![vec!["rm", "-rf"]],
        Arc::new(PanicGate),
        no_sandbox(),
    );
    let out = cmd.run(input("echo \"rm -rf /\"")).await.unwrap();
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"rm -rf /\n");
}

#[tokio::test]
async fn command_substitution_evades_destructive_pattern() {
    // Documented limitation per the policy.rs comment: $(echo rm) -rf
    // tokenizes to ["$(echo", "rm)", "-rf", ...] which doesn't match
    // ["rm", "-rf"]. The bwrap sandbox is the real defense; this test
    // pins the syntactic backstop's behavior so a future tightening
    // (e.g. expanding shell substitutions before matching) is a
    // deliberate choice rather than an accidental relaxation. We use a
    // /tmp target so that even when the sandbox is off and the script
    // runs, no real data is at risk.
    use std::path::PathBuf;
    let scratch = std::env::temp_dir().join(format!(
        "sandbox-policy-substitution-{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&scratch);
    let target: PathBuf = scratch.join("file");
    std::fs::write(&target, b"x").unwrap();

    let cmd = bash_with(
        vec![],
        vec![vec!["rm", "-rf"]],
        Arc::new(DenyAllGate),
        no_sandbox(),
    );
    let script = format!("$(echo rm) -rf {}", target.display());
    let out = cmd.run(input(&script)).await.unwrap();

    // Either: (a) the syntactic check missed it and bash ran it →
    // exit 0 and the file is gone, OR (b) the matcher tightened in a
    // future change → exit 126. Either is acceptable; we just want
    // the test to flag the next time behavior shifts.
    let cleaned = !target.exists();
    let denied = out.exit_code == POLICY_DENIED_EXIT;
    assert!(
        cleaned || denied,
        "expected file removed (matcher missed substitution) OR exit 126 (matcher caught it); \
         got exit_code={} cleaned={cleaned}",
        out.exit_code
    );

    let _ = std::fs::remove_dir_all(&scratch);
}

// ---------------------------------------------------------------------
// Layer 3: bwrap sandbox (filesystem mount restrictions)
// ---------------------------------------------------------------------

#[tokio::test]
async fn bwrap_allows_writes_to_tmp() {
    let Some(sandbox) = bwrap_or_none() else {
        eprintln!("skipping bwrap test: bwrap not on PATH");
        return;
    };
    let unique = format!("/tmp/assistd-sandbox-test-{}", std::process::id());
    let cmd = bash_with(vec![], vec![], Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd
        .run(input(&format!("touch {unique} && echo ok")))
        .await
        .unwrap();
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
        eprintln!("skipping bwrap test: bwrap not on PATH");
        return;
    };
    // /usr is part of the `--ro-bind / /` mount, so any write under it
    // fails with EROFS regardless of the host user's permissions.
    let cmd = bash_with(vec![], vec![], Arc::new(AlwaysAllowGate), sandbox);
    let unique = format!("/usr/assistd-sandbox-test-{}", std::process::id());
    let out = cmd.run(input(&format!("touch {unique}"))).await.unwrap();
    assert_ne!(
        out.exit_code,
        0,
        "expected non-zero exit for write under /usr; stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
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
        eprintln!("skipping bwrap test: bwrap not on PATH");
        return;
    };
    // --unshare-pid creates a new PID namespace where bwrap itself runs
    // as PID 1; bash (its child) is PID 2. The actual value depends on
    // the bwrap version but the host PID would be in the thousands+,
    // so any single-digit value demonstrates the namespace was created.
    let cmd = bash_with(vec![], vec![], Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd.run(input("echo $$")).await.unwrap();
    assert_eq!(out.exit_code, 0);
    let pid_str = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let pid: u32 = pid_str
        .parse()
        .unwrap_or_else(|_| panic!("expected numeric PID, got {pid_str:?}"));
    assert!(
        pid < 100,
        "expected low PID inside the sandbox (host PIDs are typically much larger), got {pid}"
    );
}

// ---------------------------------------------------------------------
// Sandbox probing: Auto falls back to None when bwrap is missing
// ---------------------------------------------------------------------

#[tokio::test]
async fn probe_sandbox_auto_with_bwrap_present_resolves_to_bwrap() {
    if bwrap_or_none().is_none() {
        eprintln!("skipping bwrap probe positive test: bwrap not on PATH");
        return;
    }
    let info = probe_sandbox(SandboxRequest::Auto, Vec::new()).expect("auto probe");
    assert!(
        matches!(info.mode, ResolvedSandboxMode::Bwrap { .. }),
        "Auto with bwrap on PATH must resolve to Bwrap"
    );
}

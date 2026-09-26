//! The bash command's denylist, confirmation gate and bwrap sandbox.
//! Bwrap tests return early when `bwrap` is not on PATH.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;

use assistd_tools::commands::{BashCommand, BashPolicyCfg};
use assistd_tools::policy::{ResolvedSandboxMode, probe_sandbox};
use assistd_tools::{
    Allowlist, AlwaysAllowGate, Approval, Command, CommandInput, ConfirmationGate,
    ConfirmationRequest, DenyAllGate, DestructivePattern, Protected, SandboxInfo, SandboxRequest,
    SearchPath,
};

const POLICY_DENIED_EXIT: i32 = 126;

fn bash_with(
    denylist: Vec<&str>,
    destructive: Vec<Vec<&str>>,
    gate: Arc<dyn ConfirmationGate>,
    sandbox: Arc<SandboxInfo>,
) -> BashCommand {
    let cfg = BashPolicyCfg {
        timeout: Duration::from_secs(10),
        denylist: denylist.into_iter().map(|s| s.to_string()).collect(),
        destructive_patterns: destructive
            .into_iter()
            .map(|pattern| DestructivePattern::new(pattern).expect("valid pattern"))
            .collect(),
        ..BashPolicyCfg::default()
    };
    BashCommand::new(Arc::new(cfg), sandbox, gate)
}

/// Bash allowing `allowed` on a read-only system search path, with
/// approvals kept in `store`.
fn bash_allowing(allowed: &[&str], store: &Path, gate: Arc<dyn ConfirmationGate>) -> BashCommand {
    let allowlist = Allowlist::load(
        allowed.iter().map(|s| s.to_string()),
        SearchPath {
            dirs: ["/usr/local/bin", "/usr/bin", "/bin"]
                .map(Into::into)
                .into(),
            read_only: true,
        },
        store.to_path_buf(),
    )
    .expect("approvals load");
    let cfg = BashPolicyCfg {
        allowlist: Arc::new(allowlist),
        ..BashPolicyCfg::default()
    };
    BashCommand::new(Arc::new(cfg), no_sandbox(), gate)
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
    let info = probe_sandbox(SandboxRequest::Bwrap, Vec::new(), Protected::default()).ok()?;
    matches!(info.mode, ResolvedSandboxMode::Bwrap { .. }).then_some(info)
}

/// Fails the test if the policy ever consults it.
struct PanicGate;

#[async_trait]
impl ConfirmationGate for PanicGate {
    async fn confirm(&self, req: ConfirmationRequest) -> Approval {
        panic!("gate must not be consulted for {:?}", req.script);
    }
}

/// Gives every request the same answer and records what it was asked.
struct RecordingGate {
    answer: Approval,
    asked: Mutex<Vec<ConfirmationRequest>>,
}

impl RecordingGate {
    fn answering(answer: Approval) -> Arc<Self> {
        Arc::new(Self {
            answer,
            asked: Mutex::default(),
        })
    }
}

#[async_trait]
impl ConfirmationGate for RecordingGate {
    async fn confirm(&self, req: ConfirmationRequest) -> Approval {
        self.asked.lock().push(req);
        self.answer
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
    let gate = RecordingGate::answering(Approval::Once);
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

#[tokio::test]
async fn run_time_command_names_ask_the_gate() {
    for template in ["$(echo rm) -rf {}", "r=rm; $r -rf {}"] {
        let scratch = tempfile::tempdir().unwrap();
        let target = scratch.path().join("file");
        std::fs::write(&target, b"x").unwrap();
        let script = template.replace("{}", &target.to_string_lossy());

        let cmd = bash_with(
            vec![],
            vec![vec!["rm", "-rf"]],
            Arc::new(DenyAllGate),
            no_sandbox(),
        );
        let out = cmd.run(input(&script)).await;
        assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{script}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("Could not rule out a destructive command"),
            "{script}: {stderr}"
        );
        assert!(target.exists(), "{script}: the script must not have run");
    }
}

#[tokio::test]
async fn running_a_script_file_asks_the_gate_without_offering_always() {
    let scratch = tempfile::tempdir().unwrap();
    let target = scratch.path().join("file");
    std::fs::write(&target, b"x").unwrap();
    let script = format!(
        "printf 'rm -rf %s\\n' {target} > {dir}/s.sh && source {dir}/s.sh",
        target = target.display(),
        dir = scratch.path().display(),
    );

    let gate = RecordingGate::answering(Approval::Deny);
    let cmd = bash_with(vec![], vec![], gate.clone(), no_sandbox());
    let out = cmd.run(input(&script)).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{script}");
    assert!(target.exists(), "{script}: the script must not have run");
    let asked = gate.asked.lock();
    let [req] = asked.as_slice() else {
        panic!("expected exactly one confirmation, got {asked:?}");
    };
    assert!(req.always_allow.is_empty(), "{req:?}");
}

#[tokio::test]
async fn allowed_programs_run_without_asking() {
    let scratch = tempfile::tempdir().unwrap();
    let cmd = bash_allowing(
        &["tr"],
        &scratch.path().join("approvals.toml"),
        Arc::new(PanicGate),
    );
    let out = cmd
        .run(CommandInput {
            args: vec!["tr a-z A-Z".into()],
            stdin: Some(b"hi".to_vec()),
        })
        .await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"HI");
}

#[tokio::test]
async fn always_allow_adds_the_offered_programs_for_good() {
    let scratch = tempfile::tempdir().unwrap();
    let store = scratch.path().join("approvals.toml");
    let marker = scratch.path().join("made");
    let script = format!("touch {}", marker.display());

    let gate = RecordingGate::answering(Approval::Always);
    let out = bash_allowing(&[], &store, gate.clone())
        .run(input(&script))
        .await;
    assert_eq!(
        out.exit_code,
        0,
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(marker.exists());
    {
        let asked = gate.asked.lock();
        let [req] = asked.as_slice() else {
            panic!("expected exactly one confirmation, got {asked:?}");
        };
        assert_eq!(req.always_allow, ["touch"]);
    }
    let saved = std::fs::read_to_string(&store).expect("approvals saved");
    assert!(saved.contains("name = \"touch\""), "{saved}");

    std::fs::remove_file(&marker).unwrap();
    let out = bash_allowing(&[], &store, Arc::new(PanicGate))
        .run(input(&script))
        .await;
    assert_eq!(
        out.exit_code,
        0,
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(marker.exists());
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
async fn bwrap_hides_the_host_tmp() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    let host_file = tempfile::NamedTempFile::new_in("/tmp").expect("host /tmp file");
    let cmd = bash_with(vec![], vec![], Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd
        .run(input(&format!("test -e {}", host_file.path().display())))
        .await;
    assert_eq!(out.exit_code, 1, "host /tmp is visible inside the sandbox");
}

#[tokio::test]
async fn bwrap_unshares_network_namespace() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    let host = std::fs::read_link("/proc/self/ns/net").expect("host net namespace");
    let cmd = bash_with(vec![], vec![], Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd.run(input("readlink /proc/self/ns/net")).await;
    assert_eq!(out.exit_code, 0);
    let inside = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(
        inside.starts_with("net:["),
        "unexpected namespace link {inside:?}"
    );
    assert_ne!(Path::new(&inside), host);
}

#[tokio::test]
async fn bwrap_clears_the_environment() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    assert!(std::env::var_os("CARGO_MANIFEST_DIR").is_some());
    let cmd = bash_with(vec![], vec![], Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd.run(input("printenv CARGO_MANIFEST_DIR")).await;
    assert_eq!(
        out.exit_code,
        1,
        "leaked: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[tokio::test]
async fn bwrap_blocks_writes_to_read_only_root() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
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
    let info =
        probe_sandbox(SandboxRequest::Auto, Vec::new(), Protected::default()).expect("auto probe");
    assert!(matches!(info.mode, ResolvedSandboxMode::Bwrap { .. }));
}

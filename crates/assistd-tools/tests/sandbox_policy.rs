//! The bash command's denylist, confirmation gate and bwrap sandbox.
//! Bwrap tests return early when `bwrap` is not on PATH.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;

use assistd_tools::commands::{BashCommand, BashPolicyCfg, WriteCommand, WritePolicyCfg};
use assistd_tools::policy::{ToolSandbox, probe_sandbox};
use assistd_tools::{
    Allowlist, AlwaysAllowGate, Approval, Command, CommandInput, ConfirmationGate,
    ConfirmationRequest, DenyAllGate, DestructivePattern, Protected, SandboxInfo, SandboxRequest,
    SearchPath, SharedDirs,
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
        denylist: denylist.into_iter().map(ToString::to_string).collect(),
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
        allowed.iter().map(ToString::to_string),
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
    match probe_sandbox(
        SandboxRequest::Bwrap,
        Vec::new(),
        Protected::default(),
        SharedDirs::default(),
    )
    .ok()?
    {
        ToolSandbox::Bwrap(info) => Some(info),
        ToolSandbox::Disabled(_) => None,
    }
}

/// Fails the test if the policy ever consults it.
#[derive(Debug)]
struct PanicGate;

#[async_trait]
impl ConfirmationGate for PanicGate {
    async fn confirm(&self, req: ConfirmationRequest) -> Approval {
        panic!("gate must not be consulted for {:?}", req.script);
    }
}

/// Gives every request the same answer and records what it was asked.
#[derive(Debug)]
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
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("its own empty /tmp: "), "{stderr}");
}

#[tokio::test]
async fn scratch_dir_files_cross_between_write_and_bash_without_a_note() {
    let scratch = tempfile::Builder::new()
        .prefix("assistd-scratch-")
        .tempdir_in("/tmp")
        .expect("host scratch dir");
    let spill = tempfile::Builder::new()
        .prefix("assistd-spill-")
        .tempdir_in("/tmp")
        .expect("host spill dir");
    std::fs::write(spill.path().join("cmd-1.txt"), "spilled\n").expect("spill file");
    let shared = SharedDirs {
        scratch: Some(scratch.path().to_path_buf()),
        read_only: vec![spill.path().to_path_buf()],
    };
    let probed = probe_sandbox(
        SandboxRequest::Bwrap,
        Vec::new(),
        Protected::default(),
        shared,
    );
    let Ok(ToolSandbox::Bwrap(sandbox)) = probed else {
        return;
    };
    let cfg = WritePolicyCfg::new(vec![scratch.path().to_path_buf()]).expect("allowlist");
    let write = WriteCommand::new(Arc::new(cfg), Arc::new(PanicGate), sandbox.clone());
    let from_write = scratch.path().join("from-write.txt");
    let out = write
        .run(CommandInput {
            args: vec![from_write.to_string_lossy().into_owned(), "hi".into()],
            stdin: None,
        })
        .await;
    assert_eq!(out.exit_code, 0);
    assert!(
        out.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bash = bash_with(vec![], vec![], Arc::new(AlwaysAllowGate), sandbox);
    let from_bash = scratch.path().join("from-bash.txt");
    let script = format!(
        "cat {} {} > {}; touch {} 2>/dev/null || echo read-only",
        from_write.display(),
        spill.path().join("cmd-1.txt").display(),
        from_bash.display(),
        spill.path().join("cmd-2.txt").display(),
    );
    let out = bash.run(input(&script)).await;
    assert_eq!(out.stdout, b"read-only\n");
    assert!(
        out.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(&from_bash).expect("kept"), b"hispilled\n");
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

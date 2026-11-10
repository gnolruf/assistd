//! The bash command's confirmation gate and bwrap sandbox.
//! Bwrap tests return early when `bwrap` is not on PATH.

use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixListener};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;

use assistd_tools::commands::{BashCommand, BashPolicyCfg, WriteCommand, WritePolicyCfg};
use assistd_tools::policy::{ToolSandbox, probe_sandbox};
use assistd_tools::{
    Allowlist, AlwaysAllowGate, Approval, Command, CommandInput, ConfirmationGate,
    ConfirmationRequest, DenyAllGate, FileScope, Protected, SandboxInfo, SandboxRequest,
    SearchPath, SharedDirs, Unattended,
};

const POLICY_DENIED_EXIT: i32 = 126;

fn bash_with(gate: Arc<dyn ConfirmationGate>, sandbox: Arc<SandboxInfo>) -> BashCommand {
    let cfg = BashPolicyCfg {
        timeout: Duration::from_secs(10),
        ..BashPolicyCfg::default()
    };
    BashCommand::new(Arc::new(cfg), sandbox, gate)
}

/// Bash on the system search path, with "always allow" approvals kept in `store`.
fn bash_approving(store: &Path, gate: Arc<dyn ConfirmationGate>) -> BashCommand {
    let allowlist = Allowlist::load(
        SearchPath {
            dirs: ["/usr/local/bin", "/usr/bin", "/bin"]
                .map(Into::into)
                .into(),
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
        scope: FileScope::Host,
    }
}

fn bwrap_or_none() -> Option<Arc<SandboxInfo>> {
    bwrap_with(Vec::new())
}

fn bwrap_with(extra_args: Vec<String>) -> Option<Arc<SandboxInfo>> {
    match probe_sandbox(
        SandboxRequest::Bwrap,
        extra_args,
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
async fn an_unvetted_program_asks_the_gate_and_runs_when_approved() {
    let gate = RecordingGate::answering(Approval::Once);
    let cmd = bash_with(gate.clone(), no_sandbox());
    let out = cmd.run(input("nproc >/dev/null && echo ran")).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"ran\n");
    let asked = gate.asked.lock();
    let [req] = asked.as_slice() else {
        panic!("expected exactly one confirmation, got {asked:?}");
    };
    assert_eq!(req.tool, "bash");
    assert_eq!(req.script, "nproc >/dev/null && echo ran");
    assert_eq!(
        req.matched_pattern,
        "`nproc` is on neither the read-only nor the edit list"
    );
    assert_eq!(req.always_allow, ["nproc [options]"]);
}

#[tokio::test]
async fn only_edits_strictly_inside_the_edit_scope_may_run_unasked() {
    let scope = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let elsewhere = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let scope_dir = scope.path().canonicalize().unwrap();
    let elsewhere_dir = elsewhere.path().canonicalize().unwrap();
    let gate = RecordingGate::answering(Approval::Deny);
    let cfg = BashPolicyCfg {
        edit_scope: vec![scope_dir.clone()],
        ..BashPolicyCfg::default()
    };
    let cmd = BashCommand::new(Arc::new(cfg), no_sandbox(), gate.clone());
    for (script, scoped) in [
        (format!("mkdir {}/new", scope_dir.display()), true),
        (format!("mkdir {}/new", elsewhere_dir.display()), false),
        (
            format!(
                "mkdir {}/a {}/b",
                scope_dir.display(),
                elsewhere_dir.display()
            ),
            false,
        ),
        (format!("chmod -R 700 {}", scope_dir.display()), false),
        (format!("mkdir {}/repo/.git", scope_dir.display()), false),
        (
            format!(
                "printf x > {}/repo/.git/hooks/post-checkout",
                scope_dir.display()
            ),
            false,
        ),
        (
            format!(
                "cp {}/source {}/copy",
                elsewhere_dir.display(),
                scope_dir.display()
            ),
            true,
        ),
        (
            format!(
                "ln -s {}/hook {}/link; printf x > {}/link",
                elsewhere_dir.display(),
                scope_dir.display(),
                scope_dir.display()
            ),
            false,
        ),
        (
            format!(
                "cp -s {}/hook {}/link",
                elsewhere_dir.display(),
                scope_dir.display()
            ),
            false,
        ),
        (format!("ln -s ../hook {}/link", scope_dir.display()), false),
    ] {
        let out = cmd.run(input(&script)).await;
        assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{script}");
        let request = gate.asked.lock().pop().expect("an edit asks the gate");
        let is_scoped = matches!(request.unattended, Unattended::ScopedEdit);
        assert_eq!(is_scoped, scoped, "{script}: {:?}", request.unattended);
        if !scoped {
            assert!(
                matches!(request.unattended, Unattended::Judged(_)),
                "{script}: {:?}",
                request.unattended
            );
        }
    }
}

#[tokio::test]
async fn a_redirect_through_a_symlink_in_scratch_is_an_edit_of_where_it_leads() {
    let scratch = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let elsewhere = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let scratch_dir = scratch.path().canonicalize().unwrap();
    std::os::unix::fs::symlink(elsewhere.path(), scratch_dir.join("link")).unwrap();
    let gate = RecordingGate::answering(Approval::Deny);
    let cfg = BashPolicyCfg {
        scratch: Some(scratch_dir.clone()),
        ..BashPolicyCfg::default()
    };
    let cmd = BashCommand::new(Arc::new(cfg), no_sandbox(), gate.clone());

    let out = cmd
        .run(input(&format!("echo x > {}/plain", scratch_dir.display())))
        .await;
    assert_eq!(out.exit_code, 0, "{}", String::from_utf8_lossy(&out.stderr));
    assert!(gate.asked.lock().is_empty());

    let out = cmd
        .run(input(&format!(
            "echo x > {}/link/hook",
            scratch_dir.display()
        )))
        .await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert_eq!(gate.asked.lock().len(), 1);
    assert!(!elsewhere.path().join("hook").exists());
}

#[tokio::test]
async fn run_time_command_names_ask_the_gate() {
    for template in ["$(echo rm) -rf {}", "r=rm; $r -rf {}"] {
        let scratch = tempfile::tempdir().unwrap();
        let target = scratch.path().join("file");
        std::fs::write(&target, b"x").unwrap();
        let script = template.replace("{}", &target.to_string_lossy());

        let cmd = bash_with(Arc::new(DenyAllGate), no_sandbox());
        let out = cmd.run(input(&script)).await;
        assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{script}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("It could not be vetted"),
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
    let cmd = bash_with(gate.clone(), no_sandbox());
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
async fn read_only_programs_run_without_asking() {
    let cmd = bash_with(Arc::new(PanicGate), no_sandbox());
    let out = cmd
        .run(CommandInput {
            args: vec!["tr a-z A-Z".into()],
            stdin: Some(b"hi".to_vec()),
            scope: FileScope::Host,
        })
        .await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"HI");
}

#[tokio::test]
async fn always_allow_adds_the_offered_programs_for_good() {
    let scratch = tempfile::tempdir().unwrap();
    let store = scratch.path().join("approvals.toml");
    let script = "nproc --all";

    let gate = RecordingGate::answering(Approval::Always);
    let out = bash_approving(&store, gate.clone())
        .run(input(script))
        .await;
    assert_eq!(
        out.exit_code,
        0,
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    {
        let asked = gate.asked.lock();
        let [req] = asked.as_slice() else {
            panic!("expected exactly one confirmation, got {asked:?}");
        };
        assert_eq!(req.always_allow, ["nproc [options]"]);
    }
    let saved = std::fs::read_to_string(&store).expect("approvals saved");
    assert!(saved.contains("name = \"nproc\""), "{saved}");

    let out = bash_approving(&store, Arc::new(PanicGate))
        .run(input(script))
        .await;
    assert_eq!(
        out.exit_code,
        0,
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.stdout.is_empty());
}

#[tokio::test]
async fn bwrap_hides_the_host_tmp() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    let host_file = tempfile::NamedTempFile::new_in("/tmp").expect("host /tmp file");
    let cmd = bash_with(Arc::new(AlwaysAllowGate), sandbox);
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
            scope: FileScope::Host,
        })
        .await;
    assert_eq!(out.exit_code, 0);
    assert!(
        out.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bash = bash_with(Arc::new(AlwaysAllowGate), sandbox);
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
    let cmd = bash_with(Arc::new(AlwaysAllowGate), sandbox);
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
async fn bwrap_commands_run_in_a_memory_capped_scope_as_first_oom_victims() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    let cmd = bash_with(Arc::new(AlwaysAllowGate), sandbox);
    let script = "cat /proc/self/oom_score_adj \
                  /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max \
                  /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/pids.max";
    let out = cmd.run(input(script)).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.exit_code, 0, "{stderr}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        ["1000", "4294967296", "512"]
    );
}

/// Returns early without `python3`; skips the networked sandbox without
/// Landlock scoping.
#[tokio::test]
async fn bwrap_cannot_reach_a_host_unix_socket_on_a_visible_path() {
    if !Path::new("/usr/bin/python3").exists() {
        return;
    }
    for extra_args in [Vec::new(), vec!["--share-net".to_string()]] {
        if let Some(sandbox) = bwrap_with(extra_args) {
            assert_host_socket_unreachable(sandbox).await;
        }
    }
}

async fn assert_host_socket_unreachable(sandbox: Arc<SandboxInfo>) {
    let dir = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).expect("socket dir");
    let path = dir.path().join("agent.sock");
    let listener = UnixListener::bind(&path).expect("host socket");
    listener.set_nonblocking(true).expect("nonblocking socket");
    let probe = format!(
        "python3 -I -c 'import socket; socket.socket(socket.AF_UNIX).connect(\"{}\"); \
         print(\"connected\")'",
        path.display()
    );
    let cmd = bash_with(Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd.run(input(&probe)).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.stdout.is_empty(), "reached a host Unix socket");
    assert!(stderr.contains("PermissionError"), "{stderr}");
    assert!(
        listener.accept().is_err(),
        "the sandbox reached a host socket"
    );
}

/// Returns early without `python3` or Landlock scoping.
#[tokio::test]
async fn networked_bwrap_cannot_reach_abstract_sockets() {
    if !Path::new("/usr/bin/python3").exists() {
        return;
    }
    let Some(sandbox) = bwrap_with(vec!["--share-net".into()]) else {
        return;
    };
    let name = format!("assistd-abstract-test-{}", std::process::id());
    let addr = SocketAddr::from_abstract_name(&name).expect("abstract address");
    let _listener = UnixListener::bind_addr(&addr).expect("abstract listener");
    let probe = format!(
        "python3 -I -c 'import socket; socket.socket(socket.AF_UNIX).connect(\"\\0{name}\"); \
         print(\"connected\")'"
    );
    let cmd = bash_with(Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd.run(input(&probe)).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.stdout.is_empty(), "reached the host's abstract socket");
    assert!(stderr.contains("PermissionError"), "{stderr}");
}

#[tokio::test]
async fn bwrap_blocks_writes_under_usr_and_etc() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    let cmd = bash_with(Arc::new(AlwaysAllowGate), sandbox);
    for dir in ["/usr", "/etc"] {
        let unique = format!("{dir}/assistd-sandbox-test-{}", std::process::id());
        let out = cmd.run(input(&format!("touch {unique}"))).await;
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert_ne!(out.exit_code, 0, "write under {dir} succeeded: {combined}");
        let lower = combined.to_ascii_lowercase();
        assert!(
            lower.contains("read-only")
                || lower.contains("permission denied")
                || lower.contains("operation not permitted"),
            "expected EROFS/EACCES-shaped error under {dir}, got: {combined}"
        );
    }
}

#[tokio::test]
async fn bwrap_dev_holds_no_block_devices() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    let cmd = bash_with(Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd.run(input("find /dev -type b")).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.exit_code, 0, "{stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "", "{stderr}");
}

#[tokio::test]
async fn bwrap_proc_shows_only_the_sandbox_processes() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    let cmd = bash_with(Arc::new(AlwaysAllowGate), sandbox);
    let list_commands = "for p in /proc/[0-9]*; do read -r c < \"$p/comm\" && echo \"$c\"; done";
    let out = cmd.run(input(list_commands)).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.exit_code, 0, "{stderr}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.lines().any(|name| name == "bash"), "{stdout}");
    assert!(
        stdout.lines().all(|name| ["bwrap", "bash"].contains(&name)),
        "a process outside the sandbox is visible: {stdout}"
    );
}

#[tokio::test]
async fn bwrap_run_holds_no_dbus_socket() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    let cmd = bash_with(Arc::new(AlwaysAllowGate), sandbox);
    let out = cmd.run(input("find /run -type s")).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.exit_code, 0, "{stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "", "{stderr}");
}

/// The bomb's depth is bounded, so it stays harmless if the cap regresses.
#[tokio::test]
async fn bwrap_caps_a_renamed_fork_bomb() {
    let Some(sandbox) = bwrap_or_none() else {
        return;
    };
    let cmd = bash_with(Arc::new(AlwaysAllowGate), sandbox);
    let bomb = "f(){ if (( $1 < 8 )); then f $(($1+1)) | f $(($1+1)) & wait; else sleep 1; fi; }; \
                f 0; echo ended";
    let out = cmd.run(input(bomb)).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.stdout, b"ended\n", "exit {}: {stderr}", out.exit_code);
    assert!(
        stderr.contains("fork: retry: Resource temporarily unavailable"),
        "{stderr}"
    );
}

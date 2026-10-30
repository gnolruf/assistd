//! A relay's quarantine: what its commands can read and reach.
//! Returns early when `bwrap` is not on PATH.

use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::json;

use assistd_tools::commands::BashPolicyCfg;
use assistd_tools::policy::{ToolSandbox, probe_sandbox};
use assistd_tools::{
    AlwaysAllowGate, ApprovalGate, Approvals, DenyAllGate, Egress, OutputCap, PresentSpec,
    Protected, Quarantine, RunTool, SandboxInfo, SandboxRequest, SharedDirs, Tool,
};

const HOST_ONLY: &str = "host-home-only";
const INPUT: &str = "handed-over-input";
const PYTHON: &str = "/usr/bin/python3";

fn bwrap_sandbox() -> Option<Arc<SandboxInfo>> {
    match probe_sandbox(
        SandboxRequest::Bwrap,
        Vec::new(),
        Protected::default(),
        SharedDirs::default(),
    ) {
        Ok(ToolSandbox::Bwrap(sandbox)) => Some(sandbox),
        _ => None,
    }
}

fn exec() -> PathBuf {
    env!("CARGO_BIN_EXE_assistd-quarantine-exec").into()
}

fn run_tool(quarantine: &Quarantine) -> RunTool {
    let commands = quarantine.commands(&BashPolicyCfg::default(), Arc::new(AlwaysAllowGate));
    let output_cap = Arc::new(OutputCap::new(PresentSpec {
        overflow_dir: quarantine.output_dir(),
        ..PresentSpec::default()
    }));
    RunTool::new(Arc::new(commands), output_cap)
}

async fn run(tool: &RunTool, command: &str) -> (i32, String) {
    let out = tool
        .invoke(json!({ "command": command }))
        .await
        .expect("run accepts a command");
    (out.exit_code, out.output)
}

#[tokio::test]
async fn input_dirs_are_read_in_place_but_their_sockets_are_out_of_reach() {
    let Some(sandbox) = bwrap_sandbox() else {
        return;
    };
    let project = tempfile::tempdir().expect("input dir");
    std::fs::create_dir(project.path().join("src")).expect("nested dir");
    std::fs::write(project.path().join("src/notes.txt"), INPUT).expect("input");
    let socket = UnixListener::bind(project.path().join("agent.sock")).expect("host socket");
    socket.set_nonblocking(true).expect("nonblocking socket");
    let relays = tempfile::tempdir().expect("relay parent");

    let quarantine = Quarantine::create(
        relays.path(),
        &sandbox,
        exec(),
        vec![project.path().to_path_buf()],
        None,
    )
    .await
    .expect("quarantine");
    let tool = run_tool(&quarantine);
    let name = project.path().file_name().unwrap().to_string_lossy();
    let mounted = quarantine.input_dir().join(&*name);
    let mounted = mounted.display();

    for command in [
        format!("cat {mounted}/src/notes.txt"),
        format!("cat ../input/{name}/src/notes.txt"),
        format!("grep -r {INPUT} {mounted}"),
        format!("bash \"cat {mounted}/src/notes.txt\""),
    ] {
        let (exit_code, output) = run(&tool, &command).await;
        assert_eq!(exit_code, 0, "{command}: {output}");
        assert!(output.contains(INPUT), "{command}: {output}");
    }
    if Path::new(PYTHON).exists() {
        let connect = format!(
            "bash \"{PYTHON} -I -c 'import socket; \
             socket.socket(socket.AF_UNIX).connect(\\\"{mounted}/agent.sock\\\")'\""
        );
        let (exit_code, output) = run(&tool, &connect).await;
        assert_ne!(exit_code, 0, "{output}");
        assert!(output.contains("PermissionError"), "{output}");
        assert!(
            socket.accept().is_err(),
            "the sandbox reached a host socket"
        );
    }

    drop((tool, quarantine));
    let left = std::fs::read_to_string(project.path().join("src/notes.txt"));
    assert_eq!(left.ok().as_deref(), Some(INPUT), "the input was harmed");
}

#[tokio::test]
async fn quarantined_bash_and_cat_read_inputs_but_not_the_host_home() {
    let Some(sandbox) = bwrap_sandbox() else {
        return;
    };
    let home = std::env::var("HOME").expect("HOME is set");
    let host = tempfile::Builder::new()
        .prefix(".assistd-quarantine-test-")
        .tempdir_in(home)
        .expect("dir in the host home");
    let host_only = host.path().join("host-only.txt");
    std::fs::write(&host_only, HOST_ONLY).expect("host-only file");
    let input = host.path().join("notes.txt");
    std::fs::write(&input, INPUT).expect("input");
    let relays = tempfile::tempdir().expect("relay parent");

    let quarantine = Quarantine::create(relays.path(), &sandbox, exec(), vec![input], None)
        .await
        .expect("quarantine");
    let tool = run_tool(&quarantine);
    let host_only = host_only.display();
    let handed_over = quarantine.input_dir().join("notes.txt");
    let handed_over = handed_over.display();

    for command in [
        format!("cat {host_only}"),
        format!("bash \"cat {host_only}\""),
        format!("bash \"ln -s {host_only} link\" && cat ~/link"),
        format!(
            "cat {}/{}{}",
            quarantine.home_dir().display(),
            "../".repeat(32),
            host_only.to_string().trim_start_matches('/')
        ),
    ] {
        let (exit_code, output) = run(&tool, &command).await;
        assert_ne!(exit_code, 0, "{command}: {output}");
        assert!(!output.contains(HOST_ONLY), "{command}: {output}");
    }
    for command in [
        format!("cat {handed_over}"),
        format!("bash \"cat {handed_over}\""),
    ] {
        let (exit_code, output) = run(&tool, &command).await;
        assert_eq!(exit_code, 0, "{command}: {output}");
        assert!(output.contains(INPUT), "{command}: {output}");
    }

    let (exit_code, output) = run(
        &tool,
        "bash \"mkdir -p d/e && touch d/e/x && chmod 500 d/e d\"",
    )
    .await;
    assert_eq!(exit_code, 0, "{output}");
    let relay = quarantine.home_dir();
    drop((tool, quarantine));
    assert!(
        !relay.exists(),
        "the relay directory outlives its quarantine"
    );
}

#[tokio::test]
async fn relay_egress_refuses_unapproved_hosts_loopback_and_plain_http() {
    let Some(sandbox) = bwrap_sandbox() else {
        return;
    };
    let relays = tempfile::tempdir().expect("relay parent");
    let egress = Egress {
        hosts: ApprovalGate::new(Arc::new(DenyAllGate), Arc::new(Approvals::unsaved())),
        search: None,
    };
    let quarantine = Quarantine::create(relays.path(), &sandbox, exec(), Vec::new(), Some(egress))
        .await
        .expect("quarantine");
    let tool = run_tool(&quarantine);

    for (target, refusal) in [
        ("example.com:443", "did not approve example.com"),
        ("127.0.0.1:443", "127.0.0.1 is not a public address"),
        (
            "example.com:80",
            "port 80 is not one a sub-agent may connect to",
        ),
    ] {
        let command = format!(
            "bash 'exec 3<>/dev/tcp/127.0.0.1/${{HTTPS_PROXY##*:}} && \
             printf \"CONNECT {target} HTTP/1.1\\r\\n\\r\\n\" >&3 && cat <&3'"
        );
        let (exit_code, output) = run(&tool, &command).await;
        assert_eq!(exit_code, 0, "{target}: {output}");
        assert!(output.contains("403 Forbidden"), "{target}: {output}");
        assert!(output.contains(refusal), "{target}: {output}");
    }

    let (exit_code, output) = run(&tool, "web fetch http://example.com/").await;
    assert_eq!(exit_code, 126, "{output}");
    assert!(output.contains("only fetch https://"), "{output}");
}

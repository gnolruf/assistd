//! A relay's quarantine: what its commands can read and reach.
//! Returns early when `bwrap` is not on PATH.

use std::net::TcpListener;
use std::os::unix::net::UnixListener;
use std::path::Path;
use std::sync::Arc;

use serde_json::json;

use assistd_tools::commands::BashPolicyCfg;
use assistd_tools::policy::{ToolSandbox, probe_sandbox};
use assistd_tools::{
    AlwaysAllowGate, ApprovalGate, Approvals, DenyAllGate, Egress, OutputCap, PresentSpec,
    Protected, Quarantine, RunTool, SandboxInfo, SandboxRequest, SharedDirs, Tool, TurnContext,
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
        .invoke(&TurnContext::default(), json!({ "command": command }))
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
        let io_uring_setup = format!(
            "bash \"{PYTHON} -I -c 'import ctypes; libc = ctypes.CDLL(None, use_errno=True); \
             print(libc.syscall(425, 8, ctypes.create_string_buffer(120)), ctypes.get_errno())'\""
        );
        let (exit_code, output) = run(&tool, &io_uring_setup).await;
        assert_eq!(exit_code, 0, "{output}");
        assert!(output.contains("-1 38"), "io_uring is open: {output}");
    }

    drop((tool, quarantine));
    let left = std::fs::read_to_string(project.path().join("src/notes.txt"));
    assert_eq!(left.ok().as_deref(), Some(INPUT), "the input was harmed");
}

#[tokio::test]
async fn an_input_replaced_on_the_host_after_handover_stays_the_one_handed_over() {
    let Some(sandbox) = bwrap_sandbox() else {
        return;
    };
    let parent = tempfile::tempdir().expect("input parent");
    let project = parent.path().join("project");
    std::fs::create_dir(&project).expect("input dir");
    std::fs::write(project.join("notes.txt"), INPUT).expect("input");
    let relays = tempfile::tempdir().expect("relay parent");

    let quarantine = Quarantine::create(relays.path(), &sandbox, vec![project.clone()], None)
        .await
        .expect("quarantine");
    std::fs::rename(&project, parent.path().join("moved")).expect("move the input away");
    std::fs::create_dir(&project).expect("replacement dir");
    std::fs::write(project.join("notes.txt"), HOST_ONLY).expect("replacement");
    let tool = run_tool(&quarantine);
    let mounted = quarantine.input_dir().join("project/notes.txt");
    let mounted = mounted.display();

    for command in [format!("cat {mounted}"), format!("bash \"cat {mounted}\"")] {
        let (exit_code, output) = run(&tool, &command).await;
        assert_eq!(exit_code, 0, "{command}: {output}");
        assert!(output.contains(INPUT), "{command}: {output}");
        assert!(!output.contains(HOST_ONLY), "{command}: {output}");
    }
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

    let quarantine = Quarantine::create(relays.path(), &sandbox, vec![input], None)
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
async fn a_web_relay_reaches_the_network_only_through_web() {
    let Some(sandbox) = bwrap_sandbox() else {
        return;
    };
    let host = TcpListener::bind("127.0.0.1:0").expect("host listener");
    host.set_nonblocking(true).expect("nonblocking listener");
    let port = host.local_addr().expect("listener address").port();
    let relays = tempfile::tempdir().expect("relay parent");
    let egress = Egress {
        hosts: ApprovalGate::new(Arc::new(DenyAllGate), Arc::new(Approvals::unsaved())),
        search: None,
    };
    let quarantine = Quarantine::create(relays.path(), &sandbox, Vec::new(), Some(egress))
        .await
        .expect("quarantine");
    let tool = run_tool(&quarantine);

    for target in ["1.1.1.1/443".to_string(), format!("127.0.0.1/{port}")] {
        let command = format!("bash 'exec 3<>/dev/tcp/{target}'");
        let (exit_code, output) = run(&tool, &command).await;
        assert_ne!(exit_code, 0, "{target}: {output}");
    }
    assert!(
        host.accept().is_err(),
        "the sandbox reached the host network"
    );

    let (exit_code, output) = run(&tool, "web fetch http://example.com/").await;
    assert_eq!(exit_code, 126, "{output}");
    assert!(output.contains("only fetch https://"), "{output}");
}

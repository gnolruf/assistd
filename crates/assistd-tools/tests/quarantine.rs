//! A relay's quarantine: what its commands can and cannot read.
//! Returns early when `bwrap` is not on PATH.

use std::sync::Arc;

use serde_json::json;

use assistd_tools::commands::BashPolicyCfg;
use assistd_tools::policy::{ToolSandbox, probe_sandbox};
use assistd_tools::{
    AlwaysAllowGate, OutputCap, PresentSpec, Protected, Quarantine, RunTool, SandboxRequest,
    SharedDirs, Tool,
};

const HOST_ONLY: &str = "host-home-only";
const INPUT: &str = "handed-over-input";

async fn run(tool: &RunTool, command: &str) -> (i32, String) {
    let out = tool
        .invoke(json!({ "command": command }))
        .await
        .expect("run accepts a command");
    (out.exit_code, out.output)
}

#[tokio::test]
async fn quarantined_bash_and_cat_read_inputs_but_not_the_host_home() {
    let Ok(ToolSandbox::Bwrap(sandbox)) = probe_sandbox(
        SandboxRequest::Bwrap,
        Vec::new(),
        Protected::default(),
        SharedDirs::default(),
    ) else {
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

    let quarantine = Quarantine::create(relays.path(), &sandbox, vec![input])
        .await
        .expect("quarantine");
    let commands = quarantine.commands(&BashPolicyCfg::default(), Arc::new(AlwaysAllowGate));
    let output_cap = Arc::new(OutputCap::new(PresentSpec {
        overflow_dir: quarantine.output_dir(),
        ..PresentSpec::default()
    }));
    let tool = RunTool::new(Arc::new(commands), output_cap);
    let host_only = host_only.display();
    let copied = quarantine.input_dir().join("notes.txt");
    let copied = copied.display();

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
    for command in [format!("cat {copied}"), format!("bash \"cat {copied}\"")] {
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

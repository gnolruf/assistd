//! End-to-end stdio-transport tests against the in-tree
//! `fake_mcp_server` binary.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use assistd_mcp::{
    HealthState, McpClient, McpError, McpServerHandle, StdioConfig, StdioMcpClient, ToolResult,
    TransportConfig, adapt_handle_as_tools, mcp_error_line,
};
use assistd_tools::presentation::PresentSpec;
use assistd_tools::{AlwaysAllowGate, ApprovalGate, Approvals, Tool};
use serde_json::json;
use tokio::sync::watch;

fn fake_server_path() -> String {
    env!("CARGO_BIN_EXE_fake_mcp_server").to_string()
}

fn make_stdio_config(label: &str) -> TransportConfig {
    let mut cfg = StdioConfig::new(label, fake_server_path());
    cfg.request_timeout = Duration::from_secs(5);
    TransportConfig::Stdio(cfg)
}

/// [`adapt_handle_as_tools`] with every call allowed.
async fn adapt_allowing(handle: &McpServerHandle) -> Result<Vec<Box<dyn Tool>>, McpError> {
    let approvals = ApprovalGate::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved()));
    adapt_handle_as_tools(handle, "mcp__fake", PresentSpec::default(), &approvals).await
}

/// Resolves once the supervisor exits and drops its health sender.
async fn supervisor_exited(health_rx: &mut watch::Receiver<HealthState>) {
    while health_rx.changed().await.is_ok() {}
}

/// True once `pid` no longer exists or is a zombie awaiting its reaper.
fn process_is_gone(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Err(_) => true,
        Ok(stat) => stat
            .rsplit(')')
            .next()
            .is_none_or(|fields| fields.trim_start().starts_with('Z')),
    }
}

async fn wait_until_process_is_gone(pid: u32) -> bool {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !process_is_gone(pid) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .is_ok()
}

fn read_pid_file(path: &Path) -> u32 {
    std::fs::read_to_string(path)
        .expect("fixture wrote its orphan's pid")
        .trim()
        .parse()
        .expect("pid file holds a pid")
}

#[tokio::test]
async fn discovers_and_invokes_a_tool_end_to_end() {
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");

    let tools = adapt_allowing(&handle)
        .await
        .expect("discovery should succeed");
    let names: Vec<&str> = tools.iter().map(|tool| tool.name()).collect();
    assert_eq!(
        names,
        [
            "mcp__fake__echo",
            "mcp__fake__crash_me",
            "mcp__fake__oversize_reply",
            "mcp__fake__close_stdout",
            "mcp__fake__spawn_orphan_and_crash",
            "mcp__fake__env_names",
        ]
    );

    let echo = tools
        .iter()
        .find(|tool| tool.name() == "mcp__fake__echo")
        .unwrap();
    let result = echo.invoke(json!({"msg": "hi"})).await.unwrap();
    assert_eq!(result["type"], "text");
    assert_eq!(result["output"], "echo:hi");
    assert_eq!(result["exit_code"], 0);

    handle.shutdown().await;
}

#[tokio::test]
async fn external_shutdown_stops_the_supervisor() {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");
    let mut health_rx = handle.watch_health();

    shutdown_tx.send(true).unwrap();

    tokio::time::timeout(Duration::from_secs(5), supervisor_exited(&mut health_rx))
        .await
        .expect("supervisor must exit on daemon-wide shutdown");
    let err = handle.client().list_tools().await.unwrap_err();
    assert!(matches!(err, McpError::ServerDown), "{err}");

    tokio::time::timeout(Duration::from_secs(5), handle.shutdown())
        .await
        .expect("shutdown of an exited supervisor must return promptly");
}

#[tokio::test]
async fn dropping_handle_without_shutdown_aborts_supervisor() {
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");

    let mut health_rx = handle.watch_health();
    drop(handle);

    tokio::time::timeout(Duration::from_secs(2), supervisor_exited(&mut health_rx))
        .await
        .expect("supervisor must release health_tx within 2s after Drop");
}

#[tokio::test]
async fn server_crash_short_circuits_subsequent_calls() {
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");

    let tools = adapt_allowing(&handle)
        .await
        .expect("discovery should succeed");
    let echo = tools
        .iter()
        .find(|tool| tool.name() == "mcp__fake__echo")
        .expect("echo present");
    let crasher = tools
        .iter()
        .find(|tool| tool.name() == "mcp__fake__crash_me")
        .expect("crash_me present");

    let pre = echo.invoke(json!({"msg": "before"})).await.unwrap();
    assert_eq!(pre["output"], "echo:before");

    let _ = crasher.invoke(json!({})).await;

    let mut watch_health = handle.watch_health();
    let _ = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if *watch_health.borrow() != HealthState::Healthy {
                return;
            }
            let _ = watch_health.changed().await;
        }
    })
    .await;
    assert_ne!(
        handle.health(),
        HealthState::Healthy,
        "supervisor should have flipped health off Healthy after server exit"
    );

    let post = tokio::time::timeout(Duration::from_secs(2), echo.invoke(json!({"msg": "after"})))
        .await
        .expect("invoke must not hang on a dead server")
        .expect("invoke returns Ok with a typed error JSON");
    assert_eq!(post["type"], "error");
    assert_eq!(post["exit_code"], -1);
    assert_eq!(post["server_name"], "fake");
    assert_eq!(
        post["output"],
        mcp_error_line("mcp__fake__echo", &McpError::ServerDown)
    );

    handle.shutdown().await;
}

#[tokio::test]
async fn oversize_reply_fails_its_call_without_restarting_the_server() {
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");
    let watch_health = handle.watch_health();
    let client = handle.client();

    let (oversized, echoed) = tokio::join!(
        client.invoke("oversize_reply", json!({})),
        client.invoke("echo", json!({"msg": "alongside"})),
    );

    let err = oversized.expect_err("an oversize reply must fail its call");
    assert!(matches!(err, McpError::ReplyTooLarge { .. }), "{err}");
    assert!(
        matches!(echoed, Ok(ToolResult::Text(ref text)) if text == "echo:alongside"),
        "{echoed:?}"
    );
    let after = client.invoke("echo", json!({"msg": "after"})).await;
    assert!(
        matches!(after, Ok(ToolResult::Text(ref text)) if text == "echo:after"),
        "{after:?}"
    );
    assert!(
        !watch_health.has_changed().unwrap(),
        "an oversize reply must not restart the server"
    );

    handle.shutdown().await;
}

#[tokio::test]
async fn dead_read_loop_under_a_live_child_is_noticed_and_restarted() {
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");

    let tools = adapt_allowing(&handle)
        .await
        .expect("discovery should succeed");
    let echo = tools
        .iter()
        .find(|tool| tool.name() == "mcp__fake__echo")
        .expect("echo present");
    let close_stdout = tools
        .iter()
        .find(|tool| tool.name() == "mcp__fake__close_stdout")
        .expect("close_stdout present");

    let mut watch_health = handle.watch_health();

    let _ = close_stdout.invoke(json!({})).await;

    let flipped = tokio::time::timeout(Duration::from_secs(5), async {
        while watch_health.changed().await.is_ok() {
            if *watch_health.borrow_and_update() != HealthState::Healthy {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    assert!(
        flipped,
        "supervisor must leave Healthy when the read loop dies under a live child"
    );

    let recovered = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if *watch_health.borrow_and_update() == HealthState::Healthy {
                return true;
            }
            if watch_health.changed().await.is_err() {
                return false;
            }
        }
    })
    .await
    .unwrap_or(false);
    assert!(recovered, "supervisor must restart the server");

    let post = tokio::time::timeout(Duration::from_secs(5), echo.invoke(json!({"msg": "after"})))
        .await
        .expect("post-restart invoke must not hang")
        .expect("post-restart invoke should succeed");
    assert_eq!(post["output"], "echo:after");

    handle.shutdown().await;
}

#[tokio::test]
async fn crashed_server_takes_its_process_group_with_it() {
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");
    let tools = adapt_allowing(&handle)
        .await
        .expect("discovery should succeed");
    let spawner = tools
        .iter()
        .find(|tool| tool.name() == "mcp__fake__spawn_orphan_and_crash")
        .expect("spawn_orphan_and_crash present");

    let result = spawner.invoke(json!({})).await.unwrap();
    let orphan: u32 = result["output"]
        .as_str()
        .and_then(|pid| pid.parse().ok())
        .unwrap_or_else(|| panic!("fixture should answer with its orphan's pid: {result}"));

    assert!(
        wait_until_process_is_gone(orphan).await,
        "grandchild {orphan} must be killed with the crashed server's process group"
    );
    handle.shutdown().await;
}

#[tokio::test]
async fn failed_initialize_kills_the_process_group() {
    let pid_dir = tempfile::tempdir().unwrap();
    let pid_file = pid_dir.path().join("orphan.pid");
    let mut cfg = StdioConfig::new("fake", fake_server_path());
    cfg.request_timeout = Duration::from_secs(5);
    cfg.env.insert(
        "FAKE_MCP_FAIL_INIT_WITH_ORPHAN_PID_FILE".into(),
        pid_file.to_string_lossy().into_owned(),
    );
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);

    let err = McpServerHandle::start("fake".into(), TransportConfig::Stdio(cfg), shutdown_rx)
        .await
        .expect_err("fixture refuses initialize");
    assert!(
        matches!(err, McpError::RpcError { code: -32000, .. }),
        "{err}"
    );

    let orphan = read_pid_file(&pid_file);
    assert!(
        wait_until_process_is_gone(orphan).await,
        "grandchild {orphan} must be killed when the handshake fails"
    );
}

#[tokio::test]
async fn server_sees_only_inherited_and_configured_environment() {
    assert!(
        std::env::var_os("CARGO_MANIFEST_DIR").is_some(),
        "cargo sets CARGO_MANIFEST_DIR for test processes"
    );
    let mut cfg = StdioConfig::new("fake", fake_server_path());
    cfg.request_timeout = Duration::from_secs(5);
    cfg.env.insert("CONFIGURED_TOKEN".into(), "secret".into());
    let (client, lifeline) = StdioMcpClient::spawn(cfg)
        .await
        .expect("server should start");

    let result = client.invoke("env_names", json!({})).await.unwrap();
    let ToolResult::Text(listing) = result else {
        panic!("expected a text listing, got {result:?}");
    };
    let names: Vec<&str> = listing.lines().collect();
    assert!(names.contains(&"PATH"), "{names:?}");
    assert!(names.contains(&"CONFIGURED_TOKEN"), "{names:?}");
    assert!(!names.contains(&"CARGO_MANIFEST_DIR"), "{names:?}");

    lifeline.shutdown(Duration::from_secs(1)).await;
}

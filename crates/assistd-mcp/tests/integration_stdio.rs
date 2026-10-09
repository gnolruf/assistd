//! End-to-end stdio-transport tests against the in-tree
//! `fake_mcp_server` binary.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use assistd_mcp::{McpClient, McpError, McpServer, StdioConfig, adapt_client_as_tools};
use assistd_tools::presentation::PresentSpec;
use assistd_tools::{AlwaysAllowGate, ApprovalGate, Approvals, Tool, VisionGate};
use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::json;

fn fake_server_path() -> String {
    env!("CARGO_BIN_EXE_fake_mcp_server").to_string()
}

fn make_stdio_config(label: &str) -> StdioConfig {
    let mut cfg = StdioConfig::new(label, fake_server_path());
    cfg.request_timeout = Duration::from_secs(5);
    cfg
}

async fn start_fake() -> Arc<McpServer> {
    let server = McpServer::start("fake".into(), make_stdio_config("fake"))
        .await
        .expect("server should start");
    Arc::new(server)
}

/// [`adapt_client_as_tools`] with every call allowed.
async fn adapt_allowing(server: &Arc<McpServer>) -> Vec<Box<dyn Tool>> {
    let approvals = ApprovalGate::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved()));
    adapt_client_as_tools(
        server.clone(),
        "fake",
        PresentSpec::default(),
        &approvals,
        &VisionGate::new(true),
    )
    .await
    .expect("discovery should succeed")
}

fn find<'a>(tools: &'a [Box<dyn Tool>], name: &str) -> &'a dyn Tool {
    tools
        .iter()
        .find(|tool| tool.name() == name)
        .unwrap_or_else(|| panic!("{name} present"))
        .as_ref()
}

/// Calls `echo` until it answers again, as it will once a later call
/// has restarted the server past its backoff.
async fn echo_answers_again(echo: &dyn Tool) -> bool {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let result = echo.invoke(json!({"msg": "after"})).await.unwrap();
            if result["output"] == "echo:after" {
                return;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    })
    .await
    .is_ok()
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

/// The text of a result's first content block.
fn first_text(result: &CallToolResult) -> Option<&str> {
    match result.content.first() {
        Some(ContentBlock::Text(text)) => Some(&text.text),
        _ => None,
    }
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
    let server = start_fake().await;
    let tools = adapt_allowing(&server).await;
    let names: Vec<&str> = tools.iter().map(|tool| tool.name()).collect();
    assert_eq!(
        names,
        [
            "mcp__fake__echo",
            "mcp__fake__crash_me",
            "mcp__fake__never_answers",
            "mcp__fake__close_stdout",
            "mcp__fake__spawn_orphan_and_crash",
            "mcp__fake__env_names",
        ]
    );

    let result = find(&tools, "mcp__fake__echo")
        .invoke(json!({"msg": "hi"}))
        .await
        .unwrap();
    assert_eq!(result["type"], "text");
    assert_eq!(result["output"], "echo:hi");
    assert_eq!(result["exit_code"], 0);

    server.shutdown().await;
}

#[tokio::test]
async fn crashed_server_fails_fast_then_restarts_on_a_later_call() {
    let server = start_fake().await;
    let tools = adapt_allowing(&server).await;
    let echo = find(&tools, "mcp__fake__echo");

    let _ = find(&tools, "mcp__fake__crash_me").invoke(json!({})).await;

    let post = tokio::time::timeout(Duration::from_secs(2), echo.invoke(json!({"msg": "x"})))
        .await
        .expect("a call to a dead server must not hang")
        .unwrap();
    assert_eq!(post["type"], "error", "{post}");
    assert_eq!(post["exit_code"], -1, "{post}");
    assert!(
        echo_answers_again(echo).await,
        "a call after the backoff must restart the server"
    );

    server.shutdown().await;
}

#[tokio::test]
async fn unanswered_call_times_out_and_the_server_keeps_answering() {
    let mut cfg = StdioConfig::new("fake", fake_server_path());
    cfg.request_timeout = Duration::from_millis(300);
    let server = McpServer::start("fake".into(), cfg)
        .await
        .expect("server should start");

    let err = server
        .invoke("never_answers", json!({}))
        .await
        .expect_err("an unanswered call must time out");
    assert!(
        matches!(err, McpError::RequestTimeout(after) if after == Duration::from_millis(300)),
        "{err}"
    );
    let after = server
        .invoke("echo", json!({"msg": "after"}))
        .await
        .unwrap();
    assert_eq!(first_text(&after), Some("echo:after"), "{after:?}");

    server.shutdown().await;
}

#[tokio::test]
async fn closed_stdout_under_a_live_child_is_restarted() {
    let server = start_fake().await;
    let tools = adapt_allowing(&server).await;

    let _ = find(&tools, "mcp__fake__close_stdout")
        .invoke(json!({}))
        .await;

    assert!(
        echo_answers_again(find(&tools, "mcp__fake__echo")).await,
        "a server that can no longer answer must be restarted"
    );

    server.shutdown().await;
}

#[tokio::test]
async fn crashed_server_takes_its_process_group_with_it() {
    let server = start_fake().await;
    let tools = adapt_allowing(&server).await;

    let result = find(&tools, "mcp__fake__spawn_orphan_and_crash")
        .invoke(json!({}))
        .await
        .unwrap();
    let orphan: u32 = result["output"]
        .as_str()
        .and_then(|pid| pid.parse().ok())
        .unwrap_or_else(|| panic!("fixture should answer with its orphan's pid: {result}"));

    assert!(
        wait_until_process_is_gone(orphan).await,
        "grandchild {orphan} must be killed with the crashed server's process group"
    );
    server.shutdown().await;
}

#[tokio::test]
async fn failed_initialize_kills_the_process_group() {
    let pid_dir = tempfile::tempdir().unwrap();
    let pid_file = pid_dir.path().join("orphan.pid");
    let mut cfg = make_stdio_config("fake");
    cfg.env.insert(
        "FAKE_MCP_FAIL_INIT_WITH_ORPHAN_PID_FILE".into(),
        pid_file.to_string_lossy().into_owned(),
    );

    let err = McpServer::start("fake".into(), cfg)
        .await
        .expect_err("fixture refuses initialize");
    assert!(matches!(err, McpError::Initialize(_)), "{err}");

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
    let mut cfg = make_stdio_config("fake");
    cfg.env.insert("CONFIGURED_TOKEN".into(), "secret".into());
    let server = McpServer::start("fake".into(), cfg)
        .await
        .expect("server should start");

    let result = server.invoke("env_names", json!({})).await.unwrap();
    let listing = first_text(&result).expect("a text listing");
    let names: Vec<&str> = listing.lines().collect();
    assert!(names.contains(&"PATH"), "{names:?}");
    assert!(names.contains(&"CONFIGURED_TOKEN"), "{names:?}");
    assert!(!names.contains(&"CARGO_MANIFEST_DIR"), "{names:?}");

    server.shutdown().await;
}

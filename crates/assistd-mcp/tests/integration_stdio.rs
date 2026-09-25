//! End-to-end stdio-transport tests against the in-tree
//! `fake_mcp_server` binary.

use std::time::Duration;

use assistd_mcp::{
    HealthState, McpError, McpServerHandle, StdioConfig, TransportConfig, adapt_handle_as_tools,
    mcp_error_line,
};
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

/// Resolves once the supervisor exits and drops its health sender.
async fn supervisor_exited(health_rx: &mut watch::Receiver<HealthState>) {
    while health_rx.changed().await.is_ok() {}
}

#[tokio::test]
async fn discovers_and_invokes_a_tool_end_to_end() {
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");

    let tools = adapt_handle_as_tools(&handle, "mcp__fake")
        .await
        .expect("discovery should succeed");
    let names: Vec<&str> = tools.iter().map(|tool| tool.name()).collect();
    assert_eq!(
        names,
        [
            "mcp__fake__echo",
            "mcp__fake__crash_me",
            "mcp__fake__flood_stdout"
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

    let tools = adapt_handle_as_tools(&handle, "mcp__fake")
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
async fn dead_read_loop_under_a_live_child_is_noticed_and_restarted() {
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");

    let tools = adapt_handle_as_tools(&handle, "mcp__fake")
        .await
        .expect("discovery should succeed");
    let echo = tools
        .iter()
        .find(|tool| tool.name() == "mcp__fake__echo")
        .expect("echo present");
    let flood = tools
        .iter()
        .find(|tool| tool.name() == "mcp__fake__flood_stdout")
        .expect("flood_stdout present");

    let mut watch_health = handle.watch_health();

    let _ = flood.invoke(json!({})).await;

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

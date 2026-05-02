//! End-to-end stdio-transport tests against the in-tree
//! `fake_mcp_server` binary.
//!
//! Cargo sets `CARGO_BIN_EXE_<name>` for binaries declared in the same
//! crate, which we use to locate the fixture without hardcoding paths.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use assistd_mcp::{
    HealthState, McpServerHandle, StdioConfig, TransportConfig, adapt_handle_as_tools,
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

#[tokio::test]
async fn discovers_and_invokes_a_tool_end_to_end() {
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");

    let tools = adapt_handle_as_tools(&handle, "mcp__fake")
        .await
        .expect("discovery should succeed");
    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    assert!(
        names.contains(&"mcp__fake__echo"),
        "expected echo tool, got {names:?}"
    );
    assert!(
        names.contains(&"mcp__fake__crash_me"),
        "expected crash_me tool, got {names:?}"
    );

    let echo = tools
        .iter()
        .find(|t| t.name() == "mcp__fake__echo")
        .unwrap();
    let result = echo.invoke(json!({"msg": "hi"})).await.unwrap();
    assert_eq!(result["type"], "text");
    assert_eq!(result["output"], "echo:hi");
    assert_eq!(result["exit_code"], 0);

    handle.shutdown().await;
}

#[tokio::test]
async fn server_crash_short_circuits_subsequent_calls() {
    // Spin up the fake server, take a successful `echo`, then call
    // `crash_me` to simulate a transport death. After a brief grace
    // period the supervisor should flip health off `Healthy`, and the
    // health-routed wrapper around the echo tool should return the
    // dispatch-shape error JSON instead of hanging on a dead transport.
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = McpServerHandle::start("fake".into(), make_stdio_config("fake"), shutdown_rx)
        .await
        .expect("server should start");

    let tools = adapt_handle_as_tools(&handle, "mcp__fake")
        .await
        .expect("discovery should succeed");
    let echo = tools
        .iter()
        .find(|t| t.name() == "mcp__fake__echo")
        .expect("echo present");
    let crasher = tools
        .iter()
        .find(|t| t.name() == "mcp__fake__crash_me")
        .expect("crash_me present");

    // Sanity: echo works pre-crash.
    let pre = echo.invoke(json!({"msg": "before"})).await.unwrap();
    assert_eq!(pre["output"], "echo:before");

    // Trigger the crash. The fake server `exit(0)`s before sending a
    // response, so this call returns a transport-level error to the
    // adapter (which `?`-bubbles via anyhow). We tolerate either
    // outcome; we just need the supervisor to notice the death.
    let _ = crasher.invoke(json!({})).await;

    // Wait for the supervisor's lifeline-watcher to see the EOF and
    // flip health. Backoff is exponential starting at 1s, so within
    // ~2s the state should flip away from `Healthy`.
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

    // Now invoke echo: the health-routed wrapper must short-circuit
    // and produce the dispatch-shape error JSON synchronously, not
    // hang on the dead transport.
    let post = tokio::time::timeout(Duration::from_secs(2), echo.invoke(json!({"msg": "after"})))
        .await
        .expect("invoke must not hang on a dead server")
        .expect("invoke returns Ok with a typed error JSON");
    assert_eq!(post["type"], "error");
    assert_eq!(post["exit_code"], -1);
    let output = post["output"].as_str().unwrap();
    assert!(output.contains("mcp__fake__echo"));
    assert!(output.contains("'fake'"));

    handle.shutdown().await;
}

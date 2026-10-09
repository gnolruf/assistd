use std::collections::HashMap;
use std::num::NonZeroU64;

use assistd_tools::{Approvals, IpcConfirmationGate};
use assistd_utils::readiness::NotReady;

use super::*;
use crate::daemon::test_support::app_state;

fn config_with_server(command: &str) -> Config {
    let mut config = Config::default();
    config.mcp.enabled = true;
    config.mcp.servers = vec![McpServerConfig {
        name: "fs".into(),
        command: command.into(),
        args: Vec::new(),
        env: HashMap::new(),
        request_timeout_secs: NonZeroU64::new(5).expect("non-zero"),
    }];
    config
}

/// The state and background start for `config`, with the MCP statuses
/// installed on the state.
fn prepared(config: Config, approvals_dir: &tempfile::TempDir) -> (Arc<AppState>, McpStartup) {
    let approvals = ApprovalGate::new(
        Arc::new(IpcConfirmationGate),
        Arc::new(Approvals::load(approvals_dir.path().join("approved")).expect("approvals")),
    );
    let (statuses, startup) = prepare(&config, approvals, VisionGate::new(false));
    let mut state = app_state(config);
    state.subsystems.mcp_servers = statuses;
    (Arc::new(state), startup.expect("mcp is enabled"))
}

#[tokio::test]
async fn a_server_that_fails_to_start_is_unavailable_and_adds_no_tools() {
    let dir = tempfile::tempdir().unwrap();
    let (state, startup) = prepared(config_with_server("/nonexistent/mcp-server"), &dir);
    let tools_before = state.subsystems.tools.snapshot().len();

    let started = start_all(startup, state.clone(), watch::channel(false).1).await;

    assert!(started.is_empty());
    let Err(NotReady::Unavailable(reason)) = state.subsystems.mcp_servers[0].readiness() else {
        panic!("expected fs to be unavailable");
    };
    assert!(reason.starts_with("failed to start"), "{reason}");
    assert_eq!(state.subsystems.tools.snapshot().len(), tools_before);
}

#[tokio::test]
async fn shutdown_before_a_server_starts_leaves_it_starting() {
    let dir = tempfile::tempdir().unwrap();
    let (state, startup) = prepared(config_with_server("/nonexistent/mcp-server"), &dir);

    let started = start_all(startup, state.clone(), watch::channel(true).1).await;

    assert!(started.is_empty());
    assert_eq!(
        state.subsystems.mcp_servers[0].readiness(),
        Err(NotReady::Starting)
    );
}

#[test]
fn disabled_mcp_has_no_servers_to_start() {
    let dir = tempfile::tempdir().unwrap();
    let approvals = ApprovalGate::new(
        Arc::new(IpcConfirmationGate),
        Arc::new(Approvals::load(dir.path().join("approved")).expect("approvals")),
    );
    let mut config = config_with_server("/nonexistent/mcp-server");
    config.mcp.enabled = false;
    let (statuses, startup) = prepare(&config, approvals, VisionGate::new(false));
    assert!(statuses.is_empty());
    assert!(startup.is_none());
}

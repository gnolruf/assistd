//! One configured MCP server: started with the daemon, killed once it stops answering,
//! restarted by the first call after it dies, and stopped on shutdown.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use rmcp::model::{CallToolResult, Tool};
use serde_json::Value;
use tokio::sync::{Mutex, oneshot};
use tokio_util::task::AbortOnDropHandle;
use tracing::{error, info, warn};

use assistd_utils::backoff::{RESTART_WINDOW, RestartDecision, RestartPolicy};

use crate::McpClient;
use crate::error::McpError;
use crate::stdio::{ChildLifeline, StdioConfig, StdioMcpClient};

/// Wait before the next restart once either restart cap is hit.
const UNHEALTHY_RETRY_INTERVAL: Duration = Duration::from_secs(300);

/// How long a stopping server gets to exit after SIGTERM.
const TERM_TIMEOUT: Duration = Duration::from_secs(10);

/// Requests in a row that time out before a live server counts as hung
/// and is killed.
const HUNG_AFTER_TIMEOUTS: u32 = 2;

/// A configured server. A call that finds its process dead restarts it, or fails with
/// [`McpError::ServerDown`] while backoff says to wait; one whose requests keep timing out is
/// killed. Dropping it SIGKILLs the live process group.
#[derive(Debug)]
pub struct McpServer {
    name: String,
    cfg: StdioConfig,
    state: Mutex<ServerState>,
}

#[derive(Debug, Default)]
struct ServerState {
    session: Option<Session>,
    policy: RestartPolicy,
    retry_at: Option<Instant>,
    stopped: bool,
    /// The first tool listing, which the registry offers for good.
    advertised: Option<Vec<Tool>>,
}

/// One running server process. Its task kills the process group as soon
/// as the process or its MCP session ends, or when told to stop.
#[derive(Debug)]
struct Session {
    client: Arc<StdioMcpClient>,
    started: Instant,
    stop_tx: oneshot::Sender<()>,
    task: AbortOnDropHandle<()>,
    consecutive_timeouts: u32,
}

impl McpServer {
    /// Spawn the server and run its handshake; errors if either fails.
    pub async fn start(name: String, cfg: StdioConfig) -> Result<Self, McpError> {
        let session = Session::spawn(&cfg).await?;
        let state = ServerState {
            session: Some(session),
            ..ServerState::default()
        };
        Ok(Self {
            name,
            cfg,
            state: Mutex::new(state),
        })
    }

    /// Stop the live process and refuse every later call with
    /// [`McpError::ServerDown`].
    pub async fn shutdown(&self) {
        let session = {
            let mut state = self.state.lock().await;
            state.stopped = true;
            state.session.take()
        };
        if let Some(session) = session {
            session.stop().await;
        }
    }

    async fn live_client(&self) -> Result<Arc<StdioMcpClient>, McpError> {
        let mut state = self.state.lock().await;
        if state.stopped {
            return Err(McpError::ServerDown);
        }
        if let Some(session) = &state.session
            && !session.task.is_finished()
        {
            return Ok(session.client.clone());
        }
        self.restart(&mut state).await
    }

    /// Count a timeout from `client` toward killing its session as hung,
    /// or clear the count when the server answered.
    async fn record_outcome<T>(&self, client: &Arc<StdioMcpClient>, outcome: &Result<T, McpError>) {
        let mut state = self.state.lock().await;
        let Some(session) = state
            .session
            .as_mut()
            .filter(|session| Arc::ptr_eq(&session.client, client))
        else {
            return;
        };
        match outcome {
            Err(McpError::RequestTimeout(_)) => session.consecutive_timeouts += 1,
            Ok(_) | Err(McpError::RpcError { .. }) => session.consecutive_timeouts = 0,
            Err(_) => {}
        }
        if session.consecutive_timeouts < HUNG_AFTER_TIMEOUTS {
            return;
        }
        error!(
            target: "assistd::mcp",
            server = %self.name,
            timeouts = HUNG_AFTER_TIMEOUTS,
            "MCP server stopped answering; killing it so the next call restarts it",
        );
        self.retire_session(&mut state);
    }

    /// Replace a dead session, unless the backoff after the last death or
    /// failed spawn has yet to pass.
    async fn restart(&self, state: &mut ServerState) -> Result<Arc<StdioMcpClient>, McpError> {
        self.retire_session(state);
        if state.retry_at.is_some_and(|at| Instant::now() < at) {
            return Err(McpError::ServerDown);
        }
        match Session::spawn(&self.cfg).await {
            Ok(session) => {
                info!(target: "assistd::mcp", server = %self.name, "MCP server restarted");
                let client = session.client.clone();
                state.session = Some(session);
                state.retry_at = None;
                if let Some(advertised) = &state.advertised {
                    self.warn_if_tools_changed(advertised, &client).await;
                }
                Ok(client)
            }
            Err(e) => {
                warn!(target: "assistd::mcp", server = %self.name, error = %e, "MCP server restart failed");
                state.policy.record_spawn_failure();
                state.retry_at = Some(Instant::now() + self.restart_delay(&mut state.policy));
                Err(e)
            }
        }
    }

    /// Drop the current session, killing its process group with SIGKILL, and
    /// schedule the next restart.
    fn retire_session(&self, state: &mut ServerState) {
        let Some(ended) = state.session.take() else {
            return;
        };
        let ran_for = ended.started.elapsed();
        drop(ended);
        warn!(
            target: "assistd::mcp",
            server = %self.name,
            ran_for_secs = ran_for.as_secs(),
            "MCP server stopped",
        );
        state.policy.record_session_end(ran_for);
        state.retry_at = Some(Instant::now() + self.restart_delay(&mut state.policy));
    }

    /// Re-list a restarted server's tools and warn when they differ from
    /// the `advertised` ones the registry still offers.
    async fn warn_if_tools_changed(&self, advertised: &[Tool], client: &StdioMcpClient) {
        let listed = match client.list_tools().await {
            Ok(listed) => listed,
            Err(e) => {
                warn!(
                    target: "assistd::mcp",
                    server = %self.name,
                    error = %e,
                    "could not re-list MCP tools after a restart",
                );
                return;
            }
        };
        let changes = ToolChanges::between(advertised, &listed);
        if changes.is_empty() {
            return;
        }
        warn!(
            target: "assistd::mcp",
            server = %self.name,
            added = ?changes.added,
            removed = ?changes.removed,
            changed = ?changes.changed,
            "MCP server's tools changed across a restart; the registry still offers the old ones \
             until the daemon restarts",
        );
    }

    /// Register the next restart and pick how long to wait before it.
    fn restart_delay(&self, policy: &mut RestartPolicy) -> Duration {
        match policy.next_restart(Instant::now()) {
            RestartDecision::Backoff { delay, .. } => delay,
            RestartDecision::ConsecutiveCapReached { failures } => {
                error!(
                    target: "assistd::mcp",
                    server = %self.name,
                    failures,
                    retry_secs = UNHEALTHY_RETRY_INTERVAL.as_secs(),
                    "MCP server failed {failures} times in a row; slowing restarts",
                );
                UNHEALTHY_RETRY_INTERVAL
            }
            RestartDecision::WindowCapReached { restarts } => {
                error!(
                    target: "assistd::mcp",
                    server = %self.name,
                    restarts,
                    window_secs = RESTART_WINDOW.as_secs(),
                    retry_secs = UNHEALTHY_RETRY_INTERVAL.as_secs(),
                    "MCP server restarted {restarts} times in the rolling window; slowing restarts",
                );
                UNHEALTHY_RETRY_INTERVAL
            }
        }
    }
}

#[async_trait]
impl McpClient for McpServer {
    async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        let client = self.live_client().await?;
        let outcome = client.list_tools().await;
        self.record_outcome(&client, &outcome).await;
        let tools = outcome?;
        self.state
            .lock()
            .await
            .advertised
            .get_or_insert_with(|| tools.clone());
        Ok(tools)
    }

    async fn invoke(&self, name: &str, arguments: Value) -> Result<CallToolResult, McpError> {
        let client = self.live_client().await?;
        let outcome = client.invoke(name, arguments).await;
        self.record_outcome(&client, &outcome).await;
        outcome
    }
}

/// Names of the tools a server added, removed or redefined between two
/// listings.
#[derive(Debug)]
struct ToolChanges<'a> {
    added: Vec<&'a str>,
    removed: Vec<&'a str>,
    changed: Vec<&'a str>,
}

impl<'a> ToolChanges<'a> {
    fn between(before: &'a [Tool], after: &'a [Tool]) -> Self {
        let find = |tools: &'a [Tool], name: &str| tools.iter().find(|tool| tool.name == name);
        let names_missing_from = |tools: &'a [Tool], other: &'a [Tool]| {
            tools
                .iter()
                .filter(|tool| find(other, &tool.name).is_none())
                .map(|tool| tool.name.as_ref())
                .collect()
        };
        Self {
            added: names_missing_from(after, before),
            removed: names_missing_from(before, after),
            changed: after
                .iter()
                .filter(|tool| find(before, &tool.name).is_some_and(|old| old != *tool))
                .map(|tool| tool.name.as_ref())
                .collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

impl Session {
    async fn spawn(cfg: &StdioConfig) -> Result<Self, McpError> {
        let (client, lifeline) = StdioMcpClient::spawn(cfg.clone()).await?;
        let (stop_tx, stop_rx) = oneshot::channel();
        let task = AbortOnDropHandle::new(tokio::spawn(end_session(lifeline, stop_rx)));
        Ok(Self {
            client,
            started: Instant::now(),
            stop_tx,
            task,
            consecutive_timeouts: 0,
        })
    }

    async fn stop(self) {
        let _ = self.stop_tx.send(());
        let _ = self.task.await;
    }
}

async fn end_session(mut lifeline: ChildLifeline, stop_rx: oneshot::Receiver<()>) {
    tokio::select! {
        () = lifeline.wait_for_death() => {}
        _ = stop_rx => {}
    }
    lifeline.shutdown(TERM_TIMEOUT).await;
}

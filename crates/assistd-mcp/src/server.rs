//! One configured MCP server: started with the daemon, restarted by the
//! first call after it dies, and stopped on shutdown.

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

/// A configured server. A call that finds its process dead restarts it,
/// or fails with [`McpError::ServerDown`] while the restart backoff says
/// to wait. Dropping it SIGKILLs the live process group.
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
}

/// One running server process. Its task kills the process group as soon
/// as the process or its MCP session ends, or when told to stop.
#[derive(Debug)]
struct Session {
    client: Arc<StdioMcpClient>,
    started: Instant,
    stop_tx: oneshot::Sender<()>,
    task: AbortOnDropHandle<()>,
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

    /// Replace a dead session, unless the backoff after the last death or
    /// failed spawn has yet to pass.
    async fn restart(&self, state: &mut ServerState) -> Result<Arc<StdioMcpClient>, McpError> {
        if let Some(ended) = state.session.take() {
            let ran_for = ended.started.elapsed();
            warn!(
                target: "assistd::mcp",
                server = %self.name,
                ran_for_secs = ran_for.as_secs(),
                "MCP server stopped",
            );
            state.policy.record_session_end(ran_for);
            state.retry_at = Some(Instant::now() + self.restart_delay(&mut state.policy));
        }
        if state.retry_at.is_some_and(|at| Instant::now() < at) {
            return Err(McpError::ServerDown);
        }
        match Session::spawn(&self.cfg).await {
            Ok(session) => {
                info!(target: "assistd::mcp", server = %self.name, "MCP server restarted");
                let client = session.client.clone();
                state.session = Some(session);
                state.retry_at = None;
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
        self.live_client().await?.list_tools().await
    }

    async fn invoke(&self, name: &str, arguments: Value) -> Result<CallToolResult, McpError> {
        self.live_client().await?.invoke(name, arguments).await
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

//! Per-server lifecycle: spawn the transport, restart it on crash, and
//! expose a stable `Arc<dyn McpClient>` that answers `ServerDown`
//! while the transport is away.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{RwLock, watch};
use tokio_util::task::AbortOnDropHandle;
use tracing::{error, info, warn};

use crate::backoff::{RESTART_WINDOW, RestartDecision, RestartPolicy, UNHEALTHY_RETRY_INTERVAL};
use crate::error::McpError;
use crate::sse::{SseConfig, SseLifeline, SseMcpClient};
use crate::stdio::{ChildLifeline, StdioConfig, StdioMcpClient};
use crate::{McpClient, ToolResult, ToolSchema};

/// Health published by the supervisor on every state change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthState {
    Healthy,
    /// The transport died and a restart is pending.
    Restarting,
    /// A restart cap was hit; restarts continue at the slow
    /// [`UNHEALTHY_RETRY_INTERVAL`] cadence.
    Unhealthy,
}

/// Per-server transport configuration.
#[derive(Debug, Clone)]
pub enum TransportConfig {
    Stdio(StdioConfig),
    Sse(SseConfig),
}

/// Stable handle for a single MCP server; [`Self::client`] survives
/// transport restarts. Dropping it without [`Self::shutdown`] aborts the
/// supervisor and kills the live transport.
pub struct McpServerHandle {
    pub name: String,
    switch: Arc<SwitchingClient>,
    health_rx: watch::Receiver<HealthState>,
    supervisor_shutdown_tx: watch::Sender<bool>,
    supervisor_task: AbortOnDropHandle<()>,
}

impl McpServerHandle {
    /// Spawn the first transport and hand it to a supervisor task.
    /// Errors only if that first spawn fails; later failures surface
    /// through [`Self::watch_health`].
    pub async fn start(
        name: String,
        transport_cfg: TransportConfig,
        external_shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self, McpError> {
        let initial = spawn_transport(&transport_cfg).await?;
        let (health_tx, health_rx) = watch::channel(HealthState::Healthy);
        let switch = Arc::new(SwitchingClient::new(Some(initial.client.clone())));

        let (supervisor_shutdown_tx, supervisor_shutdown_rx) = watch::channel(false);

        let supervisor = Supervisor {
            name: name.clone(),
            transport_cfg,
            switch: switch.clone(),
            health_tx,
            supervisor_shutdown_rx,
            external_shutdown_rx,
            policy: RestartPolicy::default(),
            session_start: Instant::now(),
        };
        let supervisor_task =
            AbortOnDropHandle::new(tokio::spawn(supervisor.run(initial.lifeline)));

        Ok(Self {
            name,
            switch,
            health_rx,
            supervisor_shutdown_tx,
            supervisor_task,
        })
    }

    /// Stable client that follows the current transport.
    pub fn client(&self) -> Arc<dyn McpClient> {
        self.switch.clone()
    }

    /// The most recently published health.
    pub fn health(&self) -> HealthState {
        *self.health_rx.borrow()
    }

    /// A receiver that observes every health change. Its sender closes
    /// when the supervisor exits.
    pub fn watch_health(&self) -> watch::Receiver<HealthState> {
        self.health_rx.clone()
    }

    /// Stop the supervisor and wait up to 15 seconds for it to exit
    /// gracefully; past that it is aborted.
    pub async fn shutdown(self) {
        let _ = self.supervisor_shutdown_tx.send(true);
        let _ = tokio::time::timeout(Duration::from_secs(15), self.supervisor_task).await;
    }
}

/// [`McpClient`] that forwards to whichever transport is live, or
/// answers [`McpError::ServerDown`] between transports.
pub struct SwitchingClient {
    inner: RwLock<Option<Arc<dyn McpClient>>>,
}

impl SwitchingClient {
    fn new(initial: Option<Arc<dyn McpClient>>) -> Self {
        Self {
            inner: RwLock::new(initial),
        }
    }

    async fn swap(&self, next: Option<Arc<dyn McpClient>>) {
        *self.inner.write().await = next;
    }
}

#[async_trait]
impl McpClient for SwitchingClient {
    async fn list_tools(&self) -> Result<Vec<ToolSchema>, McpError> {
        let live = self.inner.read().await.clone();
        let client = live.ok_or(McpError::ServerDown)?;
        client.list_tools().await
    }

    async fn invoke(&self, name: &str, arguments: Value) -> Result<ToolResult, McpError> {
        let live = self.inner.read().await.clone();
        let client = live.ok_or(McpError::ServerDown)?;
        client.invoke(name, arguments).await
    }
}

struct TransportInstance {
    client: Arc<dyn McpClient>,
    lifeline: Lifeline,
}

enum Lifeline {
    Stdio(ChildLifeline),
    Sse(SseLifeline),
}

impl Lifeline {
    async fn wait(&mut self) {
        match self {
            Lifeline::Stdio(child) => child.wait_for_death().await,
            Lifeline::Sse(connection) => connection.wait_for_disconnect().await,
        }
    }

    async fn shutdown(self) {
        match self {
            Lifeline::Stdio(child) => child.shutdown(Duration::from_secs(10)).await,
            Lifeline::Sse(connection) => connection.shutdown().await,
        }
    }
}

async fn spawn_transport(cfg: &TransportConfig) -> Result<TransportInstance, McpError> {
    match cfg {
        TransportConfig::Stdio(stdio_cfg) => {
            let (stdio_client, lifeline) = StdioMcpClient::spawn(stdio_cfg.clone()).await?;
            let client: Arc<dyn McpClient> = stdio_client;
            Ok(TransportInstance {
                client,
                lifeline: Lifeline::Stdio(lifeline),
            })
        }
        TransportConfig::Sse(sse_cfg) => {
            let (sse_client, lifeline) = SseMcpClient::connect(sse_cfg.clone()).await?;
            let client: Arc<dyn McpClient> = sse_client;
            Ok(TransportInstance {
                client,
                lifeline: Lifeline::Sse(lifeline),
            })
        }
    }
}

struct Supervisor {
    name: String,
    transport_cfg: TransportConfig,
    switch: Arc<SwitchingClient>,
    health_tx: watch::Sender<HealthState>,
    supervisor_shutdown_rx: watch::Receiver<bool>,
    external_shutdown_rx: watch::Receiver<bool>,
    policy: RestartPolicy,
    session_start: Instant,
}

impl Supervisor {
    async fn run(mut self, initial_lifeline: Lifeline) {
        info!(
            target: "assistd::mcp",
            server = %self.name,
            transport = %transport_label(&self.transport_cfg),
            "MCP supervisor running",
        );

        let mut current_lifeline = Some(initial_lifeline);
        loop {
            if let Some(lifeline) = current_lifeline.take()
                && self.supervise_session(lifeline).await.is_break()
            {
                return;
            }

            let delay = self.restart_delay();
            tokio::select! {
                _ = tokio::time::sleep(delay) => {}
                _ = self.shutdown_requested() => return,
            }

            current_lifeline = self.respawn().await;
        }
    }

    /// Wait for `lifeline` to end; `Break` means shutdown was requested.
    async fn supervise_session(&mut self, mut lifeline: Lifeline) -> ControlFlow<()> {
        tokio::select! {
            _ = lifeline.wait() => {
                let ran_for = self.session_start.elapsed();
                warn!(
                    target: "assistd::mcp",
                    server = %self.name,
                    ran_for_secs = ran_for.as_secs(),
                    "MCP server transport died",
                );
                lifeline.shutdown().await;
                self.policy.record_session_end(ran_for);
                let _ = self.health_tx.send(HealthState::Restarting);
                self.switch.swap(None).await;
                ControlFlow::Continue(())
            }
            reason = self.shutdown_requested() => {
                info!(target: "assistd::mcp", server = %self.name, reason, "supervisor shutdown");
                self.switch.swap(None).await;
                lifeline.shutdown().await;
                ControlFlow::Break(())
            }
        }
    }

    /// Register the next restart and pick its delay, publishing
    /// `Unhealthy` when a restart cap is hit.
    fn restart_delay(&mut self) -> Duration {
        let name = &self.name;
        match self.policy.next_restart(Instant::now()) {
            RestartDecision::Backoff { delay, failures } => {
                warn!(
                    target: "assistd::mcp",
                    server = %name,
                    failures,
                    "restarting MCP server in {delay:?}",
                );
                delay
            }
            RestartDecision::ConsecutiveCapReached { failures } => {
                error!(
                    target: "assistd::mcp",
                    server = %name,
                    failures,
                    retry_secs = UNHEALTHY_RETRY_INTERVAL.as_secs(),
                    "MCP server failed {failures} times in a row; marking unhealthy and retrying at slow cadence",
                );
                let _ = self.health_tx.send(HealthState::Unhealthy);
                UNHEALTHY_RETRY_INTERVAL
            }
            RestartDecision::WindowCapReached { restarts } => {
                error!(
                    target: "assistd::mcp",
                    server = %name,
                    restarts,
                    window_secs = RESTART_WINDOW.as_secs(),
                    retry_secs = UNHEALTHY_RETRY_INTERVAL.as_secs(),
                    "MCP server restarted {restarts} times in the rolling window; marking unhealthy and retrying at slow cadence",
                );
                let _ = self.health_tx.send(HealthState::Unhealthy);
                UNHEALTHY_RETRY_INTERVAL
            }
        }
    }

    async fn respawn(&mut self) -> Option<Lifeline> {
        match spawn_transport(&self.transport_cfg).await {
            Ok(instance) => {
                self.session_start = Instant::now();
                self.switch.swap(Some(instance.client)).await;
                let _ = self.health_tx.send(HealthState::Healthy);
                info!(target: "assistd::mcp", server = %self.name, "MCP server restarted");
                Some(instance.lifeline)
            }
            Err(err) => {
                self.policy.record_spawn_failure();
                warn!(
                    target: "assistd::mcp",
                    server = %self.name,
                    error = %err,
                    "MCP server restart failed",
                );
                None
            }
        }
    }

    async fn shutdown_requested(&mut self) -> &'static str {
        shutdown_reason(
            &mut self.supervisor_shutdown_rx,
            &mut self.external_shutdown_rx,
        )
        .await
    }
}

/// Resolves when either shutdown watch turns true or its sender is
/// gone.
async fn shutdown_reason(
    handle_rx: &mut watch::Receiver<bool>,
    daemon_rx: &mut watch::Receiver<bool>,
) -> &'static str {
    tokio::select! {
        _ = handle_rx.wait_for(|stop| *stop) => "handle.shutdown",
        _ = daemon_rx.wait_for(|stop| *stop) => "daemon-wide",
    }
}

fn transport_label(cfg: &TransportConfig) -> &'static str {
    match cfg {
        TransportConfig::Stdio(_) => "stdio",
        TransportConfig::Sse(_) => "sse",
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;
    use serde_json::json;

    use super::*;

    struct FakeClient {
        invocations: Arc<Mutex<u32>>,
    }

    #[async_trait]
    impl McpClient for FakeClient {
        async fn list_tools(&self) -> Result<Vec<ToolSchema>, McpError> {
            Ok(vec![ToolSchema {
                name: "ping".into(),
                description: "ping".into(),
                input_schema: json!({"type": "object"}),
            }])
        }
        async fn invoke(&self, _name: &str, _args: Value) -> Result<ToolResult, McpError> {
            *self.invocations.lock() += 1;
            Ok(ToolResult::Text("pong".into()))
        }
    }

    #[tokio::test]
    async fn switching_client_follows_swaps() {
        let switch = SwitchingClient::new(None);
        let err = switch.list_tools().await.unwrap_err();
        assert!(matches!(err, McpError::ServerDown), "{err}");

        let invocations = Arc::new(Mutex::new(0));
        let fake: Arc<dyn McpClient> = Arc::new(FakeClient {
            invocations: invocations.clone(),
        });
        switch.swap(Some(fake)).await;
        assert_eq!(switch.list_tools().await.unwrap().len(), 1);
        switch.invoke("ping", json!({})).await.unwrap();
        assert_eq!(*invocations.lock(), 1);

        switch.swap(None).await;
        let err = switch.invoke("ping", json!({})).await.unwrap_err();
        assert!(matches!(err, McpError::ServerDown), "{err}");
        assert_eq!(*invocations.lock(), 1);
    }
}

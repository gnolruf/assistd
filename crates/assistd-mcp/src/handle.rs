//! Per-server lifecycle: spawn the transport, restart it on crash, and
//! expose a stable `Arc<dyn McpClient>` that answers `ServerDown`
//! while the transport is away.

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
    Restarting,
    Unhealthy,
}

/// Per-server transport configuration.
#[derive(Debug, Clone)]
pub enum TransportConfig {
    Stdio(StdioConfig),
    Sse(SseConfig),
}

/// Stable handle for a single MCP server. The `Arc<dyn McpClient>`
/// returned by [`Self::client`] survives transport restarts.
///
/// Dropping the handle without [`Self::shutdown`] aborts the
/// supervisor, which drops the live transport: a stdio child is
/// SIGKILLed (`kill_on_drop`, direct child only) and an SSE
/// connection's tasks are aborted.
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
            initial_lifeline: initial.lifeline,
            switch: switch.clone(),
            health_tx,
            supervisor_shutdown_rx,
            external_shutdown_rx,
        };
        let supervisor_task = AbortOnDropHandle::new(tokio::spawn(supervisor.run()));

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

    pub fn health(&self) -> HealthState {
        *self.health_rx.borrow()
    }

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
            Lifeline::Stdio(c) => c.wait_for_death().await,
            Lifeline::Sse(s) => s.wait_for_disconnect().await,
        }
    }

    async fn shutdown(self) {
        match self {
            Lifeline::Stdio(c) => c.shutdown(Duration::from_secs(10)).await,
            Lifeline::Sse(s) => s.shutdown().await,
        }
    }
}

async fn spawn_transport(cfg: &TransportConfig) -> Result<TransportInstance, McpError> {
    match cfg {
        TransportConfig::Stdio(s) => {
            let (c, l) = StdioMcpClient::spawn(s.clone()).await?;
            let client: Arc<dyn McpClient> = c;
            Ok(TransportInstance {
                client,
                lifeline: Lifeline::Stdio(l),
            })
        }
        TransportConfig::Sse(s) => {
            let (c, l) = SseMcpClient::connect(s.clone()).await?;
            let client: Arc<dyn McpClient> = c;
            Ok(TransportInstance {
                client,
                lifeline: Lifeline::Sse(l),
            })
        }
    }
}

struct Supervisor {
    name: String,
    transport_cfg: TransportConfig,
    initial_lifeline: Lifeline,
    switch: Arc<SwitchingClient>,
    health_tx: watch::Sender<HealthState>,
    supervisor_shutdown_rx: watch::Receiver<bool>,
    external_shutdown_rx: watch::Receiver<bool>,
}

impl Supervisor {
    async fn run(self) {
        let Self {
            name,
            transport_cfg,
            initial_lifeline,
            switch,
            health_tx,
            mut supervisor_shutdown_rx,
            mut external_shutdown_rx,
        } = self;

        info!(
            target: "assistd::mcp",
            server = %name,
            transport = %transport_label(&transport_cfg),
            "MCP supervisor running",
        );

        let mut policy = RestartPolicy::default();
        let mut current_lifeline: Option<Lifeline> = Some(initial_lifeline);
        let mut session_start = Instant::now();

        loop {
            if let Some(mut lifeline) = current_lifeline.take() {
                tokio::select! {
                    _ = lifeline.wait() => {
                        let ran_for = session_start.elapsed();
                        warn!(
                            target: "assistd::mcp",
                            server = %name,
                            ran_for_secs = ran_for.as_secs(),
                            "MCP server transport died",
                        );
                        lifeline.shutdown().await;
                        policy.record_session_end(ran_for);
                        let _ = health_tx.send(HealthState::Restarting);
                        switch.swap(None).await;
                    }
                    reason = shutdown_reason(&mut supervisor_shutdown_rx, &mut external_shutdown_rx) => {
                        info!(target: "assistd::mcp", server = %name, reason, "supervisor shutdown");
                        switch.swap(None).await;
                        lifeline.shutdown().await;
                        return;
                    }
                }
            }

            let delay = match policy.next_restart(Instant::now()) {
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
                    let _ = health_tx.send(HealthState::Unhealthy);
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
                    let _ = health_tx.send(HealthState::Unhealthy);
                    UNHEALTHY_RETRY_INTERVAL
                }
            };
            tokio::select! {
                _ = tokio::time::sleep(delay) => {}
                _ = shutdown_reason(&mut supervisor_shutdown_rx, &mut external_shutdown_rx) => return,
            }

            match spawn_transport(&transport_cfg).await {
                Ok(instance) => {
                    session_start = Instant::now();
                    switch.swap(Some(instance.client)).await;
                    let _ = health_tx.send(HealthState::Healthy);
                    info!(target: "assistd::mcp", server = %name, "MCP server restarted");
                    current_lifeline = Some(instance.lifeline);
                }
                Err(e) => {
                    policy.record_spawn_failure();
                    warn!(
                        target: "assistd::mcp",
                        server = %name,
                        error = %e,
                        "MCP server restart failed",
                    );
                }
            }
        }
    }
}

/// Resolves when either shutdown watch turns true or its sender is
/// gone.
async fn shutdown_reason(
    handle: &mut watch::Receiver<bool>,
    daemon: &mut watch::Receiver<bool>,
) -> &'static str {
    tokio::select! {
        _ = handle.wait_for(|v| *v) => "handle.shutdown",
        _ = daemon.wait_for(|v| *v) => "daemon-wide",
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
    use super::*;
    use parking_lot::Mutex;
    use serde_json::json;

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

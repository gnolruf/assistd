//! Per-server lifecycle: spawn the transport, supervise crashes, expose
//! a stable `Arc<dyn McpClient>` for the rest of the daemon.
//!
//! The handle is the sole boundary between the always-on registry and
//! the ephemeral transport. Even when the underlying server is down or
//! restarting, the registered `McpToolAdapter` still holds a live
//! `Arc<dyn McpClient>` — the [`SwitchingClient`] — that returns
//! `McpError::ServerDown` until the supervisor reconnects.
//!
//! Note that the `HealthRoutedTool` short-circuits before the
//! transport is consulted at all, so the `ServerDown` path here is
//! defense in depth (and useful for direct-client consumers).

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result as AnyResult;
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{RwLock, watch};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::backoff::{MAX_CONSECUTIVE_FAILURES, MIN_HEALTHY_SECONDS, backoff_delay};
use crate::error::McpError;
use crate::sse::{SseConfig, SseLifeline, SseMcpClient};
use crate::stdio::{ChildLifeline, StdioConfig, StdioMcpClient};
use crate::{McpClient, ToolResult, ToolSchema};

/// Coarse health status published by the supervisor on every state change.
/// The `HealthRoutedTool` reads this on every invoke to decide whether to
/// forward to the transport or short-circuit with a tool-error JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthState {
    Healthy,
    Restarting,
    Unhealthy,
}

/// Per-server transport configuration. The daemon builds one of these
/// from `assistd_config::McpServerConfig` and hands it to
/// [`McpServerHandle::start`].
#[derive(Debug, Clone)]
pub enum TransportConfig {
    Stdio(StdioConfig),
    Sse(SseConfig),
}

impl TransportConfig {
    /// Return the human-readable label for this transport, used in tracing logs.
    pub fn label(&self) -> &str {
        match self {
            Self::Stdio(s) => &s.label,
            Self::Sse(s) => &s.label,
        }
    }
}

/// Stable handle for a single MCP server. The `Arc<dyn McpClient>`
/// returned by [`Self::client`] survives transport restarts.
pub struct McpServerHandle {
    pub name: String,
    switch: Arc<SwitchingClient>,
    health_rx: watch::Receiver<HealthState>,
    supervisor_shutdown_tx: watch::Sender<bool>,
    supervisor_task: Option<JoinHandle<()>>,
}

impl McpServerHandle {
    /// Start the server: spawn the first transport, perform initial
    /// discovery, then hand off to a supervisor task that handles all
    /// future crashes/restarts.
    ///
    /// Returns `Err` only on the FIRST spawn attempt — once the handle
    /// is alive, the supervisor takes over and the daemon never sees
    /// transport-level failures again (they surface to the model via
    /// the health-routed adapter).
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
        let supervisor_task = tokio::spawn(supervisor.run());

        Ok(Self {
            name,
            switch,
            health_rx,
            supervisor_shutdown_tx,
            supervisor_task: Some(supervisor_task),
        })
    }

    /// Return the stable [`McpClient`] handle backed by the current transport.
    pub fn client(&self) -> Arc<dyn McpClient> {
        self.switch.clone()
    }

    /// Return the current health state without waiting for a change.
    pub fn health(&self) -> HealthState {
        *self.health_rx.borrow()
    }

    /// Return a watch receiver that fires on every health-state transition.
    pub fn watch_health(&self) -> watch::Receiver<HealthState> {
        self.health_rx.clone()
    }

    /// Signal the supervisor to stop and wait up to 15 seconds for it to exit.
    pub async fn shutdown(mut self) {
        let _ = self.supervisor_shutdown_tx.send(true);
        if let Some(task) = self.supervisor_task.take() {
            let _ = tokio::time::timeout(Duration::from_secs(15), task).await;
        }
    }
}

impl Drop for McpServerHandle {
    fn drop(&mut self) {
        // Defense-in-depth for the no-shutdown() path (panics, tests,
        // future misuse). Graceful path is `shutdown()`; here we just
        // signal + abort. Stdio children still die because
        // `Child::kill_on_drop(true)` is set in stdio.rs — aborting
        // the supervisor drops the Child future, which sends SIGKILL.
        let _ = self.supervisor_shutdown_tx.send(true);
        if let Some(task) = self.supervisor_task.take() {
            task.abort();
        }
    }
}

/// `McpClient` impl that points at the live transport via
/// `RwLock<Option<Arc<dyn McpClient>>>`. The supervisor swaps it on
/// crash/restart so the registry-side `McpToolAdapter` keeps holding
/// a stable `Arc<dyn McpClient>`.
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
    async fn list_tools(&self) -> AnyResult<Vec<ToolSchema>> {
        let live = self.inner.read().await.clone();
        let client = live.ok_or(McpError::ServerDown)?;
        client.list_tools().await
    }

    async fn invoke(&self, name: &str, arguments: Value) -> AnyResult<ToolResult> {
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
            Lifeline::Stdio(c) => {
                let _ = c.wait_for_exit().await;
            }
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

        let mut consecutive_failures: u32 = 0;
        let mut current_lifeline: Option<Lifeline> = Some(initial_lifeline);
        let mut session_start = Instant::now();

        loop {
            match current_lifeline.take() {
                Some(mut lifeline) => {
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
                            if ran_for >= Duration::from_secs(MIN_HEALTHY_SECONDS) {
                                consecutive_failures = 0;
                            }
                        }
                        _ = supervisor_shutdown_rx.changed() => {
                            if *supervisor_shutdown_rx.borrow() {
                                info!(target: "assistd::mcp", server = %name, "supervisor shutdown (handle.shutdown)");
                                switch.swap(None).await;
                                lifeline.shutdown().await;
                                return;
                            }
                        }
                        _ = external_shutdown_rx.changed() => {
                            if *external_shutdown_rx.borrow() {
                                info!(target: "assistd::mcp", server = %name, "supervisor shutdown (daemon-wide)");
                                switch.swap(None).await;
                                lifeline.shutdown().await;
                                return;
                            }
                        }
                    }
                }
                None => {}
            }

            // Swap the switching client to None so direct consumers see
            // ServerDown rather than calling a dead transport.
            let _ = health_tx.send(HealthState::Restarting);
            switch.swap(None).await;

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                error!(
                    target: "assistd::mcp",
                    server = %name,
                    attempts = consecutive_failures,
                    "MCP server reached {MAX_CONSECUTIVE_FAILURES} consecutive failures; marking permanently unhealthy",
                );
                let _ = health_tx.send(HealthState::Unhealthy);
                // Park awaiting any shutdown signal.
                tokio::select! {
                    _ = supervisor_shutdown_rx.changed() => {}
                    _ = external_shutdown_rx.changed() => {}
                }
                return;
            }

            let delay = backoff_delay(consecutive_failures);
            warn!(
                target: "assistd::mcp",
                server = %name,
                attempt = consecutive_failures + 1,
                cap = MAX_CONSECUTIVE_FAILURES,
                "restarting MCP server in {delay:?}",
            );
            tokio::select! {
                _ = tokio::time::sleep(delay) => {}
                _ = supervisor_shutdown_rx.changed() => return,
                _ = external_shutdown_rx.changed() => return,
            }

            match spawn_transport(&transport_cfg).await {
                Ok(instance) => {
                    consecutive_failures = 0;
                    session_start = Instant::now();
                    switch.swap(Some(instance.client)).await;
                    let _ = health_tx.send(HealthState::Healthy);
                    info!(target: "assistd::mcp", server = %name, "MCP server restarted");
                    current_lifeline = Some(instance.lifeline);
                }
                Err(e) => {
                    consecutive_failures += 1;
                    warn!(
                        target: "assistd::mcp",
                        server = %name,
                        attempt = consecutive_failures,
                        error = %e,
                        "MCP server restart failed",
                    );
                    // current_lifeline stays None; the loop retries with backoff.
                }
            }
        }
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
    use serde_json::json;
    use std::sync::Mutex;

    /// Trivial McpClient that returns a fixed tool list and echoes args.
    struct FakeClient {
        invocations: Arc<Mutex<u32>>,
    }

    #[async_trait]
    impl McpClient for FakeClient {
        async fn list_tools(&self) -> AnyResult<Vec<ToolSchema>> {
            Ok(vec![ToolSchema {
                name: "ping".into(),
                description: "ping".into(),
                input_schema: json!({"type": "object"}),
            }])
        }
        async fn invoke(&self, _name: &str, _args: Value) -> AnyResult<ToolResult> {
            *self.invocations.lock().unwrap() += 1;
            Ok(ToolResult::Text("pong".into()))
        }
    }

    #[tokio::test]
    async fn switching_client_returns_server_down_when_empty() {
        let switch = Arc::new(SwitchingClient::new(None));
        let err = switch.list_tools().await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.to_lowercase().contains("unavailable") || msg.contains("server is currently"));
    }

    #[tokio::test]
    async fn switching_client_forwards_when_swapped_in() {
        let invocations = Arc::new(Mutex::new(0));
        let fake: Arc<dyn McpClient> = Arc::new(FakeClient {
            invocations: invocations.clone(),
        });
        let switch = Arc::new(SwitchingClient::new(Some(fake)));
        let _ = switch.invoke("ping", json!({})).await.unwrap();
        assert_eq!(*invocations.lock().unwrap(), 1);

        // Swap to None; subsequent calls fail.
        switch.swap(None).await;
        assert!(switch.invoke("ping", json!({})).await.is_err());
    }

    #[tokio::test]
    async fn switching_client_can_be_swapped_back_in() {
        let switch = Arc::new(SwitchingClient::new(None));
        assert!(switch.list_tools().await.is_err());

        let fake: Arc<dyn McpClient> = Arc::new(FakeClient {
            invocations: Arc::new(Mutex::new(0)),
        });
        switch.swap(Some(fake)).await;
        let tools = switch.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
    }
}

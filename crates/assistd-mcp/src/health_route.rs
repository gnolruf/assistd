//! Tool wrapper that answers with a tool-error envelope, without an
//! RPC, while the owning server is unhealthy.

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::error::{McpError, mcp_error_line};
use crate::handle::HealthState;
use crate::{McpToolAdapter, Tool, ToolError};

/// Wraps an [`McpToolAdapter`] with a per-server health gate.
pub struct HealthRoutedTool {
    inner: McpToolAdapter,
    server_name: String,
    health_rx: watch::Receiver<HealthState>,
}

impl HealthRoutedTool {
    pub fn new(
        inner: McpToolAdapter,
        server_name: String,
        health_rx: watch::Receiver<HealthState>,
    ) -> Self {
        Self {
            inner,
            server_name,
            health_rx,
        }
    }
}

#[async_trait]
impl Tool for HealthRoutedTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn parameters_schema(&self) -> Value {
        self.inner.parameters_schema()
    }

    async fn invoke(&self, args: Value) -> Result<Value, ToolError> {
        let state = *self.health_rx.borrow();
        match state {
            HealthState::Healthy => self.inner.invoke(args).await,
            HealthState::Restarting | HealthState::Unhealthy => Ok(json!({
                "type": "error",
                "output": mcp_error_line(self.inner.name(), &McpError::ServerDown),
                "exit_code": -1,
                "duration_ms": 0,
                "truncated": false,
                "server_name": self.server_name,
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{McpClient, ToolResult, ToolSchema};
    use serde_json::json;
    use std::sync::Arc;

    struct FakeClient;

    #[async_trait]
    impl McpClient for FakeClient {
        async fn list_tools(&self) -> Result<Vec<ToolSchema>, McpError> {
            Ok(vec![])
        }
        async fn invoke(&self, name: &str, _args: Value) -> Result<ToolResult, McpError> {
            Ok(ToolResult::Text(format!("called {name}")))
        }
    }

    fn make_tool(state: HealthState) -> (HealthRoutedTool, watch::Sender<HealthState>) {
        let (tx, rx) = watch::channel(state);
        let inner = McpToolAdapter::new(
            Arc::new(FakeClient),
            ToolSchema {
                name: "search".into(),
                description: "search".into(),
                input_schema: json!({"type": "object"}),
            },
            "mcp__web__search".into(),
        );
        (HealthRoutedTool::new(inner, "web".into(), rx), tx)
    }

    #[tokio::test]
    async fn forwards_when_healthy() {
        let (tool, _tx) = make_tool(HealthState::Healthy);
        let result = tool.invoke(json!({})).await.unwrap();

        assert_eq!(result["type"], "text");
        assert_eq!(result["output"], "called search");
    }

    #[tokio::test]
    async fn short_circuits_with_server_down_envelope_when_not_healthy() {
        for state in [HealthState::Unhealthy, HealthState::Restarting] {
            let (tool, _tx) = make_tool(state);
            let result = tool.invoke(json!({})).await.unwrap();
            assert_eq!(
                result,
                json!({
                    "type": "error",
                    "output": mcp_error_line("mcp__web__search", &McpError::ServerDown),
                    "exit_code": -1,
                    "duration_ms": 0,
                    "truncated": false,
                    "server_name": "web",
                }),
                "{state:?}"
            );
        }
    }

    #[tokio::test]
    async fn flips_back_to_forwarding_when_health_recovers() {
        let (tool, tx) = make_tool(HealthState::Restarting);
        let r = tool.invoke(json!({})).await.unwrap();
        assert_eq!(r["type"], "error");

        tx.send(HealthState::Healthy).unwrap();
        let r = tool.invoke(json!({})).await.unwrap();
        assert_eq!(r["type"], "text");
        assert_eq!(r["exit_code"], 0);
    }
}

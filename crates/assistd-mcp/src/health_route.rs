//! Tool wrapper that intercepts invocations of MCP tools belonging to
//! an unhealthy server *before* they reach the transport.
//!
//! The tool registry is built once at daemon startup and never mutated.
//! When an MCP server crashes, its tools therefore remain in the model's
//! tool list until the next daemon restart. Without this wrapper, a
//! tool call against a dead server would propagate to the supervisor's
//! `SwitchingClient` (returning `ServerDown`) — fine in principle, but
//! the failure surfaces as an `anyhow::Error` deep in the agent loop's
//! dispatch path. Surfacing it earlier as a synthetic tool result lets
//! the model see "this tool is unavailable, try a different one" and
//! continue the turn, instead of triggering a tool-error path that
//! treats every MCP server stall the same as a daemon-side bug.

use anyhow::Result as AnyResult;
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::error::{McpError, mcp_error_line};
use crate::handle::HealthState;
use crate::{McpToolAdapter, Tool};

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

    async fn invoke(&self, args: Value) -> AnyResult<Value> {
        let state = *self.health_rx.borrow();
        match state {
            HealthState::Healthy => self.inner.invoke(args).await,
            HealthState::Restarting | HealthState::Unhealthy => {
                // Shape mirrors the JSON `RunTool` emits via
                // `assistd_tools::PresentResult` so the agent loop's
                // dispatch site (`agent.rs::dispatch_tool_call`) reads
                // `output`/`exit_code`/`duration_ms`/`truncated` straight
                // into the tool_role message body. Routing through
                // `mcp_error_line(ServerDown)` keeps the line shape
                // consistent with every other MCP failure the model
                // sees, so the recovery hint (`Try: another tool while
                // the server reconnects`) is the same regardless of
                // whether the failure was caught at the wrapper or at
                // the transport. `duration_ms` stays 0 — the wrapper
                // short-circuits before any RPC, so 0 is honest.
                Ok(json!({
                    "type": "error",
                    "output": mcp_error_line(self.inner.name(), &McpError::ServerDown),
                    "exit_code": -1,
                    "duration_ms": 0,
                    "truncated": false,
                    "server_name": self.server_name,
                }))
            }
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
        async fn list_tools(&self) -> AnyResult<Vec<ToolSchema>> {
            Ok(vec![])
        }
        async fn invoke(&self, name: &str, _args: Value) -> AnyResult<ToolResult> {
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
        // McpToolAdapter renders Text results into the dispatch envelope
        // shape (`output` field, not the old `text` field).
        assert_eq!(result["type"], "text");
        assert!(result["output"].as_str().unwrap().contains("search"));
    }

    #[tokio::test]
    async fn short_circuits_when_unhealthy() {
        let (tool, _tx) = make_tool(HealthState::Unhealthy);
        let result = tool.invoke(json!({})).await.unwrap();
        assert_eq!(result["type"], "error");
        assert_eq!(result["exit_code"], -1);
        assert_eq!(result["truncated"], false);
        // Server name carried out-of-band on a separate field so the
        // message body itself can stay convention-compliant without
        // having to embed the supervisor's identifier mid-sentence.
        assert_eq!(result["server_name"], "web");
        let output = result["output"].as_str().unwrap();
        // Convention-compliant line: `[error] <tool>: <what>. <Hint>: <recovery>\n`
        assert!(output.starts_with("[error] mcp__web__search: "), "{output}");
        assert!(
            output.contains("Try:") || output.contains("Check:"),
            "missing recovery hint: {output}"
        );
        assert!(output.ends_with('\n'));
    }

    #[tokio::test]
    async fn short_circuits_when_restarting() {
        let (tool, _tx) = make_tool(HealthState::Restarting);
        let result = tool.invoke(json!({})).await.unwrap();
        assert_eq!(result["type"], "error");
    }

    #[tokio::test]
    async fn flips_back_to_forwarding_when_health_recovers() {
        let (tool, tx) = make_tool(HealthState::Restarting);
        let r = tool.invoke(json!({})).await.unwrap();
        assert_eq!(r["type"], "error");

        let _ = tx.send(HealthState::Healthy);
        let r = tool.invoke(json!({})).await.unwrap();
        assert_eq!(r["type"], "text");
        assert_eq!(r["exit_code"], 0);
    }
}

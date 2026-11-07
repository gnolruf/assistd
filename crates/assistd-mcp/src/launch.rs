//! Bringing a configured server up: start it and list its tools, retrying failed attempts with
//! backoff until one succeeds or fails in a way retrying cannot fix.

use std::sync::Arc;

use assistd_tools::{ApprovalGate, ToolCatalog, VisionGate};
use assistd_utils::backoff::RestartPolicy;
use tracing::{info, warn};

use crate::adapt_client_as_tools;
use crate::error::McpError;
use crate::server::{McpServer, restart_delay};
use crate::stdio::StdioConfig;

/// The tools a [`launch`] offers and how their calls are gated.
#[derive(Debug, Clone)]
pub struct LaunchTarget {
    pub catalog: Arc<ToolCatalog>,
    pub approvals: ApprovalGate,
    pub vision: Arc<VisionGate>,
}

/// Start server `name` and add its tools to the target's catalog, retrying with backoff after
/// each failed attempt, which `on_retry` hears about first. Errs only on a failure retrying
/// cannot fix, such as a missing command; dropping the future cancels pending retries.
pub async fn launch(
    name: String,
    cfg: StdioConfig,
    target: &LaunchTarget,
    mut on_retry: impl FnMut(&McpError),
) -> Result<Arc<McpServer>, McpError> {
    let mut policy = RestartPolicy::default();
    loop {
        let error = match start_and_offer(&name, &cfg, target).await {
            Ok(server) => return Ok(server),
            Err(error) if error.is_permanent() => {
                warn!(target: "assistd::mcp", server = %name, %error, "MCP server cannot start");
                return Err(error);
            }
            Err(error) => error,
        };
        on_retry(&error);
        policy.record_spawn_failure();
        let delay = restart_delay(&name, &mut policy);
        warn!(
            target: "assistd::mcp",
            server = %name,
            %error,
            retry_secs = delay.as_secs(),
            "MCP server failed to start; retrying",
        );
        tokio::time::sleep(delay).await;
    }
}

/// One attempt: start the server and offer its tools, shutting it down again when listing
/// them fails.
async fn start_and_offer(
    name: &str,
    cfg: &StdioConfig,
    target: &LaunchTarget,
) -> Result<Arc<McpServer>, McpError> {
    let server = Arc::new(McpServer::start(name.to_owned(), cfg.clone()).await?);
    match adapt_client_as_tools(server.clone(), name, &target.approvals, &target.vision).await {
        Ok(tools) => {
            info!(target: "assistd::mcp", server = %name, tools = tools.len(), "MCP server ready");
            target.catalog.extend(tools);
            Ok(server)
        }
        Err(error) => {
            server.shutdown().await;
            Err(error)
        }
    }
}

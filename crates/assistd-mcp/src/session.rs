//! The transport-independent half of an MCP session: the `rmcp` peer requests go through,
//! the lifeline that ends with the session, and the connector that opens both.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ClientConfig, Implementation, Tool,
};
use rmcp::service::{Peer, RoleClient, ServiceError};
use serde_json::Value;

use crate::McpClient;
use crate::error::McpError;
use crate::stdio::{self, ChildLifeline, StdioConfig};

/// The requesting side of an open session; every request is bounded by `request_timeout`.
#[derive(Debug)]
pub(crate) struct McpPeer {
    peer: Peer<RoleClient>,
    request_timeout: Duration,
}

impl McpPeer {
    pub(crate) fn new(peer: Peer<RoleClient>, request_timeout: Duration) -> Self {
        Self {
            peer,
            request_timeout,
        }
    }

    async fn within_timeout<T>(
        &self,
        request: impl Future<Output = Result<T, ServiceError>>,
    ) -> Result<T, McpError> {
        tokio::time::timeout(self.request_timeout, request)
            .await
            .map_err(|_| McpError::RequestTimeout(self.request_timeout))?
            .map_err(McpError::from)
    }
}

#[async_trait]
impl McpClient for McpPeer {
    async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        self.within_timeout(self.peer.list_all_tools()).await
    }

    async fn invoke(&self, name: &str, arguments: Value) -> Result<CallToolResult, McpError> {
        let mut params = CallToolRequestParams::new(name.to_owned());
        params.arguments = match arguments {
            Value::Object(arguments) => Some(arguments),
            Value::Null => None,
            _ => return Err(McpError::ArgumentsNotObject),
        };
        self.within_timeout(self.peer.call_tool(params)).await
    }
}

/// What keeps a session alive: it resolves once the session ends and can be told to stop.
#[derive(Debug)]
pub(crate) enum Lifeline {
    Stdio(ChildLifeline),
}

impl Lifeline {
    /// Resolves when the session can no longer answer.
    pub(crate) async fn wait_for_end(&mut self) {
        match self {
            Self::Stdio(child) => child.wait_for_death().await,
        }
    }

    /// Close the session, giving it `term_timeout` to wind down before it is forced.
    pub(crate) async fn shutdown(self, term_timeout: Duration) {
        match self {
            Self::Stdio(child) => child.shutdown(term_timeout).await,
        }
    }
}

/// How to open a fresh session to one configured server.
#[derive(Debug)]
pub(crate) enum Connector {
    Stdio(StdioConfig),
}

impl Connector {
    /// Open a session and run the initialize handshake; errors if either fails.
    pub(crate) async fn open(&self) -> Result<(Arc<McpPeer>, Lifeline), McpError> {
        match self {
            Self::Stdio(cfg) => {
                let (peer, child) = stdio::connect(cfg).await?;
                Ok((Arc::new(peer), Lifeline::Stdio(child)))
            }
        }
    }
}

/// The client identity and capabilities sent in every initialize handshake.
pub(crate) fn client_config() -> ClientConfig {
    ClientConfig::new(
        ClientCapabilities::default(),
        Implementation::new("assistd", env!("CARGO_PKG_VERSION")),
    )
}

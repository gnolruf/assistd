//! The transport-independent half of an MCP session: the `rmcp` peer requests go through,
//! the lifeline that ends with the session, and the connector that opens both.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ClientConfig, Implementation, Tool,
};
use rmcp::service::{
    Peer, RoleClient, RunningService, RunningServiceCancellationToken, ServiceError,
};
use serde_json::Value;

use crate::error::McpError;
use crate::stdio::{self, ChildLifeline, StdioConfig};
use crate::streamable::{self, HttpConfig, HttpLifeline};
use crate::{Definition, McpClient};

/// Longest message read from a server, as a stdio line, a JSON body or one SSE event: a
/// largest accepted image base64-encoded, plus envelope.
pub(crate) const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

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

    async fn invoke(
        &self,
        name: &str,
        arguments: Value,
        _definition: Definition,
    ) -> Result<CallToolResult, McpError> {
        let mut params = CallToolRequestParams::new(name.to_owned());
        params.arguments = match arguments {
            Value::Object(arguments) => Some(arguments),
            Value::Null => None,
            _ => return Err(McpError::ArgumentsNotObject),
        };
        self.within_timeout(self.peer.call_tool(params)).await
    }
}

/// The running `rmcp` service of an open session. Waiting for its end does not give it up, so
/// a session still running can be closed.
pub(crate) struct OpenService {
    /// `None` once the service has stopped.
    running: Option<BoxFuture<'static, ()>>,
    canceller: RunningServiceCancellationToken,
}

impl OpenService {
    pub(crate) fn new(service: RunningService<RoleClient, ClientConfig>) -> Self {
        let canceller = service.cancellation_token();
        let running = async move {
            let _ = service.waiting().await;
        };
        Self {
            running: Some(running.boxed()),
            canceller,
        }
    }

    /// Resolves once the service has stopped. Dropped before then, it can be awaited again.
    pub(crate) async fn wait_for_end(&mut self) {
        if let Some(running) = &mut self.running {
            running.await;
            self.running = None;
        }
    }

    /// Stop a service still running and wait up to `timeout` for its transport to close.
    pub(crate) async fn close(self, timeout: Duration) {
        let Some(running) = self.running else {
            return;
        };
        self.canceller.cancel();
        let _ = tokio::time::timeout(timeout, running).await;
    }
}

impl fmt::Debug for OpenService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenService")
            .field("stopped", &self.running.is_none())
            .finish_non_exhaustive()
    }
}

/// What keeps a session alive: it resolves once the session ends and can be told to stop.
#[derive(Debug)]
pub(crate) enum Lifeline {
    Stdio(Box<ChildLifeline>),
    Http(HttpLifeline),
}

impl Lifeline {
    /// Resolves when the session can no longer answer.
    pub(crate) async fn wait_for_end(&mut self) {
        match self {
            Self::Stdio(child) => child.wait_for_death().await,
            Self::Http(session) => session.wait_for_end().await,
        }
    }

    /// Close the session, giving it `term_timeout` to wind down before it is forced.
    pub(crate) async fn shutdown(self, term_timeout: Duration) {
        match self {
            Self::Stdio(child) => child.shutdown(term_timeout).await,
            Self::Http(session) => session.shutdown(term_timeout).await,
        }
    }
}

/// How to open a fresh session to one configured server: spawn it, or reach it by URL.
#[derive(Debug, Clone)]
pub enum Connector {
    Stdio(StdioConfig),
    Http(HttpConfig),
}

impl Connector {
    /// Open a session and run the initialize handshake; errors if either fails.
    pub(crate) async fn open(&self) -> Result<(Arc<McpPeer>, Lifeline), McpError> {
        match self {
            Self::Stdio(cfg) => {
                let (peer, child) = stdio::connect(cfg).await?;
                Ok((Arc::new(peer), Lifeline::Stdio(Box::new(child))))
            }
            Self::Http(cfg) => {
                let (peer, session) = streamable::connect(cfg).await?;
                Ok((Arc::new(peer), Lifeline::Http(session)))
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

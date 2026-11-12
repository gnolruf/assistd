//! `McpLogin`, `McpLogout` and `McpStatus` handlers, which reach the MCP servers through the
//! daemon's [`McpAccounts`].

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::BoxFuture;
use thiserror::Error;

use assistd_ipc::{Event, EventBody, McpServerState, StartupComponent, TokenRevocation};
use assistd_tools::ToolCatalog;

use super::readiness::publish_readiness;
use super::send_error;
use super::subsystems::McpServerStatus;
use crate::bus::{EventBus, EventSink};

/// Why MCP is off, for requests that need a server.
const MCP_DISABLED: &str = "MCP is disabled (mcp.enabled = false) or has no sandbox to run in";

/// Signs OAuth MCP servers in and out, restarting or stopping each to match.
#[async_trait]
pub trait McpAccounts: fmt::Debug + Send + Sync {
    /// Start signing `server` in through the browser; its state is recorded on `host`.
    async fn begin_login(&self, host: McpHost, server: &str) -> Result<McpLogin, McpAccountError>;

    /// Stop `server` and delete its token, revoking it where the authorization server offers
    /// that; its state is recorded on `host`.
    async fn logout(&self, host: McpHost, server: &str)
    -> Result<TokenRevocation, McpAccountError>;
}

/// What an MCP server's start, sign-in or sign-out changes in the daemon: the catalog its
/// tools join and leave, and its recorded state.
#[derive(Debug, Clone)]
pub struct McpHost {
    catalog: Arc<ToolCatalog>,
    servers: Arc<[McpServerStatus]>,
    bus: EventBus,
}

impl McpHost {
    pub(super) fn new(
        catalog: Arc<ToolCatalog>,
        servers: Arc<[McpServerStatus]>,
        bus: EventBus,
    ) -> Self {
        Self {
            catalog,
            servers,
            bus,
        }
    }

    /// The catalog the servers' tools are offered through.
    pub fn catalog(&self) -> &Arc<ToolCatalog> {
        &self.catalog
    }

    /// Record `state` for the configured server `name` and broadcast its readiness; a
    /// name that is not configured is ignored.
    pub fn record_state(&self, name: &str, state: McpServerState) {
        let Some(status) = self.servers.iter().find(|server| server.name() == name) else {
            return;
        };
        status.set(state);
        let component = StartupComponent::Mcp {
            server: name.to_string(),
        };
        publish_readiness(&self.bus, component, status.readiness());
    }
}

/// A sign-in waiting on the user.
pub struct McpLogin {
    /// The URL the user opens to sign in.
    pub authorize_url: String,
    /// Resolves once the redirect is redeemed and the restarted server's first start settles.
    pub completion: BoxFuture<'static, Result<(), McpAccountError>>,
}

/// Why signing an MCP server in or out failed.
#[derive(Debug, Error)]
pub enum McpAccountError {
    #[error("no MCP server named '{0}' is configured")]
    UnknownServer(String),
    #[error("MCP server '{0}' does not use OAuth; only `auth = \"oauth\"` servers sign in")]
    NotOAuth(String),
    #[error("there is no XDG state directory to keep OAuth tokens in")]
    NoTokenStore,
    #[error("the daemon is shutting down")]
    ShuttingDown,
    #[error(transparent)]
    Mcp(Box<dyn StdError + Send + Sync>),
}

impl fmt::Debug for McpLogin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpLogin").finish_non_exhaustive()
    }
}

/// Handles the MCP sign-in, sign-out and status requests over the configured servers.
#[derive(Debug)]
pub struct McpService {
    pub(super) host: McpHost,
    /// Signs OAuth servers in and out; `None` while MCP is off.
    pub(super) accounts: Option<Arc<dyn McpAccounts>>,
}

impl McpService {
    /// What the configured servers' starts and sign-ins act on.
    pub fn host(&self) -> &McpHost {
        &self.host
    }

    pub(super) async fn handle_mcp_login(&self, id: String, server: String, tx: EventSink) {
        let Some(accounts) = self.accounts.clone() else {
            return send_error(&tx, id, MCP_DISABLED.into()).await;
        };
        let login = match accounts.begin_login(self.host.clone(), &server).await {
            Ok(login) => login,
            Err(e) => return send_error(&tx, id, e.to_string()).await,
        };
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::McpAuthorizeUrl {
                    server: server.clone(),
                    url: login.authorize_url,
                },
            ))
            .await;
        if let Err(e) = login.completion.await {
            return send_error(&tx, id, e.to_string()).await;
        }
        if let Some(event) = self.mcp_server_event(&id, &server) {
            let _ = tx.send(event).await;
        }
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }

    pub(super) async fn handle_mcp_logout(&self, id: String, server: String, tx: EventSink) {
        let Some(accounts) = self.accounts.clone() else {
            return send_error(&tx, id, MCP_DISABLED.into()).await;
        };
        match accounts.logout(self.host.clone(), &server).await {
            Ok(revocation) => {
                let _ = tx
                    .send(Event::new(
                        id.clone(),
                        EventBody::McpLoggedOut { server, revocation },
                    ))
                    .await;
                let _ = tx.send(Event::new(id, EventBody::Done)).await;
            }
            Err(e) => send_error(&tx, id, e.to_string()).await,
        }
    }

    pub(super) async fn handle_mcp_status(&self, id: String, tx: EventSink) {
        for server in self.host.servers.iter() {
            if let Some(event) = self.mcp_server_event(&id, server.name()) {
                let _ = tx.send(event).await;
            }
        }
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }

    fn mcp_server_event(&self, id: &str, name: &str) -> Option<Event> {
        let server = self
            .host
            .servers
            .iter()
            .find(|server| server.name() == name)?;
        Some(Event::new(
            id.to_owned(),
            EventBody::McpServer {
                server: name.to_owned(),
                transport: server.transport(),
                state: server.state(),
            },
        ))
    }
}

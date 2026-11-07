//! `McpLogin`, `McpLogout` and `McpStatus` handlers, which reach the MCP servers through the
//! daemon's [`McpAccounts`].

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::BoxFuture;
use thiserror::Error;
use tokio::sync::mpsc;

use assistd_ipc::{Event, TokenRevocation};

use super::{AppState, send_error};

/// Why MCP is off, for requests that need a server.
const MCP_DISABLED: &str = "MCP is disabled (mcp.enabled = false) or has no sandbox to run in";

/// Signs OAuth MCP servers in and out, restarting or stopping each to match.
#[async_trait]
pub trait McpAccounts: fmt::Debug + Send + Sync {
    /// Start signing `server` in through the browser; its readiness is recorded on `state`.
    async fn begin_login(
        &self,
        state: Arc<AppState>,
        server: &str,
    ) -> Result<McpLogin, McpAccountError>;

    /// Stop `server` and delete its token, revoking it where the authorization server offers
    /// that; its readiness is recorded on `state`.
    async fn logout(
        &self,
        state: Arc<AppState>,
        server: &str,
    ) -> Result<TokenRevocation, McpAccountError>;
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

impl AppState {
    pub(super) async fn handle_mcp_login(
        self: Arc<Self>,
        id: String,
        server: String,
        tx: mpsc::Sender<Event>,
    ) {
        let Some(accounts) = self.subsystems.mcp_accounts.clone() else {
            return send_error(&tx, id, MCP_DISABLED.into()).await;
        };
        let login = match accounts.begin_login(self.clone(), &server).await {
            Ok(login) => login,
            Err(e) => return send_error(&tx, id, e.to_string()).await,
        };
        let _ = tx
            .send(Event::McpAuthorizeUrl {
                id: id.clone(),
                server: server.clone(),
                url: login.authorize_url,
            })
            .await;
        if let Err(e) = login.completion.await {
            return send_error(&tx, id, e.to_string()).await;
        }
        if let Some(event) = self.mcp_server_event(&id, &server) {
            let _ = tx.send(event).await;
        }
        let _ = tx.send(Event::Done { id }).await;
    }

    pub(super) async fn handle_mcp_logout(
        self: Arc<Self>,
        id: String,
        server: String,
        tx: mpsc::Sender<Event>,
    ) {
        let Some(accounts) = self.subsystems.mcp_accounts.clone() else {
            return send_error(&tx, id, MCP_DISABLED.into()).await;
        };
        match accounts.logout(self.clone(), &server).await {
            Ok(revocation) => {
                let _ = tx
                    .send(Event::McpLoggedOut {
                        id: id.clone(),
                        server,
                        revocation,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
            }
            Err(e) => send_error(&tx, id, e.to_string()).await,
        }
    }

    pub(super) async fn handle_mcp_status(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        for server in &self.subsystems.mcp_servers {
            if let Some(event) = self.mcp_server_event(&id, server.name()) {
                let _ = tx.send(event).await;
            }
        }
        let _ = tx.send(Event::Done { id }).await;
    }

    fn mcp_server_event(&self, id: &str, name: &str) -> Option<Event> {
        let server = self
            .subsystems
            .mcp_servers
            .iter()
            .find(|server| server.name() == name)?;
        Some(Event::McpServer {
            id: id.to_owned(),
            server: name.to_owned(),
            transport: server.transport(),
            state: server.state(),
        })
    }
}

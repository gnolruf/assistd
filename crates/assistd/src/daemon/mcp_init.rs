//! MCP subsystem wiring for the daemon: servers start in the background, each one's tools are
//! offered once it has connected, and OAuth servers are signed in and out on request.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use assistd_core::config::{
    McpAuth, McpHttpTransport, McpOAuthClient, McpServerUrl, McpStdioTransport, McpTransport,
};
use assistd_core::{
    AppState, Config, McpAccountError, McpAccounts, McpLogin, McpServerConfig, McpServerStatus,
};
use assistd_ipc::{McpServerState, McpTransportKind, StartupComponent, TokenRevocation};
use assistd_mcp::{
    Connector, HttpAuth, HttpConfig, LaunchTarget, McpError, McpServer, OAuthConfig, Revocation,
    StdioConfig, TokenStore, begin_login, launch, sign_out, withdraw_tools,
};
use assistd_tools::{ApprovalGate, VisionGate};
use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::{Mutex as AsyncMutex, oneshot};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::info;

/// Every configured server, started in the background and signed in and out on request.
#[derive(Debug)]
pub(super) struct McpFleet {
    servers: Vec<Arc<ManagedServer>>,
    approvals: ApprovalGate,
    vision: Arc<VisionGate>,
    /// Where `auth = "oauth"` servers keep their tokens; `None` without a state directory.
    tokens: Option<TokenStore>,
    stopping: CancellationToken,
}

/// The fleet's background start, stopped at shutdown with every server it started.
pub(super) struct McpService {
    fleet: Option<Arc<McpFleet>>,
}

/// One configured server: its pending start, if any, and the session it started.
#[derive(Debug)]
struct ManagedServer {
    config: McpServerConfig,
    starting: Mutex<Option<AbortOnDropHandle<()>>>,
    running: Mutex<Option<Arc<McpServer>>>,
    /// Held while the server is replaced or stopped, so those never interleave.
    changing: AsyncMutex<()>,
}

/// Everything one start of a server needs.
struct Start {
    server: Arc<ManagedServer>,
    connector: Connector,
    target: LaunchTarget,
    state: Arc<AppState>,
}

impl McpFleet {
    /// Start every server at once, retrying failed ones and recording each one's state as it
    /// settles.
    pub(super) fn spawn(self: Arc<Self>, state: &Arc<AppState>) -> McpService {
        for server in &self.servers {
            let Some(connector) = connector(&server.config, self.tokens.clone()) else {
                record_state(state, &server.config.name, no_token_store());
                continue;
            };
            let start = Start {
                server: server.clone(),
                connector,
                target: self.target(state),
                state: state.clone(),
            };
            *server.starting.lock() = Some(start.spawn(None));
        }
        McpService { fleet: Some(self) }
    }

    fn target(&self, state: &AppState) -> LaunchTarget {
        LaunchTarget {
            catalog: state.subsystems.tools.clone(),
            approvals: self.approvals.clone(),
            vision: self.vision.clone(),
        }
    }

    fn server(&self, name: &str) -> Result<&Arc<ManagedServer>, McpAccountError> {
        self.servers
            .iter()
            .find(|server| server.config.name == name)
            .ok_or_else(|| McpAccountError::UnknownServer(name.to_owned()))
    }

    /// The OAuth client and URL of `server`, or why it cannot sign in.
    fn oauth_endpoint<'a>(
        &self,
        server: &'a ManagedServer,
    ) -> Result<(&'a McpServerUrl, OAuthConfig), McpAccountError> {
        let not_oauth = || McpAccountError::NotOAuth(server.config.name.clone());
        let McpTransport::Http(http) = &server.config.transport else {
            return Err(not_oauth());
        };
        let McpAuth::OAuth(client) = &http.auth else {
            return Err(not_oauth());
        };
        let tokens = self.tokens.clone().ok_or(McpAccountError::NoTokenStore)?;
        Ok((&http.url, oauth_config(client, tokens)))
    }

    /// Cancel every pending start and stop every server.
    async fn shutdown(&self) {
        self.stopping.cancel();
        for server in &self.servers {
            let _changing = server.changing.lock().await;
            server.stop().await;
        }
    }
}

#[async_trait]
impl McpAccounts for McpFleet {
    async fn begin_login(
        &self,
        state: Arc<AppState>,
        name: &str,
    ) -> Result<McpLogin, McpAccountError> {
        let server = self.server(name)?;
        let (url, oauth) = self.oauth_endpoint(server)?;
        let request_timeout = request_timeout(&server.config);
        let pending = begin_login(name, url.as_url(), &oauth, request_timeout)
            .await
            .map_err(mcp_failure)?;
        let connector =
            connector(&server.config, self.tokens.clone()).ok_or(McpAccountError::NoTokenStore)?;
        let start = Start {
            server: server.clone(),
            connector,
            target: self.target(&state),
            state,
        };
        let stopping = self.stopping.clone();
        Ok(McpLogin {
            authorize_url: pending.authorize_url().to_owned(),
            completion: Box::pin(async move {
                pending.complete().await.map_err(mcp_failure)?;
                start.replace(&stopping).await
            }),
        })
    }

    async fn logout(
        &self,
        state: Arc<AppState>,
        name: &str,
    ) -> Result<TokenRevocation, McpAccountError> {
        let server = self.server(name)?;
        let (url, oauth) = self.oauth_endpoint(server)?;
        let _changing = server.changing.lock().await;
        server.stop().await;
        withdraw_tools(&state.subsystems.tools, name);
        record_state(
            &state,
            name,
            McpServerState::NeedsAuthorization { scopes: Vec::new() },
        );
        let revocation = sign_out(name, url.as_url(), &oauth, request_timeout(&server.config))
            .await
            .map_err(mcp_failure)?;
        info!(target: "assistd::mcp", server = %name, "MCP server signed out");
        Ok(wire_revocation(revocation))
    }
}

impl McpService {
    pub(super) fn disabled() -> Self {
        Self { fleet: None }
    }

    /// Stop every pending start and every server. The `tools` shutdown stage must already
    /// have flipped.
    pub(super) async fn shutdown(self) {
        if let Some(fleet) = self.fleet {
            fleet.shutdown().await;
        }
    }
}

impl ManagedServer {
    /// Cancel the pending start, waiting for it to end, then stop the running session.
    async fn stop(&self) {
        let pending = self.starting.lock().take();
        if let Some(pending) = pending {
            pending.abort();
            let _ = pending.await;
        }
        let running = self.running.lock().take();
        if let Some(running) = running {
            running.shutdown().await;
        }
    }
}

impl Start {
    /// Run the start in the background, firing `settled` when its first attempt settles.
    fn spawn(self, settled: Option<oneshot::Sender<()>>) -> AbortOnDropHandle<()> {
        AbortOnDropHandle::new(tokio::spawn(self.run(settled)))
    }

    async fn run(self, mut settled: Option<oneshot::Sender<()>>) {
        let name = self.server.config.name.clone();
        let on_retry = |e: &McpError| {
            record_state(&self.state, &name, failed_state(e));
            if let Some(settled) = settled.take() {
                let _ = settled.send(());
            }
        };
        let outcome = launch(name.clone(), self.connector, &self.target, on_retry).await;
        match outcome {
            Ok(started) => {
                *self.server.running.lock() = Some(started);
                record_state(&self.state, &name, McpServerState::Ready);
            }
            Err(e) => record_state(&self.state, &name, failed_state(&e)),
        }
        if let Some(settled) = settled {
            let _ = settled.send(());
        }
    }

    /// Stop the server and withdraw its tools, then start it afresh and wait for the first
    /// attempt to settle. Refused once the daemon is shutting down.
    async fn replace(self, stopping: &CancellationToken) -> Result<(), McpAccountError> {
        let server = self.server.clone();
        let name = server.config.name.clone();
        let (settled_tx, settled_rx) = oneshot::channel();
        {
            let _changing = server.changing.lock().await;
            if stopping.is_cancelled() {
                return Err(McpAccountError::ShuttingDown);
            }
            server.stop().await;
            withdraw_tools(&self.target.catalog, &name);
            record_state(&self.state, &name, McpServerState::Starting);
            *server.starting.lock() = Some(self.spawn(Some(settled_tx)));
        }
        let _ = settled_rx.await;
        Ok(())
    }
}

/// A status per configured server and the fleet that starts and signs them in, whose tools ask
/// before each call unless `approvals` holds them and whose image results follow `vision`.
/// No fleet when MCP is disabled.
pub(super) fn prepare(
    config: &Config,
    approvals: ApprovalGate,
    vision: Arc<VisionGate>,
) -> (Vec<McpServerStatus>, Option<Arc<McpFleet>>) {
    if !config.mcp.enabled {
        info!("mcp: disabled in config (mcp.enabled = false)");
        return (Vec::new(), None);
    }
    let statuses = config
        .mcp
        .servers
        .iter()
        .map(|server| McpServerStatus::starting(server.name.clone(), transport_kind(server)))
        .collect();
    let servers = config
        .mcp
        .servers
        .iter()
        .map(|config| {
            Arc::new(ManagedServer {
                config: config.clone(),
                starting: Mutex::new(None),
                running: Mutex::new(None),
                changing: AsyncMutex::new(()),
            })
        })
        .collect();
    let fleet = McpFleet {
        servers,
        approvals,
        vision,
        tokens: token_file().map(TokenStore::new),
        stopping: CancellationToken::new(),
    };
    (statuses, Some(Arc::new(fleet)))
}

fn record_state(state: &AppState, name: &str, server_state: McpServerState) {
    let Some(status) = state
        .subsystems
        .mcp_servers
        .iter()
        .find(|s| s.name() == name)
    else {
        return;
    };
    status.set(server_state);
    state.publish_readiness(
        StartupComponent::Mcp {
            server: name.to_string(),
        },
        status.readiness(),
    );
}

fn failed_state(error: &McpError) -> McpServerState {
    match error {
        McpError::NeedsAuthorization { scopes } => McpServerState::NeedsAuthorization {
            scopes: scopes.clone(),
        },
        other => McpServerState::Unavailable {
            reason: other.to_string(),
        },
    }
}

fn no_token_store() -> McpServerState {
    McpServerState::Unavailable {
        reason: McpAccountError::NoTokenStore.to_string(),
    }
}

fn mcp_failure(error: McpError) -> McpAccountError {
    McpAccountError::Mcp(Box::new(error))
}

fn wire_revocation(revocation: Revocation) -> TokenRevocation {
    match revocation {
        Revocation::Revoked => TokenRevocation::Revoked,
        Revocation::NotOffered => TokenRevocation::NotOffered,
        Revocation::NothingStored => TokenRevocation::NothingStored,
        Revocation::Failed(reason) => TokenRevocation::Failed { reason },
    }
}

fn transport_kind(server: &McpServerConfig) -> McpTransportKind {
    match server.transport {
        McpTransport::Stdio(_) => McpTransportKind::Stdio,
        McpTransport::Http(_) => McpTransportKind::Http,
    }
}

fn request_timeout(server: &McpServerConfig) -> Duration {
    Duration::from_secs(server.request_timeout_secs.get())
}

/// How to reach `server`; `None` for an OAuth server when there is nowhere to keep `tokens`.
fn connector(server: &McpServerConfig, tokens: Option<TokenStore>) -> Option<Connector> {
    let request_timeout = request_timeout(server);
    match &server.transport {
        McpTransport::Stdio(transport) => Some(Connector::Stdio(stdio_config(
            server,
            transport,
            request_timeout,
        ))),
        McpTransport::Http(transport) => {
            let mut http = http_config(server, transport, request_timeout);
            if let McpAuth::OAuth(client) = &transport.auth {
                http.auth = HttpAuth::OAuth(oauth_config(client, tokens?));
            }
            Some(Connector::Http(http))
        }
    }
}

fn oauth_config(client: &McpOAuthClient, tokens: TokenStore) -> OAuthConfig {
    OAuthConfig {
        client_id: client.client_id.clone(),
        client_secret: client.client_secret.clone(),
        scopes: client.scopes.clone(),
        tokens,
    }
}

fn stdio_config(
    server: &McpServerConfig,
    transport: &McpStdioTransport,
    request_timeout: Duration,
) -> StdioConfig {
    let mut stdio = StdioConfig::new(
        server.name.clone(),
        transport.command.to_string_lossy().into_owned(),
    );
    stdio.args.clone_from(&transport.args);
    stdio.env.clone_from(&transport.env);
    stdio.request_timeout = request_timeout;
    stdio
}

fn http_config(
    server: &McpServerConfig,
    transport: &McpHttpTransport,
    request_timeout: Duration,
) -> HttpConfig {
    let mut http = HttpConfig::new(server.name.clone(), transport.url.as_url().clone());
    http.headers = transport.headers.to_header_map();
    http.request_timeout = request_timeout;
    http
}

/// `$XDG_STATE_HOME/assistd/mcp-tokens.json`; `None` when there is no state directory.
fn token_file() -> Option<PathBuf> {
    dirs::state_dir().map(|state| state.join("assistd").join("mcp-tokens.json"))
}

//! MCP subsystem wiring for the daemon: servers start in the background, each one's tools are
//! offered once it has connected, and OAuth servers are signed in and out on request.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use assistd_config::{
    AbsoluteDir, Config, McpAuth, McpHttpTransport, McpOAuthClient, McpServerConfig, McpServerUrl,
    McpStdioTransport, McpTransport,
};
use assistd_core::{McpAccountError, McpAccounts, McpHost, McpLogin, McpServerStatus};
use assistd_ipc::{McpServerState, McpTransportKind, TokenRevocation};
use assistd_mcp::{
    Connector, HttpAuth, HttpConfig, LaunchTarget, McpError, McpServer, OAuthConfig, Revocation,
    StdioConfig, TokenStore, begin_login, launch, sign_out, withdraw_tools,
};
use assistd_tools::{PinnedApprovalGate, SandboxInfo, SandboxedServer, ServerSandbox, VisionGate};
use async_trait::async_trait;
use futures_util::future::join_all;
use parking_lot::Mutex;
use tokio::sync::{Mutex as AsyncMutex, oneshot};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::info;

/// Every configured server, started in the background and signed in and out on request.
#[derive(Debug)]
pub(super) struct McpFleet {
    servers: Vec<Arc<ManagedServer>>,
    approvals: PinnedApprovalGate,
    vision: Arc<VisionGate>,
    /// Where `auth = "oauth"` servers keep their tokens; `None` without a state directory.
    tokens: Option<TokenStore>,
    /// What every stdio server runs in.
    sandbox: Arc<ServerSandbox>,
    /// Holds each stdio server's home, under its name; `None` without a data directory.
    homes: Option<PathBuf>,
    stopping: CancellationToken,
}

/// Why a configured server has no way to be reached.
#[derive(Debug, Clone, Copy)]
enum Unreachable {
    NoTokenStore,
    NoDataDir,
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
    host: McpHost,
}

impl McpFleet {
    /// Start every server at once, retrying failed ones and recording each one's state on
    /// `host` as it settles.
    pub(super) fn spawn(self: Arc<Self>, host: &McpHost) -> McpService {
        for server in &self.servers {
            let connector = match self.connector(&server.config) {
                Ok(connector) => connector,
                Err(unreachable) => {
                    let reason = unreachable.reason();
                    host.record_state(
                        server.config.name.as_str(),
                        McpServerState::Unavailable { reason },
                    );
                    continue;
                }
            };
            let start = Start {
                server: server.clone(),
                connector,
                target: self.target(host),
                host: host.clone(),
            };
            *server.starting.lock() = Some(start.spawn(None));
        }
        McpService { fleet: Some(self) }
    }

    fn target(&self, host: &McpHost) -> LaunchTarget {
        LaunchTarget {
            catalog: Arc::clone(host.catalog()),
            approvals: self.approvals.clone(),
            vision: self.vision.clone(),
        }
    }

    fn server(&self, name: &str) -> Result<&Arc<ManagedServer>, McpAccountError> {
        self.servers
            .iter()
            .find(|server| server.config.name.as_str() == name)
            .ok_or_else(|| McpAccountError::UnknownServer(name.to_owned()))
    }

    /// The OAuth client and URL of `server`, or why it cannot sign in.
    fn oauth_endpoint<'a>(
        &self,
        server: &'a ManagedServer,
    ) -> Result<(&'a McpServerUrl, OAuthConfig), McpAccountError> {
        let not_oauth = || McpAccountError::NotOAuth(server.config.name.to_string());
        let McpTransport::Http(http) = &server.config.transport else {
            return Err(not_oauth());
        };
        let McpAuth::OAuth(client) = &http.auth else {
            return Err(not_oauth());
        };
        let tokens = self.tokens.clone().ok_or(McpAccountError::NoTokenStore)?;
        Ok((&http.url, oauth_config(client, tokens)))
    }

    /// How to reach `server`, or why it cannot be reached.
    fn connector(&self, server: &McpServerConfig) -> Result<Connector, Unreachable> {
        let request_timeout = request_timeout(server);
        match &server.transport {
            McpTransport::Stdio(transport) => {
                let homes = self.homes.as_ref().ok_or(Unreachable::NoDataDir)?;
                let mut stdio = StdioConfig::sandboxed(
                    server.name.to_string(),
                    Arc::clone(&self.sandbox),
                    sandboxed_server(transport, homes.join(server.name.as_str())),
                );
                stdio.request_timeout = request_timeout;
                Ok(Connector::Stdio(stdio))
            }
            McpTransport::Http(transport) => {
                let mut http = http_config(server, transport, request_timeout);
                if let McpAuth::OAuth(client) = &transport.auth {
                    let tokens = self.tokens.clone().ok_or(Unreachable::NoTokenStore)?;
                    http.auth = HttpAuth::OAuth(oauth_config(client, tokens));
                }
                Ok(Connector::Http(Box::new(http)))
            }
        }
    }

    /// Cancel every pending start and stop every server.
    async fn shutdown(&self) {
        self.stopping.cancel();
        join_all(self.servers.iter().map(|server| async move {
            let _changing = server.changing.lock().await;
            server.stop().await;
        }))
        .await;
    }
}

#[async_trait]
impl McpAccounts for McpFleet {
    async fn begin_login(&self, host: McpHost, name: &str) -> Result<McpLogin, McpAccountError> {
        let server = self.server(name)?;
        let (url, oauth) = self.oauth_endpoint(server)?;
        let request_timeout = request_timeout(&server.config);
        let pending = begin_login(name, url.as_url(), &oauth, request_timeout)
            .await
            .map_err(mcp_failure)?;
        let connector = self
            .connector(&server.config)
            .map_err(|_| McpAccountError::NoTokenStore)?;
        let start = Start {
            server: server.clone(),
            connector,
            target: self.target(&host),
            host,
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

    async fn logout(&self, host: McpHost, name: &str) -> Result<TokenRevocation, McpAccountError> {
        let server = self.server(name)?;
        let (url, oauth) = self.oauth_endpoint(server)?;
        let _changing = server.changing.lock().await;
        server.stop().await;
        withdraw_tools(host.catalog(), &server.config.name);
        host.record_state(
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

impl Unreachable {
    fn reason(self) -> String {
        match self {
            Self::NoTokenStore => McpAccountError::NoTokenStore.to_string(),
            Self::NoDataDir => {
                "there is no user data directory to keep the server's home in".to_owned()
            }
        }
    }
}

impl McpService {
    pub(super) fn disabled() -> Self {
        Self { fleet: None }
    }

    /// Stop every pending start and every server.
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
            self.host.record_state(name.as_str(), failed_state(e));
            if let Some(settled) = settled.take() {
                let _ = settled.send(());
            }
        };
        let outcome = launch(name.clone(), self.connector, &self.target, on_retry).await;
        match outcome {
            Ok(started) => {
                *self.server.running.lock() = Some(started);
                self.host.record_state(name.as_str(), McpServerState::Ready);
            }
            Err(e) => self.host.record_state(name.as_str(), failed_state(&e)),
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
            self.host
                .record_state(name.as_str(), McpServerState::Starting);
            *server.starting.lock() = Some(self.spawn(Some(settled_tx)));
        }
        let _ = settled_rx.await;
        Ok(())
    }
}

/// A status per configured server and the fleet that starts and signs them in, whose tools ask
/// before each call unless `approvals` holds their definitions and whose image results follow `vision`.
/// Stdio servers run in a sandbox started as `sandbox` starts commands, with the directory of
/// `config_path` hidden. No fleet when MCP is disabled.
pub(super) fn prepare(
    config: &Config,
    config_path: &Path,
    sandbox: &SandboxInfo,
    approvals: PinnedApprovalGate,
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
        .map(|server| McpServerStatus::starting(server.name.to_string(), transport_kind(server)))
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
        sandbox: Arc::new(ServerSandbox::over(sandbox, &own_files(config_path))),
        homes: dirs::data_dir().map(|data| data.join("assistd").join("mcp")),
        stopping: CancellationToken::new(),
    };
    (statuses, Some(Arc::new(fleet)))
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

fn oauth_config(client: &McpOAuthClient, tokens: TokenStore) -> OAuthConfig {
    OAuthConfig {
        client_id: client.client_id.clone(),
        client_secret: client.client_secret.clone(),
        scopes: client.scopes.clone(),
        tokens,
    }
}

/// What `transport` runs and may reach, with `home` as its `$HOME`.
fn sandboxed_server(transport: &McpStdioTransport, home: PathBuf) -> SandboxedServer {
    let paths = |granted: &[AbsoluteDir]| granted.iter().map(|dir| dir.as_path().into()).collect();
    SandboxedServer {
        command: transport.command.clone(),
        args: transport.args.clone(),
        env: transport.env.clone(),
        home,
        read: paths(&transport.read),
        write: paths(&transport.write),
        network: transport.network,
    }
}

/// The daemon's socket and its config, data, state, cache and runtime directories, which no
/// server may be granted.
fn own_files(config_path: &Path) -> Vec<PathBuf> {
    let user_dirs = [
        dirs::config_dir(),
        dirs::data_dir(),
        dirs::state_dir(),
        dirs::cache_dir(),
        dirs::runtime_dir(),
    ];
    user_dirs
        .into_iter()
        .flatten()
        .map(|dir| dir.join("assistd"))
        .chain(config_path.parent().map(Path::to_path_buf))
        .chain([assistd_ipc::socket_path()])
        .collect()
}

fn http_config(
    server: &McpServerConfig,
    transport: &McpHttpTransport,
    request_timeout: Duration,
) -> HttpConfig {
    let mut http = HttpConfig::new(server.name.to_string(), transport.url.as_url().clone());
    http.headers = transport.headers.to_header_map();
    http.request_timeout = request_timeout;
    http
}

/// `$XDG_STATE_HOME/assistd/mcp-tokens.json`; `None` when there is no state directory.
fn token_file() -> Option<PathBuf> {
    dirs::state_dir().map(|state| state.join("assistd").join("mcp-tokens.json"))
}

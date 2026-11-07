//! MCP subsystem wiring for the daemon: servers start in the background
//! and each one's tools are offered once it has connected.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use assistd_core::config::{McpAuth, McpHttpTransport, McpStdioTransport, McpTransport};
use assistd_core::{AppState, Config, McpServerConfig, McpServerStatus};
use assistd_ipc::{ComponentReadiness, StartupComponent};
use assistd_mcp::{
    Connector, HttpAuth, HttpConfig, LaunchTarget, McpError, McpServer, OAuthConfig, StdioConfig,
    TokenStore, launch,
};
use assistd_tools::{ApprovalGate, VisionGate};
use assistd_utils::readiness::Readiness;
use tokio::sync::watch;
use tokio::task::{JoinError, JoinHandle, JoinSet};
use tracing::info;

/// What starting the configured MCP servers in the background needs.
pub(super) struct McpStartup {
    servers: Vec<McpServerConfig>,
    approvals: ApprovalGate,
    vision: Arc<VisionGate>,
    /// Where `auth = "oauth"` servers keep their tokens; `None` without a state directory.
    tokens: Option<TokenStore>,
}

/// The background start, joined at shutdown for the servers it started.
pub(super) struct McpService {
    startup: Option<JoinHandle<Vec<Arc<McpServer>>>>,
}

impl McpStartup {
    /// Start every server at once, retrying failed ones, offering each one's tools and
    /// recording its readiness as it settles. Cancels pending starts once `shutdown` flips.
    pub(super) fn spawn(self, state: Arc<AppState>, shutdown: watch::Receiver<bool>) -> McpService {
        McpService {
            startup: Some(tokio::spawn(start_all(self, state, shutdown))),
        }
    }
}

impl McpService {
    pub(super) fn disabled() -> Self {
        Self { startup: None }
    }

    /// Wait for the background start to stop, then stop every server it
    /// started. The `tools` shutdown stage must already have flipped.
    pub(super) async fn shutdown(self) {
        let Some(startup) = self.startup else {
            return;
        };
        let servers = startup.await.unwrap_or_else(|e| {
            tracing::error!("mcp startup task failed: {e}");
            Vec::new()
        });
        for server in servers {
            server.shutdown().await;
        }
    }
}

/// A status per configured server and what starting them needs, whose
/// tools ask before each call unless `approvals` holds them and whose
/// image results follow `vision`. Nothing to start when MCP is disabled.
pub(super) fn prepare(
    config: &Config,
    approvals: ApprovalGate,
    vision: Arc<VisionGate>,
) -> (Vec<McpServerStatus>, Option<McpStartup>) {
    if !config.mcp.enabled {
        info!("mcp: disabled in config (mcp.enabled = false)");
        return (Vec::new(), None);
    }
    let statuses = config
        .mcp
        .servers
        .iter()
        .map(|server| McpServerStatus::starting(server.name.clone()))
        .collect();
    let startup = McpStartup {
        servers: config.mcp.servers.clone(),
        approvals,
        vision,
        tokens: token_file().map(TokenStore::new),
    };
    (statuses, Some(startup))
}

async fn start_all(
    startup: McpStartup,
    state: Arc<AppState>,
    mut shutdown: watch::Receiver<bool>,
) -> Vec<Arc<McpServer>> {
    let target = LaunchTarget {
        catalog: state.subsystems.tools.clone(),
        approvals: startup.approvals,
        vision: startup.vision,
    };
    let mut launches = JoinSet::new();
    for config in startup.servers {
        launches.spawn(start_server(
            config,
            startup.tokens.clone(),
            state.clone(),
            target.clone(),
        ));
    }
    let mut started = Vec::new();
    loop {
        tokio::select! {
            biased;
            _ = shutdown.wait_for(|stopping| *stopping) => break,
            joined = launches.join_next() => match joined {
                Some(joined) => collect_started(joined, &mut started),
                None => return started,
            },
        }
    }
    launches.abort_all();
    while let Some(joined) = launches.join_next().await {
        collect_started(joined, &mut started);
    }
    started
}

fn collect_started(
    joined: Result<Option<Arc<McpServer>>, JoinError>,
    started: &mut Vec<Arc<McpServer>>,
) {
    match joined {
        Ok(server) => started.extend(server),
        Err(e) if e.is_cancelled() => {}
        Err(e) => tracing::error!("mcp: server start task failed: {e}"),
    }
}

fn record_readiness(state: &AppState, name: &str, readiness: Readiness<()>) {
    let Some(status) = state
        .subsystems
        .mcp_servers
        .iter()
        .find(|s| s.name() == name)
    else {
        return;
    };
    status.set(readiness);
    state.publish_readiness(
        StartupComponent::Mcp {
            server: name.to_string(),
        },
        ComponentReadiness::from(status.readiness()),
    );
}

/// Bring one server up, recording its readiness as attempts settle. `None` when it is
/// unavailable until the config changes.
async fn start_server(
    server: McpServerConfig,
    tokens: Option<TokenStore>,
    state: Arc<AppState>,
    target: LaunchTarget,
) -> Option<Arc<McpServer>> {
    let name = server.name.clone();
    let unavailable = |reason: String| Readiness::Unavailable(reason.into());
    let Some(connector) = connector(&server, tokens) else {
        let reason = "no XDG state directory to keep OAuth tokens in".to_owned();
        record_readiness(&state, &name, unavailable(reason));
        return None;
    };
    let on_retry = |e: &McpError| record_readiness(&state, &name, unavailable(e.to_string()));
    match launch(name.clone(), connector, &target, on_retry).await {
        Ok(started) => {
            record_readiness(&state, &name, Readiness::Ready(()));
            Some(started)
        }
        Err(e) => {
            record_readiness(&state, &name, unavailable(e.to_string()));
            None
        }
    }
}

/// How to reach `server`; `None` for an OAuth server when there is nowhere to keep `tokens`.
fn connector(server: &McpServerConfig, tokens: Option<TokenStore>) -> Option<Connector> {
    let request_timeout = Duration::from_secs(server.request_timeout_secs.get());
    match &server.transport {
        McpTransport::Stdio(transport) => Some(Connector::Stdio(stdio_config(
            server,
            transport,
            request_timeout,
        ))),
        McpTransport::Http(transport) => {
            let mut http = http_config(server, transport, request_timeout);
            if let McpAuth::OAuth(client) = &transport.auth {
                http.auth = HttpAuth::OAuth(OAuthConfig {
                    client_id: client.client_id.clone(),
                    client_secret: client.client_secret.clone(),
                    scopes: client.scopes.clone(),
                    tokens: tokens?,
                });
            }
            Some(Connector::Http(http))
        }
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

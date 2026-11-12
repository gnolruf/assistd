//! Web search wiring for the daemon: the SearXNG instance starts in the
//! background as soon as the socket opens, since it needs no GPU.

use std::path::Path;
use std::sync::Arc;

use assistd_config::SearchConfig;
use assistd_core::{AppState, Config};
use assistd_ipc::{ComponentReadiness, StartupComponent};
use assistd_search::SearxngServerSpec;
use assistd_utils::child_server::{ChildServer, ChildServerStatus};
use assistd_utils::program::find_program;
use assistd_utils::readiness::{Readiness, ReadinessCell};
use tokio::sync::watch;
use tracing::info;

use super::background_start::BackgroundStart;

const SUBSYSTEM: &str = "web search";

/// What starting the SearXNG instance in the background needs.
pub(super) struct SearchStartup {
    config: SearchConfig,
}

/// The background start, joined at shutdown for the instance it brought up.
pub(super) struct SearchService {
    startup: BackgroundStart<ChildServer>,
}

impl SearchStartup {
    /// Start the instance and record its readiness once it settles. Gives up
    /// when `shutdown` flips first.
    pub(super) fn spawn(
        self,
        state: Arc<AppState>,
        shutdown: watch::Receiver<bool>,
    ) -> SearchService {
        let panicked_state = Arc::clone(&state);
        let startup = BackgroundStart::spawn(
            SUBSYSTEM,
            async move { start_and_record(self, &state, shutdown).await },
            move || {
                record_readiness(
                    &panicked_state,
                    Readiness::Unavailable("startup panicked".into()),
                );
            },
        );
        SearchService { startup }
    }
}

impl SearchService {
    pub(super) fn disabled() -> Self {
        Self {
            startup: BackgroundStart::disabled(SUBSYSTEM),
        }
    }

    /// Wait for the background start to stop, then stop the instance. The
    /// `search` shutdown stage must already have flipped.
    pub(super) async fn shutdown(self) {
        if let Some(server) = self.startup.join().await
            && let Err(e) = server.shutdown().await
        {
            tracing::warn!("search-server shutdown error: {e:#}");
        }
    }
}

/// The readiness requests see from startup and, when search is enabled,
/// what starting the instance needs.
pub(super) fn prepare(
    config: &Config,
) -> (Arc<ReadinessCell<ChildServerStatus>>, Option<SearchStartup>) {
    if !config.search.enabled {
        const REASON: &str = "disabled in config (search.enabled = false)";
        info!("web search: {REASON}");
        return (
            Arc::new(ReadinessCell::new(Readiness::Unavailable(REASON.into()))),
            None,
        );
    }
    let startup = SearchStartup {
        config: config.search.clone(),
    };
    (Arc::new(ReadinessCell::starting()), Some(startup))
}

async fn start_and_record(
    startup: SearchStartup,
    state: &AppState,
    shutdown: watch::Receiver<bool>,
) -> Option<ChildServer> {
    let port = startup.config.port;
    let (server, readiness) = match start_server(startup.config, shutdown).await {
        Ok(server) => {
            info!("web search: ready (port={port})");
            let status = server.status();
            (Some(server), Readiness::Ready(status))
        }
        Err(reason) => {
            tracing::warn!("web search: {reason}; web search disabled this run");
            (None, Readiness::Unavailable(reason.into()))
        }
    };
    record_readiness(state, readiness);
    server
}

fn record_readiness(state: &AppState, readiness: Readiness<ChildServerStatus>) {
    state.subsystems.web_search.set(readiness);
    state.status.publish_readiness(
        StartupComponent::WebSearch,
        ComponentReadiness::from(state.subsystems.web_search.get()),
    );
}

/// `Err` is why web search is unavailable. A program missing from `$PATH`
/// fails here rather than through the supervisor's restart attempts.
async fn start_server(
    config: SearchConfig,
    shutdown: watch::Receiver<bool>,
) -> Result<ChildServer, String> {
    let program = config.command.program();
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    find_program(Path::new(program), &path_env)
        .map_err(|e| format!("cannot run search.command `{program}`: {e}"))?;
    let spec = SearxngServerSpec::prepare(config).map_err(|e| e.to_string())?;
    ChildServer::start(spec, shutdown)
        .await
        .map_err(|e| format!("server failed to start: {e:#}"))
}

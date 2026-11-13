//! One configured MCP server: started with the daemon, killed once it stops answering,
//! restarted by the first call after it dies, and stopped on shutdown.

use std::mem;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex;
use rmcp::model::{CallToolResult, Tool};
use serde_json::Value;
use tokio::sync::{Mutex as AsyncMutex, oneshot};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{error, info, warn};

use assistd_utils::backoff::{MIN_HEALTHY_SECONDS, RestartDecision, RestartPolicy};

use crate::McpClient;
use crate::error::McpError;
use crate::session::{Connector, Lifeline, McpPeer};

/// How long a stopping server gets to exit after SIGTERM.
const TERM_TIMEOUT: Duration = Duration::from_secs(10);

/// Requests in a row that time out before a live server counts as hung
/// and is killed.
const HUNG_AFTER_TIMEOUTS: u32 = 2;

/// A configured server. A call that finds its process dead restarts it, or fails with
/// [`McpError::ServerDown`] while backoff says to wait; one whose requests keep timing out is
/// killed, and one that needs authorization has its session closed and fails every later call
/// with that. Calls made while another one restarts it wait for that restart. Dropping it
/// SIGKILLs the live process group.
#[derive(Debug)]
pub struct McpServer {
    name: String,
    connector: Connector,
    state: Mutex<ServerState>,
    /// Held to open a session in place of one that ended, or to stop for good; the lifecycle
    /// leaves `Down` only under it.
    reopening: AsyncMutex<()>,
    stopping: CancellationToken,
}

#[derive(Debug)]
struct ServerState {
    lifecycle: Lifecycle,
    policy: RestartPolicy,
    /// The first tool listing, which the registry offers for good.
    advertised: Option<Vec<Tool>>,
}

#[derive(Debug)]
enum Lifecycle {
    Live(Session),
    /// No session; the first call from `retry_at` on opens one.
    Down {
        retry_at: Instant,
    },
    /// Refusing calls for good, having been asked for `scopes`.
    NeedsAuthorization {
        scopes: Vec<String>,
    },
    Stopped,
}

/// One open session. Its task shuts the session down as soon as its
/// lifeline ends, or when told to stop.
#[derive(Debug)]
struct Session {
    client: Arc<McpPeer>,
    started: Instant,
    /// Set by the task when the lifeline ends by itself.
    ended_at: Arc<OnceLock<Instant>>,
    stop_tx: oneshot::Sender<()>,
    task: AbortOnDropHandle<()>,
    consecutive_timeouts: u32,
}

/// Why a session is being retired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionEnd {
    /// The server exited or ended the session itself.
    Died,
    /// The daemon killed it for not answering.
    Hung,
}

impl McpServer {
    /// Open a session through `connector` and run its handshake; errors if either fails.
    pub async fn start(name: String, connector: Connector) -> Result<Self, McpError> {
        let session = Session::open(&connector).await?;
        let state = ServerState {
            lifecycle: Lifecycle::Live(session),
            policy: RestartPolicy::default(),
            advertised: None,
        };
        Ok(Self {
            name,
            connector,
            state: Mutex::new(state),
            reopening: AsyncMutex::new(()),
            stopping: CancellationToken::new(),
        })
    }

    /// Stop the live process, abandon a restart in progress, and refuse every later call with
    /// [`McpError::ServerDown`].
    pub async fn shutdown(&self) {
        self.stopping.cancel();
        let lifecycle = {
            let _reopening = self.reopening.lock().await;
            mem::replace(&mut self.state.lock().lifecycle, Lifecycle::Stopped)
        };
        if let Lifecycle::Live(session) = lifecycle {
            session.stop().await;
        }
    }

    async fn live_client(&self) -> Result<Arc<McpPeer>, McpError> {
        match self.current_client()? {
            Some(client) => Ok(client),
            None => self.reopen().await,
        }
    }

    /// The live session's client, or `None` when a session is due to be opened in place of
    /// one that ended.
    fn current_client(&self) -> Result<Option<Arc<McpPeer>>, McpError> {
        let mut state = self.state.lock();
        let ServerState {
            lifecycle, policy, ..
        } = &mut *state;
        if let Lifecycle::Live(session) = lifecycle
            && session.task.is_finished()
        {
            *lifecycle = self.down_after(policy, session, SessionEnd::Died);
        }
        match lifecycle {
            Lifecycle::Live(session) => Ok(Some(session.client.clone())),
            Lifecycle::Down { retry_at } if Instant::now() < *retry_at => Err(McpError::ServerDown),
            Lifecycle::Down { .. } => Ok(None),
            Lifecycle::NeedsAuthorization { scopes } => Err(McpError::NeedsAuthorization {
                scopes: scopes.clone(),
            }),
            Lifecycle::Stopped => Err(McpError::ServerDown),
        }
    }

    /// Apply the outcome of a request made through `client`, unless its session is no longer
    /// the live one: a timeout counts toward killing the session as hung, an answer clears
    /// that count, and a session the server ended or refused authorization to is dropped.
    fn record_outcome<T>(&self, client: &Arc<McpPeer>, outcome: &Result<T, McpError>) {
        let mut state = self.state.lock();
        let ServerState {
            lifecycle, policy, ..
        } = &mut *state;
        let Lifecycle::Live(session) = lifecycle else {
            return;
        };
        if !Arc::ptr_eq(&session.client, client) {
            return;
        }
        match outcome {
            Err(McpError::NeedsAuthorization { scopes }) => {
                *lifecycle = self.held_for_authorization(scopes);
            }
            Err(McpError::SessionExpired) => {
                info!(
                    target: "assistd::mcp",
                    server = %self.name,
                    "MCP server ended the session; the next call opens a new one",
                );
                *lifecycle = self.down_after(policy, session, SessionEnd::Died);
            }
            Err(McpError::RequestTimeout(_)) => {
                session.consecutive_timeouts += 1;
                if session.consecutive_timeouts < HUNG_AFTER_TIMEOUTS {
                    return;
                }
                error!(
                    target: "assistd::mcp",
                    server = %self.name,
                    timeouts = HUNG_AFTER_TIMEOUTS,
                    "MCP server stopped answering; killing it so the next call restarts it",
                );
                *lifecycle = self.down_after(policy, session, SessionEnd::Hung);
            }
            Ok(_) | Err(McpError::RpcError { .. }) => session.consecutive_timeouts = 0,
            Err(_) => {}
        }
    }

    /// Open a session in place of the one that ended, unless a concurrent call just has.
    /// A shutdown abandons the attempt, which then fails with [`McpError::ServerDown`].
    async fn reopen(&self) -> Result<Arc<McpPeer>, McpError> {
        let client = {
            let _reopening = self.reopening.lock().await;
            if let Some(client) = self.current_client()? {
                return Ok(client);
            }
            let opened = tokio::select! {
                biased;
                () = self.stopping.cancelled() => return Err(McpError::ServerDown),
                opened = Session::open(&self.connector) => opened,
            };
            self.settle_reopen(opened)?
        };
        let advertised = self.state.lock().advertised.clone();
        if let Some(advertised) = advertised {
            self.warn_if_tools_changed(&advertised, &client).await;
        }
        Ok(client)
    }

    /// Make an `opened` session the live one, or back off again if opening it failed.
    fn settle_reopen(&self, opened: Result<Session, McpError>) -> Result<Arc<McpPeer>, McpError> {
        let mut state = self.state.lock();
        match opened {
            Ok(session) => {
                info!(target: "assistd::mcp", server = %self.name, "MCP server restarted");
                let client = session.client.clone();
                state.lifecycle = Lifecycle::Live(session);
                Ok(client)
            }
            Err(McpError::NeedsAuthorization { scopes }) => {
                state.lifecycle = self.held_for_authorization(&scopes);
                Err(McpError::NeedsAuthorization { scopes })
            }
            Err(e) => {
                warn!(target: "assistd::mcp", server = %self.name, error = %e, "MCP server restart failed");
                state.policy.record_spawn_failure();
                let delay = restart_delay(&self.name, &mut state.policy);
                state.lifecycle = Lifecycle::Down {
                    retry_at: Instant::now() + delay,
                };
                Err(e)
            }
        }
    }

    /// The state that refuses every later call until the daemon replaces this server with one
    /// holding new credentials.
    fn held_for_authorization(&self, scopes: &[String]) -> Lifecycle {
        warn!(
            target: "assistd::mcp",
            server = %self.name,
            ?scopes,
            "MCP server needs authorization; refusing its calls until it is authorized",
        );
        Lifecycle::NeedsAuthorization {
            scopes: scopes.to_vec(),
        }
    }

    /// The state that follows `ended`, whose restart it registers with `policy`. Counted from
    /// when the session ended, the restart is due at once after a healthy run that died on its
    /// own, unless a restart cap is hit, and after the backoff otherwise.
    fn down_after(
        &self,
        policy: &mut RestartPolicy,
        ended: &Session,
        end: SessionEnd,
    ) -> Lifecycle {
        let ended_at = ended.ended_at();
        let ran_for = ended_at.duration_since(ended.started);
        warn!(
            target: "assistd::mcp",
            server = %self.name,
            ran_for_secs = ran_for.as_secs(),
            "MCP server stopped",
        );
        policy.record_session_end(ran_for);
        let ran_healthy = ran_for >= Duration::from_secs(MIN_HEALTHY_SECONDS);
        let delay = match policy.next_restart(Instant::now()) {
            RestartDecision::Backoff { .. } if end == SessionEnd::Died && ran_healthy => {
                Duration::ZERO
            }
            decision => decision_delay(&self.name, decision),
        };
        Lifecycle::Down {
            retry_at: ended_at + delay,
        }
    }

    /// Re-list a restarted server's tools and warn when they differ from
    /// the `advertised` ones the registry still offers.
    async fn warn_if_tools_changed(&self, advertised: &[Tool], client: &McpPeer) {
        let listed = match client.list_tools().await {
            Ok(listed) => listed,
            Err(e) => {
                warn!(
                    target: "assistd::mcp",
                    server = %self.name,
                    error = %e,
                    "could not re-list MCP tools after a restart",
                );
                return;
            }
        };
        let changes = ToolChanges::between(advertised, &listed);
        if changes.is_empty() {
            return;
        }
        warn!(
            target: "assistd::mcp",
            server = %self.name,
            added = ?changes.added,
            removed = ?changes.removed,
            changed = ?changes.changed,
            "MCP server's tools changed across a restart; the registry still offers the old ones \
             until the daemon restarts",
        );
    }
}

#[async_trait]
impl McpClient for McpServer {
    async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        let client = self.live_client().await?;
        let outcome = client.list_tools().await;
        self.record_outcome(&client, &outcome);
        let tools = outcome?;
        self.state
            .lock()
            .advertised
            .get_or_insert_with(|| tools.clone());
        Ok(tools)
    }

    async fn invoke(&self, name: &str, arguments: Value) -> Result<CallToolResult, McpError> {
        let client = self.live_client().await?;
        let outcome = client.invoke(name, arguments).await;
        self.record_outcome(&client, &outcome);
        outcome
    }
}

/// Names of the tools a server added, removed or redefined between two
/// listings.
#[derive(Debug)]
struct ToolChanges<'a> {
    added: Vec<&'a str>,
    removed: Vec<&'a str>,
    changed: Vec<&'a str>,
}

impl<'a> ToolChanges<'a> {
    fn between(before: &'a [Tool], after: &'a [Tool]) -> Self {
        let find = |tools: &'a [Tool], name: &str| tools.iter().find(|tool| tool.name == name);
        let names_missing_from = |tools: &'a [Tool], other: &'a [Tool]| {
            tools
                .iter()
                .filter(|tool| find(other, &tool.name).is_none())
                .map(|tool| tool.name.as_ref())
                .collect()
        };
        Self {
            added: names_missing_from(after, before),
            removed: names_missing_from(before, after),
            changed: after
                .iter()
                .filter(|tool| find(before, &tool.name).is_some_and(|old| old != *tool))
                .map(|tool| tool.name.as_ref())
                .collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

impl Session {
    async fn open(connector: &Connector) -> Result<Self, McpError> {
        let (client, lifeline) = connector.open().await?;
        let (stop_tx, stop_rx) = oneshot::channel();
        let ended_at = Arc::new(OnceLock::new());
        let task = AbortOnDropHandle::new(tokio::spawn(end_session(
            lifeline,
            stop_rx,
            ended_at.clone(),
        )));
        Ok(Self {
            client,
            started: Instant::now(),
            ended_at,
            stop_tx,
            task,
            consecutive_timeouts: 0,
        })
    }

    async fn stop(self) {
        let _ = self.stop_tx.send(());
        let _ = self.task.await;
    }

    /// When the lifeline ended, or now for a session that is being ended.
    fn ended_at(&self) -> Instant {
        self.ended_at.get().copied().unwrap_or_else(Instant::now)
    }
}

async fn end_session(
    mut lifeline: Lifeline,
    stop_rx: oneshot::Receiver<()>,
    ended_at: Arc<OnceLock<Instant>>,
) {
    tokio::select! {
        () = lifeline.wait_for_end() => {
            let _ = ended_at.set(Instant::now());
        }
        _ = stop_rx => {}
    }
    lifeline.shutdown(TERM_TIMEOUT).await;
}

/// Register the next restart of server `name` and pick how long to wait before it.
pub(crate) fn restart_delay(name: &str, policy: &mut RestartPolicy) -> Duration {
    decision_delay(name, policy.next_restart(Instant::now()))
}

/// How long `decision` says server `name` waits before its restart, logging a hit cap.
fn decision_delay(name: &str, decision: RestartDecision) -> Duration {
    if let RestartDecision::SlowRetry { delay, cap } = decision {
        error!(
            target: "assistd::mcp",
            server = %name,
            retry_secs = delay.as_secs(),
            "MCP server {cap}; slowing restarts",
        );
    }
    decision.delay()
}

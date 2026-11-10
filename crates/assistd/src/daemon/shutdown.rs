//! Staged daemon teardown: signal handling, intake drain, and subsystem
//! shutdown in dependency order.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use assistd_core::{AppState, Component, PresenceManager, TaskRunner, spawn_supervised};
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::{Instant, timeout_at};
use tracing::info;

use super::embed_init::EmbeddingService;
use super::mcp_init::McpService;
use super::memory_init::MemorySubsystem;
use super::search_init::SearchService;
use super::voice_init::VoiceService;
use super::wm_init::WindowSubsystem;

/// Whole-teardown budget, leaving systemd's 15 s `TimeoutStopSec` room to
/// close the memory database and exit before it sends SIGKILL.
const TEARDOWN_BUDGET: Duration = Duration::from_secs(12);
const PERSISTENCE_DRAIN_BUDGET: Duration = Duration::from_secs(5);
const TASK_RUN_STOP_BUDGET: Duration = Duration::from_secs(10);

/// One watch per teardown stage. A signal flips only `intake`; the rest
/// flip in dependency order once in-flight work has drained.
pub(super) struct ShutdownStages {
    pub(super) intake: watch::Sender<bool>,
    pub(super) voice: watch::Sender<bool>,
    pub(super) llm: watch::Sender<bool>,
    pub(super) tools: watch::Sender<bool>,
    pub(super) search: watch::Sender<bool>,
    pub(super) embed_worker: watch::Sender<bool>,
    pub(super) embed_server: watch::Sender<bool>,
}

impl ShutdownStages {
    pub(super) fn new() -> Self {
        Self {
            intake: watch::channel(false).0,
            voice: watch::channel(false).0,
            llm: watch::channel(false).0,
            tools: watch::channel(false).0,
            search: watch::channel(false).0,
            embed_worker: watch::channel(false).0,
            embed_server: watch::channel(false).0,
        }
    }

    /// Flip every stage at once, for a startup abandoned mid-way.
    pub(super) fn cancel_all(&self) {
        for stage in [
            &self.intake,
            &self.voice,
            &self.llm,
            &self.tools,
            &self.search,
            &self.embed_worker,
            &self.embed_server,
        ] {
            stage.send_replace(true);
        }
    }
}

pub(super) struct DaemonShutdown {
    pub(super) state: Arc<AppState>,
    pub(super) task_runner: Arc<TaskRunner>,
    pub(super) presence: Arc<PresenceManager>,
    pub(super) memory: MemorySubsystem,
    pub(super) voice: VoiceService,
    pub(super) embed: EmbeddingService,
    pub(super) search: SearchService,
    pub(super) window: WindowSubsystem,
    pub(super) mcp: McpService,
    pub(super) intake_tasks: IntakeTasks,
}

impl DaemonShutdown {
    /// Tear down after the socket has stopped: stop intake, finish intake
    /// tasks and persistence, then stop each subsystem before its dependencies.
    /// Every stage before closing memory shares one [`TEARDOWN_BUDGET`]
    /// deadline, and a stage that overruns it is abandoned.
    pub(super) async fn shutdown(self, stages: &ShutdownStages) {
        let deadline = TeardownDeadline::after(TEARDOWN_BUDGET);
        stages.intake.send_replace(true);
        deadline
            .bound("intake tasks", join_intake_tasks(self.intake_tasks))
            .await;
        stop_task_runs(&self.task_runner, deadline).await;
        drain_persistence(&self.state, deadline).await;

        stages.voice.send_replace(true);
        deadline.bound("voice worker", self.voice.shutdown()).await;

        stages.llm.send_replace(true);
        if let Some(Err(e)) = deadline.bound("presence", self.presence.shutdown()).await {
            tracing::error!("presence shutdown error: {e:#}");
        }

        stages.tools.send_replace(true);
        stages.search.send_replace(true);
        tokio::join!(
            deadline.bound("window manager", self.window.shutdown()),
            deadline.bound("MCP servers", self.mcp.shutdown()),
            deadline.bound("web search", self.search.shutdown()),
            deadline.bound(
                "embedding",
                self.embed
                    .shutdown(&stages.embed_worker, &stages.embed_server, deadline.at),
            ),
        );
        self.memory.shutdown().await;
    }
}

/// The instant every teardown stage before closing memory must finish by.
#[derive(Clone, Copy)]
struct TeardownDeadline {
    at: Instant,
}

impl TeardownDeadline {
    fn after(budget: Duration) -> Self {
        Self {
            at: Instant::now() + budget,
        }
    }

    /// The earlier of the deadline and `budget` from now.
    fn capped(self, budget: Duration) -> Instant {
        self.at.min(Instant::now() + budget)
    }

    /// Run `stage`, abandoning it with a warning at the deadline.
    async fn bound<F: Future>(self, stage: &'static str, work: F) -> Option<F::Output> {
        let outcome = timeout_at(self.at, work).await.ok();
        if outcome.is_none() {
            tracing::warn!("{stage} did not stop before the shutdown deadline; abandoning it");
        }
        outcome
    }
}

/// Tasks that start new work and stop on the `intake` stage.
pub(super) struct IntakeTasks {
    pub(super) gpu_monitor: Option<JoinHandle<()>>,
    pub(super) idle_monitor: Option<JoinHandle<()>>,
    /// Also runs the listen dispatcher once voice capture is up.
    pub(super) warmup: JoinHandle<()>,
}

pub(super) fn spawn_signal_handler(shutdown_tx: &watch::Sender<bool>) {
    spawn_supervised(
        "signal_handler",
        Component::Daemon,
        forward_signals(shutdown_tx.clone()),
    );
}

/// Flag shutdown on the first SIGINT/SIGTERM; a second signal exits
/// immediately without cleanup.
async fn forward_signals(shutdown_tx: watch::Sender<bool>) {
    let (mut int, mut term) = match (
        signal(SignalKind::interrupt()),
        signal(SignalKind::terminate()),
    ) {
        (Ok(int), Ok(term)) => (int, term),
        (Err(e), _) | (_, Err(e)) => {
            tracing::error!("failed to install signal handlers: {e}");
            return;
        }
    };
    loop {
        let (name, exit_code) = tokio::select! {
            _ = int.recv() => ("SIGINT", 130),
            _ = term.recv() => ("SIGTERM", 143),
        };
        if shutdown_tx.send_replace(true) {
            tracing::warn!("received {name} again; exiting without cleanup");
            std::process::exit(exit_code);
        }
        info!("received {name}; shutting down (send again to force exit)");
    }
}

async fn join_intake_tasks(tasks: IntakeTasks) {
    for handle in [tasks.gpu_monitor, tasks.idle_monitor]
        .into_iter()
        .flatten()
        .chain([tasks.warmup])
    {
        let _ = handle.await;
    }
}

async fn stop_task_runs(task_runner: &TaskRunner, deadline: TeardownDeadline) {
    if timeout_at(deadline.capped(TASK_RUN_STOP_BUDGET), task_runner.stop())
        .await
        .is_err()
    {
        tracing::warn!(
            target: "assistd::tasks",
            "task runs did not stop in time at shutdown; abandoning them"
        );
    }
}

/// Wait for queued transcript writes and background persistence tasks,
/// abandoning them after [`PERSISTENCE_DRAIN_BUDGET`] or at the deadline.
async fn drain_persistence(state: &AppState, deadline: TeardownDeadline) {
    let drained = timeout_at(
        deadline.capped(PERSISTENCE_DRAIN_BUDGET),
        state.finish_persistence(),
    );
    if drained.await.is_err() {
        tracing::warn!(
            target: "assistd::memory",
            "persistence drain timed out at shutdown; abandoning the remaining writes"
        );
    }
}

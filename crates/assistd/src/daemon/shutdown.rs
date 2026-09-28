//! Staged daemon teardown: signal handling, intake drain, and subsystem
//! shutdown in dependency order.

use std::sync::Arc;
use std::time::Duration;

use assistd_core::{Component, PresenceManager, spawn_supervised};
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::task::TaskTracker;
use tracing::info;

use super::embed_init::EmbeddingSubsystem;
use super::listen_dispatcher::ListenDispatcherHandles;
use super::mcp_init::McpSubsystem;
use super::memory_init::MemorySubsystem;
use super::wm_init::WindowSubsystem;

const PERSISTENCE_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// One watch per teardown stage. A signal flips only `intake`; the rest
/// flip in dependency order once in-flight work has drained.
pub(super) struct ShutdownStages {
    pub(super) intake: watch::Sender<bool>,
    pub(super) llm: watch::Sender<bool>,
    pub(super) tools: watch::Sender<bool>,
    pub(super) embed_worker: watch::Sender<bool>,
    pub(super) embed_server: watch::Sender<bool>,
    pub(super) memory_writer: watch::Sender<bool>,
}

impl ShutdownStages {
    pub(super) fn new() -> Self {
        Self {
            intake: watch::channel(false).0,
            llm: watch::channel(false).0,
            tools: watch::channel(false).0,
            embed_worker: watch::channel(false).0,
            embed_server: watch::channel(false).0,
            memory_writer: watch::channel(false).0,
        }
    }

    /// Flip every stage at once, for a startup abandoned mid-way.
    pub(super) fn cancel_all(&self) {
        for stage in [
            &self.intake,
            &self.llm,
            &self.tools,
            &self.embed_worker,
            &self.embed_server,
            &self.memory_writer,
        ] {
            stage.send_replace(true);
        }
    }
}

pub(super) struct DaemonShutdown {
    pub(super) persistence_tracker: TaskTracker,
    pub(super) presence: Arc<PresenceManager>,
    pub(super) memory: MemorySubsystem,
    pub(super) embed: EmbeddingSubsystem,
    pub(super) window: WindowSubsystem,
    pub(super) mcp: McpSubsystem,
    pub(super) intake_tasks: IntakeTasks,
}

impl DaemonShutdown {
    /// Tear down after the socket has drained: finish intake tasks and
    /// persistence, then stop each subsystem before the ones it depends on.
    pub(super) async fn shutdown(self, stages: &ShutdownStages) {
        join_intake_tasks(self.intake_tasks).await;
        drain_persistence(&self.persistence_tracker).await;

        stages.llm.send_replace(true);
        if let Err(e) = self.presence.sleep().await {
            tracing::error!("presence shutdown error: {e:#}");
        }

        stages.tools.send_replace(true);
        self.window.shutdown().await;
        self.mcp.shutdown().await;

        self.embed
            .shutdown(&stages.embed_worker, &stages.embed_server)
            .await;
        self.memory.shutdown(&stages.memory_writer).await;
    }
}

/// Tasks that start new work and stop on the `intake` stage.
pub(super) struct IntakeTasks {
    pub(super) hotkey: Option<JoinHandle<()>>,
    pub(super) gpu_monitor: Option<JoinHandle<()>>,
    pub(super) idle_monitor: Option<JoinHandle<()>>,
    pub(super) listen: Option<ListenDispatcherHandles>,
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
    for handle in [tasks.hotkey, tasks.gpu_monitor, tasks.idle_monitor]
        .into_iter()
        .flatten()
    {
        let _ = handle.await;
    }
    if let Some(listen) = tasks.listen {
        let _ = listen.forwarder.await;
        let _ = listen.presence_gate.await;
    }
}

/// Wait for fire-and-forget persistence tasks, abandoning them after
/// [`PERSISTENCE_DRAIN_BUDGET`].
async fn drain_persistence(tracker: &TaskTracker) {
    tracker.close();
    if tokio::time::timeout(PERSISTENCE_DRAIN_BUDGET, tracker.wait())
        .await
        .is_err()
    {
        tracing::warn!(
            target: "assistd::memory",
            in_flight = tracker.len(),
            "persistence task drain timed out at shutdown; abandoning remaining tasks"
        );
    }
}

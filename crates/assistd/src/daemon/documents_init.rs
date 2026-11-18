//! Document indexing for the daemon: the extraction worker runs under supervision
//! from the daemon's own binary, sandboxed over the index roots, and its readiness
//! is published whenever it changes.

use std::path::Path;
use std::sync::Arc;

use assistd_config::Config;
use assistd_core::{AppState, Component, DocumentsUnavailable, spawn_supervised};
use assistd_documents::{Extractor, WorkerCommand, supervise};
use assistd_ipc::{ComponentReadiness, StartupComponent};
use assistd_tools::{SandboxInfo, ToolsDisabled};
use assistd_utils::readiness::NotReady;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use super::SELF_EXE;

/// The hidden subcommand the worker runs as.
const WORKER_SUBCOMMAND: &str = "extract-worker";

/// What starting the extraction worker needs.
pub(super) struct DocumentsStartup {
    command: WorkerCommand,
}

/// The background task that runs the extraction worker, or none when
/// document indexing is unavailable.
pub(super) struct DocumentsService {
    task: Option<JoinHandle<()>>,
}

impl DocumentsStartup {
    /// Run the worker until `shutdown` flips.
    pub(super) fn spawn(
        self,
        state: Arc<AppState>,
        shutdown: watch::Receiver<bool>,
    ) -> DocumentsService {
        let task = spawn_supervised(
            "extract_worker",
            Component::Documents,
            run(self.command, state, shutdown),
        );
        DocumentsService { task: Some(task) }
    }
}

impl DocumentsService {
    pub(super) fn disabled() -> Self {
        Self { task: None }
    }

    /// Wait for the worker to stop. Its shutdown stage must already have flipped.
    pub(super) async fn shutdown(self) {
        if let Some(task) = self.task
            && let Err(e) = task.await
        {
            tracing::error!("extraction worker task failed: {e}");
        }
    }
}

/// The extractor requests see from startup and, when a worker can run in
/// `sandbox`, what starting it needs.
pub(super) fn prepare(
    config: &Config,
    sandbox: Result<&SandboxInfo, &ToolsDisabled>,
) -> (Arc<Extractor>, Option<DocumentsStartup>) {
    let command = assistd_core::extract_worker_command(
        &config.documents,
        sandbox,
        Path::new(SELF_EXE),
        WORKER_SUBCOMMAND,
    );
    match command {
        Ok(command) => (
            Arc::new(Extractor::starting()),
            Some(DocumentsStartup { command }),
        ),
        Err(reason) => {
            match reason {
                DocumentsUnavailable::Disabled => info!("document indexing: {reason}"),
                _ => warn!("document indexing is unavailable: {reason}"),
            }
            (Arc::new(Extractor::unavailable(reason.to_string())), None)
        }
    }
}

async fn run(command: WorkerCommand, state: Arc<AppState>, shutdown: watch::Receiver<bool>) {
    let extractor = &state.subsystems.documents;
    let changes = extractor.subscribe_readiness();
    tokio::join!(
        supervise(command, Arc::clone(extractor), shutdown.clone()),
        publish_changes(&state, changes, shutdown),
    );
}

async fn publish_changes(
    state: &AppState,
    mut changes: watch::Receiver<Result<(), NotReady>>,
    mut shutdown: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            biased;
            _ = shutdown.wait_for(|stop| *stop) => return,
            changed = changes.changed() => {
                if changed.is_err() {
                    return;
                }
                let readiness = ComponentReadiness::from(changes.borrow_and_update().clone());
                info!("document indexing: {readiness}");
                state
                    .status
                    .publish_readiness(StartupComponent::DocumentIndexing, readiness);
            }
        }
    }
}

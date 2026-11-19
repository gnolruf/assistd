//! Document indexing for the daemon: the extraction worker runs under supervision
//! from the daemon's own binary, sandboxed over the index roots, its readiness is
//! published whenever it changes, and passes over the roots feed it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_config::Config;
use assistd_core::{
    AppState, Component, DocumentsUnavailable, IndexDeps, IndexPasses, remove_dropped_roots,
    spawn_supervised,
};
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

/// What document indexing starts with while `documents.enabled`.
pub(super) struct DocumentsStartup {
    /// `documents.roots`; a stored root not among them is removed.
    roots: Vec<PathBuf>,
    /// The worker's command and the passes, or none when indexing is unavailable.
    indexing: Option<(WorkerCommand, IndexPasses)>,
}

/// The background task that runs the extraction worker and the index passes, or
/// none when document indexing is off.
pub(super) struct DocumentsService {
    task: Option<JoinHandle<()>>,
}

impl DocumentsStartup {
    /// Run the worker and the passes until `shutdown` flips.
    pub(super) fn spawn(
        self,
        state: Arc<AppState>,
        shutdown: watch::Receiver<bool>,
    ) -> DocumentsService {
        let task = spawn_supervised(
            "document_indexing",
            Component::Documents,
            run(self, state, shutdown),
        );
        DocumentsService { task: Some(task) }
    }
}

impl DocumentsService {
    pub(super) fn disabled() -> Self {
        Self { task: None }
    }

    /// Wait for the worker and the pass under way to stop. Their shutdown stage
    /// must already have flipped, and the memory database must still be open.
    pub(super) async fn shutdown(self) {
        if let Some(task) = self.task
            && let Err(e) = task.await
        {
            tracing::error!("document indexing task failed: {e}");
        }
    }
}

/// The extractor requests see from startup and, while `documents.enabled`, what
/// document indexing starts with in `sandbox`.
pub(super) fn prepare(
    config: &Config,
    sandbox: Result<&Arc<SandboxInfo>, &ToolsDisabled>,
) -> (Arc<Extractor>, Option<DocumentsStartup>) {
    let startup = |indexing| DocumentsStartup {
        roots: config
            .documents
            .roots
            .iter()
            .map(|root| root.as_path().to_path_buf())
            .collect(),
        indexing,
    };
    let indexing = assistd_core::document_indexing(
        &config.documents,
        sandbox,
        Path::new(SELF_EXE),
        WORKER_SUBCOMMAND,
    );
    match indexing {
        Ok(indexing) => (
            Arc::new(Extractor::starting()),
            Some(startup(Some(indexing))),
        ),
        Err(reason @ DocumentsUnavailable::Disabled) => {
            info!("document indexing: {reason}");
            (Arc::new(Extractor::unavailable(reason.to_string())), None)
        }
        Err(reason) => {
            warn!("document indexing is unavailable: {reason}");
            (
                Arc::new(Extractor::unavailable(reason.to_string())),
                Some(startup(None)),
            )
        }
    }
}

async fn run(startup: DocumentsStartup, state: Arc<AppState>, shutdown: watch::Receiver<bool>) {
    remove_dropped_roots(state.memory.semantic.as_ref(), &startup.roots).await;
    let Some((worker, passes)) = startup.indexing else {
        return;
    };
    let extractor = &state.subsystems.documents;
    let changes = extractor.subscribe_readiness();
    let deps = IndexDeps {
        store: Arc::clone(&state.memory.semantic),
        extractor: Arc::clone(extractor),
        embed_wake: Arc::clone(&state.memory.embed_wake),
    };
    tokio::join!(
        supervise(worker, Arc::clone(extractor), shutdown.clone()),
        publish_changes(&state, changes, shutdown.clone()),
        passes.run(deps, &state.subsystems.document_index, shutdown),
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

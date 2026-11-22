//! Document indexing for the daemon: the extraction worker runs under supervision
//! from the daemon's own binary, sandboxed over the index roots, its readiness is
//! published whenever it changes, passes over the roots feed it, what the index
//! is doing is broadcast, and an OCR model reads the pages that have no text.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use assistd_config::Config;
use assistd_core::{
    AppState, BorrowedModel, DocumentIndex, DocumentsUnavailable, IndexDeps, IndexPasses,
    TranscriptionDeps, remove_dropped_roots, transcribe_while_drowsy,
};
use assistd_documents::{Extractor, WorkerCommand, supervise};
use assistd_ipc::{ComponentReadiness, StartupComponent};
use assistd_llm::{LlamaServerControl, LlmHealthProbe, OCR_MODEL_ALIAS, PageTranscriber};
use assistd_tools::{SandboxInfo, ToolsDisabled};
use assistd_utils::child_server::ApiKey;
use assistd_utils::readiness::NotReady;
use tokio::sync::watch;
use tracing::{info, warn};

use super::SELF_EXE;
use super::background_service::{BackgroundService, publish_changes};

pub(super) const SUBSYSTEM: &str = "document indexing";

/// The hidden subcommand the worker runs as.
const WORKER_SUBCOMMAND: &str = "extract-worker";

/// What document indexing starts with while `documents.enabled`.
pub(super) struct DocumentsStartup {
    /// The worker's command and the passes, or none when indexing is unavailable.
    indexing: Option<(WorkerCommand, IndexPasses)>,
    /// The OCR model, or none when `documents.ocr.model` is unset.
    ocr: Option<OcrModel>,
}

/// The model that reads pages without text, and the clients that reach it on the
/// model server.
struct OcrModel {
    model: BorrowedModel,
    transcriber: PageTranscriber,
    control: LlamaServerControl,
}

impl OcrModel {
    fn from_config(
        config: &Config,
        api_key: &ApiKey,
        health_probe: &Arc<dyn LlmHealthProbe>,
    ) -> Result<Option<Self>> {
        let Some(source) = config.documents.ocr_model() else {
            return Ok(None);
        };
        let addr = config.model.listen_addr();
        let prompt = config.documents.ocr.prompt.clone();
        let transcriber = PageTranscriber::new(addr, Some(api_key), prompt)
            .context("failed to construct the OCR model's client")?;
        let control = LlamaServerControl::new(addr, Some(api_key), Some(Arc::clone(health_probe)))
            .context("failed to construct llama-server control client for the OCR model")?;
        Ok(Some(Self {
            model: BorrowedModel {
                source: source.to_owned(),
                served_as: OCR_MODEL_ALIAS.to_owned(),
            },
            transcriber,
            control,
        }))
    }
}

impl DocumentsStartup {
    /// Run the worker and the passes until `shutdown` flips. The memory database
    /// must stay open until the service is joined.
    pub(super) fn spawn(
        self,
        state: Arc<AppState>,
        shutdown: watch::Receiver<bool>,
    ) -> BackgroundService<()> {
        BackgroundService::spawn(SUBSYSTEM, run(self, state, shutdown))
    }
}

/// The extractor requests see from startup and, while `documents.enabled`, what
/// document indexing starts with in `sandbox`: the passes, and the OCR model on the
/// model server that takes `api_key`.
pub(super) fn prepare(
    config: &Config,
    sandbox: Result<&Arc<SandboxInfo>, &ToolsDisabled>,
    api_key: &ApiKey,
    health_probe: &Arc<dyn LlmHealthProbe>,
) -> Result<(Arc<Extractor>, Option<DocumentsStartup>)> {
    let indexing = assistd_core::document_indexing(
        &config.documents,
        sandbox,
        Path::new(SELF_EXE),
        WORKER_SUBCOMMAND,
    );
    let prepared = match indexing {
        Ok(indexing) => (
            Arc::new(Extractor::starting()),
            Some(DocumentsStartup {
                indexing: Some(indexing),
                ocr: OcrModel::from_config(config, api_key, health_probe)?,
            }),
        ),
        Err(reason) => {
            let startup = match reason {
                DocumentsUnavailable::Disabled => {
                    info!("document indexing: {reason}");
                    None
                }
                _ => {
                    warn!("document indexing is unavailable: {reason}");
                    Some(DocumentsStartup {
                        indexing: None,
                        ocr: None,
                    })
                }
            };
            (
                Arc::new(Extractor::unavailable(reason.to_string())),
                startup,
            )
        }
    };
    Ok(prepared)
}

async fn run(startup: DocumentsStartup, state: Arc<AppState>, shutdown: watch::Receiver<bool>) {
    let roots = &state.config.documents.roots;
    remove_dropped_roots(state.memory.document_store.as_ref(), roots).await;
    let Some((worker, passes)) = startup.indexing else {
        return;
    };
    let extractor = &state.subsystems.documents;
    let changes = extractor.subscribe_readiness();
    let deps = IndexDeps {
        store: Arc::clone(&state.memory.document_store),
        extractor: Arc::clone(extractor),
    };
    let transcription = startup.ocr.map(|ocr| TranscriptionDeps {
        store: Arc::clone(&state.memory.document_store),
        extractor: Arc::clone(extractor),
        presence: Arc::clone(&state.subsystems.presence),
        model: ocr.model,
        transcriber: ocr.transcriber,
        control: ocr.control,
        max_file_bytes: state.config.documents.max_file_bytes(),
    });
    let index = &state.subsystems.document_index;
    tokio::join!(
        supervise(worker, Arc::clone(extractor), shutdown.clone()),
        publish_changes(changes, shutdown.clone(), |readiness| {
            publish_readiness(&state, readiness.clone());
        }),
        state.documents.broadcast_activity(shutdown.clone()),
        transcribe_pages(transcription, index, shutdown.clone()),
        passes.run(deps, index, shutdown),
    );
}

/// Read the pages that wait for the OCR model until `shutdown` flips; returns at
/// once when there is no model.
async fn transcribe_pages(
    transcription: Option<TranscriptionDeps>,
    index: &DocumentIndex,
    shutdown: watch::Receiver<bool>,
) {
    if let Some(deps) = transcription {
        transcribe_while_drowsy(deps, index, shutdown).await;
    }
}

fn publish_readiness(state: &AppState, readiness: Result<(), NotReady>) {
    let readiness = ComponentReadiness::from(readiness);
    info!("document indexing: {readiness}");
    state
        .status
        .publish_readiness(StartupComponent::DocumentIndexing, readiness);
}

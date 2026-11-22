//! Embedding subsystem wiring for the daemon: handles requests use from
//! startup, and the embedding server that comes up behind them afterwards.

use std::sync::Arc;
use std::time::Duration;

use assistd_config::{Config, NonBlankString};
use assistd_core::AppState;
use assistd_embed::{
    EmbedServerSpec, Embedder, EmbedderHandle, LlamaEmbedder, TaskPrefixes, spawn_embedder_task,
};
use assistd_ipc::{ComponentReadiness, StartupComponent};
use assistd_memory::{
    EmbeddingQueue, EmbeddingSpace, NoSemanticStore, SemanticStore, SqliteHandle,
    SqliteSemanticStore,
};
use assistd_utils::child_server::{ApiKey, ChildServer};
use assistd_utils::readiness::Readiness;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::info;

use super::background_service::BackgroundService;

pub(super) const SUBSYSTEM: &str = "embedding";

/// Longest the worker gets at shutdown to finish the batch it is embedding.
const EMBED_DRAIN_BUDGET: Duration = Duration::from_secs(10);

/// The handles the daemon serves from startup.
pub(super) struct EmbeddingHandles {
    pub embedder: Arc<EmbedderHandle>,
    pub semantic_store: Arc<dyn SemanticStore>,
    pub embed_wake: Arc<Notify>,
}

/// What starting the embedding server in the background still needs.
pub(super) struct EmbeddingStartup {
    config: assistd_config::EmbeddingConfig,
    server_binary: NonBlankString,
    api_key: ApiKey,
    embed_wake: Arc<Notify>,
}

/// The shutdown signals the embedding server and worker stop on.
pub(super) struct EmbeddingStages {
    pub worker: watch::Receiver<bool>,
    pub server: watch::Receiver<bool>,
}

/// The background start, joined at shutdown for what it brought up.
pub(super) type EmbeddingService = BackgroundService<Option<RunningEmbedding>>;

/// The server and, when its client probe passed, the worker embedding
/// rows missing an embedding.
pub(super) struct RunningEmbedding {
    server: ChildServer,
    worker: Option<EmbedWorker>,
}

/// The worker, and the task keeping the gate on its document chunks; that task
/// ends when the worker does.
struct EmbedWorker {
    worker: JoinHandle<()>,
    documents_gate: JoinHandle<()>,
}

impl EmbeddingStartup {
    /// Start the server once `model_settled` flips, then fill the embedder
    /// handle and start the worker. Gives up when `model_settled` closes.
    pub(super) fn spawn(
        self,
        state: Arc<AppState>,
        model_settled: watch::Receiver<bool>,
        stages: EmbeddingStages,
    ) -> EmbeddingService {
        let panicked_state = Arc::clone(&state);
        BackgroundService::spawn_marking(
            SUBSYSTEM,
            async move { start_and_record(self, &state, model_settled, stages).await },
            move || {
                panicked_state
                    .memory
                    .embedder
                    .set(Readiness::Unavailable("startup panicked".into()));
                publish_readiness(&panicked_state);
            },
        )
    }
}

/// Let the worker finish its batch while the server is still up, abandoning
/// it after [`EMBED_DRAIN_BUDGET`] or at `deadline`, then stop the server. A
/// start still under way is cancelled. The memory database must still be open.
pub(super) async fn shutdown(
    service: EmbeddingService,
    worker_shutdown: &watch::Sender<bool>,
    server_shutdown: &watch::Sender<bool>,
    deadline: Instant,
) {
    worker_shutdown.send_replace(true);
    if !service.is_finished() {
        server_shutdown.send_replace(true);
    }
    let Some(RunningEmbedding { server, worker }) = service.join().await.flatten() else {
        return;
    };
    if let Some(worker) = worker {
        join_worker_within_budget(worker, deadline).await;
    }
    server_shutdown.send_replace(true);
    if let Err(e) = server.shutdown().await {
        tracing::warn!("embed-server shutdown error: {e:#}");
    }
}

async fn join_worker_within_budget(embed_worker: EmbedWorker, deadline: Instant) {
    let EmbedWorker {
        mut worker,
        documents_gate,
    } = embed_worker;
    let drain_until = deadline.min(Instant::now() + EMBED_DRAIN_BUDGET);
    if tokio::time::timeout_at(drain_until, &mut worker)
        .await
        .is_err()
    {
        tracing::warn!(
            target: "assistd::embed",
            "embed worker still busy at shutdown; its rows are embedded on the next start"
        );
        worker.abort();
        let _ = worker.await;
    }
    let _ = documents_gate.await;
}

/// The handles requests use from startup and, when embedding is enabled,
/// what starting its server needs.
pub(super) fn prepare(
    config: &Config,
    api_key: &ApiKey,
    sqlite_handle: &SqliteHandle,
) -> (EmbeddingHandles, Option<EmbeddingStartup>) {
    if !config.embedding.enabled {
        const REASON: &str = "disabled in config (embedding.enabled = false)";
        info!("embedding: {REASON}");
        return (disabled_handles(REASON), None);
    }
    let semantic_store: Arc<dyn SemanticStore> =
        Arc::new(SqliteSemanticStore::new(sqlite_handle.clone()));
    let embed_wake = Arc::new(Notify::new());
    let handles = EmbeddingHandles {
        embedder: Arc::new(EmbedderHandle::new(Readiness::Starting)),
        semantic_store,
        embed_wake: embed_wake.clone(),
    };
    let startup = EmbeddingStartup {
        config: config.embedding.clone(),
        server_binary: config.model.server_binary.clone(),
        api_key: api_key.clone(),
        embed_wake,
    };
    (handles, Some(startup))
}

fn disabled_handles(reason: &str) -> EmbeddingHandles {
    EmbeddingHandles {
        embedder: Arc::new(EmbedderHandle::new(Readiness::Unavailable(reason.into()))),
        semantic_store: Arc::new(NoSemanticStore),
        embed_wake: Arc::default(),
    }
}

async fn start_and_record(
    startup: EmbeddingStartup,
    state: &AppState,
    mut model_settled: watch::Receiver<bool>,
    stages: EmbeddingStages,
) -> Option<RunningEmbedding> {
    if model_settled.wait_for(|settled| *settled).await.is_err() {
        return None;
    }
    let running = match start_server(&startup, stages.server).await {
        Ok((server, embedder)) => {
            info!(
                "embedding: ready (model={}, dim={}, port={})",
                embedder.model(),
                embedder.dim(),
                startup.config.port,
            );
            let worker = adopt_space_then_serve(state, embedder, startup, stages.worker).await;
            Some(RunningEmbedding { server, worker })
        }
        Err(failure) => {
            tracing::warn!(
                "embedding: {}; semantic search disabled this run",
                failure.reason
            );
            state
                .memory
                .embedder
                .set(Readiness::Unavailable(failure.reason.into()));
            failure.server.map(|server| RunningEmbedding {
                server,
                worker: None,
            })
        }
    };
    publish_readiness(state);
    running
}

fn publish_readiness(state: &AppState) {
    state.status.publish_readiness(
        StartupComponent::Embedding,
        ComponentReadiness::from(state.memory.embedder.readiness()),
    );
}

/// Drop vectors from any other embedding space, then start the worker and mark the
/// embedder ready. When that fails the embedder stays unavailable.
async fn adopt_space_then_serve(
    state: &AppState,
    embedder: Arc<dyn Embedder>,
    startup: EmbeddingStartup,
    worker_shutdown: watch::Receiver<bool>,
) -> Option<EmbedWorker> {
    let space = EmbeddingSpace {
        model: embedder.model().to_owned(),
        dim: embedder.dim(),
        server_args: startup.config.custom_args.clone().into(),
        document_prefix: startup.config.document_prefix.clone(),
    };
    if let Err(reason) = adopt_space(state.memory.embedding_queue.as_ref(), &space).await {
        tracing::warn!("embedding: {reason}; semantic search disabled this run");
        state
            .memory
            .embedder
            .set(Readiness::Unavailable(reason.into()));
        return None;
    }
    Some(serve_embedder(state, embedder, startup, worker_shutdown))
}

fn serve_embedder(
    state: &AppState,
    embedder: Arc<dyn Embedder>,
    startup: EmbeddingStartup,
    worker_shutdown: watch::Receiver<bool>,
) -> EmbedWorker {
    let (documents_gate_open, documents_gate) = state.status.document_embed_gate();
    let worker = spawn_embedder_task(
        embedder.clone(),
        state.memory.embedding_queue.clone(),
        startup.embed_wake,
        documents_gate_open,
        worker_shutdown,
    );
    state.memory.embedder.set(Readiness::Ready(embedder));
    state.subsystems.document_index.request_pass();
    EmbedWorker {
        worker,
        documents_gate,
    }
}

/// Why the embedder did not come up, with the server when it started
/// but its client probe failed.
struct StartFailure {
    reason: String,
    server: Option<ChildServer>,
}

async fn start_server(
    startup: &EmbeddingStartup,
    server_shutdown: watch::Receiver<bool>,
) -> Result<(ChildServer, Arc<dyn Embedder>), StartFailure> {
    let EmbeddingStartup {
        config,
        server_binary,
        api_key,
        ..
    } = startup;
    let spec = EmbedServerSpec::new(config.clone(), server_binary.clone(), api_key.clone());
    let server = ChildServer::start(spec, server_shutdown)
        .await
        .map_err(|e| StartFailure {
            reason: format!("server failed to start: {e:#}"),
            server: None,
        })?;
    match LlamaEmbedder::new(
        config.listen_addr(),
        config.model.to_string(),
        TaskPrefixes {
            document: config.document_prefix.clone(),
            query: config.query_prefix.clone(),
        },
        assistd_embed::REQUEST_TIMEOUT,
        Some(api_key),
        Some(server.status()),
    )
    .await
    {
        Ok(client) => Ok((server, Arc::new(client))),
        Err(e) => Err(StartFailure {
            reason: format!("client probe failed: {e:#}"),
            server: Some(server),
        }),
    }
}

async fn adopt_space(queue: &dyn EmbeddingQueue, space: &EmbeddingSpace) -> Result<(), String> {
    match queue.adopt_embedding_space(space).await {
        Ok(None) => Ok(()),
        Ok(Some(replaced)) => {
            info!(
                "embedding: dropped every stored vector for re-embedding; they came from \
                 model={}, dim={}, args={:?}, document_prefix={:?}",
                replaced.model, replaced.dim, replaced.server_args, replaced.document_prefix
            );
            Ok(())
        }
        Err(e) => Err(format!("checking the stored vectors' model failed: {e}")),
    }
}

//! Embedding subsystem wiring for the daemon: handles requests use from
//! startup, and the embedding server that comes up behind them afterwards.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use assistd_core::{AppState, Config};
use assistd_embed::{
    EmbedJob, EmbedServerSpec, Embedder, EmbedderHandle, LlamaEmbedder, spawn_embedder_task,
};
use assistd_ipc::{ComponentReadiness, StartupComponent};
use assistd_memory::{NoSemanticStore, SemanticStore, SqliteHandle, SqliteSemanticStore, WriteOp};
use assistd_utils::child_server::{ApiKey, ChildServer};
use assistd_utils::readiness::Readiness;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::info;

const EMBED_QUEUE_CAPACITY: usize = 256;

/// Longest the worker gets at shutdown to embed and store what is still queued.
const EMBED_DRAIN_BUDGET: Duration = Duration::from_secs(10);

/// The handles the daemon serves from startup.
pub(super) struct EmbeddingHandles {
    pub embedder: Arc<EmbedderHandle>,
    pub semantic_store: Arc<dyn SemanticStore>,
    pub embed_tx: mpsc::Sender<EmbedJob>,
}

/// What starting the embedding server in the background still needs.
/// Jobs queued before it is up wait in `embed_rx`.
pub(super) struct EmbeddingStartup {
    config: assistd_config::EmbeddingConfig,
    api_key: ApiKey,
    embed_rx: mpsc::Receiver<EmbedJob>,
    writer_tx: Arc<mpsc::Sender<WriteOp>>,
}

/// The shutdown signals the embedding server and worker stop on.
pub(super) struct EmbeddingStages {
    pub worker: watch::Receiver<bool>,
    pub server: watch::Receiver<bool>,
}

/// The background start, joined at shutdown for what it brought up.
pub(super) struct EmbeddingService {
    startup: Option<JoinHandle<Option<RunningEmbedding>>>,
}

/// The server and, when its client probe passed, the worker embedding
/// queued jobs.
struct RunningEmbedding {
    server: ChildServer,
    worker: Option<JoinHandle<()>>,
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
        let startup = spawn_startup(self, state, model_settled, stages);
        EmbeddingService {
            startup: Some(startup),
        }
    }
}

impl EmbeddingService {
    pub(super) fn disabled() -> Self {
        Self { startup: None }
    }

    /// Drain queued jobs while the server is still up, abandoning them after
    /// [`EMBED_DRAIN_BUDGET`], then stop it. A start still under way is
    /// cancelled. The memory writer must still be running.
    pub(super) async fn shutdown(
        self,
        worker_shutdown: &watch::Sender<bool>,
        server_shutdown: &watch::Sender<bool>,
    ) {
        worker_shutdown.send_replace(true);
        let Some(startup) = self.startup else {
            return;
        };
        if !startup.is_finished() {
            server_shutdown.send_replace(true);
        }
        let running = match startup.await {
            Ok(running) => running,
            Err(e) => {
                tracing::error!("embedding startup task failed: {e}");
                None
            }
        };
        let Some(RunningEmbedding { server, worker }) = running else {
            return;
        };
        if let Some(worker) = worker {
            join_worker_within_budget(worker).await;
        }
        server_shutdown.send_replace(true);
        if let Err(e) = server.shutdown().await {
            tracing::warn!("embed-server shutdown error: {e:#}");
        }
    }
}

async fn join_worker_within_budget(mut worker: JoinHandle<()>) {
    if tokio::time::timeout(EMBED_DRAIN_BUDGET, &mut worker)
        .await
        .is_err()
    {
        tracing::warn!(
            target: "assistd::embed",
            "embed queue drain timed out at shutdown; queued rows stay unindexed"
        );
        worker.abort();
        let _ = worker.await;
    }
}

/// The handles requests use from startup and, when embedding is enabled,
/// what starting its server needs.
pub(super) fn prepare(
    config: &Config,
    api_key: &ApiKey,
    sqlite_handle: Option<&Arc<SqliteHandle>>,
) -> (EmbeddingHandles, Option<EmbeddingStartup>) {
    if !config.embedding.enabled {
        info!("embedding: disabled in config (embedding.enabled = false)");
        return (disabled_handles(), None);
    }
    let semantic_store: Arc<dyn SemanticStore> = match sqlite_handle {
        Some(h) => Arc::new(SqliteSemanticStore::new(h.clone())),
        None => Arc::new(NoSemanticStore),
    };
    let writer_tx = sqlite_handle.map_or_else(closed_writer, |h| h.writer_tx());
    let (embed_tx, embed_rx) = mpsc::channel(EMBED_QUEUE_CAPACITY);
    let handles = EmbeddingHandles {
        embedder: Arc::new(EmbedderHandle::new(Readiness::Starting)),
        semantic_store,
        embed_tx,
    };
    let startup = EmbeddingStartup {
        config: config.embedding.clone(),
        api_key: api_key.clone(),
        embed_rx,
        writer_tx,
    };
    (handles, Some(startup))
}

fn disabled_handles() -> EmbeddingHandles {
    let (embed_tx, embed_rx) = mpsc::channel(1);
    drop(embed_rx);
    EmbeddingHandles {
        embedder: Arc::new(EmbedderHandle::new(Readiness::Unavailable(
            "disabled in config (embedding.enabled = false)".into(),
        ))),
        semantic_store: Arc::new(NoSemanticStore),
        embed_tx,
    }
}

fn closed_writer() -> Arc<mpsc::Sender<WriteOp>> {
    let (tx, rx) = mpsc::channel(1);
    drop(rx);
    Arc::new(tx)
}

fn spawn_startup(
    startup: EmbeddingStartup,
    state: Arc<AppState>,
    mut model_settled: watch::Receiver<bool>,
    stages: EmbeddingStages,
) -> JoinHandle<Option<RunningEmbedding>> {
    tokio::spawn(async move {
        if model_settled.wait_for(|settled| *settled).await.is_err() {
            return None;
        }
        let running = match start_server(&startup.config, &startup.api_key, stages.server).await {
            Ok((server, embedder)) => {
                info!(
                    "embedding: ready (model={}, dim={}, port={})",
                    embedder.model(),
                    embedder.dim(),
                    startup.config.port,
                );
                let worker = serve_embedder(&state, embedder.clone(), startup, stages.worker);
                report_unindexed_rows(state.memory.semantic.as_ref(), embedder.model()).await;
                Some(RunningEmbedding {
                    server,
                    worker: Some(worker),
                })
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
        state.publish_readiness(
            StartupComponent::Embedding,
            ComponentReadiness::from(state.memory.embedder.readiness()),
        );
        running
    })
}

fn serve_embedder(
    state: &AppState,
    embedder: Arc<dyn Embedder>,
    startup: EmbeddingStartup,
    worker_shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    let worker = spawn_embedder_task(
        embedder.clone(),
        startup.writer_tx,
        startup.embed_rx,
        worker_shutdown,
    );
    state.memory.embedder.set(Readiness::Ready(embedder));
    worker
}

/// Why the embedder did not come up, with the server when it started
/// but its client probe failed.
struct StartFailure {
    reason: String,
    server: Option<ChildServer>,
}

async fn start_server(
    config: &assistd_config::EmbeddingConfig,
    api_key: &ApiKey,
    server_shutdown: watch::Receiver<bool>,
) -> Result<(ChildServer, Arc<dyn Embedder>), StartFailure> {
    let spec = EmbedServerSpec::new(config.clone(), api_key.clone());
    let server = ChildServer::start(spec, server_shutdown)
        .await
        .map_err(|e| StartFailure {
            reason: format!("server failed to start: {e:#}"),
            server: None,
        })?;
    match LlamaEmbedder::new(
        SocketAddr::new(config.host, config.port.get()),
        config.model.clone(),
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

async fn report_unindexed_rows(semantic: &dyn SemanticStore, model_name: &str) {
    warn_if_stale_rows(semantic, model_name).await;
    warn_if_missing_rows(semantic, model_name).await;
}

async fn warn_if_stale_rows(semantic: &dyn SemanticStore, model_name: &str) {
    match semantic.count_stale(model_name).await {
        Ok((n, models)) if n > 0 => {
            tracing::warn!(
                "embedding: {n} rows exist under non-current model(s) {models:?}; \
                 run `assistd memory reindex` to rebuild against {model_name}"
            );
        }
        Ok(_) => {}
        Err(e) => {
            tracing::debug!("embedding: count_stale check failed ({e:#}); skipping diagnostic");
        }
    }
}

async fn warn_if_missing_rows(semantic: &dyn SemanticStore, model_name: &str) {
    match semantic.count_missing(model_name).await {
        Ok((chunks, memories)) if chunks + memories > 0 => {
            tracing::warn!(
                "embedding: {chunks} conversation chunks and {memories} memories have no \
                 {model_name} embedding and are invisible to semantic recall; \
                 run `assistd memory reindex` to index them"
            );
        }
        Ok(_) => {}
        Err(e) => {
            tracing::debug!("embedding: count_missing check failed ({e:#}); skipping diagnostic");
        }
    }
}

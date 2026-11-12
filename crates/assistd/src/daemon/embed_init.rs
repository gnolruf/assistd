//! Embedding subsystem wiring for the daemon: handles requests use from
//! startup, and the embedding server that comes up behind them afterwards.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use assistd_core::{AppState, Config};
use assistd_embed::{
    EmbedServerSpec, Embedder, EmbedderHandle, LlamaEmbedder, spawn_embedder_task,
};
use assistd_ipc::{ComponentReadiness, StartupComponent};
use assistd_memory::{NoSemanticStore, SemanticStore, SqliteHandle, SqliteSemanticStore};
use assistd_utils::child_server::{ApiKey, ChildServer};
use assistd_utils::readiness::Readiness;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::info;

use super::background_start::BackgroundStart;

const SUBSYSTEM: &str = "embedding";

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
    api_key: ApiKey,
    embed_wake: Arc<Notify>,
}

/// The shutdown signals the embedding server and worker stop on.
pub(super) struct EmbeddingStages {
    pub worker: watch::Receiver<bool>,
    pub server: watch::Receiver<bool>,
}

/// The background start, joined at shutdown for what it brought up.
pub(super) struct EmbeddingService {
    startup: BackgroundStart<RunningEmbedding>,
}

/// The server and, when its client probe passed, the worker embedding
/// rows missing an embedding.
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
        let panicked_state = Arc::clone(&state);
        let startup = BackgroundStart::spawn(
            SUBSYSTEM,
            async move { start_and_record(self, &state, model_settled, stages).await },
            move || {
                panicked_state
                    .memory
                    .embedder
                    .set(Readiness::Unavailable("startup panicked".into()));
                publish_readiness(&panicked_state);
            },
        );
        EmbeddingService { startup }
    }
}

impl EmbeddingService {
    pub(super) fn disabled() -> Self {
        Self {
            startup: BackgroundStart::disabled(SUBSYSTEM),
        }
    }

    /// Let the worker finish its batch while the server is still up, abandoning
    /// it after [`EMBED_DRAIN_BUDGET`] or at `deadline`, then stop the server. A
    /// start still under way is cancelled. The memory database must still be open.
    pub(super) async fn shutdown(
        self,
        worker_shutdown: &watch::Sender<bool>,
        server_shutdown: &watch::Sender<bool>,
        deadline: Instant,
    ) {
        worker_shutdown.send_replace(true);
        if !self.startup.is_finished() {
            server_shutdown.send_replace(true);
        }
        let Some(RunningEmbedding { server, worker }) = self.startup.join().await else {
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
}

async fn join_worker_within_budget(mut worker: JoinHandle<()>, deadline: Instant) {
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
    let running = match start_server(&startup.config, &startup.api_key, stages.server).await {
        Ok((server, embedder)) => {
            info!(
                "embedding: ready (model={}, dim={}, port={})",
                embedder.model(),
                embedder.dim(),
                startup.config.port,
            );
            let worker =
                drop_other_models_then_serve(state, embedder, startup, stages.worker).await;
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

/// Drop vectors from other models, then start the worker and mark the embedder
/// ready. When dropping fails the embedder stays unavailable.
async fn drop_other_models_then_serve(
    state: &AppState,
    embedder: Arc<dyn Embedder>,
    startup: EmbeddingStartup,
    worker_shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    let semantic = state.memory.semantic.as_ref();
    if let Err(reason) = drop_other_models(semantic, embedder.model()).await {
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
) -> JoinHandle<()> {
    let worker = spawn_embedder_task(
        embedder.clone(),
        state.memory.semantic.clone(),
        startup.embed_wake,
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
        SocketAddr::new(config.host.get(), config.port.get()),
        config.model.to_string(),
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

async fn drop_other_models(semantic: &dyn SemanticStore, model_name: &str) -> Result<(), String> {
    match semantic.drop_other_models(model_name).await {
        Ok(0) => Ok(()),
        Ok(dropped) => {
            info!("embedding: dropped {dropped} vectors from models other than {model_name}");
            Ok(())
        }
        Err(e) => Err(format!("dropping other models' vectors failed: {e}")),
    }
}

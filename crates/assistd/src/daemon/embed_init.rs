//! Embedding subsystem wiring for the daemon.

use std::sync::Arc;
use std::time::Duration;

use assistd_core::Config;
use assistd_embed::{
    EmbedJob, EmbedServerSpec, Embedder, LlamaEmbedder, NoEmbedder, spawn_embedder_task,
};
use assistd_memory::{NoSemanticStore, SemanticStore, SqliteHandle, SqliteSemanticStore};
use assistd_utils::child_server::ChildServer;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::info;

pub(super) struct EmbeddingSubsystem {
    pub embedder: Arc<dyn Embedder>,
    pub semantic_store: Arc<dyn SemanticStore>,
    pub embed_tx: mpsc::Sender<EmbedJob>,
    pub service_handle: Option<ChildServer>,
    pub task_handle: Option<JoinHandle<()>>,
    pub model_name: String,
}

impl EmbeddingSubsystem {
    fn disabled(service_handle: Option<ChildServer>) -> Self {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        Self {
            embedder: Arc::new(NoEmbedder),
            semantic_store: Arc::new(NoSemanticStore),
            embed_tx: tx,
            service_handle,
            task_handle: None,
            model_name: String::new(),
        }
    }

    /// Drain queued jobs while the embed server is still up, then stop the
    /// server. The memory writer must still be running.
    pub(super) async fn shutdown(
        self,
        worker_shutdown: &watch::Sender<bool>,
        server_shutdown: &watch::Sender<bool>,
    ) {
        worker_shutdown.send_replace(true);
        if let Some(h) = self.task_handle {
            let _ = h.await;
        }
        server_shutdown.send_replace(true);
        if let Some(service) = self.service_handle
            && let Err(e) = service.shutdown().await
        {
            tracing::warn!("embed-server shutdown error: {e:#}");
        }
    }
}

/// Degrades to a no-op subsystem when disabled, when the embed server
/// fails to start, or when the client probe fails.
pub(super) async fn init(
    config: &Config,
    sqlite_handle: Option<&Arc<SqliteHandle>>,
    worker_shutdown: &watch::Sender<bool>,
    server_shutdown: &watch::Sender<bool>,
) -> EmbeddingSubsystem {
    if !config.embedding.enabled {
        info!("embedding: disabled in config (embedding.enabled = false)");
        return EmbeddingSubsystem::disabled(None);
    }

    let service = match ChildServer::start(
        EmbedServerSpec::new(
            config.embedding.clone(),
            config.llama_server.binary_path.clone(),
            Duration::from_secs(config.llama_server.ready_timeout_secs.get()),
        ),
        server_shutdown.subscribe(),
    )
    .await
    {
        Ok(service) => service,
        Err(e) => {
            tracing::warn!("embedding: failed to start ({e:#}); semantic search disabled this run");
            return EmbeddingSubsystem::disabled(None);
        }
    };

    let client = match LlamaEmbedder::new(
        &config.embedding.host.to_string(),
        config.embedding.port.get(),
        config.embedding.model.clone(),
        assistd_embed::REQUEST_TIMEOUT,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                "embedding: client probe failed ({e:#}); semantic search disabled this run"
            );
            return EmbeddingSubsystem::disabled(Some(service));
        }
    };

    let model_name = config.embedding.model.clone();
    let embedder: Arc<dyn Embedder> = Arc::new(client);
    let semantic_store: Arc<dyn SemanticStore> = match sqlite_handle {
        Some(h) => Arc::new(SqliteSemanticStore::new(h.clone())),
        None => Arc::new(NoSemanticStore),
    };
    let writer_tx = sqlite_handle.map_or_else(
        || {
            let (tx, rx) = mpsc::channel(1);
            drop(rx);
            Arc::new(tx)
        },
        |h| h.writer_tx(),
    );
    let (embed_tx, embed_rx) = mpsc::channel(256);
    let task = spawn_embedder_task(
        embedder.clone(),
        writer_tx,
        embed_rx,
        worker_shutdown.subscribe(),
    );

    info!(
        "embedding: ready (model={}, dim={}, port={})",
        embedder.model(),
        embedder.dim(),
        config.embedding.port,
    );

    warn_if_stale_rows(semantic_store.as_ref(), &model_name).await;

    EmbeddingSubsystem {
        embedder,
        semantic_store,
        embed_tx,
        service_handle: Some(service),
        task_handle: Some(task),
        model_name,
    }
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

//! Embedding subsystem wiring for the daemon.
//!
//! Boots the local embed-server, builds the [`LlamaEmbedder`] client,
//! and spawns the writer-fed embedder task that consumes [`EmbedJob`]s.
//! When the SQLite store is available the produced
//! [`SqliteSemanticStore`] is plumbed through; otherwise we fall back
//! to a no-op semantic store so semantic-search tools degrade
//! gracefully. A diagnostic logs how many existing rows were embedded
//! against a stale model, prompting the operator to reindex.

use std::sync::Arc;
use std::time::Duration;

use assistd_core::Config;
use assistd_embed::{
    EmbedJob, EmbedService, Embedder, LlamaEmbedder, NoEmbedder, spawn_embedder_task,
};
use assistd_memory::{NoSemanticStore, SemanticStore, SqliteHandle, SqliteSemanticStore};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::info;

/// Live handles for the embedding subsystem, returned by [`init`].
pub struct EmbeddingSubsystem {
    /// Embedder client used to produce vectors from text.
    pub embedder: Arc<dyn Embedder>,
    /// Semantic search store backed by SQLite (or a no-op if unavailable).
    pub semantic_store: Arc<dyn SemanticStore>,
    /// Channel for submitting background [`EmbedJob`]s to the writer task.
    pub embed_tx: mpsc::Sender<EmbedJob>,
    /// Running embed-server process handle, shut down on [`EmbeddingSubsystem::shutdown`].
    pub service_handle: Option<EmbedService>,
    /// Background embedder writer task.
    pub task_handle: Option<JoinHandle<()>>,
    /// Model name reported by the running embed server.
    pub model_name: String,
}

impl EmbeddingSubsystem {
    fn disabled(service_handle: Option<EmbedService>) -> Self {
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

    /// Await the embedder task and shut down the embed server gracefully.
    pub async fn shutdown(self) {
        if let Some(h) = self.task_handle {
            let _ = h.await;
        }
        if let Some(svc) = self.service_handle {
            if let Err(e) = svc.shutdown().await {
                tracing::warn!("embed-server shutdown error: {e:#}");
            }
        }
    }
}

/// Initialise the embedding subsystem from config.
///
/// Starts the embed server, probes the model, and wires the embedder task.
/// Degrades gracefully to a no-op subsystem when the server fails to start
/// or the client probe fails.
pub async fn init(
    config: &Config,
    sqlite_handle: Option<&Arc<SqliteHandle>>,
    shutdown_tx: &watch::Sender<bool>,
) -> EmbeddingSubsystem {
    if !config.embedding.enabled {
        info!("embedding: disabled in config (embedding.enabled = false)");
        return EmbeddingSubsystem::disabled(None);
    }

    let svc = match EmbedService::start(config.embedding.clone(), shutdown_tx.subscribe()).await {
        Ok(svc) => svc,
        Err(e) => {
            tracing::warn!("embedding: failed to start ({e:#}); semantic search disabled this run");
            return EmbeddingSubsystem::disabled(None);
        }
    };

    let client = match LlamaEmbedder::new(
        &config.embedding.host,
        config.embedding.port,
        config.embedding.model.clone(),
        Duration::from_secs(config.embedding.request_timeout_secs),
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                "embedding: client probe failed ({e:#}); semantic search disabled this run"
            );
            return EmbeddingSubsystem::disabled(Some(svc));
        }
    };

    let model_name = config.embedding.model.clone();
    let embedder: Arc<dyn Embedder> = Arc::new(client);
    let semantic: Arc<dyn SemanticStore> = match sqlite_handle {
        Some(h) => Arc::new(SqliteSemanticStore::new(h.clone())),
        None => Arc::new(NoSemanticStore),
    };
    let writer_tx = sqlite_handle.map(|h| h.writer_tx()).unwrap_or_else(|| {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        Arc::new(tx)
    });
    let (embed_tx, embed_rx) = mpsc::channel(256);
    let task = spawn_embedder_task(
        embedder.clone(),
        writer_tx,
        embed_rx,
        shutdown_tx.subscribe(),
    );

    info!(
        "embedding: ready (model={}, dim={}, port={})",
        embedder.model(),
        embedder.dim(),
        config.embedding.port,
    );

    match semantic.count_stale(&model_name).await {
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

    EmbeddingSubsystem {
        embedder,
        semantic_store: semantic,
        embed_tx,
        service_handle: Some(svc),
        task_handle: Some(task),
        model_name,
    }
}

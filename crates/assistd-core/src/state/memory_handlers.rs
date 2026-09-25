//! Handlers for the `Memory*` variants of `Request`.

use std::future::Future;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::warn;

use assistd_embed::{BATCH_SIZE, embed_each};
use assistd_ipc::{Event, ReindexKind};
use assistd_memory::{MemoryError, vector_to_blob};
use assistd_tools::DEFAULT_SEARCH_LIMIT;

use super::{AppState, DispatchError, send_error, wire_role};

impl AppState {
    pub(super) async fn handle_memory_semantic_search(
        self: Arc<Self>,
        id: String,
        query: String,
        limit: u32,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let model = self.memory.embedder.model().to_string();
        if model.is_empty() {
            let _ = tx.send(Event::Done { id }).await;
            return Ok(());
        }
        let limit = if limit == 0 {
            DEFAULT_SEARCH_LIMIT
        } else {
            limit as usize
        };
        let embedding = match self.memory.embedder.embed(query).await {
            Ok(embedding) => embedding,
            Err(e) => {
                send_error(&tx, id, format!("embed failed: {e}")).await;
                return Err(e.into());
            }
        };
        match self
            .memory
            .semantic
            .nearest_chunks(embedding, limit, &model, None)
            .await
        {
            Ok(hits) => {
                for hit in hits {
                    let _ = tx
                        .send(Event::SemanticHit {
                            id: id.clone(),
                            conversation_id: hit.conversation_id,
                            chunk_id: hit.chunk_id,
                            session_id: hit.session_id,
                            timestamp: hit.timestamp,
                            role: wire_role(hit.role),
                            content: hit.content,
                            similarity: hit.similarity,
                        })
                        .await;
                }
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("semantic search failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_memory_save(
        self: Arc<Self>,
        id: String,
        key: String,
        value: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match self.memory.memory_ops.save(&key, value).await {
            Ok(_id) => {
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("memory save failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_memory_load(
        self: Arc<Self>,
        id: String,
        key: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match self.memory.memory_ops.load(&key).await {
            Ok(value) => {
                let _ = tx
                    .send(Event::MemoryValue {
                        id: id.clone(),
                        key,
                        value,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("memory load failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_memory_list(
        self: Arc<Self>,
        id: String,
        prefix: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match self.memory.memory_ops.list(&prefix).await {
            Ok(keys) => {
                let _ = tx
                    .send(Event::MemoryKeys {
                        id: id.clone(),
                        keys,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("memory list failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_memory_delete(
        self: Arc<Self>,
        id: String,
        key: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match self.memory.memory_ops.delete(&key).await {
            Ok(()) => {
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("memory delete failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_memory_list_all(
        self: Arc<Self>,
        id: String,
        prefix: String,
        limit: u32,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match self.memory.memory_ops.list_full(&prefix).await {
            Ok(rows) => {
                let cap = if limit == 0 {
                    rows.len()
                } else {
                    rows.len().min(limit as usize)
                };
                for row in rows.into_iter().take(cap) {
                    let _ = tx
                        .send(Event::MemoryRow {
                            id: id.clone(),
                            memory_id: row.id,
                            key: row.key,
                            value: row.value,
                        })
                        .await;
                }
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("memory list_all failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    pub(super) async fn handle_memory_forget(
        self: Arc<Self>,
        id: String,
        memory_id: i64,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match self.memory.memory_ops.forget(memory_id).await {
            Ok(removed) => {
                let _ = tx
                    .send(Event::MemoryForgetResult {
                        id: id.clone(),
                        deleted: removed.is_some(),
                        key: removed,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, format!("memory forget failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    /// Embed every memory and chunk lacking an embedding under the current
    /// model, streaming `ReindexProgress`. Per-item failures are logged and
    /// counted as done.
    pub(super) async fn handle_memory_reindex(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let model = self.memory.embedder.model().to_string();
        if model.is_empty() {
            send_error(
                &tx,
                id,
                "embedding subsystem disabled; cannot reindex".to_string(),
            )
            .await;
            return Ok(());
        }
        let dim = self.memory.embedder.dim() as i64;

        let chunks = match self.memory.semantic.chunks_missing_embedding(&model).await {
            Ok(chunks) => chunks,
            Err(e) => {
                send_error(&tx, id, format!("reindex: list missing chunks: {e}")).await;
                return Err(e.into());
            }
        };
        let memories = match self
            .memory
            .semantic
            .memories_missing_embedding(&model)
            .await
        {
            Ok(memories) => memories,
            Err(e) => {
                send_error(&tx, id, format!("reindex: list missing memories: {e}")).await;
                return Err(e.into());
            }
        };
        let chunks_total = chunks.len() as u32;
        let memories_total = memories.len() as u32;

        let _ = tx
            .send(Event::ReindexProgress {
                id: id.clone(),
                kind: ReindexKind::Chunks,
                done: 0,
                total: chunks_total,
            })
            .await;
        let _ = tx
            .send(Event::ReindexProgress {
                id: id.clone(),
                kind: ReindexKind::Memories,
                done: 0,
                total: memories_total,
            })
            .await;

        let semantic = &self.memory.semantic;
        self.reindex_items(&id, &tx, ReindexKind::Chunks, chunks, |chunk_id, blob| {
            semantic.store_chunk_embedding(chunk_id, model.clone(), dim, blob)
        })
        .await;
        self.reindex_items(
            &id,
            &tx,
            ReindexKind::Memories,
            memories,
            |memory_id, blob| semantic.store_memory_embedding(memory_id, model.clone(), dim, blob),
        )
        .await;

        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    async fn reindex_items<F, Fut>(
        &self,
        id: &str,
        tx: &mpsc::Sender<Event>,
        kind: ReindexKind,
        items: Vec<(i64, String)>,
        store: F,
    ) where
        F: Fn(i64, Vec<u8>) -> Fut,
        Fut: Future<Output = Result<(), MemoryError>>,
    {
        let total = items.len() as u32;
        let mut done = 0u32;
        for batch in items.chunks(BATCH_SIZE) {
            let texts: Vec<&str> = batch.iter().map(|(_, text)| text.as_str()).collect();
            let results = embed_each(&*self.memory.embedder, &texts).await;
            for (&(item_id, _), result) in batch.iter().zip(results) {
                match result {
                    Ok(embedding) => {
                        if let Err(e) = store(item_id, vector_to_blob(&embedding)).await {
                            warn!(
                                target: "assistd::memory",
                                kind = kind.as_str(),
                                item_id,
                                error = %e,
                                "reindex: store embedding failed"
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            target: "assistd::memory",
                            kind = kind.as_str(),
                            item_id,
                            error = %e,
                            "reindex: embed failed"
                        );
                    }
                }
                done += 1;
                let _ = tx
                    .send(Event::ReindexProgress {
                        id: id.to_string(),
                        kind,
                        done,
                        total,
                    })
                    .await;
            }
        }
    }
}

//! Handlers for the `Memory*` variants of `Request`.

use super::AppState;
use anyhow::Result;
use assistd_ipc::Event;
use std::sync::Arc;
use tokio::sync::mpsc;

impl AppState {
    pub(super) async fn handle_memory_semantic_search(
        self: Arc<Self>,
        id: String,
        query: String,
        limit: u32,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let model = self.memory.embedder.model().to_string();
        if model.is_empty() {
            let _ = tx.send(Event::Done { id }).await;
            return Ok(());
        }
        let limit = if limit == 0 {
            assistd_tools::DEFAULT_SEARCH_LIMIT
        } else {
            limit as usize
        };
        let vec = match self.memory.embedder.embed(query).await {
            Ok(v) => v,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("embed failed: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };
        match self
            .memory
            .semantic
            .nearest_chunks(vec, limit, &model)
            .await
        {
            Ok(hits) => {
                for h in hits {
                    let _ = tx
                        .send(Event::SemanticHit {
                            id: id.clone(),
                            conversation_id: h.conversation_id,
                            chunk_id: h.chunk_id,
                            session_id: h.session_id,
                            timestamp: h.timestamp,
                            role: h.role.as_wire().into(),
                            content: h.content,
                            similarity: h.similarity,
                        })
                        .await;
                }
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("semantic search failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_memory_save(
        self: Arc<Self>,
        id: String,
        key: String,
        value: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.memory.memory_ops.save(&key, value).await {
            Ok(_id) => {
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory save failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_memory_load(
        self: Arc<Self>,
        id: String,
        key: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
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
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory load failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_memory_list(
        self: Arc<Self>,
        id: String,
        prefix: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
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
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory list failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_memory_delete(
        self: Arc<Self>,
        id: String,
        key: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.memory.memory_ops.delete(&key).await {
            Ok(()) => {
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory delete failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_memory_list_all(
        self: Arc<Self>,
        id: String,
        prefix: String,
        limit: u32,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
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
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory list_all failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    pub(super) async fn handle_memory_forget(
        self: Arc<Self>,
        id: String,
        memory_id: i64,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
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
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory forget failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    /// Embed every memory and every conversation chunk that has no
    /// embedding under the daemon's currently-configured model. Streams
    /// `ReindexProgress` events as items complete; finishes with `Done`
    /// (or `Error` if no embedder is configured). Per-item failures
    /// during embed/write are logged and counted as still-done so a
    /// single bad row doesn't wedge the whole run; same log-and-drop
    /// posture as the background embedder task.
    pub(super) async fn handle_memory_reindex(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let model = self.memory.embedder.model().to_string();
        if model.is_empty() {
            let _ = tx
                .send(Event::Error {
                    id,
                    message: "embedding subsystem disabled; cannot reindex".to_string(),
                })
                .await;
            return Ok(());
        }
        let dim = self.memory.embedder.dim() as i64;

        let chunks = match self.memory.semantic.chunks_missing_embedding(&model).await {
            Ok(v) => v,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("reindex: list missing chunks: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };
        let memories = match self
            .memory
            .semantic
            .memories_missing_embedding(&model)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("reindex: list missing memories: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };
        let chunks_total = chunks.len() as u32;
        let memories_total = memories.len() as u32;

        let _ = tx
            .send(Event::ReindexProgress {
                id: id.clone(),
                kind: "chunks".to_string(),
                done: 0,
                total: chunks_total,
            })
            .await;
        let _ = tx
            .send(Event::ReindexProgress {
                id: id.clone(),
                kind: "memories".to_string(),
                done: 0,
                total: memories_total,
            })
            .await;

        let mut done = 0u32;
        for (chunk_id, text) in chunks {
            match self.memory.embedder.embed(text).await {
                Ok(vec) => {
                    let blob = assistd_memory::vector_to_blob(&vec);
                    if let Err(e) = self
                        .memory
                        .semantic
                        .store_chunk_embedding(chunk_id, model.clone(), dim, blob)
                        .await
                    {
                        tracing::warn!(
                            target: "assistd::memory",
                            chunk_id,
                            error = %e,
                            "reindex: store_chunk_embedding failed"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "assistd::memory",
                        chunk_id,
                        error = %e,
                        "reindex: embed chunk failed"
                    );
                }
            }
            done = done.saturating_add(1);
            let _ = tx
                .send(Event::ReindexProgress {
                    id: id.clone(),
                    kind: "chunks".to_string(),
                    done,
                    total: chunks_total,
                })
                .await;
        }

        let mut done = 0u32;
        for (memory_id, value) in memories {
            match self.memory.embedder.embed(value).await {
                Ok(vec) => {
                    let blob = assistd_memory::vector_to_blob(&vec);
                    if let Err(e) = self
                        .memory
                        .semantic
                        .store_memory_embedding(memory_id, model.clone(), dim, blob)
                        .await
                    {
                        tracing::warn!(
                            target: "assistd::memory",
                            memory_id,
                            error = %e,
                            "reindex: store_memory_embedding failed"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "assistd::memory",
                        memory_id,
                        error = %e,
                        "reindex: embed memory failed"
                    );
                }
            }
            done = done.saturating_add(1);
            let _ = tx
                .send(Event::ReindexProgress {
                    id: id.clone(),
                    kind: "memories".to_string(),
                    done,
                    total: memories_total,
                })
                .await;
        }

        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }
}

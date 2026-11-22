//! Handlers for the `Memory*` variants of `Request`.

use std::time::Duration;

use assistd_ipc::{Event, EventBody};
use assistd_memory::{ChunkLocator, MemoryId, SessionScope, UnindexedRows};
use assistd_tools::DEFAULT_SEARCH_LIMIT;

use super::memory_stack::MemoryStack;
use super::{DispatchError, FailedStep, wire_role, wire_timestamp};
use crate::bus::EventSink;

/// How often a reindex request counts the rows still waiting.
const REINDEX_COUNT_INTERVAL: Duration = Duration::from_secs(1);

impl MemoryStack {
    pub(super) async fn handle_semantic_search(
        &self,
        id: String,
        query: String,
        limit: u32,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let embedder = self.embedder.get().context("semantic search failed")?;
        let limit = if limit == 0 {
            DEFAULT_SEARCH_LIMIT
        } else {
            limit as usize
        };
        let embedding = embedder.embed_query(&query).await.context("embed failed")?;
        let hits = self
            .recall
            .nearest_chunks(embedding, limit, SessionScope::All)
            .await
            .context("semantic search failed")?;
        for hit in hits {
            let ChunkLocator::Message {
                session_id,
                message_id,
                role,
                timestamp,
                ..
            } = hit.locator
            else {
                continue;
            };
            let _ = tx
                .send(Event::new(
                    id.clone(),
                    EventBody::SemanticHit {
                        message_id: message_id.0,
                        chunk_id: hit.chunk_id.0,
                        session_id: session_id.to_string(),
                        timestamp: wire_timestamp(timestamp),
                        role: wire_role(role),
                        content: hit.text,
                        similarity: hit.similarity,
                    },
                ))
                .await;
        }
        Ok(())
    }

    pub(super) async fn handle_memory_save(
        &self,
        key: String,
        value: String,
    ) -> Result<(), DispatchError> {
        self.memory_ops
            .save(&key, value)
            .await
            .context("memory save failed")?;
        Ok(())
    }

    pub(super) async fn handle_memory_load(
        &self,
        id: String,
        key: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let value = self
            .memory_ops
            .load(&key)
            .await
            .context("memory load failed")?;
        tx.reply(id, EventBody::MemoryValue { key, value }).await;
        Ok(())
    }

    pub(super) async fn handle_memory_delete(&self, key: String) -> Result<(), DispatchError> {
        self.memory_ops
            .delete(&key)
            .await
            .context("memory delete failed")?;
        Ok(())
    }

    pub(super) async fn handle_memory_list_all(
        &self,
        id: String,
        prefix: String,
        limit: u32,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let rows = self
            .memory_ops
            .list_full(&prefix)
            .await
            .context("memory list_all failed")?;
        let cap = if limit == 0 {
            rows.len()
        } else {
            rows.len().min(limit as usize)
        };
        for row in rows.into_iter().take(cap) {
            let _ = tx
                .send(Event::new(
                    id.clone(),
                    EventBody::MemoryRow {
                        memory_id: row.id.0,
                        key: row.key,
                        value: row.value,
                    },
                ))
                .await;
        }
        Ok(())
    }

    pub(super) async fn handle_memory_forget(
        &self,
        id: String,
        memory_id: i64,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let removed = self
            .memory_ops
            .forget(MemoryId(memory_id))
            .await
            .context("memory forget failed")?;
        let result = EventBody::MemoryForgetResult {
            deleted: removed.is_some(),
            key: removed,
        };
        tx.reply(id, result).await;
        Ok(())
    }

    /// Clear the refusal marks, wake the embedding worker, and stream the rows still
    /// waiting as `ReindexProgress` until none is or the client leaves.
    pub(super) async fn handle_memory_reindex(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        self.embedder.get().context("reindex failed")?;
        self.embedding_queue
            .clear_unembeddable()
            .await
            .context("reindex: clear refusal marks")?;
        self.embed_wake.notify_one();
        let mut reported = None;
        loop {
            let waiting = self
                .embedding_queue
                .count_missing()
                .await
                .context("reindex: count waiting rows")?;
            if reported != Some(waiting) {
                let body = EventBody::ReindexProgress {
                    message_chunks: waiting.message_chunks,
                    document_chunks: waiting.document_chunks,
                    memories: waiting.memories,
                };
                if tx.send(Event::new(id.clone(), body)).await.is_err() {
                    return Ok(());
                }
                reported = Some(waiting);
            }
            if waiting == UnindexedRows::default() {
                return Ok(());
            }
            tokio::select! {
                () = tx.closed() => return Ok(()),
                () = tokio::time::sleep(REINDEX_COUNT_INTERVAL) => {}
            }
        }
    }
}

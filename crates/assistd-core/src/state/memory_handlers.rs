//! Handlers for the `Memory*` variants of `Request`.

use std::future::Future;

use tracing::warn;

use assistd_embed::{BATCH_SIZE, Embedder, embed_rows};
use assistd_ipc::{Event, EventBody, ReindexKind};
use assistd_memory::{
    ChunkLocator, ChunkScope, EmbeddingTarget, MemoryError, MemoryId, SemanticStore, SessionScope,
    UnindexedRows,
};
use assistd_tools::DEFAULT_SEARCH_LIMIT;

use super::memory_stack::MemoryStack;
use super::{DispatchError, FailedStep, wire_role, wire_timestamp};
use crate::bus::EventSink;

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
        let embedding = embedder.embed(query).await.context("embed failed")?;
        let hits = self
            .semantic
            .nearest_chunks(embedding, limit, ChunkScope::Chats(SessionScope::All))
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
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    pub(super) async fn handle_memory_save(
        &self,
        id: String,
        key: String,
        value: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        self.memory_ops
            .save(&key, value)
            .await
            .context("memory save failed")?;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
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
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::MemoryValue { key, value },
            ))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    pub(super) async fn handle_memory_delete(
        &self,
        id: String,
        key: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        self.memory_ops
            .delete(&key)
            .await
            .context("memory delete failed")?;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
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
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
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
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::MemoryForgetResult {
                    deleted: removed.is_some(),
                    key: removed,
                },
            ))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    /// Embed every memory and chunk lacking an embedding under the current
    /// model, rows marked unembeddable included, streaming `ReindexProgress`.
    /// Per-item failures are logged and counted as done.
    pub(super) async fn handle_memory_reindex(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let embedder = self.embedder.get().context("reindex failed")?;
        let unindexed = self
            .unindexed_rows_with_marks_cleared()
            .await
            .context("reindex: count missing rows")?;
        let chunks_total = u32::try_from(unindexed.chunks).unwrap_or(u32::MAX);
        let memories_total = u32::try_from(unindexed.memories).unwrap_or(u32::MAX);

        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::ReindexProgress {
                    kind: ReindexKind::Chunks,
                    done: 0,
                    total: chunks_total,
                },
            ))
            .await;
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::ReindexProgress {
                    kind: ReindexKind::Memories,
                    done: 0,
                    total: memories_total,
                },
            ))
            .await;

        let semantic = self.semantic.as_ref();
        let embedder = embedder.as_ref();
        reindex_items(
            embedder,
            semantic,
            KindProgress::new(&id, &tx, ReindexKind::Chunks, chunks_total),
            |after, limit| semantic.chunks_missing_embedding(after, limit),
            EmbeddingTarget::Chunk,
        )
        .await
        .context("reindex: list missing chunks")?;
        reindex_items(
            embedder,
            semantic,
            KindProgress::new(&id, &tx, ReindexKind::Memories, memories_total),
            |after, limit| semantic.memories_missing_embedding(after, limit),
            EmbeddingTarget::Memory,
        )
        .await
        .context("reindex: list missing memories")?;

        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    async fn unindexed_rows_with_marks_cleared(&self) -> Result<UnindexedRows, MemoryError> {
        self.semantic.clear_unembeddable().await?;
        self.semantic.count_missing().await
    }
}

/// Progress through the `total` rows of one [`ReindexKind`], streamed
/// to `tx` as `ReindexProgress` events.
struct KindProgress<'a> {
    id: &'a str,
    tx: &'a EventSink,
    kind: ReindexKind,
    total: u32,
    done: u32,
}

impl<'a> KindProgress<'a> {
    fn new(id: &'a str, tx: &'a EventSink, kind: ReindexKind, total: u32) -> Self {
        Self {
            id,
            tx,
            kind,
            total,
            done: 0,
        }
    }

    fn remaining(&self) -> usize {
        usize::try_from(self.total - self.done).unwrap_or(usize::MAX)
    }

    async fn advance(&mut self, rows: usize) {
        self.done = self
            .done
            .saturating_add(u32::try_from(rows).unwrap_or(u32::MAX));
        let _ = self
            .tx
            .send(Event::new(
                self.id.to_string(),
                EventBody::ReindexProgress {
                    kind: self.kind,
                    done: self.done,
                    total: self.total,
                },
            ))
            .await;
    }
}

/// Embed and store up to `progress.total` rows, listed a page at a time in
/// id order. Failed embeds and stores are logged; a listing failure aborts.
async fn reindex_items<Id, L, LFut>(
    embedder: &dyn Embedder,
    semantic: &dyn SemanticStore,
    mut progress: KindProgress<'_>,
    list: L,
    to_target: fn(Id) -> EmbeddingTarget,
) -> Result<(), MemoryError>
where
    Id: Copy,
    L: Fn(Option<Id>, usize) -> LFut,
    LFut: Future<Output = Result<Vec<(Id, String)>, MemoryError>>,
{
    let mut after = None;
    while progress.remaining() > 0 {
        let batch = list(after, progress.remaining().min(BATCH_SIZE)).await?;
        let Some(&(last_id, _)) = batch.last() else {
            break;
        };
        after = Some(last_id);
        let row_count = batch.len();
        let rows = batch
            .into_iter()
            .map(|(id, text)| (to_target(id), text))
            .collect();
        let embedded = embed_rows(embedder, rows).await;
        if let Some(e) = embedded.deferred {
            warn!(
                target: "assistd::memory",
                kind = progress.kind.as_str(),
                error = %e,
                "reindex: embedding failed; rows stay unindexed"
            );
        }
        if let Err(e) = semantic.store_embeddings(embedded.embeddings).await {
            warn!(
                target: "assistd::memory",
                kind = progress.kind.as_str(),
                error = %e,
                "reindex: store embeddings failed"
            );
        }
        progress.advance(row_count).await;
    }
    Ok(())
}

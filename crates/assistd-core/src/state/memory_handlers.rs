//! Handlers for the `Memory*` variants of `Request`.

use std::future::Future;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::warn;

use assistd_embed::{BATCH_SIZE, EmbedJob, Embedder, embed_jobs};
use assistd_ipc::{Event, ReindexKind};
use assistd_memory::{EmbeddingTarget, MemoryError, MemoryId, SemanticStore};
use assistd_tools::DEFAULT_SEARCH_LIMIT;

use super::{AppState, DispatchError, send_error, wire_role, wire_timestamp};

impl AppState {
    pub(super) async fn handle_memory_semantic_search(
        self: Arc<Self>,
        id: String,
        query: String,
        limit: u32,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let embedder = match self.memory.embedder.get() {
            Ok(embedder) => embedder,
            Err(e) => {
                send_error(&tx, id, format!("semantic search failed: {e}")).await;
                return Err(e.into());
            }
        };
        let limit = if limit == 0 {
            DEFAULT_SEARCH_LIMIT
        } else {
            limit as usize
        };
        let embedding = match embedder.embed(query).await {
            Ok(embedding) => embedding,
            Err(e) => {
                send_error(&tx, id, format!("embed failed: {e}")).await;
                return Err(e.into());
            }
        };
        match self
            .memory
            .semantic
            .nearest_chunks(embedding, limit, embedder.model(), None)
            .await
        {
            Ok(hits) => {
                for hit in hits {
                    let _ = tx
                        .send(Event::SemanticHit {
                            id: id.clone(),
                            message_id: hit.message_id.0,
                            chunk_id: hit.chunk_id.0,
                            session_id: hit.session_id.to_string(),
                            timestamp: wire_timestamp(hit.timestamp),
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
                            memory_id: row.id.0,
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
        match self.memory.memory_ops.forget(MemoryId(memory_id)).await {
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
        let embedder = match self.memory.embedder.get() {
            Ok(embedder) => embedder,
            Err(e) => {
                send_error(&tx, id, format!("reindex failed: {e}")).await;
                return Err(e.into());
            }
        };
        let model = embedder.model().to_string();

        let unindexed = match self.memory.semantic.count_missing(&model).await {
            Ok(unindexed) => unindexed,
            Err(e) => {
                send_error(&tx, id, format!("reindex: count missing rows: {e}")).await;
                return Err(e.into());
            }
        };
        let chunks_total = u32::try_from(unindexed.chunks).unwrap_or(u32::MAX);
        let memories_total = u32::try_from(unindexed.memories).unwrap_or(u32::MAX);

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

        let semantic = self.memory.semantic.as_ref();
        let embedder = embedder.as_ref();
        let chunks = reindex_items(
            embedder,
            semantic,
            KindProgress::new(&id, &tx, ReindexKind::Chunks, chunks_total),
            |after, limit| semantic.chunks_missing_embedding(&model, after, limit),
            EmbeddingTarget::Chunk,
        )
        .await;
        if let Err(e) = chunks {
            send_error(&tx, id, format!("reindex: list missing chunks: {e}")).await;
            return Err(e.into());
        }
        let memories = reindex_items(
            embedder,
            semantic,
            KindProgress::new(&id, &tx, ReindexKind::Memories, memories_total),
            |after, limit| semantic.memories_missing_embedding(&model, after, limit),
            EmbeddingTarget::Memory,
        )
        .await;
        if let Err(e) = memories {
            send_error(&tx, id, format!("reindex: list missing memories: {e}")).await;
            return Err(e.into());
        }

        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }
}

/// Progress through the `total` rows of one [`ReindexKind`], streamed
/// to `tx` as `ReindexProgress` events.
struct KindProgress<'a> {
    id: &'a str,
    tx: &'a mpsc::Sender<Event>,
    kind: ReindexKind,
    total: u32,
    done: u32,
}

impl<'a> KindProgress<'a> {
    fn new(id: &'a str, tx: &'a mpsc::Sender<Event>, kind: ReindexKind, total: u32) -> Self {
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
            .send(Event::ReindexProgress {
                id: self.id.to_string(),
                kind: self.kind,
                done: self.done,
                total: self.total,
            })
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
    let mut jobs = Vec::with_capacity(BATCH_SIZE);
    while progress.remaining() > 0 {
        let batch = list(after, progress.remaining().min(BATCH_SIZE)).await?;
        let Some(&(last_id, _)) = batch.last() else {
            break;
        };
        after = Some(last_id);
        jobs.extend(batch.into_iter().map(|(id, text)| EmbedJob {
            target: to_target(id),
            text,
        }));
        let job_count = jobs.len();
        let embeddings = embed_jobs(embedder, jobs.drain(..)).await;
        if let Err(e) = semantic
            .store_embeddings(embedder.model(), embeddings)
            .await
        {
            warn!(
                target: "assistd::memory",
                kind = progress.kind.as_str(),
                error = %e,
                "reindex: store embeddings failed"
            );
        }
        progress.advance(job_count).await;
    }
    Ok(())
}

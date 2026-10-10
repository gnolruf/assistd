//! Handlers for the `Memory*` variants of `Request`.

use std::future::Future;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::warn;

use assistd_embed::{BATCH_SIZE, EmbedError, Embedder, embed_each};
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
        let embedder = match self.memory.embedder.get() {
            Ok(embedder) => embedder,
            Err(e) => {
                send_error(&tx, id, format!("reindex failed: {e}")).await;
                return Err(e.into());
            }
        };
        let model = embedder.model().to_string();
        let dim = i64::try_from(embedder.dim()).unwrap_or(i64::MAX);

        let (chunks_total, memories_total) = match self.memory.semantic.count_missing(&model).await
        {
            Ok(counts) => counts,
            Err(e) => {
                send_error(&tx, id, format!("reindex: count missing rows: {e}")).await;
                return Err(e.into());
            }
        };
        let chunks_total = u32::try_from(chunks_total).unwrap_or(u32::MAX);
        let memories_total = u32::try_from(memories_total).unwrap_or(u32::MAX);

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
        let embedder = embedder.as_ref();
        let chunks = reindex_items(
            embedder,
            KindProgress::new(&id, &tx, ReindexKind::Chunks, chunks_total),
            |after, limit| semantic.chunks_missing_embedding(&model, after, limit),
            |chunk_id, blob| semantic.store_chunk_embedding(chunk_id, model.clone(), dim, blob),
        )
        .await;
        if let Err(e) = chunks {
            send_error(&tx, id, format!("reindex: list missing chunks: {e}")).await;
            return Err(e.into());
        }
        let memories = reindex_items(
            embedder,
            KindProgress::new(&id, &tx, ReindexKind::Memories, memories_total),
            |after, limit| semantic.memories_missing_embedding(&model, after, limit),
            |memory_id, blob| semantic.store_memory_embedding(memory_id, model.clone(), dim, blob),
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

    async fn advance(&mut self) {
        self.done += 1;
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
/// id order. Per-item failures are logged; a listing failure aborts.
async fn reindex_items<L, LFut, S, SFut>(
    embedder: &dyn Embedder,
    mut progress: KindProgress<'_>,
    list: L,
    store: S,
) -> Result<(), MemoryError>
where
    L: Fn(i64, usize) -> LFut,
    LFut: Future<Output = Result<Vec<(i64, String)>, MemoryError>>,
    S: Fn(i64, Vec<u8>) -> SFut,
    SFut: Future<Output = Result<(), MemoryError>>,
{
    let mut after = 0;
    while progress.remaining() > 0 {
        let batch = list(after, progress.remaining().min(BATCH_SIZE)).await?;
        let Some(&(last_id, _)) = batch.last() else {
            break;
        };
        after = last_id;
        let texts: Vec<&str> = batch.iter().map(|(_, text)| text.as_str()).collect();
        let results = embed_each(embedder, &texts).await;
        for (&(item_id, _), result) in batch.iter().zip(results) {
            store_embedding(progress.kind, item_id, result, &store).await;
            progress.advance().await;
        }
    }
    Ok(())
}

async fn store_embedding<S, SFut>(
    kind: ReindexKind,
    item_id: i64,
    embedded: Result<Vec<f32>, EmbedError>,
    store: &S,
) where
    S: Fn(i64, Vec<u8>) -> SFut,
    SFut: Future<Output = Result<(), MemoryError>>,
{
    let embedding = match embedded {
        Ok(embedding) => embedding,
        Err(e) => {
            warn!(
                target: "assistd::memory",
                kind = kind.as_str(),
                item_id,
                error = %e,
                "reindex: embed failed"
            );
            return;
        }
    };
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

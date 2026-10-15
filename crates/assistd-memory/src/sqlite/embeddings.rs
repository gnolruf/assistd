//! Vector retrieval over `chunk_embeddings` and `memory_embeddings`.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::fmt;
use std::hash::Hash;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::Row;
use rusqlite::types::FromSql;

use super::connection::SqliteHandle;
use super::ids::{ChunkId, MemoryId, MessageId, SessionId};
use super::timestamp::timestamp_at;
use super::transcripts::PersistedRole;
use super::writer::{WriteOp, dispatch_write};
use crate::{MemoryError, Result};

/// One message-chunk hit. `content` is the full parent message,
/// not the chunk text, since chunks may cut mid-sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingHit {
    pub message_id: MessageId,
    pub chunk_id: ChunkId,
    pub session_id: SessionId,
    pub timestamp: DateTime<Utc>,
    pub role: PersistedRole,
    pub content: String,
    pub similarity: f32,
}

/// One saved-memory hit, ranked by cosine similarity to the query.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryHit {
    pub memory_id: MemoryId,
    pub key: String,
    pub value: String,
    pub similarity: f32,
}

/// The row an embedding belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTarget {
    Chunk(ChunkId),
    Memory(MemoryId),
}

impl fmt::Display for EmbeddingTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chunk(id) => write!(f, "chunk {id}"),
            Self::Memory(id) => write!(f, "memory {id}"),
        }
    }
}

/// An L2-normalised embedding of `target`'s text.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub target: EmbeddingTarget,
    pub vector: Vec<f32>,
}

/// Rows with no embedding under a given model, by table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnindexedRows {
    pub chunks: u64,
    pub memories: u64,
}

/// Rows embedded under models other than the current one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StaleEmbeddings {
    pub rows: u64,
    /// The other models' names, sorted.
    pub models: Vec<String>,
}

/// Top-K vector retrieval over embedded chunks and memories.
#[async_trait]
pub trait SemanticStore: fmt::Debug + Send + Sync + 'static {
    /// Top-K message chunks by cosine to the L2-normalised `query_vector`.
    /// `exclude_session`'s chunks are dropped before ranking, so never count toward `top_k`.
    async fn nearest_chunks(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        model: &str,
        exclude_session: Option<&SessionId>,
    ) -> Result<Vec<EmbeddingHit>>;

    /// Top-K saved memories by cosine to the L2-normalised `query_vector`.
    async fn nearest_memories(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        model: &str,
    ) -> Result<Vec<MemoryHit>>;

    /// Rows embedded under a model other than `current`.
    async fn count_stale(&self, current: &str) -> Result<StaleEmbeddings>;

    /// Rows with no embedding under `current`.
    async fn count_missing(&self, current: &str) -> Result<UnindexedRows>;

    /// Up to `limit` memories after `after` (from the start when `None`) with no
    /// embedding under `current`, in id order, as `(id, value)`.
    async fn memories_missing_embedding(
        &self,
        current: &str,
        after: Option<MemoryId>,
        limit: usize,
    ) -> Result<Vec<(MemoryId, String)>>;

    /// Up to `limit` message chunks after `after` (from the start when `None`) with
    /// no embedding under `current`, in id order, as `(id, chunk text)`.
    async fn chunks_missing_embedding(
        &self,
        current: &str,
        after: Option<ChunkId>,
        limit: usize,
    ) -> Result<Vec<(ChunkId, String)>>;

    /// Upsert `embeddings` under `model` in one transaction, skipping targets
    /// deleted since they were embedded.
    async fn store_embeddings(&self, model: &str, embeddings: Vec<Embedding>) -> Result<()>;
}

/// No-op store: writes are discarded and searches find nothing.
#[derive(Debug)]
pub struct NoSemanticStore;

#[async_trait]
impl SemanticStore for NoSemanticStore {
    async fn nearest_chunks(
        &self,
        _query_vector: Vec<f32>,
        _top_k: usize,
        _model: &str,
        _exclude_session: Option<&SessionId>,
    ) -> Result<Vec<EmbeddingHit>> {
        Ok(Vec::new())
    }
    async fn nearest_memories(
        &self,
        _query_vector: Vec<f32>,
        _top_k: usize,
        _model: &str,
    ) -> Result<Vec<MemoryHit>> {
        Ok(Vec::new())
    }
    async fn count_stale(&self, _current: &str) -> Result<StaleEmbeddings> {
        Ok(StaleEmbeddings::default())
    }
    async fn count_missing(&self, _current: &str) -> Result<UnindexedRows> {
        Ok(UnindexedRows::default())
    }
    async fn memories_missing_embedding(
        &self,
        _current: &str,
        _after: Option<MemoryId>,
        _limit: usize,
    ) -> Result<Vec<(MemoryId, String)>> {
        Ok(Vec::new())
    }
    async fn chunks_missing_embedding(
        &self,
        _current: &str,
        _after: Option<ChunkId>,
        _limit: usize,
    ) -> Result<Vec<(ChunkId, String)>> {
        Ok(Vec::new())
    }
    async fn store_embeddings(&self, _model: &str, _embeddings: Vec<Embedding>) -> Result<()> {
        Ok(())
    }
}

/// SQLite-backed [`SemanticStore`]: a linear scan into a bounded min-heap, then one
/// batched lookup of the winners.
#[derive(Debug, Clone)]
pub struct SqliteSemanticStore {
    handle: SqliteHandle,
}

impl SqliteSemanticStore {
    /// Store over the shared database handle.
    pub fn new(handle: SqliteHandle) -> Self {
        Self { handle }
    }

    /// Load the parent messages of `ranked` chunks, keeping rank order.
    async fn hydrate_chunk_hits(&self, ranked: Vec<(ChunkId, f32)>) -> Result<Vec<EmbeddingHit>> {
        let sql = format!(
            "SELECT cc.id, m.id, s.uuid, m.created_at, m.role, m.content
             FROM message_chunks cc
             JOIN messages m ON m.id = cc.message_id
             JOIN sessions s ON s.id = m.session_id
             WHERE cc.id IN ({})",
            placeholders(ranked.len())
        );
        let chunk_ids: Vec<ChunkId> = ranked.iter().map(|(id, _)| *id).collect();
        let hits = self
            .handle
            .reader()
            .call(move |c| query_by_id(c, &sql, chunk_ids, chunk_hit))
            .await
            .map_err(MemoryError::sqlite("nearest_chunks: hydrate winners"))?;
        Ok(in_rank_order(ranked, hits, |hit, similarity| {
            hit.similarity = similarity;
        }))
    }

    /// Load the `ranked` memories, keeping rank order.
    async fn hydrate_memory_hits(&self, ranked: Vec<(MemoryId, f32)>) -> Result<Vec<MemoryHit>> {
        let sql = format!(
            "SELECT id, key, value FROM memories WHERE id IN ({})",
            placeholders(ranked.len())
        );
        let memory_ids: Vec<MemoryId> = ranked.iter().map(|(id, _)| *id).collect();
        let hits = self
            .handle
            .reader()
            .call(move |c| query_by_id(c, &sql, memory_ids, memory_hit))
            .await
            .map_err(MemoryError::sqlite("nearest_memories: hydrate winners"))?;
        Ok(in_rank_order(ranked, hits, |hit, similarity| {
            hit.similarity = similarity;
        }))
    }
}

#[async_trait]
impl SemanticStore for SqliteSemanticStore {
    async fn nearest_chunks(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        model: &str,
        exclude_session: Option<&SessionId>,
    ) -> Result<Vec<EmbeddingHit>> {
        if top_k == 0 || query_vector.is_empty() {
            return Ok(Vec::new());
        }
        let model = model.to_string();
        let excluded = exclude_session.copied();
        let ranked = self
            .handle
            .reader()
            .call(move |c| {
                scan_top_k(
                    c,
                    "SELECT e.chunk_id, e.vector
                     FROM chunk_embeddings e
                     JOIN embedding_models em ON em.id = e.model_id
                     JOIN message_chunks cc ON cc.id = e.chunk_id
                     JOIN messages m ON m.id = cc.message_id
                     WHERE em.name = ?1
                       AND m.session_id IS NOT (SELECT id FROM sessions WHERE uuid = ?2)",
                    rusqlite::params![model, excluded],
                    &query_vector,
                    top_k,
                )
            })
            .await
            .map_err(MemoryError::sqlite("nearest_chunks: scan embeddings"))?;
        if ranked.is_empty() {
            return Ok(Vec::new());
        }
        self.hydrate_chunk_hits(ranked).await
    }

    async fn nearest_memories(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        model: &str,
    ) -> Result<Vec<MemoryHit>> {
        if top_k == 0 || query_vector.is_empty() {
            return Ok(Vec::new());
        }
        let model = model.to_string();
        let ranked = self
            .handle
            .reader()
            .call(move |c| {
                scan_top_k(
                    c,
                    "SELECT e.memory_id, e.vector
                     FROM memory_embeddings e JOIN embedding_models m ON m.id = e.model_id
                     WHERE m.name = ?1",
                    rusqlite::params![model],
                    &query_vector,
                    top_k,
                )
            })
            .await
            .map_err(MemoryError::sqlite(
                "nearest_memories: scan memory_embeddings",
            ))?;
        if ranked.is_empty() {
            return Ok(Vec::new());
        }
        self.hydrate_memory_hits(ranked).await
    }

    async fn count_stale(&self, current: &str) -> Result<StaleEmbeddings> {
        let current = current.to_string();
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let mut rows: u64 = 0;
                let mut models = BTreeSet::new();
                let sql = "SELECT m.name, count(*)
                           FROM chunk_embeddings e JOIN embedding_models m ON m.id = e.model_id
                           WHERE m.name != ?1 GROUP BY m.name
                           UNION ALL
                           SELECT m.name, count(*)
                           FROM memory_embeddings e JOIN embedding_models m ON m.id = e.model_id
                           WHERE m.name != ?1 GROUP BY m.name";
                let mut stmt = c.prepare(sql)?;
                let model_counts = stmt.query_map(rusqlite::params![current], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?))
                })?;
                for model_count in model_counts {
                    let (model, count) = model_count?;
                    rows += count;
                    models.insert(model);
                }
                Ok(StaleEmbeddings {
                    rows,
                    models: models.into_iter().collect(),
                })
            })
            .await
            .map_err(MemoryError::sqlite("count_stale"))
    }

    async fn count_missing(&self, current: &str) -> Result<UnindexedRows> {
        let current = current.to_string();
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let chunks = c.query_row(
                    "SELECT count(*)
                     FROM message_chunks cc
                     WHERE NOT EXISTS (
                         SELECT 1 FROM chunk_embeddings e
                         JOIN embedding_models m ON m.id = e.model_id
                         WHERE e.chunk_id = cc.id AND m.name = ?1
                     )",
                    rusqlite::params![current],
                    |r| r.get(0),
                )?;
                let memories = c.query_row(
                    "SELECT count(*)
                     FROM memories mem
                     WHERE NOT EXISTS (
                         SELECT 1 FROM memory_embeddings e
                         JOIN embedding_models m ON m.id = e.model_id
                         WHERE e.memory_id = mem.id AND m.name = ?1
                     )",
                    rusqlite::params![current],
                    |r| r.get(0),
                )?;
                Ok(UnindexedRows { chunks, memories })
            })
            .await
            .map_err(MemoryError::sqlite("count_missing"))
    }

    async fn memories_missing_embedding(
        &self,
        current: &str,
        after: Option<MemoryId>,
        limit: usize,
    ) -> Result<Vec<(MemoryId, String)>> {
        let current = current.to_string();
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(
                    "SELECT mem.id, mem.value
                     FROM memories mem
                     WHERE mem.id > ifnull(?2, 0) AND NOT EXISTS (
                         SELECT 1 FROM memory_embeddings e
                         JOIN embedding_models m ON m.id = e.model_id
                         WHERE e.memory_id = mem.id AND m.name = ?1
                     )
                     ORDER BY mem.id
                     LIMIT ?3",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![current, after, limit], |r| {
                        Ok((r.get(0)?, r.get(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("memories_missing_embedding"))
    }

    async fn chunks_missing_embedding(
        &self,
        current: &str,
        after: Option<ChunkId>,
        limit: usize,
    ) -> Result<Vec<(ChunkId, String)>> {
        let current = current.to_string();
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(
                    "SELECT cc.id,
                            CAST(substr(CAST(msg.content AS BLOB), cc.byte_start + 1,
                                        cc.byte_end - cc.byte_start) AS TEXT)
                     FROM message_chunks cc
                     JOIN messages msg ON msg.id = cc.message_id
                     WHERE cc.id > ifnull(?2, 0) AND NOT EXISTS (
                         SELECT 1 FROM chunk_embeddings e
                         JOIN embedding_models m ON m.id = e.model_id
                         WHERE e.chunk_id = cc.id AND m.name = ?1
                     )
                     ORDER BY cc.id
                     LIMIT ?3",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![current, after, limit], |r| {
                        Ok((r.get(0)?, r.get(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("chunks_missing_embedding"))
    }

    async fn store_embeddings(&self, model: &str, embeddings: Vec<Embedding>) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }
        let model = model.to_string();
        dispatch_write(self.handle.writer(), |ack| WriteOp::StoreEmbeddings {
            model,
            embeddings,
            ack,
        })
        .await
    }
}

/// Ordered in reverse by similarity so `BinaryHeap` acts as a min-heap. Only finite
/// similarities are ever pushed, so `total_cmp` agrees with `>` on every entry.
struct HeapEntry<Id> {
    similarity: f32,
    id: Id,
}

impl<Id> PartialEq for HeapEntry<Id> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<Id> Eq for HeapEntry<Id> {}
impl<Id> PartialOrd for HeapEntry<Id> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<Id> Ord for HeapEntry<Id> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.similarity.total_cmp(&self.similarity)
    }
}

fn scan_top_k<Id: FromSql>(
    conn: &rusqlite::Connection,
    sql: &str,
    params: impl rusqlite::Params,
    query: &[f32],
    top_k: usize,
) -> rusqlite::Result<Vec<(Id, f32)>> {
    let mut stmt = conn.prepare_cached(sql)?;
    let mut heap = BinaryHeap::new();
    let mut rows = stmt.query(params)?;
    while let Some(row) = rows.next()? {
        let bytes = row.get_ref(1)?.as_blob()?;
        if let Some(similarity) = score_against(query, bytes) {
            push_top_k(&mut heap, top_k, row.get(0)?, similarity);
        }
    }
    Ok(heap_to_sorted(heap))
}

/// Rows of `sql` (whose `IN` list binds `ids`) keyed by the id in column 0.
fn query_by_id<Id, T>(
    conn: &rusqlite::Connection,
    sql: &str,
    ids: Vec<Id>,
    to_hit: fn(&Row<'_>) -> rusqlite::Result<T>,
) -> rusqlite::Result<HashMap<Id, T>>
where
    Id: FromSql + rusqlite::ToSql + Eq + Hash,
{
    let mut stmt = conn.prepare(sql)?;
    stmt.query_map(rusqlite::params_from_iter(ids), |row| {
        Ok((row.get(0)?, to_hit(row)?))
    })?
    .collect()
}

/// `hits` reordered to follow `ranked`, each given its score by `set_similarity`;
/// ids with no hit (deleted since the scan) are skipped.
fn in_rank_order<Id: Eq + Hash, T>(
    ranked: Vec<(Id, f32)>,
    mut hits: HashMap<Id, T>,
    set_similarity: fn(&mut T, f32),
) -> Vec<T> {
    ranked
        .into_iter()
        .filter_map(|(id, similarity)| {
            let mut hit = hits.remove(&id)?;
            set_similarity(&mut hit, similarity);
            Some(hit)
        })
        .collect()
}

fn chunk_hit(row: &Row<'_>) -> rusqlite::Result<EmbeddingHit> {
    Ok(EmbeddingHit {
        chunk_id: row.get(0)?,
        message_id: row.get(1)?,
        session_id: row.get(2)?,
        timestamp: timestamp_at(row, 3)?,
        role: row.get(4)?,
        content: row.get(5)?,
        similarity: 0.0,
    })
}

fn memory_hit(row: &Row<'_>) -> rusqlite::Result<MemoryHit> {
    Ok(MemoryHit {
        memory_id: row.get(0)?,
        key: row.get(1)?,
        value: row.get(2)?,
        similarity: 0.0,
    })
}

fn placeholders(count: usize) -> String {
    vec!["?"; count].join(",")
}

fn push_top_k<Id>(heap: &mut BinaryHeap<HeapEntry<Id>>, top_k: usize, id: Id, similarity: f32) {
    if heap.len() < top_k {
        heap.push(HeapEntry { similarity, id });
    } else if let Some(weakest) = heap.peek()
        && similarity > weakest.similarity
    {
        heap.pop();
        heap.push(HeapEntry { similarity, id });
    }
}

fn heap_to_sorted<Id>(heap: BinaryHeap<HeapEntry<Id>>) -> Vec<(Id, f32)> {
    let mut ranked: Vec<(Id, f32)> = heap
        .into_iter()
        .map(|entry| (entry.id, entry.similarity))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked
}

/// Dot product of `query` against the LE-packed f32 BLOB `bytes`, or `None` when
/// the BLOB is malformed, its dimension differs, or the product is not finite.
fn score_against(query: &[f32], bytes: &[u8]) -> Option<f32> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let dim = bytes.len() / 4;
    if dim != query.len() {
        return None;
    }
    let (words, _) = bytes.as_chunks::<4>();
    let similarity: f32 = words
        .iter()
        .zip(query)
        .map(|(word, q)| f32::from_le_bytes(*word) * q)
        .sum();
    similarity.is_finite().then_some(similarity)
}

/// Encode `vector` as the little-endian `f32` BLOB the embedding tables store.
pub(super) fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

#[cfg(test)]
mod tests;

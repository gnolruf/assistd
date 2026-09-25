//! Vector retrieval over `embeddings` (chunk-keyed) and `memory_embeddings`.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::connection::SqliteHandle;
use super::conversations::{PersistedRole, SessionId};
use super::writer::{WriteOp, dispatch_write};
use crate::{MemoryError, Result};

/// One conversation-chunk hit. `content` is the full parent message,
/// not the chunk text, since chunks may cut mid-sentence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingHit {
    pub conversation_id: i64,
    pub chunk_id: i64,
    pub session_id: String,
    pub timestamp: String,
    pub role: PersistedRole,
    pub content: String,
    pub similarity: f32,
}

/// One saved-memory hit, ranked by cosine similarity to the query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryHit {
    pub memory_id: i64,
    pub key: String,
    pub value: String,
    pub similarity: f32,
}

/// Top-K vector retrieval over embedded chunks and memories.
#[async_trait]
pub trait SemanticStore: Send + Sync + 'static {
    /// Top-K conversation chunks by cosine to the L2-normalised `query_vector`.
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

    /// Embedding row counts under `model`, as `(chunks, memories)`.
    async fn count_for_model(&self, model: &str) -> Result<(i64, i64)>;

    /// Count of rows embedded under a model other than `current`, plus those model names sorted.
    async fn count_stale(&self, current: &str) -> Result<(i64, Vec<String>)>;

    /// Memories with no embedding under `current`, as `(id, value)`.
    async fn memories_missing_embedding(&self, current: &str) -> Result<Vec<(i64, String)>>;

    /// Conversation chunks with no embedding under `current`, as `(id, content)`.
    async fn chunks_missing_embedding(&self, current: &str) -> Result<Vec<(i64, String)>>;

    /// Upsert the embedding for a `conversation_chunks` row.
    async fn store_chunk_embedding(
        &self,
        chunk_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
    ) -> Result<()>;

    /// Upsert the embedding for a `memories` row.
    async fn store_memory_embedding(
        &self,
        memory_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
    ) -> Result<()>;
}

/// No-op store: writes are discarded and searches find nothing.
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
    async fn count_for_model(&self, _model: &str) -> Result<(i64, i64)> {
        Ok((0, 0))
    }
    async fn count_stale(&self, _current: &str) -> Result<(i64, Vec<String>)> {
        Ok((0, Vec::new()))
    }
    async fn memories_missing_embedding(&self, _current: &str) -> Result<Vec<(i64, String)>> {
        Ok(Vec::new())
    }
    async fn chunks_missing_embedding(&self, _current: &str) -> Result<Vec<(i64, String)>> {
        Ok(Vec::new())
    }
    async fn store_chunk_embedding(
        &self,
        _chunk_id: i64,
        _model: String,
        _dim: i64,
        _vector: Vec<u8>,
    ) -> Result<()> {
        Ok(())
    }
    async fn store_memory_embedding(
        &self,
        _memory_id: i64,
        _model: String,
        _dim: i64,
        _vector: Vec<u8>,
    ) -> Result<()> {
        Ok(())
    }
}

/// SQLite-backed [`SemanticStore`]: a linear scan into a bounded min-heap, then one
/// batched lookup of the winners.
#[derive(Clone)]
pub struct SqliteSemanticStore {
    handle: Arc<SqliteHandle>,
}

impl SqliteSemanticStore {
    /// Store over the shared database handle.
    pub fn new(handle: Arc<SqliteHandle>) -> Self {
        Self { handle }
    }

    /// Load the parent messages of `ranked` chunks, keeping rank order.
    async fn hydrate_chunk_hits(&self, ranked: Vec<(i64, f32)>) -> Result<Vec<EmbeddingHit>> {
        let chunk_ids: Vec<i64> = ranked.iter().map(|(id, _)| *id).collect();
        let similarity_by_id: HashMap<i64, f32> = ranked.into_iter().collect();
        let sql = format!(
            "SELECT cc.id, c.id, c.session_id, c.timestamp, c.role, c.content
             FROM conversation_chunks cc
             JOIN conversations c ON c.id = cc.conversation_id
             WHERE cc.id IN ({})",
            placeholders(chunk_ids.len())
        );
        let chunk_ids_for_query = chunk_ids.clone();
        let rows: Vec<(i64, i64, String, String, String, String)> = self
            .handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(&sql)?;
                let rows = stmt
                    .query_map(rusqlite::params_from_iter(chunk_ids_for_query), |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                            row.get::<_, String>(5)?,
                        ))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("nearest_chunks: hydrate winners"))?;
        let by_id: HashMap<i64, (i64, String, String, String, String)> = rows
            .into_iter()
            .map(
                |(chunk_id, conversation_id, session_id, timestamp, role, content)| {
                    (
                        chunk_id,
                        (conversation_id, session_id, timestamp, role, content),
                    )
                },
            )
            .collect();
        let mut hits = Vec::with_capacity(chunk_ids.len());
        for chunk_id in chunk_ids {
            let Some((conversation_id, session_id, timestamp, role_str, content)) =
                by_id.get(&chunk_id)
            else {
                continue;
            };
            let role = PersistedRole::parse(role_str)
                .ok_or_else(|| MemoryError::UnknownRole(role_str.clone()))?;
            hits.push(EmbeddingHit {
                conversation_id: *conversation_id,
                chunk_id,
                session_id: session_id.clone(),
                timestamp: timestamp.clone(),
                role,
                content: content.clone(),
                similarity: *similarity_by_id.get(&chunk_id).unwrap_or(&0.0),
            });
        }
        Ok(hits)
    }

    /// Load the `ranked` memories, keeping rank order.
    async fn hydrate_memory_hits(&self, ranked: Vec<(i64, f32)>) -> Result<Vec<MemoryHit>> {
        let memory_ids: Vec<i64> = ranked.iter().map(|(id, _)| *id).collect();
        let similarity_by_id: HashMap<i64, f32> = ranked.into_iter().collect();
        let sql = format!(
            "SELECT id, key, value FROM memories WHERE id IN ({})",
            placeholders(memory_ids.len())
        );
        let memory_ids_for_query = memory_ids.clone();
        let rows: Vec<(i64, String, String)> = self
            .handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(&sql)?;
                let rows = stmt
                    .query_map(rusqlite::params_from_iter(memory_ids_for_query), |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("nearest_memories: hydrate winners"))?;
        let by_id: HashMap<i64, (String, String)> = rows
            .into_iter()
            .map(|(id, key, value)| (id, (key, value)))
            .collect();
        let mut hits = Vec::with_capacity(memory_ids.len());
        for id in memory_ids {
            let Some((key, value)) = by_id.get(&id) else {
                continue;
            };
            hits.push(MemoryHit {
                memory_id: id,
                key: key.clone(),
                value: value.clone(),
                similarity: *similarity_by_id.get(&id).unwrap_or(&0.0),
            });
        }
        Ok(hits)
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
        let excluded = exclude_session.map(|s| s.0.clone());
        let ranked = self
            .handle
            .conn()
            .call(move |c| {
                scan_top_k(
                    c,
                    "SELECT e.conversation_chunk_id, e.vector
                     FROM embeddings e
                     JOIN conversation_chunks cc ON cc.id = e.conversation_chunk_id
                     JOIN conversations conv ON conv.id = cc.conversation_id
                     WHERE e.model = ?1 AND (?2 IS NULL OR conv.session_id <> ?2)",
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
            .conn()
            .call(move |c| {
                scan_top_k(
                    c,
                    "SELECT memory_id, vector FROM memory_embeddings WHERE model = ?1",
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

    async fn count_for_model(&self, model: &str) -> Result<(i64, i64)> {
        let model = model.to_string();
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let chunks: i64 = c.query_row(
                    "SELECT count(*) FROM embeddings WHERE model = ?1",
                    rusqlite::params![model],
                    |r| r.get(0),
                )?;
                let memories: i64 = c.query_row(
                    "SELECT count(*) FROM memory_embeddings WHERE model = ?1",
                    rusqlite::params![model],
                    |r| r.get(0),
                )?;
                Ok((chunks, memories))
            })
            .await
            .map_err(MemoryError::sqlite("count_for_model"))
    }

    async fn count_stale(&self, current: &str) -> Result<(i64, Vec<String>)> {
        let current = current.to_string();
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut total: i64 = 0;
                let mut models = BTreeSet::new();
                let sql = "SELECT model, count(*)
                           FROM embeddings WHERE model != ?1 GROUP BY model
                           UNION ALL
                           SELECT model, count(*)
                           FROM memory_embeddings WHERE model != ?1 GROUP BY model";
                let mut stmt = c.prepare(sql)?;
                let rows = stmt.query_map(rusqlite::params![current], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })?;
                for row in rows {
                    let (model, count) = row?;
                    total += count;
                    models.insert(model);
                }
                Ok((total, models.into_iter().collect::<Vec<_>>()))
            })
            .await
            .map_err(MemoryError::sqlite("count_stale"))
    }

    async fn memories_missing_embedding(&self, current: &str) -> Result<Vec<(i64, String)>> {
        let current = current.to_string();
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(
                    "SELECT m.id, m.value
                     FROM memories m
                     LEFT JOIN memory_embeddings e
                       ON e.memory_id = m.id AND e.model = ?1
                     WHERE e.id IS NULL
                     ORDER BY m.id",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![current], |r| {
                        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("memories_missing_embedding"))
    }

    async fn chunks_missing_embedding(&self, current: &str) -> Result<Vec<(i64, String)>> {
        let current = current.to_string();
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(
                    "SELECT cc.id, cc.content
                     FROM conversation_chunks cc
                     LEFT JOIN embeddings e
                       ON e.conversation_chunk_id = cc.id AND e.model = ?1
                     WHERE e.id IS NULL
                     ORDER BY cc.id",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![current], |r| {
                        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("chunks_missing_embedding"))
    }

    async fn store_chunk_embedding(
        &self,
        chunk_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
    ) -> Result<()> {
        dispatch_write(self.handle.writer(), |ack| WriteOp::StoreChunkEmbedding {
            chunk_id,
            model,
            dim,
            vector,
            ack,
        })
        .await
    }

    async fn store_memory_embedding(
        &self,
        memory_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
    ) -> Result<()> {
        dispatch_write(self.handle.writer(), |ack| WriteOp::StoreMemoryEmbedding {
            memory_id,
            model,
            dim,
            vector,
            ack,
        })
        .await
    }
}

/// Ordered in reverse by similarity so `BinaryHeap` acts as a min-heap; NaN compares
/// equal to everything.
struct HeapEntry {
    similarity: f32,
    rowid: i64,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

fn scan_top_k(
    conn: &rusqlite::Connection,
    sql: &str,
    params: impl rusqlite::Params,
    query: &[f32],
    top_k: usize,
) -> rusqlite::Result<Vec<(i64, f32)>> {
    let mut stmt = conn.prepare(sql)?;
    let mut heap = BinaryHeap::new();
    let mut rows = stmt.query(params)?;
    while let Some(row) = rows.next()? {
        let rowid: i64 = row.get(0)?;
        let bytes: Vec<u8> = row.get(1)?;
        if let Some(similarity) = score_against(query, &bytes) {
            push_top_k(&mut heap, top_k, rowid, similarity);
        }
    }
    Ok(heap_to_sorted(heap))
}

fn placeholders(count: usize) -> String {
    vec!["?"; count].join(",")
}

fn push_top_k(heap: &mut BinaryHeap<HeapEntry>, top_k: usize, rowid: i64, similarity: f32) {
    if heap.len() < top_k {
        heap.push(HeapEntry { similarity, rowid });
    } else if let Some(weakest) = heap.peek()
        && similarity > weakest.similarity
    {
        heap.pop();
        heap.push(HeapEntry { similarity, rowid });
    }
}

fn heap_to_sorted(heap: BinaryHeap<HeapEntry>) -> Vec<(i64, f32)> {
    let mut ranked: Vec<(i64, f32)> = heap
        .into_iter()
        .map(|entry| (entry.rowid, entry.similarity))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    ranked
}

/// Dot product of `query` against the LE-packed f32 BLOB `bytes`, or
/// `None` when the BLOB is malformed or its dimension differs.
fn score_against(query: &[f32], bytes: &[u8]) -> Option<f32> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let dim = bytes.len() / 4;
    if dim != query.len() {
        return None;
    }
    let (words, _) = bytes.as_chunks::<4>();
    Some(
        words
            .iter()
            .zip(query)
            .map(|(word, q)| f32::from_le_bytes(*word) * q)
            .sum(),
    )
}

/// Encode `vector` as the little-endian `f32` BLOB the embedding tables store. It must
/// be L2-normalised, since similarity is scored as a plain dot product.
pub fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

#[cfg(test)]
mod tests;

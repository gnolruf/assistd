//! Vector retrieval over `embeddings` (chunk-keyed) and
//! `memory_embeddings` (memory-keyed).
//!
//! Vectors are little-endian `f32` BLOBs, L2-normalised at write time,
//! so cosine similarity is a plain dot product. Retrieval is a linear
//! scan into a bounded min-heap, then one batched JOIN to hydrate the
//! winners; that is ample for a single user's history.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::connection::SqliteHandle;
use super::conversations::{PersistedRole, SessionId};

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
    /// Top-K conversation chunks by cosine. `query_vector` must already
    /// be L2-normalised. `exclude_session` drops one session's chunks
    /// before ranking so the caller still gets `top_k` hits from other
    /// conversations.
    async fn nearest_chunks(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        model: &str,
        exclude_session: Option<&SessionId>,
    ) -> Result<Vec<EmbeddingHit>>;

    /// Top-K saved memories by cosine. `query_vector` must already be
    /// L2-normalised. Empty when nothing is embedded under `model`.
    async fn nearest_memories(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        model: &str,
    ) -> Result<Vec<MemoryHit>>;

    /// Embedding row counts under `model`, as `(chunks, memories)`.
    async fn count_for_model(&self, model: &str) -> Result<(i64, i64)>;

    /// Rows embedded under a model other than `current`: the total
    /// count plus the distinct stale model names, sorted.
    async fn count_stale(&self, current: &str) -> Result<(i64, Vec<String>)>;

    /// Memories with no embedding under `current`, as `(id, value)`.
    async fn memories_missing_embedding(&self, current: &str) -> Result<Vec<(i64, String)>>;

    /// Conversation chunks with no embedding under `current`, as
    /// `(id, content)`.
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

/// No-op fallback used when the embedding subsystem is disabled.
pub struct NoSemanticStore;

#[async_trait]
impl SemanticStore for NoSemanticStore {
    async fn nearest_chunks(
        &self,
        _q: Vec<f32>,
        _k: usize,
        _model: &str,
        _exclude_session: Option<&SessionId>,
    ) -> Result<Vec<EmbeddingHit>> {
        Ok(Vec::new())
    }
    async fn nearest_memories(
        &self,
        _q: Vec<f32>,
        _k: usize,
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

/// SQLite-backed [`SemanticStore`].
#[derive(Clone)]
pub struct SqliteSemanticStore {
    handle: Arc<SqliteHandle>,
}

impl SqliteSemanticStore {
    pub fn new(handle: Arc<SqliteHandle>) -> Self {
        Self { handle }
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
        let q = query_vector;
        let ranked: Vec<(i64, f32)> = self
            .handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                // Filter before ranking so the excluded session doesn't
                // eat into `top_k`.
                let mut stmt = c.prepare(
                    "SELECT e.conversation_chunk_id, e.vector
                     FROM embeddings e
                     JOIN conversation_chunks cc ON cc.id = e.conversation_chunk_id
                     JOIN conversations conv ON conv.id = cc.conversation_id
                     WHERE e.model = ?1 AND (?2 IS NULL OR conv.session_id <> ?2)",
                )?;
                let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(top_k + 1);
                let mut rows = stmt.query(rusqlite::params![model, excluded])?;
                while let Some(row) = rows.next()? {
                    let chunk_id: i64 = row.get(0)?;
                    let bytes: Vec<u8> = row.get(1)?;
                    let Some(sim) = score_against(&q, &bytes) else {
                        continue;
                    };
                    push_top_k(&mut heap, top_k, chunk_id, sim);
                }
                Ok(heap_to_sorted(heap))
            })
            .await
            .context("nearest_chunks: scan embeddings")?;
        if ranked.is_empty() {
            return Ok(Vec::new());
        }
        let chunk_ids: Vec<i64> = ranked.iter().map(|(id, _)| *id).collect();
        let sims: std::collections::HashMap<i64, f32> = ranked.iter().copied().collect();
        let placeholders = vec!["?"; chunk_ids.len()].join(",");
        let sql = format!(
            "SELECT cc.id, c.id, c.session_id, c.timestamp, c.role, c.content
             FROM conversation_chunks cc
             JOIN conversations c ON c.id = cc.conversation_id
             WHERE cc.id IN ({placeholders})"
        );
        let chunk_ids_for_query = chunk_ids.clone();
        let raw: Vec<(i64, i64, String, String, String, String)> = self
            .handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(&sql)?;
                let params: Vec<&dyn rusqlite::ToSql> = chunk_ids_for_query
                    .iter()
                    .map(|id| id as &dyn rusqlite::ToSql)
                    .collect();
                let rows = stmt
                    .query_map(rusqlite::params_from_iter(params), |row| {
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
            .context("nearest_chunks: hydrate winners")?;
        let by_id: std::collections::HashMap<i64, (i64, String, String, String, String)> = raw
            .into_iter()
            .map(|(cc_id, c_id, sess, ts, role, content)| (cc_id, (c_id, sess, ts, role, content)))
            .collect();
        let mut out = Vec::with_capacity(chunk_ids.len());
        for cc_id in chunk_ids {
            let Some((c_id, sess, ts, role_str, content)) = by_id.get(&cc_id) else {
                continue;
            };
            let role = PersistedRole::parse(role_str)
                .with_context(|| format!("unknown role in DB: {role_str}"))?;
            out.push(EmbeddingHit {
                conversation_id: *c_id,
                chunk_id: cc_id,
                session_id: sess.clone(),
                timestamp: ts.clone(),
                role,
                content: content.clone(),
                similarity: *sims.get(&cc_id).unwrap_or(&0.0),
            });
        }
        Ok(out)
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
        let q = query_vector;
        let ranked: Vec<(i64, f32)> = self
            .handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt =
                    c.prepare("SELECT memory_id, vector FROM memory_embeddings WHERE model = ?1")?;
                let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(top_k + 1);
                let mut rows = stmt.query(rusqlite::params![model])?;
                while let Some(row) = rows.next()? {
                    let memory_id: i64 = row.get(0)?;
                    let bytes: Vec<u8> = row.get(1)?;
                    let Some(sim) = score_against(&q, &bytes) else {
                        continue;
                    };
                    push_top_k(&mut heap, top_k, memory_id, sim);
                }
                Ok(heap_to_sorted(heap))
            })
            .await
            .context("nearest_memories: scan memory_embeddings")?;
        if ranked.is_empty() {
            return Ok(Vec::new());
        }
        let memory_ids: Vec<i64> = ranked.iter().map(|(id, _)| *id).collect();
        let sims: std::collections::HashMap<i64, f32> = ranked.iter().copied().collect();
        let placeholders = vec!["?"; memory_ids.len()].join(",");
        let sql = format!("SELECT id, key, value FROM memories WHERE id IN ({placeholders})");
        let memory_ids_for_query = memory_ids.clone();
        let raw: Vec<(i64, String, String)> = self
            .handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(&sql)?;
                let params: Vec<&dyn rusqlite::ToSql> = memory_ids_for_query
                    .iter()
                    .map(|id| id as &dyn rusqlite::ToSql)
                    .collect();
                let rows = stmt
                    .query_map(rusqlite::params_from_iter(params), |row| {
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
            .context("nearest_memories: hydrate winners")?;
        let by_id: std::collections::HashMap<i64, (String, String)> = raw
            .into_iter()
            .map(|(id, key, value)| (id, (key, value)))
            .collect();
        let mut out = Vec::with_capacity(memory_ids.len());
        for id in memory_ids {
            let Some((key, value)) = by_id.get(&id) else {
                continue;
            };
            out.push(MemoryHit {
                memory_id: id,
                key: key.clone(),
                value: value.clone(),
                similarity: *sims.get(&id).unwrap_or(&0.0),
            });
        }
        Ok(out)
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
            .context("count_for_model")
    }

    async fn count_stale(&self, current: &str) -> Result<(i64, Vec<String>)> {
        let current = current.to_string();
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut total: i64 = 0;
                let mut models: std::collections::BTreeSet<String> =
                    std::collections::BTreeSet::new();
                let sql = "SELECT model, count(*)
                           FROM embeddings WHERE model != ?1 GROUP BY model
                           UNION ALL
                           SELECT model, count(*)
                           FROM memory_embeddings WHERE model != ?1 GROUP BY model";
                let mut stmt = c.prepare(sql)?;
                let rows = stmt.query_map(rusqlite::params![current], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })?;
                for r in rows {
                    let (model, n) = r?;
                    total += n;
                    models.insert(model);
                }
                Ok((total, models.into_iter().collect::<Vec<_>>()))
            })
            .await
            .context("count_stale")
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
            .context("memories_missing_embedding")
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
            .context("chunks_missing_embedding")
    }

    async fn store_chunk_embedding(
        &self,
        chunk_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
    ) -> Result<()> {
        use super::writer::{WriteCall, WriteOp};
        WriteCall::run(self.handle.writer(), |ack| WriteOp::StoreChunkEmbedding {
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
        use super::writer::{WriteCall, WriteOp};
        WriteCall::run(self.handle.writer(), |ack| WriteOp::StoreMemoryEmbedding {
            memory_id,
            model,
            dim,
            vector,
            ack,
        })
        .await
    }
}

struct HeapEntry {
    sim: f32,
    rowid: i64,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.sim == other.sim
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
        // Reversed so the max-heap acts as a min-heap; NaN sorts
        // smallest so it is evicted first.
        match other.sim.partial_cmp(&self.sim) {
            Some(o) => o,
            None => Ordering::Equal,
        }
    }
}

fn push_top_k(heap: &mut BinaryHeap<HeapEntry>, k: usize, rowid: i64, sim: f32) {
    if heap.len() < k {
        heap.push(HeapEntry { sim, rowid });
    } else if let Some(top) = heap.peek()
        && sim > top.sim
    {
        heap.pop();
        heap.push(HeapEntry { sim, rowid });
    }
}

fn heap_to_sorted(heap: BinaryHeap<HeapEntry>) -> Vec<(i64, f32)> {
    let mut v: Vec<(i64, f32)> = heap.into_iter().map(|e| (e.rowid, e.sim)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    v
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
            .map(|(w, q)| f32::from_le_bytes(*w) * q)
            .sum(),
    )
}

/// Encode an `f32` slice as the little-endian BLOB the embedding tables
/// store.
pub fn vector_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests;

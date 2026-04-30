//! Vector retrieval over `embeddings` (chunk-keyed) and
//! `memory_embeddings` (memory-keyed).
//!
//! Both vector collections are stored as little-endian `f32` BLOBs and
//! L2-normalised at write time, so cosine similarity collapses to a
//! plain dot product. At query time we:
//!
//! 1. Read every `(rowid, vector)` for the configured model. Reads
//!    bypass the writer task via `conn.call(...)` directly; SQLite WAL
//!    mode supports concurrent readers alongside a single writer.
//! 2. Decode each BLOB with safe `chunks_exact(4)` arithmetic — no
//!    `unsafe`, and the workspace lints deny it anyway.
//! 3. Maintain a min-heap of size ≤ K so we don't sort the whole list.
//! 4. Hydrate the K winners with one batched JOIN against the parent
//!    table to surface the row's content / key / value.
//!
//! Linear scan is fine for the expected DB scale: even 50K chunks at
//! 768-dim is ~150 MB of f32 — microseconds to dot-product on modern
//! hardware. If/when a heavy user pushes beyond that, an HNSW / sqlite-vec
//! extension can drop in behind this same trait without touching callers.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::connection::SqliteHandle;
use super::conversations::PersistedRole;

/// One conversation-chunk hit, hydrated with the *full* parent message
/// content (chunks may cut mid-sentence — for the LLM-facing surface we
/// surface the whole message so the model isn't reading a torso).
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

/// One saved-memory hit, ranked by semantic similarity to the query.
/// Same shape `RecallTool` already produces from prefix mode (key/value
/// pair + similarity score), so the LLM-facing output stays uniform.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryHit {
    pub memory_id: i64,
    pub key: String,
    pub value: String,
    pub similarity: f32,
}

/// Top-K vector retrieval. Sibling to [`crate::ConversationStore`];
/// keeping it on its own trait means callers that only need FTS5 can
/// hold `Arc<dyn ConversationStore>` without a `dim()` requirement.
#[async_trait]
pub trait SemanticStore: Send + Sync + 'static {
    /// Top-K conversation chunks ranked by cosine. `query_vector` must
    /// already be L2-normalised (consumers go through `LlamaEmbedder`
    /// which normalises before returning).
    async fn nearest_chunks(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        model: &str,
    ) -> Result<Vec<EmbeddingHit>>;

    /// Top-K saved memories ranked by cosine. Same normalisation
    /// expectation as above. Returns empty when no memories have been
    /// embedded yet (e.g. fresh DB or model just changed).
    async fn nearest_memories(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        model: &str,
    ) -> Result<Vec<MemoryHit>>;

    /// Diagnostic / health: how many embedding rows exist for the
    /// configured model? Returned as `(chunks, memories)`. Useful in
    /// tests and a future `assistd memory stats` view.
    async fn count_for_model(&self, model: &str) -> Result<(i64, i64)>;

    /// Diagnostic: rows whose `model` does NOT equal `current`. Used by
    /// the daemon at startup to warn when a config change has stranded
    /// pre-existing embeddings under a different model name (the
    /// retrieval queries filter by model, so stale rows are invisible
    /// until reindexed). Returns total stale row count + the distinct
    /// model names that own them, sorted lexicographically.
    async fn count_stale(&self, current: &str) -> Result<(i64, Vec<String>)>;

    /// Memories that have no embedding under `current`. The reindex
    /// handler embeds and stores each `(id, value)` to bring them back
    /// into the recall index. A row appears here when (a) the memory
    /// was saved while the embedder subsystem was down, or (b) the
    /// configured embedding model has changed since the row was
    /// indexed.
    async fn memories_missing_embedding(&self, current: &str) -> Result<Vec<(i64, String)>>;

    /// Conversation chunks that have no embedding under `current`.
    /// Symmetric to [`SemanticStore::memories_missing_embedding`].
    async fn chunks_missing_embedding(&self, current: &str) -> Result<Vec<(i64, String)>>;

    /// Persist a freshly-computed embedding for a `conversation_chunks`
    /// row. Idempotent on `(chunk_id)` via the same UPSERT the
    /// background embedder task uses. Used by the reindex handler so
    /// it can dispatch directly without going through the embedder
    /// task's mpsc.
    async fn store_chunk_embedding(
        &self,
        chunk_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
    ) -> Result<()>;

    /// Persist a freshly-computed embedding for a `memories` row.
    /// Symmetric to [`SemanticStore::store_chunk_embedding`].
    async fn store_memory_embedding(
        &self,
        memory_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
    ) -> Result<()>;
}

/// Successful-no-op fallback used when the embedding subsystem is
/// disabled. Mirrors `NoMemoryStore` / `NoConversationStore`.
pub struct NoSemanticStore;

#[async_trait]
impl SemanticStore for NoSemanticStore {
    async fn nearest_chunks(
        &self,
        _q: Vec<f32>,
        _k: usize,
        _model: &str,
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

/// SQLite-backed implementation. Holds an `Arc<SqliteHandle>` so it
/// shares the connection + writer with the other stores.
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
    ) -> Result<Vec<EmbeddingHit>> {
        if top_k == 0 || query_vector.is_empty() {
            return Ok(Vec::new());
        }
        let model = model.to_string();
        let q = query_vector;
        // Phase 1: scan + score. Returns Vec<(chunk_id, similarity)> for
        // the top-K results, ordered descending by similarity.
        let ranked: Vec<(i64, f32)> = self
            .handle
            .conn()
            .call(move |c| {
                let mut stmt = c.prepare(
                    "SELECT conversation_chunk_id, vector FROM embeddings WHERE model = ?1",
                )?;
                let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(top_k + 1);
                let mut rows = stmt.query(rusqlite::params![model])?;
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
        // Phase 2: hydrate winners with one batched JOIN. Batching keeps
        // it O(1) round-trips regardless of K.
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
            .call(move |c| {
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
        // Re-order to match the rank order (the IN-clause query may
        // return rows in any order; we want best-first).
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
            .call(move |c| {
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
            .call(move |c| {
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
            .call(move |c| {
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
            .call(move |c| {
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
            .call(move |c| {
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
            .call(move |c| {
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

// Min-heap entry: smallest similarity at the top so we can pop it when
// a better candidate arrives. `OrderedFloat` would do this with less
// boilerplate, but we already avoid extra deps elsewhere — wrap by hand.
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
        // Reverse the natural f32 ordering so `BinaryHeap` (which is a
        // max-heap) acts as a min-heap. NaN treated as the smallest
        // value so it gets evicted first.
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
    // Best-first.
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    v
}

/// Compute cosine (== dot product, given both sides L2-normalised) of
/// `query` against the LE-packed f32 BLOB `bytes`. Returns `None` if
/// the BLOB length is malformed (not a multiple of 4, or dim mismatch);
/// the row is then silently skipped.
fn score_against(query: &[f32], bytes: &[u8]) -> Option<f32> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let dim = bytes.len() / 4;
    if dim != query.len() {
        // Embedding from a different model (or model swap mid-deploy);
        // skip rather than poison the heap.
        return None;
    }
    let mut sum = 0.0f32;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        sum += f * query[i];
    }
    Some(sum)
}

/// Encode an `f32` slice as a LE-packed `Vec<u8>` for storage. Used by
/// callers that want to construct `WriteOp::StoreChunkEmbedding` /
/// `StoreMemoryEmbedding` payloads without copy-pasting the byte math.
pub fn vector_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PersistedMessage;
    use crate::sqlite::SqliteHandle;
    use crate::sqlite::writer::WriteOp;
    use std::sync::Arc;
    use tokio::sync::{oneshot, watch};

    async fn fresh() -> (Arc<SqliteHandle>, tokio::task::JoinHandle<()>) {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        std::mem::forget(temp);
        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
        (Arc::new(handle), writer)
    }

    fn unit_vec(angle: f32) -> Vec<f32> {
        // 2-d unit vector at the given angle.
        vec![angle.cos(), angle.sin()]
    }

    async fn insert_chunk_with_vec(
        handle: &SqliteHandle,
        conv_id: i64,
        chunk_index: i64,
        v: &[f32],
        model: &str,
    ) -> i64 {
        // store_chunk
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::StoreChunk {
                conversation_id: conv_id,
                chunk_index,
                content: format!("chunk{chunk_index}"),
                token_count: None,
                ack: tx,
            })
            .await
            .unwrap();
        let chunk_id = rx.await.unwrap().unwrap();
        // store embedding
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::StoreChunkEmbedding {
                chunk_id,
                model: model.into(),
                dim: v.len() as i64,
                vector: vector_to_blob(v),
                ack: tx,
            })
            .await
            .unwrap();
        rx.await.unwrap().unwrap();
        chunk_id
    }

    #[tokio::test]
    async fn nearest_chunks_empty_store_returns_empty() {
        let (handle, _w) = fresh().await;
        let s = SqliteSemanticStore::new(handle);
        let hits = s
            .nearest_chunks(unit_vec(0.0), 5, "test-model")
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn nearest_chunks_ranks_by_similarity() {
        let (handle, _w) = fresh().await;
        // Create one conversation row to FK the chunks against.
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::BeginSession {
                session_id: "sess-1".into(),
                daemon_pid: 1,
                ack: tx,
            })
            .await
            .unwrap();
        rx.await.unwrap().unwrap();
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::AppendMessage {
                session_id: "sess-1".into(),
                turn_id: None,
                msg: PersistedMessage::user("hello world"),
                ack: tx,
            })
            .await
            .unwrap();
        let conv_id = rx.await.unwrap().unwrap();

        // Insert three chunks with vectors at different angles.
        // Query points at 0; expect chunk at angle 0 to win, then 0.3, then 1.5.
        let _c1 = insert_chunk_with_vec(&handle, conv_id, 0, &unit_vec(0.0), "m").await;
        let _c2 = insert_chunk_with_vec(&handle, conv_id, 1, &unit_vec(0.3), "m").await;
        let _c3 = insert_chunk_with_vec(&handle, conv_id, 2, &unit_vec(1.5), "m").await;

        let s = SqliteSemanticStore::new(handle);
        let hits = s.nearest_chunks(unit_vec(0.0), 3, "m").await.unwrap();
        assert_eq!(hits.len(), 3);
        // Best-first.
        assert!(hits[0].similarity > hits[1].similarity);
        assert!(hits[1].similarity > hits[2].similarity);
        // First hit should be ~1.0.
        assert!((hits[0].similarity - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn nearest_chunks_top_k_caps_results() {
        let (handle, _w) = fresh().await;
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::BeginSession {
                session_id: "s".into(),
                daemon_pid: 0,
                ack: tx,
            })
            .await
            .unwrap();
        rx.await.unwrap().unwrap();
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::AppendMessage {
                session_id: "s".into(),
                turn_id: None,
                msg: PersistedMessage::user("x"),
                ack: tx,
            })
            .await
            .unwrap();
        let conv_id = rx.await.unwrap().unwrap();
        for i in 0..10 {
            insert_chunk_with_vec(&handle, conv_id, i, &unit_vec((i as f32) * 0.1), "m").await;
        }
        let s = SqliteSemanticStore::new(handle);
        let hits = s.nearest_chunks(unit_vec(0.0), 3, "m").await.unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[tokio::test]
    async fn nearest_chunks_filters_by_model() {
        let (handle, _w) = fresh().await;
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::BeginSession {
                session_id: "s".into(),
                daemon_pid: 0,
                ack: tx,
            })
            .await
            .unwrap();
        rx.await.unwrap().unwrap();
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::AppendMessage {
                session_id: "s".into(),
                turn_id: None,
                msg: PersistedMessage::user("x"),
                ack: tx,
            })
            .await
            .unwrap();
        let conv_id = rx.await.unwrap().unwrap();
        insert_chunk_with_vec(&handle, conv_id, 0, &unit_vec(0.0), "old-model").await;
        let s = SqliteSemanticStore::new(handle);
        // Query with the new model name — old-model rows must not appear.
        let hits = s
            .nearest_chunks(unit_vec(0.0), 5, "new-model")
            .await
            .unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn nearest_memories_round_trips() {
        let (handle, _w) = fresh().await;
        // Save a memory; capture its row id from the writer ack.
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::SaveMemory {
                key: "editor".into(),
                value: "vim".into(),
                source_conversation_id: None,
                ack: tx,
            })
            .await
            .unwrap();
        let mem_id = rx.await.unwrap().unwrap();
        // Embed it.
        let (tx, rx) = oneshot::channel();
        let v = unit_vec(0.0);
        handle
            .writer()
            .send(WriteOp::StoreMemoryEmbedding {
                memory_id: mem_id,
                model: "m".into(),
                dim: v.len() as i64,
                vector: vector_to_blob(&v),
                ack: tx,
            })
            .await
            .unwrap();
        rx.await.unwrap().unwrap();
        // Retrieve.
        let s = SqliteSemanticStore::new(handle);
        let hits = s.nearest_memories(unit_vec(0.0), 5, "m").await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].key, "editor");
        assert_eq!(hits[0].value, "vim");
        assert!((hits[0].similarity - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn upsert_replaces_memory_embedding_in_place() {
        let (handle, _w) = fresh().await;
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::SaveMemory {
                key: "k".into(),
                value: "v1".into(),
                source_conversation_id: None,
                ack: tx,
            })
            .await
            .unwrap();
        let mem_id_1 = rx.await.unwrap().unwrap();
        // Re-save under same key — id should be stable.
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::SaveMemory {
                key: "k".into(),
                value: "v2".into(),
                source_conversation_id: None,
                ack: tx,
            })
            .await
            .unwrap();
        let mem_id_2 = rx.await.unwrap().unwrap();
        assert_eq!(mem_id_1, mem_id_2, "UPSERT must keep same row id");
    }

    #[tokio::test]
    async fn no_semantic_store_returns_empty() {
        let s = NoSemanticStore;
        assert!(
            s.nearest_chunks(vec![1.0], 5, "m")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            s.nearest_memories(vec![1.0], 5, "m")
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(s.count_for_model("m").await.unwrap(), (0, 0));
        let (n, models) = s.count_stale("m").await.unwrap();
        assert_eq!(n, 0);
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn missing_embedding_lists_only_unindexed_rows_for_current_model() {
        let (handle, _w) = fresh().await;
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::BeginSession {
                session_id: "sess-mx".into(),
                daemon_pid: 1,
                ack: tx,
            })
            .await
            .unwrap();
        rx.await.unwrap().unwrap();
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::AppendMessage {
                session_id: "sess-mx".into(),
                turn_id: None,
                msg: PersistedMessage::user("x"),
                ack: tx,
            })
            .await
            .unwrap();
        let conv_id = rx.await.unwrap().unwrap();

        // Two chunks: one indexed under "new", one indexed under "old".
        let _ = insert_chunk_with_vec(&handle, conv_id, 0, &unit_vec(0.0), "new").await;
        let _ = insert_chunk_with_vec(&handle, conv_id, 1, &unit_vec(0.5), "old").await;
        // One unindexed chunk (no embedding row at all).
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::StoreChunk {
                conversation_id: conv_id,
                chunk_index: 2,
                content: "naked-chunk".into(),
                token_count: None,
                ack: tx,
            })
            .await
            .unwrap();
        let naked_chunk = rx.await.unwrap().unwrap();

        // Two memories: one indexed under "new", one bare.
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::SaveMemory {
                key: "indexed".into(),
                value: "v1".into(),
                source_conversation_id: None,
                ack: tx,
            })
            .await
            .unwrap();
        let indexed_mem = rx.await.unwrap().unwrap();
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::StoreMemoryEmbedding {
                memory_id: indexed_mem,
                model: "new".into(),
                dim: 2,
                vector: vector_to_blob(&unit_vec(0.0)),
                ack: tx,
            })
            .await
            .unwrap();
        rx.await.unwrap().unwrap();
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::SaveMemory {
                key: "bare".into(),
                value: "v2".into(),
                source_conversation_id: None,
                ack: tx,
            })
            .await
            .unwrap();
        let bare_mem = rx.await.unwrap().unwrap();

        let s = SqliteSemanticStore::new(handle);
        // Under current = "new":
        // - Chunks missing: the "old"-indexed chunk + the naked one.
        // - Memories missing: just the bare memory.
        let chunks = s.chunks_missing_embedding("new").await.unwrap();
        assert_eq!(chunks.len(), 2);
        let chunk_contents: Vec<&str> = chunks.iter().map(|(_, t)| t.as_str()).collect();
        assert!(chunk_contents.contains(&"chunk1")); // old-indexed
        assert!(chunk_contents.contains(&"naked-chunk"));
        assert!(chunks.iter().any(|(id, _)| *id == naked_chunk));

        let memories = s.memories_missing_embedding("new").await.unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].0, bare_mem);
        assert_eq!(memories[0].1, "v2");

        // store_*_embedding should be idempotent: write under "new"
        // and the row drops out of the missing list.
        s.store_memory_embedding(
            bare_mem,
            "new".to_string(),
            2,
            vector_to_blob(&unit_vec(0.0)),
        )
        .await
        .unwrap();
        let memories = s.memories_missing_embedding("new").await.unwrap();
        assert!(memories.is_empty());
    }

    #[tokio::test]
    async fn count_stale_aggregates_across_chunks_and_memories() {
        let (handle, _w) = fresh().await;
        // Create a conversation row to FK chunk inserts.
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::BeginSession {
                session_id: "sess-stale".into(),
                daemon_pid: 1,
                ack: tx,
            })
            .await
            .unwrap();
        rx.await.unwrap().unwrap();
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::AppendMessage {
                session_id: "sess-stale".into(),
                turn_id: None,
                msg: PersistedMessage::user("x"),
                ack: tx,
            })
            .await
            .unwrap();
        let conv_id = rx.await.unwrap().unwrap();

        // Two chunks under "old-A", one under "old-B", one under "new".
        let _ = insert_chunk_with_vec(&handle, conv_id, 0, &unit_vec(0.0), "old-A").await;
        let _ = insert_chunk_with_vec(&handle, conv_id, 1, &unit_vec(0.5), "old-A").await;
        let _ = insert_chunk_with_vec(&handle, conv_id, 2, &unit_vec(1.0), "old-B").await;
        let _ = insert_chunk_with_vec(&handle, conv_id, 3, &unit_vec(1.5), "new").await;

        // One memory embedding under "old-A".
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::SaveMemory {
                key: "k".into(),
                value: "v".into(),
                source_conversation_id: None,
                ack: tx,
            })
            .await
            .unwrap();
        let mem_id = rx.await.unwrap().unwrap();
        let (tx, rx) = oneshot::channel();
        handle
            .writer()
            .send(WriteOp::StoreMemoryEmbedding {
                memory_id: mem_id,
                model: "old-A".into(),
                dim: 2,
                vector: vector_to_blob(&unit_vec(0.0)),
                ack: tx,
            })
            .await
            .unwrap();
        rx.await.unwrap().unwrap();

        let s = SqliteSemanticStore::new(handle);
        // Current = "new" → 2 chunk rows under old-A + 1 chunk under old-B
        // + 1 memory under old-A = 4 stale rows, two distinct models.
        let (n, models) = s.count_stale("new").await.unwrap();
        assert_eq!(n, 4);
        assert_eq!(models, vec!["old-A".to_string(), "old-B".to_string()]);

        // Switching current to "old-A" should leave only the "old-B"
        // chunk + the "new" chunk as stale = 2 rows, two models.
        let (n, models) = s.count_stale("old-A").await.unwrap();
        assert_eq!(n, 2);
        assert_eq!(models, vec!["new".to_string(), "old-B".to_string()]);
    }

    #[test]
    fn vector_to_blob_round_trips() {
        let v = vec![0.5f32, -0.25, 0.125, 1e-6];
        let b = vector_to_blob(&v);
        let mut decoded = Vec::with_capacity(v.len());
        for chunk in b.chunks_exact(4) {
            decoded.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        assert_eq!(decoded, v);
    }

    #[test]
    fn score_against_dim_mismatch_returns_none() {
        let q = vec![1.0f32, 0.0];
        let v_3d = vector_to_blob(&[1.0, 0.0, 0.0]);
        assert!(score_against(&q, &v_3d).is_none());
    }
}

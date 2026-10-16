//! SQLite-backed [`SemanticStore`] over `chunk_embeddings` and `memory_embeddings`.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::fmt;
use std::hash::Hash;

use async_trait::async_trait;
use rusqlite::types::FromSql;
use rusqlite::{OptionalExtension, Row, Transaction};

use crate::{
    ChunkId, Embedding, EmbeddingHit, EmbeddingTarget, MemoryError, MemoryHit, MemoryId, Result,
    SemanticStore, SessionId, StaleEmbeddings, UnindexedRows,
};

use super::connection::SqliteHandle;
use super::timestamp::timestamp_at;

/// Every chunk vector under model `?1`.
const ALL_CHUNK_VECTORS: &str = "
    SELECT chunk_id, vector
    FROM chunk_embeddings
    WHERE model_id = (SELECT id FROM embedding_models WHERE name = ?1)";

/// Chunk vectors under model `?1` outside the session with UUID `?2`.
const CHUNK_VECTORS_OUTSIDE_SESSION: &str = "
    SELECT e.chunk_id, e.vector
    FROM chunk_embeddings e
    JOIN message_chunks cc ON cc.id = e.chunk_id
    JOIN messages m ON m.id = cc.message_id
    WHERE e.model_id = (SELECT id FROM embedding_models WHERE name = ?1)
      AND m.session_id IS NOT (SELECT id FROM sessions WHERE uuid = ?2)";

/// Independent partial sums in the dot product, so the compiler can vectorise it.
const SCORE_LANES: usize = 8;

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
}

#[async_trait]
impl SemanticStore for SqliteSemanticStore {
    async fn nearest_chunks(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        model: &str,
        exclude_session: Option<SessionId>,
    ) -> Result<Vec<EmbeddingHit>> {
        if top_k == 0 || query_vector.is_empty() {
            return Ok(Vec::new());
        }
        let model = model.to_string();
        self.handle
            .search_reader()
            .call(move |c| {
                let ranked = match exclude_session {
                    None => scan_top_k::<ChunkId>(
                        c,
                        ALL_CHUNK_VECTORS,
                        rusqlite::params![model],
                        &query_vector,
                        top_k,
                    )?,
                    Some(session) => scan_top_k::<ChunkId>(
                        c,
                        CHUNK_VECTORS_OUTSIDE_SESSION,
                        rusqlite::params![model, session],
                        &query_vector,
                        top_k,
                    )?,
                };
                hydrate_in_rank_order(
                    c,
                    "SELECT cc.id, m.id, s.uuid, m.created_at, m.role, m.content
                     FROM message_chunks cc
                     JOIN messages m ON m.id = cc.message_id
                     JOIN sessions s ON s.id = m.session_id
                     WHERE cc.id IN (SELECT value FROM json_each(?1))",
                    ranked,
                    chunk_hit,
                    |hit, similarity| hit.similarity = similarity,
                )
            })
            .await
            .map_err(MemoryError::sqlite("nearest_chunks"))
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
        self.handle
            .search_reader()
            .call(move |c| {
                let ranked = scan_top_k::<MemoryId>(
                    c,
                    "SELECT memory_id, vector
                     FROM memory_embeddings
                     WHERE model_id = (SELECT id FROM embedding_models WHERE name = ?1)",
                    rusqlite::params![model],
                    &query_vector,
                    top_k,
                )?;
                hydrate_in_rank_order(
                    c,
                    "SELECT id, key, value FROM memories WHERE id IN (SELECT value FROM json_each(?1))",
                    ranked,
                    memory_hit,
                    |hit, similarity| hit.similarity = similarity,
                )
            })
            .await
            .map_err(MemoryError::sqlite("nearest_memories"))
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
                let mut stmt = c.prepare_cached(sql)?;
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
                         WHERE e.chunk_id = cc.id
                           AND e.model_id = (SELECT id FROM embedding_models WHERE name = ?1)
                     )",
                    rusqlite::params![current],
                    |r| r.get(0),
                )?;
                let memories = c.query_row(
                    "SELECT count(*)
                     FROM memories mem
                     WHERE NOT EXISTS (
                         SELECT 1 FROM memory_embeddings e
                         WHERE e.memory_id = mem.id
                           AND e.model_id = (SELECT id FROM embedding_models WHERE name = ?1)
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
                let mut stmt = c.prepare_cached(
                    "SELECT mem.id, mem.value
                     FROM memories mem
                     WHERE mem.id > ifnull(?2, 0) AND NOT EXISTS (
                         SELECT 1 FROM memory_embeddings e
                         WHERE e.memory_id = mem.id
                           AND e.model_id = (SELECT id FROM embedding_models WHERE name = ?1)
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
                let mut stmt = c.prepare_cached(
                    "SELECT cc.id,
                            CAST(substr(CAST(msg.content AS BLOB), cc.byte_start + 1,
                                        cc.byte_end - cc.byte_start) AS TEXT)
                     FROM message_chunks cc
                     JOIN messages msg ON msg.id = cc.message_id
                     WHERE cc.id > ifnull(?2, 0) AND NOT EXISTS (
                         SELECT 1 FROM chunk_embeddings e
                         WHERE e.chunk_id = cc.id
                           AND e.model_id = (SELECT id FROM embedding_models WHERE name = ?1)
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
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let model_id = embedding_model_id(&tx, &model)?;
                for embedding in &embeddings {
                    upsert_embedding(&tx, model_id, embedding)?;
                }
                tx.commit()
            })
            .await
            .map_err(MemoryError::sqlite("store_embeddings"))
    }
}

/// Ordered in reverse by similarity so `BinaryHeap` acts as a min-heap and
/// `into_sorted_vec` yields the best match first. Only finite
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
    Ok(heap
        .into_sorted_vec()
        .into_iter()
        .map(|entry| (entry.id, entry.similarity))
        .collect())
}

/// The rows `select` finds for the `ranked` ids, bound to `?1` as a JSON array and
/// keyed by column 0, in rank order and given their scores; ids deleted since the scan are skipped.
fn hydrate_in_rank_order<Id, T>(
    conn: &rusqlite::Connection,
    select: &str,
    ranked: Vec<(Id, f32)>,
    to_hit: fn(&Row<'_>) -> rusqlite::Result<T>,
    set_similarity: fn(&mut T, f32),
) -> rusqlite::Result<Vec<T>>
where
    Id: FromSql + fmt::Display + Eq + Hash,
{
    if ranked.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare_cached(select)?;
    let mut hits = stmt
        .query_map(rusqlite::params![json_id_array(&ranked)], |row| {
            Ok((row.get::<_, Id>(0)?, to_hit(row)?))
        })?
        .collect::<rusqlite::Result<HashMap<_, _>>>()?;
    Ok(ranked
        .into_iter()
        .filter_map(|(id, similarity)| {
            let mut hit = hits.remove(&id)?;
            set_similarity(&mut hit, similarity);
            Some(hit)
        })
        .collect())
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

/// `ranked`'s ids as a JSON array of integers.
fn json_id_array<Id: fmt::Display>(ranked: &[(Id, f32)]) -> String {
    let ids: Vec<String> = ranked.iter().map(|(id, _)| id.to_string()).collect();
    format!("[{}]", ids.join(","))
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

/// Dot product of `query` against the LE-packed f32 BLOB `bytes`, or `None` when
/// the BLOB is malformed, its dimension differs, or the product is not finite.
fn score_against(query: &[f32], bytes: &[u8]) -> Option<f32> {
    let (words, trailing) = bytes.as_chunks::<4>();
    if !trailing.is_empty() || words.len() != query.len() {
        return None;
    }
    let (word_blocks, word_tail) = words.as_chunks::<SCORE_LANES>();
    let (query_blocks, query_tail) = query.as_chunks::<SCORE_LANES>();
    let mut lanes = [0.0f32; SCORE_LANES];
    for (word_block, query_block) in word_blocks.iter().zip(query_blocks) {
        for ((lane, word), q) in lanes.iter_mut().zip(word_block).zip(query_block) {
            *lane += f32::from_le_bytes(*word) * q;
        }
    }
    let similarity = lanes.iter().sum::<f32>() + dot_le_words(word_tail, query_tail);
    similarity.is_finite().then_some(similarity)
}

fn dot_le_words(words: &[[u8; 4]], query: &[f32]) -> f32 {
    words
        .iter()
        .zip(query)
        .map(|(word, q)| f32::from_le_bytes(*word) * q)
        .sum()
}

/// Upsert `embedding` into its target's table. The `SELECT` makes a since-deleted
/// target, or a memory whose value has changed since, a no-op rather than an error
/// or a vector for the wrong text.
fn upsert_embedding(
    tx: &Transaction<'_>,
    model_id: i64,
    embedding: &Embedding,
) -> rusqlite::Result<usize> {
    let blob = vector_to_blob(&embedding.vector);
    match embedding.target {
        EmbeddingTarget::Chunk(id) => tx
            .prepare_cached(
                "INSERT INTO chunk_embeddings (chunk_id, model_id, vector)
                 SELECT id, ?2, ?3 FROM message_chunks WHERE id = ?1
                 ON CONFLICT(chunk_id) DO UPDATE
                 SET model_id = excluded.model_id, vector = excluded.vector",
            )?
            .execute(rusqlite::params![id, model_id, blob]),
        EmbeddingTarget::Memory(id) => tx
            .prepare_cached(
                "INSERT INTO memory_embeddings (memory_id, model_id, vector)
                 SELECT id, ?2, ?3 FROM memories WHERE id = ?1 AND value = ?4
                 ON CONFLICT(memory_id) DO UPDATE
                 SET model_id = excluded.model_id, vector = excluded.vector",
            )?
            .execute(rusqlite::params![id, model_id, blob, embedding.text]),
    }
}

/// The `embedding_models` id of `name`, inserting the model on first use.
fn embedding_model_id(tx: &Transaction<'_>, name: &str) -> rusqlite::Result<i64> {
    let existing = tx
        .prepare_cached("SELECT id FROM embedding_models WHERE name = ?1")?
        .query_row(rusqlite::params![name], |r| r.get(0))
        .optional()?;
    match existing {
        Some(id) => Ok(id),
        None => tx
            .prepare_cached("INSERT INTO embedding_models (name) VALUES (?1) RETURNING id")?
            .query_row(rusqlite::params![name], |r| r.get(0)),
    }
}

/// Encode `vector` as the little-endian `f32` BLOB the embedding tables store.
fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

#[cfg(test)]
mod tests;

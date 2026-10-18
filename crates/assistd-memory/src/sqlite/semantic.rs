//! SQLite-backed [`SemanticStore`] over `chunk_embeddings` and `memory_embeddings`.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;

use async_trait::async_trait;
use rusqlite::types::FromSql;
use rusqlite::{OptionalExtension, Row, Transaction};

use crate::{
    ChunkId, Embedding, EmbeddingHit, EmbeddingTarget, MemoryError, MemoryHit, MemoryId, MessageId,
    ModelAdoption, Result, SemanticStore, SessionId, UnindexedRows,
};

use super::connection::SqliteHandle;
use super::timestamp::timestamp_at;

/// Every chunk vector, grouped by its message.
const ALL_CHUNK_VECTORS: &str = "
    SELECT cc.message_id, e.chunk_id, e.vector
    FROM chunk_embeddings e JOIN message_chunks cc ON cc.id = e.chunk_id";

/// Every memory vector, each its own group.
const ALL_MEMORY_VECTORS: &str = "SELECT memory_id, memory_id, vector FROM memory_embeddings";

/// Independent partial sums in the dot product, so the compiler can vectorise it.
const SCORE_LANES: usize = 8;

/// SQLite-backed [`SemanticStore`]: a linear scan into a bounded top-K, then one
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
        exclude_session: Option<SessionId>,
    ) -> Result<Vec<EmbeddingHit>> {
        if top_k == 0 || query_vector.is_empty() {
            return Ok(Vec::new());
        }
        self.handle
            .search_reader()
            .call(move |c| {
                let excluded = match exclude_session {
                    None => HashSet::new(),
                    Some(session) => session_message_ids(c, session)?,
                };
                let ranked = scan_top_k::<MessageId, ChunkId>(
                    c,
                    ALL_CHUNK_VECTORS,
                    &query_vector,
                    top_k,
                    &excluded,
                )?;
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
    ) -> Result<Vec<MemoryHit>> {
        if top_k == 0 || query_vector.is_empty() {
            return Ok(Vec::new());
        }
        self.handle
            .search_reader()
            .call(move |c| {
                let ranked = scan_top_k::<MemoryId, MemoryId>(
                    c,
                    ALL_MEMORY_VECTORS,
                    &query_vector,
                    top_k,
                    &HashSet::new(),
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

    async fn adopt_model(&self, model: &str) -> Result<ModelAdoption> {
        let model = model.to_string();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let adoption = reconcile_model(&tx, &model)?;
                tx.commit()?;
                Ok(adoption)
            })
            .await
            .map_err(MemoryError::sqlite("adopt_model"))
    }

    async fn count_missing(&self) -> Result<UnindexedRows> {
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let chunks = c.query_row(
                    "SELECT count(*)
                     FROM message_chunks cc
                     WHERE NOT EXISTS (SELECT 1 FROM chunk_embeddings e WHERE e.chunk_id = cc.id)",
                    [],
                    |r| r.get(0),
                )?;
                let memories = c.query_row(
                    "SELECT count(*)
                     FROM memories mem
                     WHERE NOT EXISTS (SELECT 1 FROM memory_embeddings e WHERE e.memory_id = mem.id)",
                    [],
                    |r| r.get(0),
                )?;
                Ok(UnindexedRows { chunks, memories })
            })
            .await
            .map_err(MemoryError::sqlite("count_missing"))
    }

    async fn memories_missing_embedding(
        &self,
        after: Option<MemoryId>,
        limit: usize,
    ) -> Result<Vec<(MemoryId, String)>> {
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare_cached(
                    "SELECT mem.id, mem.value
                     FROM memories mem
                     WHERE mem.id > ifnull(?1, 0) AND NOT EXISTS (
                         SELECT 1 FROM memory_embeddings e WHERE e.memory_id = mem.id
                     )
                     ORDER BY mem.id
                     LIMIT ?2",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![after, limit], |r| {
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
        after: Option<ChunkId>,
        limit: usize,
    ) -> Result<Vec<(ChunkId, String)>> {
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
                     WHERE cc.id > ifnull(?1, 0) AND NOT EXISTS (
                         SELECT 1 FROM chunk_embeddings e WHERE e.chunk_id = cc.id
                     )
                     ORDER BY cc.id
                     LIMIT ?2",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![after, limit], |r| {
                        Ok((r.get(0)?, r.get(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("chunks_missing_embedding"))
    }

    async fn store_embeddings(&self, embeddings: Vec<Embedding>) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                for embedding in &embeddings {
                    upsert_embedding(&tx, embedding)?;
                }
                tx.commit()
            })
            .await
            .map_err(MemoryError::sqlite("store_embeddings"))
    }
}

/// The `capacity` best-scoring ids seen, at most one per group.
struct TopK<Group, Id> {
    capacity: usize,
    entries: Vec<Ranked<Group, Id>>,
    /// The weakest kept similarity once full; `NEG_INFINITY` until then.
    floor: f32,
}

struct Ranked<Group, Id> {
    group: Group,
    id: Id,
    similarity: f32,
}

impl<Group: PartialEq, Id> TopK<Group, Id> {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::new(),
            floor: f32::NEG_INFINITY,
        }
    }

    /// Keep `id` if it beats `group`'s current best, or when it has none, the
    /// weakest entry once full.
    fn offer(&mut self, group: Group, id: Id, similarity: f32) {
        if similarity <= self.floor {
            return;
        }
        let candidate = Ranked {
            group,
            id,
            similarity,
        };
        let same_group = self.entries.iter().position(|e| e.group == candidate.group);
        match same_group {
            Some(index) if similarity <= self.entries[index].similarity => return,
            Some(index) => self.entries[index] = candidate,
            None if self.entries.len() < self.capacity => self.entries.push(candidate),
            None => {
                let weakest = self.weakest_index();
                self.entries[weakest] = candidate;
            }
        }
        self.refresh_floor();
    }

    fn weakest_index(&self) -> usize {
        self.entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.similarity.total_cmp(&b.similarity))
            .map_or(0, |(index, _)| index)
    }

    fn refresh_floor(&mut self) {
        if self.entries.len() == self.capacity {
            self.floor = self
                .entries
                .iter()
                .map(|e| e.similarity)
                .fold(f32::INFINITY, f32::min);
        }
    }

    /// The kept ids with their scores, best first.
    fn into_ranked(mut self) -> Vec<(Id, f32)> {
        self.entries
            .sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        self.entries
            .into_iter()
            .map(|e| (e.id, e.similarity))
            .collect()
    }
}

/// The `top_k` best `(group, id, vector)` rows of `sql` against `query`, one per
/// group, best first, skipping groups in `excluded`.
fn scan_top_k<Group: FromSql + Eq + Hash, Id: FromSql>(
    conn: &rusqlite::Connection,
    sql: &str,
    query: &[f32],
    top_k: usize,
    excluded: &HashSet<Group>,
) -> rusqlite::Result<Vec<(Id, f32)>> {
    let mut stmt = conn.prepare_cached(sql)?;
    let mut best = TopK::new(top_k);
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let group = row.get(0)?;
        if excluded.contains(&group) {
            continue;
        }
        let bytes = row.get_ref(2)?.as_blob()?;
        if let Some(similarity) = score_against(query, bytes) {
            best.offer(group, row.get(1)?, similarity);
        }
    }
    Ok(best.into_ranked())
}

/// Every message in `session`.
fn session_message_ids(
    conn: &rusqlite::Connection,
    session: SessionId,
) -> rusqlite::Result<HashSet<MessageId>> {
    let mut stmt = conn.prepare_cached(
        "SELECT m.id
         FROM sessions s JOIN messages m ON m.session_id = s.id
         WHERE s.uuid = ?1",
    )?;
    stmt.query_map(rusqlite::params![session], |r| r.get(0))?
        .collect()
}

/// The rows `select` finds for the `ranked` ids, bound to `?1` as a JSON array and
/// keyed by column 0, in rank order and given their scores; ids deleted since the
/// scan are skipped.
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
fn upsert_embedding(tx: &Transaction<'_>, embedding: &Embedding) -> rusqlite::Result<usize> {
    let blob = vector_to_blob(&embedding.vector);
    match embedding.target {
        EmbeddingTarget::Chunk(id) => tx
            .prepare_cached(
                "INSERT INTO chunk_embeddings (chunk_id, vector)
                 SELECT id, ?2 FROM message_chunks WHERE id = ?1
                 ON CONFLICT(chunk_id) DO UPDATE SET vector = excluded.vector",
            )?
            .execute(rusqlite::params![id, blob]),
        EmbeddingTarget::Memory(id) => tx
            .prepare_cached(
                "INSERT INTO memory_embeddings (memory_id, vector)
                 SELECT id, ?2 FROM memories WHERE id = ?1 AND value = ?3
                 ON CONFLICT(memory_id) DO UPDATE SET vector = excluded.vector",
            )?
            .execute(rusqlite::params![id, blob, embedding.text]),
    }
}

/// Record `model` in `embedding_model`, deleting every vector unless it is
/// already the recorded model.
fn reconcile_model(tx: &Transaction<'_>, model: &str) -> rusqlite::Result<ModelAdoption> {
    let previous: Option<String> = tx
        .query_row("SELECT name FROM embedding_model", [], |r| r.get(0))
        .optional()?;
    if previous.as_deref() == Some(model) {
        return Ok(ModelAdoption::Unchanged);
    }
    let dropped = tx.execute("DELETE FROM chunk_embeddings", [])?
        + tx.execute("DELETE FROM memory_embeddings", [])?;
    tx.execute(
        "INSERT INTO embedding_model (id, name) VALUES (1, ?1)
         ON CONFLICT(id) DO UPDATE SET name = excluded.name",
        rusqlite::params![model],
    )?;
    Ok(match previous {
        None => ModelAdoption::Recorded,
        Some(previous) => ModelAdoption::Replaced {
            previous,
            dropped: u64::try_from(dropped).unwrap_or(u64::MAX),
        },
    })
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

//! SQLite-backed [`SemanticStore`] over `chunk_embeddings` and `memory_embeddings`.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fmt;
use std::hash::Hash;
use std::ops::Range;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use rusqlite::types::FromSql;
use rusqlite::{OptionalExtension, Row, Transaction};

use crate::chunking::{ChunkingConfig, chunk_ranges};
use crate::{
    AdjacentChunks, ChunkId, ChunkLocator, ChunkScope, ContextChunk, DocumentId, Embedding,
    EmbeddingHit, EmbeddingTarget, MemoryError, MemoryHit, MemoryId, MessageId, Result,
    SemanticStore, SessionId, SessionScope, UnindexedRows,
};

use super::connection::SqliteHandle;
use super::timestamp::{now_millis, timestamp_at};

/// Every message chunk vector, grouped by its message.
const MESSAGE_CHUNK_VECTORS: &str = "
    SELECT cc.message_id, e.chunk_id, e.vector
    FROM chunk_embeddings e JOIN chunks cc ON cc.id = e.chunk_id
    WHERE cc.message_id IS NOT NULL";

/// Every document chunk vector, grouped by its document.
const DOCUMENT_CHUNK_VECTORS: &str = "
    SELECT cc.document_id, e.chunk_id, e.vector
    FROM chunk_embeddings e JOIN chunks cc ON cc.id = e.chunk_id
    WHERE cc.document_id IS NOT NULL";

/// Every memory vector, each its own group.
const ALL_MEMORY_VECTORS: &str = "SELECT memory_id, memory_id, vector FROM memory_embeddings";

/// A message chunk row's columns, read by [`message_context_chunk`]: chunk id, message
/// id, session uuid, message time, role, session title, the session's opening user text,
/// chunk text.
const MESSAGE_CHUNK_COLUMNS: &str = "
    cc.id, m.id, s.uuid, m.created_at, m.role, s.title,
    (SELECT substr(f.content, 1, 256) FROM messages f
     WHERE f.session_id = s.id AND f.role = 0 ORDER BY f.id LIMIT 1),
    CAST(substr(CAST(m.content AS BLOB), cc.byte_start + 1,
                cc.byte_end - cc.byte_start) AS TEXT)";

/// A document chunk row's columns, read by [`document_context_chunk`]: chunk id,
/// document id, path, byte range, index time, chunk text.
const DOCUMENT_CHUNK_COLUMNS: &str = "
    cc.id, d.id, d.path, cc.byte_start, cc.byte_end, d.indexed_at,
    CAST(substr(CAST(d.content AS BLOB), cc.byte_start + 1,
                cc.byte_end - cc.byte_start) AS TEXT)";

/// Characters of an untitled session's opening user message kept in its label.
const SESSION_LABEL_CHARS: usize = 60;

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

    /// Index `content` as the text of the document at `path`, replacing its chunks
    /// and their vectors when the text changed. Returns the new chunks as
    /// `(id, chunk text)` for embedding; empty when the text is unchanged.
    pub async fn index_document(
        &self,
        path: PathBuf,
        content: String,
    ) -> Result<Vec<(ChunkId, String)>> {
        let ranges = chunk_ranges(&content, &ChunkingConfig::DOCUMENTS);
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let chunks = replace_document(&tx, &path, &content, &ranges)?;
                tx.commit()?;
                Ok(chunks)
            })
            .await
            .map_err(MemoryError::sqlite("index_document"))
    }
}

#[async_trait]
impl SemanticStore for SqliteSemanticStore {
    async fn nearest_chunks(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        scope: ChunkScope,
    ) -> Result<Vec<EmbeddingHit>> {
        if top_k == 0 || query_vector.is_empty() {
            return Ok(Vec::new());
        }
        self.handle
            .search_reader()
            .call(move |c| match scope {
                ChunkScope::Chats(sessions) => {
                    nearest_message_chunks(c, &query_vector, top_k, sessions)
                }
                ChunkScope::Documents => nearest_document_chunks(c, &query_vector, top_k),
            })
            .await
            .map_err(MemoryError::sqlite("nearest_chunks"))
    }

    async fn adjacent_chunks(&self, locator: &ChunkLocator) -> Result<AdjacentChunks> {
        let locator = locator.clone();
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                Ok(AdjacentChunks {
                    before: adjacent_chunk(c, &locator, Side::Before)?,
                    after: adjacent_chunk(c, &locator, Side::After)?,
                })
            })
            .await
            .map_err(MemoryError::sqlite("adjacent_chunks"))
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
                    |_| true,
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

    async fn drop_other_models(&self, model: &str) -> Result<u64> {
        let model = model.to_string();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let dropped = tx.execute(
                    "DELETE FROM chunk_embeddings WHERE model != ?1",
                    rusqlite::params![model],
                )? + tx.execute(
                    "DELETE FROM memory_embeddings WHERE model != ?1",
                    rusqlite::params![model],
                )?;
                tx.commit()?;
                Ok(u64::try_from(dropped).unwrap_or(u64::MAX))
            })
            .await
            .map_err(MemoryError::sqlite("drop_other_models"))
    }

    async fn count_missing(&self) -> Result<UnindexedRows> {
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let chunks = c.query_row(
                    "SELECT count(*)
                     FROM chunks cc
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
                            CAST(substr(CAST(coalesce(msg.content, doc.content) AS BLOB),
                                        cc.byte_start + 1, cc.byte_end - cc.byte_start) AS TEXT)
                     FROM chunks cc
                     LEFT JOIN messages msg ON msg.id = cc.message_id
                     LEFT JOIN documents doc ON doc.id = cc.document_id
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

    async fn store_embeddings(&self, model: &str, embeddings: Vec<Embedding>) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }
        let model = model.to_string();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                for embedding in &embeddings {
                    upsert_embedding(&tx, &model, embedding)?;
                }
                tx.commit()
            })
            .await
            .map_err(MemoryError::sqlite("store_embeddings"))
    }
}

/// The messages a chunk search may rank.
enum MessageFilter {
    Any,
    Skip(HashSet<MessageId>),
    Keep(HashSet<MessageId>),
}

impl MessageFilter {
    fn for_scope(conn: &rusqlite::Connection, scope: SessionScope) -> rusqlite::Result<Self> {
        Ok(match scope {
            SessionScope::All => Self::Any,
            SessionScope::AllExcept(session) => Self::Skip(session_message_ids(conn, session)?),
            SessionScope::AllExceptInContext(session) => {
                Self::Skip(in_context_message_ids(conn, session)?)
            }
            SessionScope::Only(session) => Self::Keep(session_message_ids(conn, session)?),
        })
    }

    fn admits(&self, message: MessageId) -> bool {
        match self {
            Self::Any => true,
            Self::Skip(skipped) => !skipped.contains(&message),
            Self::Keep(kept) => kept.contains(&message),
        }
    }
}

/// Which neighbour of a chunk's source position [`adjacent_chunk`] looks up.
#[derive(Clone, Copy)]
enum Side {
    Before,
    After,
}

impl Side {
    /// The comparison against the hit's position and the sort order that puts the
    /// nearest chunk on this side first.
    fn test_and_order(self) -> (&'static str, &'static str) {
        match self {
            Self::Before => ("<", "DESC"),
            Self::After => (">", "ASC"),
        }
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
/// group, best first, skipping groups `admits` rejects.
fn scan_top_k<Group: FromSql + Eq + Hash, Id: FromSql>(
    conn: &rusqlite::Connection,
    sql: &str,
    query: &[f32],
    top_k: usize,
    admits: impl Fn(&Group) -> bool,
) -> rusqlite::Result<Vec<(Id, f32)>> {
    let mut stmt = conn.prepare_cached(sql)?;
    let mut best = TopK::new(top_k);
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let group = row.get(0)?;
        if !admits(&group) {
            continue;
        }
        let bytes = row.get_ref(2)?.as_blob()?;
        if let Some(similarity) = score_against(query, bytes) {
            best.offer(group, row.get(1)?, similarity);
        }
    }
    Ok(best.into_ranked())
}

/// The `top_k` messages in `sessions` with the best chunks against `query`, best first.
fn nearest_message_chunks(
    conn: &rusqlite::Connection,
    query: &[f32],
    top_k: usize,
    sessions: SessionScope,
) -> rusqlite::Result<Vec<EmbeddingHit>> {
    let filter = MessageFilter::for_scope(conn, sessions)?;
    let ranked =
        scan_top_k::<MessageId, ChunkId>(conn, MESSAGE_CHUNK_VECTORS, query, top_k, |message| {
            filter.admits(*message)
        })?;
    let select = format!(
        "SELECT {MESSAGE_CHUNK_COLUMNS}
         FROM chunks cc
         JOIN messages m ON m.id = cc.message_id
         JOIN sessions s ON s.id = m.session_id
         WHERE cc.id IN (SELECT value FROM json_each(?1))"
    );
    hydrate_in_rank_order(conn, &select, ranked, message_chunk_hit, set_similarity)
}

/// The `top_k` documents with the best chunks against `query`, best first.
fn nearest_document_chunks(
    conn: &rusqlite::Connection,
    query: &[f32],
    top_k: usize,
) -> rusqlite::Result<Vec<EmbeddingHit>> {
    let ranked =
        scan_top_k::<DocumentId, ChunkId>(conn, DOCUMENT_CHUNK_VECTORS, query, top_k, |_| true)?;
    let select = format!(
        "SELECT {DOCUMENT_CHUNK_COLUMNS}
         FROM chunks cc
         JOIN documents d ON d.id = cc.document_id
         WHERE cc.id IN (SELECT value FROM json_each(?1))"
    );
    hydrate_in_rank_order(conn, &select, ranked, document_chunk_hit, set_similarity)
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

/// The messages of `session` not folded into its current branch's summary.
fn in_context_message_ids(
    conn: &rusqlite::Connection,
    session: SessionId,
) -> rusqlite::Result<HashSet<MessageId>> {
    let mut stmt = conn.prepare_cached(
        "SELECT m.id
         FROM sessions s JOIN messages m ON m.session_id = s.id
         WHERE s.uuid = ?1
           AND m.id NOT IN (
               SELECT bm.message_id
               FROM branch_messages bm
               JOIN branch_summaries bs ON bs.branch_id = bm.branch_id
               WHERE bm.branch_id = s.current_branch_id AND bm.seq < bs.kept_from_seq
           )",
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

/// The chunk next to `locator` on `side` within its source.
fn adjacent_chunk(
    conn: &rusqlite::Connection,
    locator: &ChunkLocator,
    side: Side,
) -> rusqlite::Result<Option<ContextChunk>> {
    match locator {
        ChunkLocator::Message { message_id, .. } => adjacent_message_chunk(conn, *message_id, side),
        ChunkLocator::Document {
            document_id, bytes, ..
        } => adjacent_document_chunk(conn, *document_id, bytes.start, side),
    }
}

/// The chunk of the nearest user or assistant message on `side` of `message` along
/// its branch: the session's current one when it holds `message`, else its oldest.
fn adjacent_message_chunk(
    conn: &rusqlite::Connection,
    message: MessageId,
    side: Side,
) -> rusqlite::Result<Option<ContextChunk>> {
    let (seq_test, order) = side.test_and_order();
    let sql = format!(
        "WITH here AS (
             SELECT bm.branch_id, bm.seq
             FROM branch_messages bm
             JOIN messages hm ON hm.id = bm.message_id
             JOIN sessions hs ON hs.id = hm.session_id
             WHERE bm.message_id = ?1
             ORDER BY bm.branch_id IS hs.current_branch_id DESC, bm.branch_id
             LIMIT 1
         )
         SELECT {MESSAGE_CHUNK_COLUMNS}
         FROM here
         JOIN branch_messages bm ON bm.branch_id = here.branch_id
         JOIN messages m ON m.id = bm.message_id
         JOIN chunks cc ON cc.message_id = m.id
         JOIN sessions s ON s.id = m.session_id
         WHERE bm.seq {seq_test} here.seq AND m.role IN (0, 1)
         ORDER BY bm.seq {order}, cc.byte_start {order}
         LIMIT 1"
    );
    conn.prepare_cached(&sql)?
        .query_row(rusqlite::params![message], message_context_chunk)
        .optional()
}

/// The chunk of `document` starting nearest `byte_start` on `side`.
fn adjacent_document_chunk(
    conn: &rusqlite::Connection,
    document: DocumentId,
    byte_start: usize,
    side: Side,
) -> rusqlite::Result<Option<ContextChunk>> {
    let (byte_test, order) = side.test_and_order();
    let sql = format!(
        "SELECT {DOCUMENT_CHUNK_COLUMNS}
         FROM chunks cc
         JOIN documents d ON d.id = cc.document_id
         WHERE cc.document_id = ?1 AND cc.byte_start {byte_test} ?2
         ORDER BY cc.byte_start {order}
         LIMIT 1"
    );
    conn.prepare_cached(&sql)?
        .query_row(
            rusqlite::params![document, byte_start],
            document_context_chunk,
        )
        .optional()
}

fn message_chunk_hit(row: &Row<'_>) -> rusqlite::Result<EmbeddingHit> {
    message_context_chunk(row).map(unscored_hit)
}

fn document_chunk_hit(row: &Row<'_>) -> rusqlite::Result<EmbeddingHit> {
    document_context_chunk(row).map(unscored_hit)
}

fn unscored_hit(chunk: ContextChunk) -> EmbeddingHit {
    EmbeddingHit {
        chunk_id: chunk.chunk_id,
        locator: chunk.locator,
        text: chunk.text,
        similarity: 0.0,
    }
}

fn set_similarity(hit: &mut EmbeddingHit, similarity: f32) {
    hit.similarity = similarity;
}

fn message_context_chunk(row: &Row<'_>) -> rusqlite::Result<ContextChunk> {
    Ok(ContextChunk {
        chunk_id: row.get(0)?,
        locator: ChunkLocator::Message {
            message_id: row.get(1)?,
            session_id: row.get(2)?,
            timestamp: timestamp_at(row, 3)?,
            role: row.get(4)?,
            session_label: session_label(row.get(5)?, row.get::<_, Option<String>>(6)?.as_deref()),
        },
        text: row.get(7)?,
    })
}

fn document_context_chunk(row: &Row<'_>) -> rusqlite::Result<ContextChunk> {
    Ok(ContextChunk {
        chunk_id: row.get(0)?,
        locator: ChunkLocator::Document {
            document_id: row.get(1)?,
            path: PathBuf::from(OsString::from_vec(row.get(2)?)),
            bytes: row.get(3)?..row.get(4)?,
            indexed_at: timestamp_at(row, 5)?,
        },
        text: row.get(6)?,
    })
}

/// `title`, or else the first line of `opening` cut to [`SESSION_LABEL_CHARS`].
fn session_label(title: Option<String>, opening: Option<&str>) -> String {
    if let Some(title) = title {
        return title;
    }
    let first_line = opening.unwrap_or_default().lines().next();
    let first_line = first_line.unwrap_or_default().trim();
    match first_line.char_indices().nth(SESSION_LABEL_CHARS) {
        Some((cut, _)) => format!("{}…", &first_line[..cut]),
        None => first_line.to_string(),
    }
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
fn upsert_embedding(
    tx: &Transaction<'_>,
    model: &str,
    embedding: &Embedding,
) -> rusqlite::Result<usize> {
    let blob = vector_to_blob(&embedding.vector);
    match embedding.target {
        EmbeddingTarget::Chunk(id) => tx
            .prepare_cached(
                "INSERT INTO chunk_embeddings (chunk_id, model, vector)
                 SELECT id, ?2, ?3 FROM chunks WHERE id = ?1
                 ON CONFLICT(chunk_id) DO UPDATE
                 SET model = excluded.model, vector = excluded.vector",
            )?
            .execute(rusqlite::params![id, model, blob]),
        EmbeddingTarget::Memory(id) => tx
            .prepare_cached(
                "INSERT INTO memory_embeddings (memory_id, model, vector)
                 SELECT id, ?2, ?3 FROM memories WHERE id = ?1 AND value = ?4
                 ON CONFLICT(memory_id) DO UPDATE
                 SET model = excluded.model, vector = excluded.vector",
            )?
            .execute(rusqlite::params![id, model, blob, embedding.text]),
    }
}

/// Store `content` as the document at `path` and cut it into `ranges`, returning the
/// new chunks; nothing changes when the stored text already matches.
fn replace_document(
    tx: &Transaction<'_>,
    path: &Path,
    content: &str,
    ranges: &[Range<usize>],
) -> rusqlite::Result<Vec<(ChunkId, String)>> {
    let changed: Option<DocumentId> = tx
        .prepare_cached(
            "INSERT INTO documents (path, content, indexed_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(path) DO UPDATE
             SET content = excluded.content, indexed_at = excluded.indexed_at
             WHERE content IS NOT excluded.content
             RETURNING id",
        )?
        .query_row(
            rusqlite::params![path.as_os_str().as_bytes(), content, now_millis()],
            |r| r.get(0),
        )
        .optional()?;
    let Some(document) = changed else {
        return Ok(Vec::new());
    };
    tx.prepare_cached("DELETE FROM chunks WHERE document_id = ?1")?
        .execute(rusqlite::params![document])?;
    let mut insert = tx.prepare_cached(
        "INSERT INTO chunks (document_id, byte_start, byte_end)
         VALUES (?1, ?2, ?3)
         RETURNING id",
    )?;
    ranges
        .iter()
        .map(|bytes| {
            let id = insert
                .query_row(rusqlite::params![document, bytes.start, bytes.end], |r| {
                    r.get(0)
                })?;
            Ok((id, content[bytes.clone()].to_string()))
        })
        .collect()
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

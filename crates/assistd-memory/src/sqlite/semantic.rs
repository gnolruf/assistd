//! SQLite-backed [`SemanticStore`] over one embeddings table each for message
//! chunks, document chunks and memories.

use std::collections::HashMap;
use std::ffi::OsString;
use std::fmt;
use std::hash::Hash;
use std::os::unix::ffi::OsStringExt;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use num_traits::ToPrimitive;
use rusqlite::types::FromSql;
use rusqlite::{OptionalExtension, Params, Row, Transaction};

use crate::{
    AdjacentChunks, ChunkId, ChunkLocator, ChunkScope, ContextChunk, DocumentId, Embedding,
    EmbeddingHit, EmbeddingSpace, EmbeddingTarget, FileStamp, IndexPass, MemoryHit, MemoryId,
    MessageId, Result, SemanticStore, SessionId, SessionScope, UnindexedRows,
};

use super::connection::SqliteHandle;
use super::timestamp::timestamp_at;

mod documents;

/// Every message chunk vector, grouped by its message.
const MESSAGE_CHUNK_VECTORS: &str = "
    SELECT cc.message_id, e.chunk_id, e.vector
    FROM message_chunk_embeddings e JOIN chunks cc ON cc.id = e.chunk_id
    WHERE e.vector IS NOT NULL";

/// Restricts [`MESSAGE_CHUNK_VECTORS`] to the messages of session `?1`.
const IN_SESSION: &str = "
    AND cc.message_id IN (
        SELECT m.id FROM sessions s JOIN messages m ON m.session_id = s.id
        WHERE s.uuid = ?1)";

/// Restricts [`MESSAGE_CHUNK_VECTORS`] to the messages outside session `?1`.
const OUTSIDE_SESSION: &str = "
    AND cc.message_id NOT IN (
        SELECT m.id FROM sessions s JOIN messages m ON m.session_id = s.id
        WHERE s.uuid = ?1)";

/// Restricts [`MESSAGE_CHUNK_VECTORS`] to the messages outside session `?1`'s live
/// context: other sessions' messages, and those its current branch's summary folded.
const OUTSIDE_SESSION_CONTEXT: &str = "
    AND cc.message_id NOT IN (
        SELECT m.id FROM sessions s JOIN messages m ON m.session_id = s.id
        WHERE s.uuid = ?1
          AND m.id NOT IN (
              SELECT bm.message_id
              FROM branch_messages bm
              JOIN branch_summaries bs ON bs.branch_id = bm.branch_id
              WHERE bm.branch_id = s.current_branch_id AND bm.seq < bs.kept_from_seq))";

/// Every document chunk's quantised vector, grouped by its document.
const DOCUMENT_CHUNK_VECTORS: &str = "
    SELECT cc.document_id, e.chunk_id, e.vector
    FROM document_chunk_embeddings e JOIN chunks cc ON cc.id = e.chunk_id
    WHERE e.vector IS NOT NULL";

/// Every memory vector, each its own group.
const ALL_MEMORY_VECTORS: &str =
    "SELECT memory_id, memory_id, vector FROM memory_embeddings WHERE vector IS NOT NULL";

/// Every table holding vectors.
const EMBEDDING_TABLES: [&str; 3] = [
    "message_chunk_embeddings",
    "document_chunk_embeddings",
    "memory_embeddings",
];

/// Holds for a chunk `cc` that is embedded or marked unembeddable.
const CHUNK_HAS_EMBEDDING: &str = "(
    EXISTS (SELECT 1 FROM message_chunk_embeddings e WHERE e.chunk_id = cc.id)
    OR EXISTS (SELECT 1 FROM document_chunk_embeddings e WHERE e.chunk_id = cc.id))";

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

/// The largest count of steps a quantised vector component holds, either side of zero.
const QUANTISED_STEPS: f32 = 127.0;

/// Scores the stored vector BLOB against the query; `None` for a BLOB to skip.
type ScoreBlob = fn(&[f32], &[u8]) -> Option<f32>;

/// SQLite-backed [`SemanticStore`]: a linear scan of one source kind's vectors into a
/// bounded top-K, then one batched lookup of the winners. Document vectors are stored
/// quantised to one byte per dimension, so their similarities are approximate.
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
        scope: ChunkScope,
    ) -> Result<Vec<EmbeddingHit>> {
        if top_k == 0 || query_vector.is_empty() {
            return Ok(Vec::new());
        }
        match scope {
            ChunkScope::Chats(sessions) => {
                self.handle
                    .search("nearest_chunks", move |c| {
                        nearest_message_chunks(c, &query_vector, top_k, sessions)
                    })
                    .await
            }
            ChunkScope::Documents => {
                self.handle
                    .search_documents("nearest_chunks", move |c| {
                        nearest_document_chunks(c, &query_vector, top_k)
                    })
                    .await
            }
        }
    }

    async fn adjacent_chunks(&self, locator: &ChunkLocator) -> Result<AdjacentChunks> {
        let locator = locator.clone();
        self.handle
            .read("adjacent_chunks", move |c| {
                Ok(AdjacentChunks {
                    before: adjacent_chunk(c, &locator, Side::Before)?,
                    after: adjacent_chunk(c, &locator, Side::After)?,
                })
            })
            .await
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
            .search("nearest_memories", move |c| {
                let ranked = scan_top_k::<MemoryId, MemoryId>(
                    c,
                    ALL_MEMORY_VECTORS,
                    [],
                    &query_vector,
                    score_against,
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
    }

    async fn adopt_embedding_space(
        &self,
        space: &EmbeddingSpace,
    ) -> Result<Option<EmbeddingSpace>> {
        let space = space.clone();
        self.handle
            .write("adopt_embedding_space", move |tx| {
                let recorded = recorded_space(tx)?;
                if recorded.as_ref() == Some(&space) {
                    return Ok(None);
                }
                recreate_embedding_tables(tx)?;
                record_space(tx, &space)?;
                Ok(recorded)
            })
            .await
    }

    async fn count_missing(&self) -> Result<UnindexedRows> {
        self.handle
            .read("count_missing", |c| {
                let chunks = c.query_row(
                    &format!("SELECT count(*) FROM chunks cc WHERE NOT {CHUNK_HAS_EMBEDDING}"),
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
    }

    async fn memories_missing_embedding(
        &self,
        after: Option<MemoryId>,
        limit: usize,
    ) -> Result<Vec<(MemoryId, String)>> {
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        self.handle
            .read("memories_missing_embedding", move |c| {
                c.prepare_cached(
                    "SELECT mem.id, mem.value
                     FROM memories mem
                     WHERE mem.id > ifnull(?1, 0) AND NOT EXISTS (
                         SELECT 1 FROM memory_embeddings e WHERE e.memory_id = mem.id
                     )
                     ORDER BY mem.id
                     LIMIT ?2",
                )?
                .query_map(rusqlite::params![after, limit], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })?
                .collect()
            })
            .await
    }

    async fn chunks_missing_embedding(
        &self,
        after: Option<ChunkId>,
        limit: usize,
    ) -> Result<Vec<(ChunkId, String)>> {
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        self.handle
            .read("chunks_missing_embedding", move |c| {
                c.prepare_cached(&format!(
                    "SELECT cc.id,
                            CAST(substr(CAST(coalesce(msg.content, doc.content) AS BLOB),
                                        cc.byte_start + 1, cc.byte_end - cc.byte_start) AS TEXT)
                     FROM chunks cc
                     LEFT JOIN messages msg ON msg.id = cc.message_id
                     LEFT JOIN documents doc ON doc.id = cc.document_id
                     WHERE cc.id > ifnull(?1, 0) AND NOT {CHUNK_HAS_EMBEDDING}
                     ORDER BY cc.id
                     LIMIT ?2"
                ))?
                .query_map(rusqlite::params![after, limit], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })?
                .collect()
            })
            .await
    }

    async fn store_embeddings(&self, embeddings: Vec<Embedding>) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }
        self.handle
            .write("store_embeddings", move |tx| {
                embeddings
                    .iter()
                    .try_for_each(|embedding| upsert_embedding(tx, embedding).map(drop))
            })
            .await
    }

    async fn clear_unembeddable(&self) -> Result<u64> {
        self.handle
            .write("clear_unembeddable", |tx| {
                let cleared = EMBEDDING_TABLES
                    .iter()
                    .map(|table| {
                        tx.execute(&format!("DELETE FROM {table} WHERE vector IS NULL"), [])
                    })
                    .sum::<rusqlite::Result<usize>>()?;
                Ok(u64::try_from(cleared).unwrap_or(u64::MAX))
            })
            .await
    }

    async fn begin_index_pass(&self, root: &Path) -> Result<IndexPass> {
        let root = root.to_path_buf();
        self.handle
            .write("begin_index_pass", move |tx| {
                documents::begin_pass(tx, &root)
            })
            .await
    }

    async fn document_unchanged(
        &self,
        pass: IndexPass,
        path: &Path,
        stamp: FileStamp,
    ) -> Result<bool> {
        let path = path.to_path_buf();
        self.handle
            .write("document_unchanged", move |tx| {
                documents::mark_seen_if_unchanged(tx, pass, &path, stamp)
            })
            .await
    }

    async fn index_document(
        &self,
        pass: IndexPass,
        path: PathBuf,
        stamp: FileStamp,
        content: String,
    ) -> Result<Vec<ChunkId>> {
        self.handle
            .write("index_document", move |tx| {
                documents::store_document(tx, pass, &path, stamp, &content)
            })
            .await
    }

    async fn finish_index_pass(&self, pass: IndexPass) -> Result<u64> {
        self.handle
            .write("finish_index_pass", move |tx| {
                documents::remove_unseen(tx, pass)
            })
            .await
    }

    async fn remove_document(&self, path: &Path) -> Result<bool> {
        let path = path.to_path_buf();
        self.handle
            .write("remove_document", move |tx| {
                documents::remove_document(tx, &path)
            })
            .await
    }

    async fn remove_index_root(&self, root: &Path) -> Result<u64> {
        let root = root.to_path_buf();
        self.handle
            .write("remove_index_root", move |tx| {
                documents::remove_root(tx, &root)
            })
            .await
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

/// The `top_k` best `(group, id, vector)` rows of `sql` bound to `params`, their
/// vectors scored against `query` by `score`, one per group, best first.
fn scan_top_k<Group: FromSql + Eq + Hash, Id: FromSql>(
    conn: &rusqlite::Connection,
    sql: &str,
    params: impl Params,
    query: &[f32],
    score: ScoreBlob,
    top_k: usize,
) -> rusqlite::Result<Vec<(Id, f32)>> {
    let mut stmt = conn.prepare_cached(sql)?;
    let mut best = TopK::new(top_k);
    let mut rows = stmt.query(params)?;
    while let Some(row) = rows.next()? {
        let bytes = row.get_ref(2)?.as_blob()?;
        if let Some(similarity) = score(query, bytes) {
            best.offer(row.get::<_, Group>(0)?, row.get(1)?, similarity);
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
    let ranked = match scope_clause(sessions) {
        None => scan_top_k::<MessageId, ChunkId>(
            conn,
            MESSAGE_CHUNK_VECTORS,
            [],
            query,
            score_against,
            top_k,
        )?,
        Some((clause, session)) => scan_top_k::<MessageId, ChunkId>(
            conn,
            &format!("{MESSAGE_CHUNK_VECTORS}{clause}"),
            [session],
            query,
            score_against,
            top_k,
        )?,
    };
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
    let ranked = scan_top_k::<DocumentId, ChunkId>(
        conn,
        DOCUMENT_CHUNK_VECTORS,
        [],
        query,
        score_against_quantised,
        top_k,
    )?;
    let select = format!(
        "SELECT {DOCUMENT_CHUNK_COLUMNS}
         FROM chunks cc
         JOIN documents d ON d.id = cc.document_id
         WHERE cc.id IN (SELECT value FROM json_each(?1))"
    );
    hydrate_in_rank_order(conn, &select, ranked, document_chunk_hit, set_similarity)
}

/// The clause restricting message chunk vectors to `scope` and the session it binds
/// to `?1`; `None` when every message is in scope.
fn scope_clause(scope: SessionScope) -> Option<(&'static str, SessionId)> {
    match scope {
        SessionScope::All => None,
        SessionScope::AllExcept(session) => Some((OUTSIDE_SESSION, session)),
        SessionScope::AllExceptInContext(session) => Some((OUTSIDE_SESSION_CONTEXT, session)),
        SessionScope::Only(session) => Some((IN_SESSION, session)),
    }
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
    let similarity = dot(query, words, |word| f32::from_le_bytes(*word));
    similarity.is_finite().then_some(similarity)
}

/// Approximate dot product of `query` against the quantised BLOB `bytes`, or `None`
/// when the BLOB is malformed, its dimension differs, or the product is not finite.
fn score_against_quantised(query: &[f32], bytes: &[u8]) -> Option<f32> {
    let (step, steps) = bytes.split_first_chunk::<4>()?;
    if steps.len() != query.len() {
        return None;
    }
    let similarity =
        f32::from_le_bytes(*step) * dot(query, steps, |byte| f32::from(i8::from_le_bytes([*byte])));
    similarity.is_finite().then_some(similarity)
}

/// Dot product of `query` against `stored`, each stored component decoded by `component`.
fn dot<T>(query: &[f32], stored: &[T], component: impl Fn(&T) -> f32) -> f32 {
    let (stored_blocks, stored_tail) = stored.as_chunks::<SCORE_LANES>();
    let (query_blocks, query_tail) = query.as_chunks::<SCORE_LANES>();
    let mut lanes = [0.0f32; SCORE_LANES];
    for (stored_block, query_block) in stored_blocks.iter().zip(query_blocks) {
        for ((lane, value), q) in lanes.iter_mut().zip(stored_block).zip(query_block) {
            *lane += component(value) * q;
        }
    }
    let tail: f32 = stored_tail
        .iter()
        .zip(query_tail)
        .map(|(value, q)| component(value) * q)
        .sum();
    lanes.iter().sum::<f32>() + tail
}

/// The embedding space the stored vectors are in; `None` before one is adopted.
fn recorded_space(tx: &Transaction<'_>) -> rusqlite::Result<Option<EmbeddingSpace>> {
    tx.query_row(
        "SELECT model, dim, server_args FROM embedding_meta",
        [],
        |row| {
            Ok(EmbeddingSpace {
                model: row.get(0)?,
                dim: row.get(1)?,
                server_args: row.get(2)?,
            })
        },
    )
    .optional()
}

fn record_space(tx: &Transaction<'_>, space: &EmbeddingSpace) -> rusqlite::Result<usize> {
    tx.execute(
        "INSERT INTO embedding_meta (id, model, dim, server_args) VALUES (0, ?1, ?2, ?3)
         ON CONFLICT(id) DO UPDATE
         SET model = excluded.model, dim = excluded.dim, server_args = excluded.server_args",
        rusqlite::params![space.model, space.dim, space.server_args],
    )
}

/// Empty every table in [`EMBEDDING_TABLES`] by dropping it and creating it again
/// from its own definition, which costs nothing per row.
fn recreate_embedding_tables(tx: &Transaction<'_>) -> rusqlite::Result<()> {
    EMBEDDING_TABLES.iter().try_for_each(|table| {
        let definition: String = tx.query_row(
            "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = ?1",
            [table],
            |row| row.get(0),
        )?;
        tx.execute_batch(&format!("DROP TABLE {table}; {definition};"))
    })
}

/// Upsert `embedding` into its target's table, with a `NULL` vector when it has
/// none. The `SELECT` makes a since-deleted
/// target, or a memory whose value has changed since, a no-op rather than an error
/// or a vector for the wrong text.
fn upsert_embedding(tx: &Transaction<'_>, embedding: &Embedding) -> rusqlite::Result<usize> {
    let vector = embedding.vector.as_deref();
    match embedding.target {
        EmbeddingTarget::Chunk(id) => upsert_chunk_embedding(tx, id, vector),
        EmbeddingTarget::Memory(id) => tx
            .prepare_cached(
                "INSERT INTO memory_embeddings (memory_id, vector)
                 SELECT id, ?2 FROM memories WHERE id = ?1 AND value = ?3
                 ON CONFLICT(memory_id) DO UPDATE SET vector = excluded.vector",
            )?
            .execute(rusqlite::params![
                id,
                vector.map(vector_to_blob),
                embedding.text
            ]),
    }
}

/// Upsert `vector` for `chunk` into the table of its owner's kind: exact for a
/// message chunk, quantised for a document chunk.
fn upsert_chunk_embedding(
    tx: &Transaction<'_>,
    chunk: ChunkId,
    vector: Option<&[f32]>,
) -> rusqlite::Result<usize> {
    let stored_for_message = tx
        .prepare_cached(
            "INSERT INTO message_chunk_embeddings (chunk_id, vector)
             SELECT id, ?2 FROM chunks WHERE id = ?1 AND message_id IS NOT NULL
             ON CONFLICT(chunk_id) DO UPDATE SET vector = excluded.vector",
        )?
        .execute(rusqlite::params![chunk, vector.map(vector_to_blob)])?;
    if stored_for_message > 0 {
        return Ok(stored_for_message);
    }
    tx.prepare_cached(
        "INSERT INTO document_chunk_embeddings (chunk_id, vector)
         SELECT id, ?2 FROM chunks WHERE id = ?1 AND document_id IS NOT NULL
         ON CONFLICT(chunk_id) DO UPDATE SET vector = excluded.vector",
    )?
    .execute(rusqlite::params![chunk, vector.map(quantised_blob)])
}

/// Encode `vector` as the little-endian `f32` BLOB the message chunk and memory
/// embedding tables store.
fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

/// Encode `vector` as the BLOB `document_chunk_embeddings` stores: a little-endian
/// `f32` step sized so the largest component is [`QUANTISED_STEPS`] steps, then each
/// component as an `i8` count of steps.
fn quantised_blob(vector: &[f32]) -> Vec<u8> {
    let largest = vector
        .iter()
        .fold(0.0f32, |largest, value| largest.max(value.abs()));
    let step = largest / QUANTISED_STEPS;
    let mut blob = Vec::with_capacity(4 + vector.len());
    blob.extend_from_slice(&step.to_le_bytes());
    blob.extend(
        vector
            .iter()
            .flat_map(|value| steps_in(*value, step).to_le_bytes()),
    );
    blob
}

/// `value` as a whole count of `step`s; zero when the count is not a number.
fn steps_in(value: f32, step: f32) -> i8 {
    (value / step).round().to_i8().unwrap_or(0)
}

#[cfg(test)]
mod tests;

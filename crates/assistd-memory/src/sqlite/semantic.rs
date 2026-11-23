//! SQLite-backed [`SemanticStore`](crate::SemanticStore) over the chunks cut from messages,
//! memories and documents: one vector table for the texts of each kind, and one
//! keyword index of them all.

use std::collections::HashMap;
use std::ffi::OsString;
use std::fmt;
use std::hash::Hash;
use std::ops::Range;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use num_traits::ToPrimitive;
use rusqlite::types::FromSql;
use rusqlite::{OptionalExtension, Params, Row, Transaction};
use tokio::sync::Notify;

use crate::{
    AdjacentChunks, ChunkId, ChunkLocator, ChunkSource, ContextChunk, DocumentHit, DocumentId,
    DocumentStore, DocumentText, Embedding, EmbeddingHit, EmbeddingQueue, EmbeddingSpace,
    EmptyPages, FileStamp, IndexPass, IndexRootCounts, MemoryHit, MemoryId, MessageId, PageReading,
    RecallStore, Result, SessionId, SessionScope, SkippedRead, TranscriptionCounts, UnindexedFile,
    UnindexedRows, WaitingPage,
};

use crate::chunking::text_hash;

use super::chunks::{CHUNK_SOURCES, kind_code, vector_table};
use super::connection::SqliteHandle;
use super::timestamp::timestamp_at;

mod document_search;
mod documents;

/// Every message chunk with the vector of its text, grouped by its message.
const MESSAGE_CHUNK_VECTORS: &str = "
    SELECT cc.source, cc.id, v.vector
    FROM chunks cc JOIN message_vectors v ON v.hash = cc.hash
    WHERE cc.kind = 0 AND v.vector IS NOT NULL";

/// Restricts [`MESSAGE_CHUNK_VECTORS`] to the messages of session `?1`.
const IN_SESSION: &str = "
    AND cc.source IN (
        SELECT m.id FROM sessions s JOIN messages m ON m.session_id = s.id
        WHERE s.uuid = ?1)";

/// Restricts [`MESSAGE_CHUNK_VECTORS`] to the messages outside session `?1`.
const OUTSIDE_SESSION: &str = "
    AND cc.source NOT IN (
        SELECT m.id FROM sessions s JOIN messages m ON m.session_id = s.id
        WHERE s.uuid = ?1)";

/// Restricts [`MESSAGE_CHUNK_VECTORS`] to the messages outside session `?1`'s live
/// context: other sessions' messages, and those its current branch's summary folded.
const OUTSIDE_SESSION_CONTEXT: &str = "
    AND cc.source NOT IN (
        SELECT m.id FROM sessions s JOIN messages m ON m.session_id = s.id
        WHERE s.uuid = ?1
          AND m.id NOT IN (
              SELECT bm.message_id
              FROM branch_messages bm
              JOIN branch_summaries bs ON bs.branch_id = bm.branch_id
              WHERE bm.branch_id = s.current_branch_id AND bm.seq < bs.kept_from_seq))";

/// The vector of every text the document chunks hold, each text its own group.
const DOCUMENT_TEXT_VECTORS: &str =
    "SELECT hash, hash, vector FROM document_vectors WHERE vector IS NOT NULL";

/// Every memory's chunk with the vector of its text, each memory its own group.
const MEMORY_VECTORS: &str = "
    SELECT cc.source, cc.source, v.vector
    FROM chunks cc JOIN memory_vectors v ON v.hash = cc.hash
    WHERE cc.kind = 1 AND v.vector IS NOT NULL";

/// Every index root `r` with how many documents it holds, the chunks they are cut into,
/// and those chunks whose text is neither embedded nor marked unembeddable.
const INDEX_ROOT_COUNTS: &str = "
    SELECT r.path, count(DISTINCT d.id), count(cc.id), count(cc.id) - count(v.hash)
    FROM index_roots r
    LEFT JOIN documents d ON d.root_id = r.id
    LEFT JOIN chunks cc ON cc.kind = 2 AND cc.source = d.id
    LEFT JOIN document_vectors v ON v.hash = cc.hash
    GROUP BY r.id ORDER BY r.id";

/// A message chunk row's columns, read by [`message_context_chunk`]: chunk id, message
/// id, session uuid, message time, role, session title, the session's opening user text,
/// chunk text.
const MESSAGE_CHUNK_COLUMNS: &str = "
    cc.id, m.id, s.uuid, m.created_at, m.role, s.title,
    (SELECT substr(f.content, 1, 256) FROM messages f
     WHERE f.session_id = s.id AND f.role = 0 ORDER BY f.id LIMIT 1),
    CAST(substr(CAST(m.content AS BLOB), cc.byte_start + 1,
                cc.byte_end - cc.byte_start) AS TEXT)";

/// A document chunk row's columns: chunk id, document id, path, page, page count,
/// byte range, index time, chunk text.
const DOCUMENT_CHUNK_COLUMNS: &str = "
    cc.id, d.id, d.path, cc.page,
    (SELECT count(*) FROM document_pages n WHERE n.document_id = d.id),
    cc.byte_start, cc.byte_end, d.indexed_at,
    CAST(substr(CAST(p.content AS BLOB), cc.byte_start + 1,
                cc.byte_end - cc.byte_start) AS TEXT)";

/// Joins a document chunk `cc` to its document `d` and its page `p`.
const DOCUMENT_CHUNK_OWNERS: &str = "
    JOIN documents d ON d.id = cc.source
    JOIN document_pages p ON p.document_id = cc.source AND p.page = cc.page";

/// Characters of an untitled session's opening user message kept in its label.
const SESSION_LABEL_CHARS: usize = 60;

/// Independent partial sums in the dot product, so the compiler can vectorise it.
const SCORE_LANES: usize = 8;

/// The largest count of steps a quantised vector component holds, either side of zero.
const QUANTISED_STEPS: f32 = 127.0;

/// SQLite-backed [`SemanticStore`](crate::SemanticStore): a linear scan of one source kind's vectors into a
/// bounded top-K, then one batched lookup of the winners. Vectors are stored
/// quantised, so similarities are approximate; documents rank by keyword as well.
#[derive(Debug, Clone)]
pub struct SqliteSemanticStore {
    handle: SqliteHandle,
}

impl SqliteSemanticStore {
    /// Store over the shared database handle.
    pub fn new(handle: SqliteHandle) -> Self {
        Self { handle }
    }

    /// Wake the embedding worker when `unembedded` holds chunks to embed.
    fn signal_chunks(&self, unembedded: &[ChunkId]) {
        if !unembedded.is_empty() {
            self.handle.rows_to_embed().notify_one();
        }
    }
}

#[async_trait]
impl RecallStore for SqliteSemanticStore {
    async fn nearest_chunks(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        scope: SessionScope,
    ) -> Result<Vec<EmbeddingHit>> {
        if top_k == 0 || query_vector.is_empty() {
            return Ok(Vec::new());
        }
        self.handle
            .search("nearest_chunks", move |c| {
                nearest_message_chunks(c, &query_vector, top_k, scope)
            })
            .await
    }

    async fn search_documents(
        &self,
        query: String,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<DocumentHit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        self.handle
            .search_documents("search_documents", move |c| {
                document_search::search(c, &query, &query_vector, top_k)
            })
            .await
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
                    MEMORY_VECTORS,
                    [],
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
    }

    async fn document_text(
        &self,
        path: &Path,
        page: usize,
        bytes: Range<usize>,
    ) -> Result<Option<DocumentText>> {
        let path = path.to_path_buf();
        self.handle
            .read("document_text", move |c| {
                c.prepare_cached(
                    "SELECT substr(CAST(p.content AS BLOB), ?3 + 1, ?4),
                            ifnull(length(CAST(p.content AS BLOB)), 0),
                            (SELECT count(*) FROM document_pages n WHERE n.document_id = d.id),
                            d.indexed_at
                     FROM documents d
                     LEFT JOIN document_pages p ON p.document_id = d.id AND p.page = ?2
                     WHERE d.path = ?1",
                )?
                .query_row(
                    rusqlite::params![path.as_os_str().as_bytes(), page, bytes.start, bytes.len()],
                    |row| {
                        let within = row.get_ref(0)?.as_blob_or_null()?.unwrap_or_default();
                        let (skipped, text) = whole_characters(within);
                        let start = bytes.start + skipped;
                        Ok(DocumentText {
                            text: text.to_owned(),
                            page,
                            page_count: row.get(2)?,
                            bytes: start..start + text.len(),
                            page_bytes: row.get(1)?,
                            indexed_at: timestamp_at(row, 3)?,
                        })
                    },
                )
                .optional()
            })
            .await
    }
}

#[async_trait]
impl EmbeddingQueue for SqliteSemanticStore {
    async fn adopt_embedding_space(
        &self,
        space: &EmbeddingSpace,
    ) -> Result<Option<EmbeddingSpace>> {
        let space = space.clone();
        let replaced = self
            .handle
            .write("adopt_embedding_space", move |tx| {
                let recorded = recorded_space(tx)?;
                if recorded.as_ref() == Some(&space) {
                    return Ok(None);
                }
                recreate_embedding_tables(tx)?;
                record_space(tx, &space)?;
                Ok(recorded)
            })
            .await?;
        if replaced.is_some() {
            self.handle.rows_to_embed().notify_one();
        }
        Ok(replaced)
    }

    fn rows_to_embed(&self) -> Arc<Notify> {
        Arc::clone(self.handle.rows_to_embed())
    }

    async fn count_missing(&self) -> Result<UnindexedRows> {
        self.handle
            .read("count_missing", |c| {
                let count_chunks = |source| {
                    let missing = ChunkKind::of(source).missing_embedding();
                    c.query_row(&format!("SELECT count(*) {missing}"), [], |r| r.get(0))
                };
                Ok(UnindexedRows {
                    message_chunks: count_chunks(ChunkSource::Message)?,
                    document_chunks: count_chunks(ChunkSource::Document)?,
                    memories: count_chunks(ChunkSource::Memory)?,
                })
            })
            .await
    }

    async fn chunks_missing_embedding(
        &self,
        source: ChunkSource,
        after: Option<ChunkId>,
        limit: usize,
    ) -> Result<Vec<(ChunkId, String)>> {
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let kind = ChunkKind::of(source);
        let (missing, text) = (kind.missing_embedding(), kind.text_column);
        self.handle
            .read("chunks_missing_embedding", move |c| {
                c.prepare_cached(&format!(
                    "SELECT cc.id,
                            CAST(substr(CAST(src.{text} AS BLOB),
                                        cc.byte_start + 1, cc.byte_end - cc.byte_start) AS TEXT)
                     {missing} AND cc.id > ifnull(?1, 0)
                       AND NOT EXISTS (SELECT 1 FROM chunks earlier
                                       WHERE earlier.hash = cc.hash AND earlier.id < cc.id
                                         AND earlier.kind = cc.kind)
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
                    .try_for_each(|embedding| upsert_text_vector(tx, embedding).map(drop))
            })
            .await
    }

    async fn clear_unembeddable(&self) -> Result<u64> {
        let cleared = self
            .handle
            .write("clear_unembeddable", |tx| {
                let cleared = CHUNK_SOURCES
                    .map(vector_table)
                    .iter()
                    .map(|table| {
                        tx.execute(&format!("DELETE FROM {table} WHERE vector IS NULL"), [])
                    })
                    .sum::<rusqlite::Result<usize>>()?;
                Ok(u64::try_from(cleared).unwrap_or(u64::MAX))
            })
            .await?;
        if cleared > 0 {
            self.handle.rows_to_embed().notify_one();
        }
        Ok(cleared)
    }
}

#[async_trait]
impl DocumentStore for SqliteSemanticStore {
    async fn begin_index_pass(&self, root: &Path) -> Result<IndexPass> {
        let root = root.to_path_buf();
        self.handle
            .write("begin_index_pass", move |tx| {
                documents::begin_pass(tx, &root)
            })
            .await
    }

    async fn unindexed_files(
        &self,
        pass: IndexPass,
        files: Vec<(PathBuf, FileStamp)>,
    ) -> Result<Vec<UnindexedFile>> {
        self.handle
            .write("unindexed_files", move |tx| {
                let mut unindexed = Vec::new();
                for (path, stamp) in files {
                    if documents::mark_seen_if_unchanged(tx, pass, &path, stamp)? {
                        continue;
                    }
                    let skipped = documents::mark_seen_if_skipped(tx, pass, &path, stamp)?;
                    unindexed.push(UnindexedFile {
                        path,
                        stamp,
                        skipped,
                    });
                }
                Ok(unindexed)
            })
            .await
    }

    async fn record_skipped_file(
        &self,
        pass: IndexPass,
        path: PathBuf,
        stamp: FileStamp,
        cause: SkippedRead,
    ) -> Result<()> {
        self.handle
            .write("record_skipped_file", move |tx| {
                documents::record_skipped(tx, pass, &path, stamp, cause)
            })
            .await
    }

    async fn forget_skipped_files(&self) -> Result<u64> {
        self.handle
            .write("forget_skipped_files", |tx| documents::forget_skipped(tx))
            .await
    }

    async fn index_document(
        &self,
        pass: IndexPass,
        path: PathBuf,
        stamp: FileStamp,
        pages: Vec<String>,
        empty_pages: EmptyPages,
    ) -> Result<Vec<ChunkId>> {
        let stored = self
            .handle
            .write("index_document", move |tx| {
                documents::store_document(tx, pass, &path, stamp, &pages, empty_pages)
            })
            .await?;
        self.signal_chunks(&stored.unembedded);
        if stored.pages_wait {
            self.handle.pages_to_transcribe().notify_one();
        }
        Ok(stored.unembedded)
    }

    fn pages_to_transcribe(&self) -> Arc<Notify> {
        Arc::clone(self.handle.pages_to_transcribe())
    }

    async fn next_waiting_page(&self) -> Result<Option<WaitingPage>> {
        self.handle
            .read("next_waiting_page", documents::next_waiting_page)
            .await
    }

    async fn store_page_reading(
        &self,
        page: WaitingPage,
        reading: PageReading,
    ) -> Result<Vec<ChunkId>> {
        let unembedded = self
            .handle
            .write("store_page_reading", move |tx| {
                documents::store_page_reading(tx, &page, &reading)
            })
            .await?;
        self.signal_chunks(&unembedded);
        Ok(unembedded)
    }

    async fn count_transcriptions(&self) -> Result<TranscriptionCounts> {
        self.handle
            .read("count_transcriptions", documents::count_transcriptions)
            .await
    }

    async fn retry_failed_transcriptions(&self) -> Result<u64> {
        let waiting = self
            .handle
            .write("retry_failed_transcriptions", |tx| {
                documents::retry_failed_transcriptions(tx)
            })
            .await?;
        if waiting > 0 {
            self.handle.pages_to_transcribe().notify_one();
        }
        Ok(waiting)
    }

    async fn finish_index_pass(&self, pass: IndexPass) -> Result<u64> {
        self.handle
            .write("finish_index_pass", move |tx| {
                documents::remove_unseen(tx, pass)
            })
            .await
    }

    async fn finish_index_pass_under(&self, pass: IndexPass, paths: Vec<PathBuf>) -> Result<u64> {
        self.handle
            .write("finish_index_pass_under", move |tx| {
                paths
                    .iter()
                    .map(|path| documents::remove_unseen_under(tx, pass, path))
                    .sum()
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

    async fn index_roots(&self) -> Result<Vec<PathBuf>> {
        self.handle
            .read("index_roots", |c| {
                c.prepare_cached("SELECT path FROM index_roots ORDER BY id")?
                    .query_map([], |r| Ok(PathBuf::from(OsString::from_vec(r.get(0)?))))?
                    .collect()
            })
            .await
    }

    async fn index_root_counts(&self) -> Result<Vec<IndexRootCounts>> {
        self.handle
            .read("index_root_counts", |c| {
                c.prepare_cached(INDEX_ROOT_COUNTS)?
                    .query_map([], |r| {
                        Ok(IndexRootCounts {
                            root: PathBuf::from(OsString::from_vec(r.get(0)?)),
                            documents: r.get(1)?,
                            chunks: r.get(2)?,
                            unembedded_chunks: r.get(3)?,
                        })
                    })?
                    .collect()
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

    async fn count_document_chunks(&self) -> Result<u64> {
        self.handle
            .read("count_document_chunks", |c| {
                c.query_row("SELECT count(*) FROM chunks WHERE kind = 2", [], |r| {
                    r.get(0)
                })
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

/// What differs between the chunks of each kind in the statements that read them.
struct ChunkKind {
    /// The `chunks.kind` code.
    code: i64,
    /// Joins a chunk `cc` to `src`, the row holding the text it is a range of.
    owner_join: &'static str,
    /// The column of `src` holding that text.
    text_column: &'static str,
    /// The table holding the vector of each text chunks of this kind hold.
    vectors: &'static str,
}

impl ChunkKind {
    const fn of(source: ChunkSource) -> Self {
        let (owner_join, text_column) = match source {
            ChunkSource::Message => ("JOIN messages src ON src.id = cc.source", "content"),
            ChunkSource::Memory => ("JOIN memories src ON src.id = cc.source", "value"),
            ChunkSource::Document => (
                "JOIN document_pages src ON src.document_id = cc.source AND src.page = cc.page",
                "content",
            ),
        };
        Self {
            code: kind_code(source),
            owner_join,
            text_column,
            vectors: vector_table(source),
        }
    }

    /// The `FROM` and `WHERE` clauses selecting the chunks `cc` of this kind whose
    /// text is neither embedded nor marked unembeddable, each with its owner `src`.
    fn missing_embedding(&self) -> String {
        let Self {
            code,
            owner_join,
            vectors,
            ..
        } = self;
        format!(
            "FROM chunks cc {owner_join}
             WHERE cc.kind = {code}
               AND NOT EXISTS (SELECT 1 FROM {vectors} v WHERE v.hash = cc.hash)"
        )
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
/// quantised vectors scored against `query`, one per group, best first.
fn scan_top_k<Group: FromSql + Eq + Hash, Id: FromSql>(
    conn: &rusqlite::Connection,
    sql: &str,
    params: impl Params,
    query: &[f32],
    top_k: usize,
) -> rusqlite::Result<Vec<(Id, f32)>> {
    let mut stmt = conn.prepare_cached(sql)?;
    let mut best = TopK::new(top_k);
    let mut rows = stmt.query(params)?;
    while let Some(row) = rows.next()? {
        let bytes = row.get_ref(2)?.as_blob()?;
        if let Some(similarity) = score_against_quantised(query, bytes) {
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
        None => scan_top_k::<MessageId, ChunkId>(conn, MESSAGE_CHUNK_VECTORS, [], query, top_k)?,
        Some((clause, session)) => scan_top_k::<MessageId, ChunkId>(
            conn,
            &format!("{MESSAGE_CHUNK_VECTORS}{clause}"),
            [session],
            query,
            top_k,
        )?,
    };
    let select = format!(
        "SELECT {MESSAGE_CHUNK_COLUMNS}
         FROM chunks cc
         JOIN messages m ON m.id = cc.source AND cc.kind = 0
         JOIN sessions s ON s.id = m.session_id
         WHERE cc.id IN (SELECT value FROM json_each(?1))"
    );
    hydrate_in_rank_order(conn, &select, ranked, message_chunk_hit, set_similarity)
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
        .query_map(
            rusqlite::params![json_id_array(ranked.iter().map(|(id, _)| id))],
            |row| Ok((row.get::<_, Id>(0)?, to_hit(row)?)),
        )?
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
            document_id,
            page,
            bytes,
            ..
        } => adjacent_document_chunk(conn, *document_id, (*page, bytes.start), side),
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
         JOIN chunks cc ON cc.kind = 0 AND cc.source = m.id
         JOIN sessions s ON s.id = m.session_id
         WHERE bm.seq {seq_test} here.seq AND m.role IN (0, 1)
         ORDER BY bm.seq {order}, cc.byte_start {order}
         LIMIT 1"
    );
    conn.prepare_cached(&sql)?
        .query_row(rusqlite::params![message], message_context_chunk)
        .optional()
}

/// The chunk of `document` starting nearest `start`, a page and a byte of it, on
/// `side`, on that page or another.
fn adjacent_document_chunk(
    conn: &rusqlite::Connection,
    document: DocumentId,
    start: (usize, usize),
    side: Side,
) -> rusqlite::Result<Option<ContextChunk>> {
    let (start_test, order) = side.test_and_order();
    let sql = format!(
        "SELECT {DOCUMENT_CHUNK_COLUMNS}
         FROM chunks cc {DOCUMENT_CHUNK_OWNERS}
         WHERE cc.kind = 2 AND cc.source = ?1
           AND (cc.page, cc.byte_start) {start_test} (?2, ?3)
         ORDER BY cc.page {order}, cc.byte_start {order}
         LIMIT 1"
    );
    conn.prepare_cached(&sql)?
        .query_row(
            rusqlite::params![document, start.0, start.1],
            document_context_chunk,
        )
        .optional()
}

fn message_chunk_hit(row: &Row<'_>) -> rusqlite::Result<EmbeddingHit> {
    message_context_chunk(row).map(unscored_hit)
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
            page: row.get(3)?,
            page_count: row.get(4)?,
            bytes: row.get(5)?..row.get(6)?,
            indexed_at: timestamp_at(row, 7)?,
        },
        text: row.get(8)?,
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

/// The whole characters of `bytes`, a slice of UTF-8 text cut anywhere, and how many
/// bytes of a character cut at its start come before them.
fn whole_characters(bytes: &[u8]) -> (usize, &str) {
    let skipped = bytes
        .iter()
        .take_while(|&&byte| byte & 0xC0 == 0x80)
        .count();
    let whole = bytes[skipped..]
        .utf8_chunks()
        .next()
        .map_or("", |chunk| chunk.valid());
    (skipped, whole)
}

fn memory_hit(row: &Row<'_>) -> rusqlite::Result<MemoryHit> {
    Ok(MemoryHit {
        memory_id: row.get(0)?,
        key: row.get(1)?,
        value: row.get(2)?,
        similarity: 0.0,
    })
}

/// `ids` as a JSON array of integers.
fn json_id_array<Id: fmt::Display>(ids: impl IntoIterator<Item = Id>) -> String {
    let ids: Vec<String> = ids.into_iter().map(|id| id.to_string()).collect();
    format!("[{}]", ids.join(","))
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
        "SELECT model, dim, server_args, document_prefix FROM embedding_meta",
        [],
        |row| {
            Ok(EmbeddingSpace {
                model: row.get(0)?,
                dim: row.get(1)?,
                server_args: row.get(2)?,
                document_prefix: row.get(3)?,
            })
        },
    )
    .optional()
}

fn record_space(tx: &Transaction<'_>, space: &EmbeddingSpace) -> rusqlite::Result<usize> {
    tx.execute(
        "INSERT INTO embedding_meta (id, model, dim, server_args, document_prefix)
         VALUES (0, ?1, ?2, ?3, ?4)
         ON CONFLICT(id) DO UPDATE
         SET model = excluded.model, dim = excluded.dim, server_args = excluded.server_args,
             document_prefix = excluded.document_prefix",
        rusqlite::params![
            space.model,
            space.dim,
            space.server_args,
            space.document_prefix
        ],
    )
}

/// Empty every vector table by dropping it and creating it again from its own
/// definition, which costs nothing per row.
fn recreate_embedding_tables(tx: &Transaction<'_>) -> rusqlite::Result<()> {
    CHUNK_SOURCES
        .map(vector_table)
        .iter()
        .try_for_each(|table| {
            let definition: String = tx.query_row(
                "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = ?1",
                [table],
                |row| row.get(0),
            )?;
            tx.execute_batch(&format!("DROP TABLE {table}; {definition};"))
        })
}

/// Upsert the vector of `embedding` as that of its text for the chunks cut from its
/// source, `NULL` when it has none; a no-op when none of them holds the text any more.
fn upsert_text_vector(tx: &Transaction<'_>, embedding: &Embedding) -> rusqlite::Result<usize> {
    let ChunkKind { code, vectors, .. } = ChunkKind::of(embedding.source);
    tx.prepare_cached(&format!(
        "INSERT INTO {vectors} (hash, vector)
         SELECT ?1, ?2
         WHERE EXISTS (SELECT 1 FROM chunks WHERE hash = ?1 AND kind = {code})
         ON CONFLICT(hash) DO UPDATE SET vector = excluded.vector"
    ))?
    .execute(rusqlite::params![
        text_hash(embedding.text.as_bytes()),
        embedding.vector.as_deref().map(quantised_blob)
    ])
}

/// Encode `vector` as the BLOB the vector tables store: a little-endian
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

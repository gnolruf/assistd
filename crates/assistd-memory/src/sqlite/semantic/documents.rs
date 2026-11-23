//! Writes to the document index: passes over index roots, each document's pages,
//! chunks and keyword index, and the pages that wait for transcription.

use std::ffi::OsString;
use std::ops::Range;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, Transaction};
use sha2::{Digest, Sha256};

use crate::chunking::{ChunkingConfig, chunk_ranges};
use crate::sqlite::chunks::{ChunkOwner, delete_unheld_vectors, insert_chunk};
use crate::sqlite::page_readings::{delete_unheld_readings, keep_reading, kept_reading};
use crate::sqlite::timestamp::now_millis;
use crate::{
    ChunkId, ChunkSource, DocumentId, EmptyPages, FileHash, FileStamp, IndexPass, PageReading,
    SkippedRead, TranscriptionCounts, UnheldRemoved, WaitingPage,
};

/// The byte after `/`: every path inside a directory sorts between the directory's
/// path followed by `/` and followed by this.
const PAST_SEPARATOR: &[u8] = b"0";

/// `page_transcriptions.state` of a page waiting for the OCR model.
const WAITING: i64 = 0;

/// `page_transcriptions.state` of a page the OCR model has read.
const READ: i64 = 1;

/// `page_transcriptions.state` of a page whose transcription failed.
const FAILED: i64 = 2;

/// A `documents` row as one pass read it from disk.
struct DocumentRow<'a> {
    pass: IndexPass,
    path: &'a [u8],
    stamp: FileStamp,
    hash: [u8; 32],
    /// The file's bytes, when a model may read its pages.
    file: Option<FileHash>,
    chunking: ChunkingConfig,
}

impl DocumentRow<'_> {
    /// Record the stamp and pass on the stored row when it already holds these pages
    /// cut with this chunking, returning whether it does. A document with pages to
    /// transcribe never does: its images may have changed under the same text.
    fn refresh_stamp_if_same_pages(&self, tx: &Transaction<'_>) -> rusqlite::Result<bool> {
        let refreshed = tx
            .prepare_cached(
                "UPDATE documents
                 SET root_id = ?1, seen_pass = ?2, mtime_ns = ?3, size = ?4, file_hash = ?10
                 WHERE path = ?5 AND content_hash = ?6
                   AND chunker = ?7 AND chunk_chars = ?8 AND overlap_chars = ?9
                   AND NOT EXISTS (SELECT 1 FROM page_transcriptions t
                                   WHERE t.document_id = documents.id)",
            )?
            .execute(rusqlite::params![
                self.pass.root,
                self.pass.pass,
                self.stamp.mtime_ns,
                self.stamp.size,
                self.path,
                self.hash,
                self.chunking.chunker(),
                self.chunking.chunk_chars(),
                self.chunking.overlap_chars(),
                self.file,
            ])?;
        Ok(refreshed > 0)
    }

    /// Insert the row, or overwrite the one stored for its path, which keeps its id.
    fn upsert(&self, tx: &Transaction<'_>) -> rusqlite::Result<DocumentId> {
        tx.prepare_cached(
            "INSERT INTO documents (root_id, seen_pass, mtime_ns, size, path, content_hash,
                                    chunker, chunk_chars, overlap_chars, indexed_at, file_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(path) DO UPDATE
             SET root_id = excluded.root_id, seen_pass = excluded.seen_pass,
                 mtime_ns = excluded.mtime_ns, size = excluded.size,
                 content_hash = excluded.content_hash, file_hash = excluded.file_hash,
                 chunker = excluded.chunker,
                 chunk_chars = excluded.chunk_chars, overlap_chars = excluded.overlap_chars,
                 indexed_at = excluded.indexed_at
             RETURNING id",
        )?
        .query_row(
            rusqlite::params![
                self.pass.root,
                self.pass.pass,
                self.stamp.mtime_ns,
                self.stamp.size,
                self.path,
                self.hash,
                self.chunking.chunker(),
                self.chunking.chunk_chars(),
                self.chunking.overlap_chars(),
                now_millis(),
                self.file,
            ],
            |r| r.get(0),
        )
    }
}

/// A page that had no text when its file was read, by its position from zero, and
/// whether a reading kept for its file gave it its text.
struct TextlessPage {
    page: usize,
    read: bool,
}

/// Where a chunk sits in a document: a byte range of one page's text.
struct PageCut {
    /// The page's position in the document, from zero.
    page: usize,
    bytes: Range<usize>,
}

/// Register `root` if it is new and start its next pass.
pub(super) fn begin_pass(conn: &Connection, root: &Path) -> rusqlite::Result<IndexPass> {
    conn.prepare_cached(
        "INSERT INTO index_roots (path, pass) VALUES (?1, 1)
         ON CONFLICT(path) DO UPDATE SET pass = pass + 1
         RETURNING id, pass",
    )?
    .query_row(rusqlite::params![root.as_os_str().as_bytes()], |r| {
        Ok(IndexPass {
            root: r.get(0)?,
            pass: r.get(1)?,
        })
    })
}

/// Mark the document at `path` seen by `pass` when it is stored with `stamp` and
/// the current chunking, returning whether it was.
pub(super) fn mark_seen_if_unchanged(
    conn: &Connection,
    pass: IndexPass,
    path: &Path,
    stamp: FileStamp,
) -> rusqlite::Result<bool> {
    let chunking = ChunkingConfig::DOCUMENTS;
    let marked = conn
        .prepare_cached(
            "UPDATE documents SET root_id = ?1, seen_pass = ?2
             WHERE path = ?3 AND mtime_ns = ?4 AND size = ?5
               AND chunker = ?6 AND chunk_chars = ?7 AND overlap_chars = ?8",
        )?
        .execute(rusqlite::params![
            pass.root,
            pass.pass,
            path.as_os_str().as_bytes(),
            stamp.mtime_ns,
            stamp.size,
            chunking.chunker(),
            chunking.chunk_chars(),
            chunking.overlap_chars(),
        ])?;
    Ok(marked > 0)
}

/// Mark the file at `path` seen by `pass` when it is recorded as skipped with
/// `stamp`, returning why it was.
pub(super) fn mark_seen_if_skipped(
    conn: &Connection,
    pass: IndexPass,
    path: &Path,
    stamp: FileStamp,
) -> rusqlite::Result<Option<SkippedRead>> {
    conn.prepare_cached(
        "UPDATE skipped_files SET root_id = ?1, seen_pass = ?2
         WHERE path = ?3 AND mtime_ns = ?4 AND size = ?5
         RETURNING cause",
    )?
    .query_row(
        rusqlite::params![
            pass.root,
            pass.pass,
            path.as_os_str().as_bytes(),
            stamp.mtime_ns,
            stamp.size,
        ],
        |r| r.get(0),
    )
    .optional()
}

/// Record the file at `path` as skipped for `cause` with `stamp`, seen by `pass`,
/// and remove the document stored at that path.
pub(super) fn record_skipped(
    tx: &Transaction<'_>,
    pass: IndexPass,
    path: &Path,
    stamp: FileStamp,
    cause: SkippedRead,
) -> rusqlite::Result<()> {
    let path = path.as_os_str().as_bytes();
    tx.prepare_cached("DELETE FROM documents WHERE path = ?1")?
        .execute(rusqlite::params![path])?;
    tx.prepare_cached(
        "INSERT INTO skipped_files (path, root_id, mtime_ns, size, cause, seen_pass)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(path) DO UPDATE
         SET root_id = excluded.root_id, mtime_ns = excluded.mtime_ns, size = excluded.size,
             cause = excluded.cause, seen_pass = excluded.seen_pass",
    )?
    .execute(rusqlite::params![
        path,
        pass.root,
        stamp.mtime_ns,
        stamp.size,
        cause,
        pass.pass
    ])?;
    Ok(())
}

/// Forget every skipped file, returning how many.
pub(super) fn forget_skipped(conn: &Connection) -> rusqlite::Result<u64> {
    let forgotten = conn
        .prepare_cached("DELETE FROM skipped_files")?
        .execute([])?;
    Ok(u64::try_from(forgotten).unwrap_or(u64::MAX))
}

/// What storing a document left for the workers that follow it.
#[derive(Default)]
pub(super) struct StoredDocument {
    /// Its chunks whose text has no vector.
    pub unembedded: Vec<ChunkId>,
    /// Whether any of its pages now waits for transcription.
    pub pages_wait: bool,
}

/// Store `pages` as the document at `path`, seen by `pass`, a page without text taking
/// the reading kept for it when `empty_pages` names its file. Pages with the stored hash
/// and chunking only have their stamp refreshed. A skip recorded for the path is dropped.
pub(super) fn store_document(
    tx: &Transaction<'_>,
    pass: IndexPass,
    path: &Path,
    stamp: FileStamp,
    mut pages: Vec<String>,
    empty_pages: EmptyPages,
) -> rusqlite::Result<StoredDocument> {
    tx.prepare_cached("DELETE FROM skipped_files WHERE path = ?1")?
        .execute(rusqlite::params![path.as_os_str().as_bytes()])?;
    let (file, textless) = match empty_pages {
        EmptyPages::Final => (None, Vec::new()),
        EmptyPages::AwaitTranscription { file } => {
            (Some(file), take_kept_readings(tx, file, &mut pages)?)
        }
    };
    let document = DocumentRow {
        pass,
        path: path.as_os_str().as_bytes(),
        stamp,
        hash: pages_hash(&pages),
        file,
        chunking: ChunkingConfig::DOCUMENTS,
    };
    if document.refresh_stamp_if_same_pages(tx)? {
        return Ok(StoredDocument::default());
    }
    let id = document.upsert(tx)?;
    replace_pages(tx, id, &pages)?;
    record_textless_pages(tx, id, &textless)?;
    let cuts = page_cuts(&pages, document.chunking);
    let unembedded = add_chunks(tx, id, &pages, &cuts)?;
    Ok(StoredDocument {
        unembedded,
        pages_wait: textless.iter().any(|page| !page.read),
    })
}

/// The first page waiting for transcription, by document and then position.
pub(super) fn next_waiting_page(conn: &Connection) -> rusqlite::Result<Option<WaitingPage>> {
    conn.prepare_cached(
        "SELECT t.document_id, d.path, t.page, d.mtime_ns, d.size
         FROM page_transcriptions t JOIN documents d ON d.id = t.document_id
         WHERE t.state = ?1
         ORDER BY t.document_id, t.page LIMIT 1",
    )?
    .query_row(rusqlite::params![WAITING], |r| {
        Ok(WaitingPage {
            document_id: r.get(0)?,
            path: PathBuf::from(OsString::from_vec(r.get(1)?)),
            page: r.get(2)?,
            stamp: FileStamp {
                mtime_ns: r.get(3)?,
                size: r.get(4)?,
            },
        })
    })
    .optional()
}

/// Record `reading` of `waiting` when it still waits and its document still has its
/// stamp, keeping its text for the document's file, and cut the document's uncut
/// pages once none of them waits. Returns the chunks added whose text has no vector.
pub(super) fn store_page_reading(
    tx: &Transaction<'_>,
    waiting: &WaitingPage,
    reading: &PageReading,
) -> rusqlite::Result<Vec<ChunkId>> {
    let (state, text) = match reading {
        PageReading::Text(text) => (READ, text.as_str()),
        PageReading::Failed => (FAILED, ""),
    };
    let WaitingPage {
        document_id,
        page,
        stamp,
        ..
    } = waiting;
    let recorded = tx
        .prepare_cached(
            "UPDATE page_transcriptions SET state = ?3
             WHERE document_id = ?1 AND page = ?2 AND state = ?4
               AND EXISTS (SELECT 1 FROM documents d
                           WHERE d.id = ?1 AND d.mtime_ns = ?5 AND d.size = ?6)",
        )?
        .execute(rusqlite::params![
            document_id,
            page,
            state,
            WAITING,
            stamp.mtime_ns,
            stamp.size
        ])?;
    if recorded == 0 {
        return Ok(Vec::new());
    }
    tx.prepare_cached(
        "UPDATE document_pages SET content = ?3 WHERE document_id = ?1 AND page = ?2",
    )?
    .execute(rusqlite::params![document_id, page, text])?;
    if let PageReading::Text(text) = reading {
        keep_reading(tx, *document_id, *page, text)?;
    }
    if has_waiting_page(tx, *document_id)? {
        return Ok(Vec::new());
    }
    cut_uncut_pages(tx, *document_id)
}

pub(super) fn count_transcriptions(conn: &Connection) -> rusqlite::Result<TranscriptionCounts> {
    conn.prepare_cached(
        "SELECT count(*) FILTER (WHERE state = ?1), count(*) FILTER (WHERE state = ?2)
         FROM page_transcriptions",
    )?
    .query_row(rusqlite::params![WAITING, FAILED], |r| {
        Ok(TranscriptionCounts {
            waiting: r.get(0)?,
            failed: r.get(1)?,
        })
    })
}

/// Remove the documents of `pass`'s root that it did not see, returning how many,
/// and the skipped files it did not see, and record that it walked the whole root.
pub(super) fn remove_unseen(conn: &Connection, pass: IndexPass) -> rusqlite::Result<u64> {
    conn.prepare_cached("UPDATE index_roots SET finished_pass = ?2 WHERE id = ?1")?
        .execute(rusqlite::params![pass.root, pass.pass])?;
    conn.prepare_cached("DELETE FROM skipped_files WHERE root_id = ?1 AND seen_pass < ?2")?
        .execute(rusqlite::params![pass.root, pass.pass])?;
    let removed = conn
        .prepare_cached("DELETE FROM documents WHERE root_id = ?1 AND seen_pass < ?2")?
        .execute(rusqlite::params![pass.root, pass.pass])?;
    Ok(u64::try_from(removed).unwrap_or(u64::MAX))
}

/// Remove the documents of `pass`'s root at or under `path` that it did not see,
/// returning how many, and the skipped files there it did not see. The `+` keeps the
/// planner on the index over `path`.
pub(super) fn remove_unseen_under(
    conn: &Connection,
    pass: IndexPass,
    path: &Path,
) -> rusqlite::Result<u64> {
    let path = path.as_os_str().as_bytes();
    let directory = path.strip_suffix(b"/").unwrap_or(path);
    let unseen_under = rusqlite::params![
        path,
        [directory, b"/"].concat(),
        [directory, PAST_SEPARATOR].concat(),
        pass.root,
        pass.pass
    ];
    conn.prepare_cached(
        "DELETE FROM skipped_files
         WHERE (path = ?1 OR (path > ?2 AND path < ?3))
           AND +root_id = ?4 AND seen_pass < ?5",
    )?
    .execute(unseen_under)?;
    let removed = conn
        .prepare_cached(
            "DELETE FROM documents
             WHERE (path = ?1 OR (path > ?2 AND path < ?3))
               AND +root_id = ?4 AND seen_pass < ?5",
        )?
        .execute(unseen_under)?;
    Ok(u64::try_from(removed).unwrap_or(u64::MAX))
}

/// Remove the document at `path`, returning whether one was stored, and any skip
/// recorded for it.
pub(super) fn remove_document(conn: &Connection, path: &Path) -> rusqlite::Result<bool> {
    let path = path.as_os_str().as_bytes();
    conn.prepare_cached("DELETE FROM skipped_files WHERE path = ?1")?
        .execute(rusqlite::params![path])?;
    let removed = conn
        .prepare_cached("DELETE FROM documents WHERE path = ?1")?
        .execute(rusqlite::params![path])?;
    Ok(removed > 0)
}

/// Remove index root `root` and its documents, returning how many documents.
pub(super) fn remove_root(tx: &Transaction<'_>, root: &Path) -> rusqlite::Result<u64> {
    let root = root.as_os_str().as_bytes();
    let removed = tx
        .prepare_cached(
            "DELETE FROM documents
             WHERE root_id = (SELECT id FROM index_roots WHERE path = ?1)",
        )?
        .execute(rusqlite::params![root])?;
    tx.prepare_cached("DELETE FROM index_roots WHERE path = ?1")?
        .execute(rusqlite::params![root])?;
    Ok(u64::try_from(removed).unwrap_or(u64::MAX))
}

/// Remove the document vectors whose text no document chunk holds and the readings
/// of files no document is stored from; none unless a pass has walked the whole of
/// each of `roots`.
pub(super) fn remove_unheld(
    conn: &Connection,
    roots: &[PathBuf],
) -> rusqlite::Result<UnheldRemoved> {
    let mut walked_whole = conn.prepare_cached(
        "SELECT EXISTS (SELECT 1 FROM index_roots WHERE path = ?1 AND finished_pass > 0)",
    )?;
    for root in roots {
        let root = root.as_os_str().as_bytes();
        if !walked_whole.query_row(rusqlite::params![root], |r| r.get::<_, bool>(0))? {
            return Ok(UnheldRemoved::default());
        }
    }
    let vectors = delete_unheld_vectors(conn, ChunkSource::Document)?;
    let readings = delete_unheld_readings(conn)?;
    Ok(UnheldRemoved {
        vectors: u64::try_from(vectors).unwrap_or(u64::MAX),
        readings: u64::try_from(readings).unwrap_or(u64::MAX),
    })
}

/// Have every page whose transcription failed wait again, returning how many.
pub(super) fn retry_failed_transcriptions(conn: &Connection) -> rusqlite::Result<u64> {
    let waiting = conn
        .prepare_cached("UPDATE page_transcriptions SET state = ?1 WHERE state = ?2")?
        .execute(rusqlite::params![WAITING, FAILED])?;
    Ok(u64::try_from(waiting).unwrap_or(u64::MAX))
}

/// Give each of `pages` without text the reading kept for that page of `file`,
/// returning every such page in order.
fn take_kept_readings(
    tx: &Transaction<'_>,
    file: FileHash,
    pages: &mut [String],
) -> rusqlite::Result<Vec<TextlessPage>> {
    let mut textless = Vec::new();
    for (page, text) in pages.iter_mut().enumerate() {
        if !text.trim().is_empty() {
            continue;
        }
        let kept = kept_reading(tx, file, page)?;
        textless.push(TextlessPage {
            page,
            read: kept.is_some(),
        });
        if let Some(kept) = kept {
            *text = kept;
        }
    }
    Ok(textless)
}

/// Record each of `textless`, stored pages of `document`, as read or as waiting for
/// transcription.
fn record_textless_pages(
    tx: &Transaction<'_>,
    document: DocumentId,
    textless: &[TextlessPage],
) -> rusqlite::Result<()> {
    let mut record = tx.prepare_cached(
        "INSERT INTO page_transcriptions (document_id, page, state) VALUES (?1, ?2, ?3)",
    )?;
    for TextlessPage { page, read } in textless {
        let state = if *read { READ } else { WAITING };
        record.execute(rusqlite::params![document, page, state])?;
    }
    Ok(())
}

fn has_waiting_page(tx: &Transaction<'_>, document: DocumentId) -> rusqlite::Result<bool> {
    tx.prepare_cached(
        "SELECT EXISTS (SELECT 1 FROM page_transcriptions
                        WHERE document_id = ?1 AND state = ?2)",
    )?
    .query_row(rusqlite::params![document, WAITING], |r| r.get(0))
}

/// Cut the pages stored for `document` that have no chunk and record the hash of all
/// its pages. Returns the chunks added whose text has no vector.
fn cut_uncut_pages(tx: &Transaction<'_>, document: DocumentId) -> rusqlite::Result<Vec<ChunkId>> {
    let (pages, uncut): (Vec<String>, Vec<bool>) = tx
        .prepare_cached(
            "SELECT p.content,
                    NOT EXISTS (SELECT 1 FROM chunks cc
                                WHERE cc.kind = ?2 AND cc.source = p.document_id
                                  AND cc.page = p.page)
             FROM document_pages p WHERE p.document_id = ?1 ORDER BY p.page",
        )?
        .query_map(rusqlite::params![document, ChunkSource::Document], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })?
        .collect::<rusqlite::Result<Vec<(String, bool)>>>()?
        .into_iter()
        .unzip();
    tx.prepare_cached("UPDATE documents SET content_hash = ?2 WHERE id = ?1")?
        .execute(rusqlite::params![document, pages_hash(&pages)])?;
    let mut cuts = page_cuts(&pages, ChunkingConfig::DOCUMENTS);
    cuts.retain(|cut| uncut[cut.page]);
    add_chunks(tx, document, &pages, &cuts)
}

/// The chunks each of `pages` is cut into under `chunking`, in page order.
fn page_cuts(pages: &[String], chunking: ChunkingConfig) -> Vec<PageCut> {
    pages
        .iter()
        .enumerate()
        .flat_map(|(page, text)| {
            chunk_ranges(text, chunking)
                .into_iter()
                .map(move |bytes| PageCut { page, bytes })
        })
        .collect()
}

/// SHA-256 of `pages` in order, each behind its length, so text that moves across
/// a page break changes the hash.
fn pages_hash(pages: &[String]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for page in pages {
        hash.update((page.len() as u64).to_le_bytes());
        hash.update(page.as_bytes());
    }
    hash.finalize().into()
}

/// Store `pages` as the pages of `document` in place of those it has, whose chunks
/// and transcriptions go with them.
fn replace_pages(
    tx: &Transaction<'_>,
    document: DocumentId,
    pages: &[String],
) -> rusqlite::Result<()> {
    tx.prepare_cached("DELETE FROM document_pages WHERE document_id = ?1")?
        .execute(rusqlite::params![document])?;
    let mut write = tx.prepare_cached(
        "INSERT INTO document_pages (document_id, page, content) VALUES (?1, ?2, ?3)",
    )?;
    for (page, content) in pages.iter().enumerate() {
        write.execute(rusqlite::params![document, page, content])?;
    }
    Ok(())
}

/// Store a chunk of `document` for each of `cuts` of its stored `pages`, with the
/// words of its text for keyword search, returning those whose text has no vector.
fn add_chunks(
    tx: &Transaction<'_>,
    document: DocumentId,
    pages: &[String],
    cuts: &[PageCut],
) -> rusqlite::Result<Vec<ChunkId>> {
    let mut has_vector =
        tx.prepare_cached("SELECT EXISTS (SELECT 1 FROM document_vectors WHERE hash = ?1)")?;
    let mut unembedded = Vec::new();
    for cut in cuts {
        let owner = ChunkOwner::Page(document, cut.page);
        let (chunk, hash) = insert_chunk(tx, owner, &pages[cut.page], &cut.bytes)?;
        if !has_vector.query_row(rusqlite::params![hash], |r| r.get::<_, bool>(0))? {
            unembedded.push(chunk);
        }
    }
    Ok(unembedded)
}

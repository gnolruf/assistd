//! Writes to the document index: passes over index roots, each document's pages,
//! chunks and keyword index, and the pages that wait for transcription.

use std::collections::HashMap;
use std::ffi::OsString;
use std::ops::Range;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, Transaction};
use sha2::{Digest, Sha256};

use crate::chunking::{ChunkingConfig, chunk_ranges};
use crate::sqlite::timestamp::now_millis;
use crate::{
    ChunkId, DocumentId, EmptyPages, FileStamp, IndexPass, PageReading, SkippedRead,
    TranscriptionCounts, WaitingPage,
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
    chunking: ChunkingConfig,
}

impl DocumentRow<'_> {
    /// Record the stamp and pass on the stored row when it already holds these pages
    /// cut with this chunking, returning whether it does. A document with pages to
    /// transcribe never does: its images may have changed under the same text.
    fn refresh_stamp_if_same_pages(&self, tx: &Transaction<'_>) -> rusqlite::Result<bool> {
        let refreshed = tx
            .prepare_cached(
                "UPDATE documents SET root_id = ?1, seen_pass = ?2, mtime_ns = ?3, size = ?4
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
            ])?;
        Ok(refreshed > 0)
    }

    /// Insert the row, or overwrite the one stored for its path, which keeps its id.
    fn upsert(&self, tx: &Transaction<'_>) -> rusqlite::Result<DocumentId> {
        tx.prepare_cached(
            "INSERT INTO documents (root_id, seen_pass, mtime_ns, size, path, content_hash,
                                    chunker, chunk_chars, overlap_chars, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(path) DO UPDATE
             SET root_id = excluded.root_id, seen_pass = excluded.seen_pass,
                 mtime_ns = excluded.mtime_ns, size = excluded.size,
                 content_hash = excluded.content_hash, chunker = excluded.chunker,
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
            ],
            |r| r.get(0),
        )
    }
}

/// Where a chunk sits in a document: a byte range of one page's text.
struct PageCut {
    /// The page's position in the document, from zero.
    page: usize,
    bytes: Range<usize>,
}

impl PageCut {
    /// The text this cut covers in `pages`; `None` when they have no such text.
    fn text_in<'t>(&self, pages: &'t [String]) -> Option<&'t str> {
        pages.get(self.page)?.get(self.bytes.clone())
    }
}

/// The pages and chunks stored for a document before it is indexed again; empty for
/// a document not stored yet.
struct StoredPages {
    pages: Vec<String>,
    chunks: Vec<(ChunkId, PageCut)>,
}

impl StoredPages {
    fn load(tx: &Transaction<'_>, document: DocumentId) -> rusqlite::Result<Self> {
        let pages = tx
            .prepare_cached(
                "SELECT content FROM document_pages WHERE document_id = ?1 ORDER BY page",
            )?
            .query_map(rusqlite::params![document], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let chunks = tx
            .prepare_cached(
                "SELECT id, page, byte_start, byte_end FROM chunks
                 WHERE document_id = ?1
                 ORDER BY page, byte_start",
            )?
            .query_map(rusqlite::params![document], |r| {
                let cut = PageCut {
                    page: r.get(1)?,
                    bytes: r.get(2)?..r.get(3)?,
                };
                Ok((r.get(0)?, cut))
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(Self { pages, chunks })
    }

    /// Match the stored chunks to `cuts` of the new `pages` by the text each covers,
    /// whatever page it is on. Chunks with the same text are matched in order of
    /// position.
    fn recut<'c>(&self, pages: &[String], cuts: &'c [PageCut]) -> Recut<'c> {
        let mut recut = Recut::default();
        let mut stored_by_text: HashMap<&str, Vec<ChunkId>> = HashMap::new();
        for (chunk, cut) in self.chunks.iter().rev() {
            match cut.text_in(&self.pages) {
                Some(text) => stored_by_text.entry(text).or_default().push(*chunk),
                None => recut.dropped.push(*chunk),
            }
        }
        for cut in cuts {
            let same_text = cut
                .text_in(pages)
                .and_then(|text| stored_by_text.get_mut(text))
                .and_then(Vec::pop);
            match same_text {
                Some(chunk) => recut.kept.push((chunk, cut)),
                None => recut.added.push(cut),
            }
        }
        recut.dropped.extend(stored_by_text.into_values().flatten());
        recut
    }
}

/// What becomes of a stored document's chunks when its new pages are cut.
#[derive(Default)]
struct Recut<'c> {
    /// Stored chunks whose text is unchanged, each with where that text now sits.
    kept: Vec<(ChunkId, &'c PageCut)>,
    /// Stored chunks whose text is gone.
    dropped: Vec<ChunkId>,
    /// Cuts whose text no stored chunk covered.
    added: Vec<&'c PageCut>,
}

impl Recut<'_> {
    /// Delete the dropped chunks of `document` and move the kept ones to their cuts,
    /// whose pages must be stored. Each is first shifted up by `clearance`, at least
    /// any old or new page's length, so none lands on a start another has yet to leave.
    fn apply(
        &self,
        tx: &Transaction<'_>,
        document: DocumentId,
        clearance: usize,
    ) -> rusqlite::Result<()> {
        let mut delete = tx.prepare_cached("DELETE FROM chunks WHERE id = ?1")?;
        for chunk in &self.dropped {
            delete.execute(rusqlite::params![chunk])?;
        }
        tx.prepare_cached(
            "UPDATE chunks SET byte_start = byte_start + ?2, byte_end = byte_end + ?2
             WHERE document_id = ?1",
        )?
        .execute(rusqlite::params![document, clearance])?;
        let mut place = tx.prepare_cached(
            "UPDATE chunks SET page = ?2, byte_start = ?3, byte_end = ?4 WHERE id = ?1",
        )?;
        for (chunk, cut) in &self.kept {
            place.execute(rusqlite::params![
                chunk,
                cut.page,
                cut.bytes.start,
                cut.bytes.end
            ])?;
        }
        Ok(())
    }
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

/// Store `pages` as the document at `path`, seen by `pass`, returning the chunks
/// added. Pages with the stored hash and chunking only have their stamp refreshed.
/// A skip recorded for the path is dropped.
pub(super) fn store_document(
    tx: &Transaction<'_>,
    pass: IndexPass,
    path: &Path,
    stamp: FileStamp,
    pages: &[String],
    empty_pages: EmptyPages,
) -> rusqlite::Result<Vec<ChunkId>> {
    tx.prepare_cached("DELETE FROM skipped_files WHERE path = ?1")?
        .execute(rusqlite::params![path.as_os_str().as_bytes()])?;
    let document = DocumentRow {
        pass,
        path: path.as_os_str().as_bytes(),
        stamp,
        hash: pages_hash(pages),
        chunking: ChunkingConfig::DOCUMENTS,
    };
    if document.refresh_stamp_if_same_pages(tx)? {
        return Ok(Vec::new());
    }
    let id = document.upsert(tx)?;
    let stored = StoredPages::load(tx, id)?;
    let cuts = page_cuts(pages, document.chunking);
    let recut = stored.recut(pages, &cuts);
    write_changed_pages(tx, id, &stored.pages, pages)?;
    recut.apply(tx, id, longest_page(&stored.pages).max(longest_page(pages)))?;
    remove_pages_from(tx, id, pages.len())?;
    reset_transcriptions(tx, id, pages, empty_pages)?;
    add_chunks(tx, id, pages, &recut.added)
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
/// stamp, and cut the document's pages again once none of them waits. Returns the
/// chunks added.
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
    if has_waiting_page(tx, *document_id)? {
        return Ok(Vec::new());
    }
    cut_stored_pages(tx, *document_id)
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
/// and the skipped files it did not see.
pub(super) fn remove_unseen(conn: &Connection, pass: IndexPass) -> rusqlite::Result<u64> {
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

/// Have every page whose transcription failed wait again, returning how many.
pub(super) fn retry_failed_transcriptions(conn: &Connection) -> rusqlite::Result<u64> {
    let waiting = conn
        .prepare_cached("UPDATE page_transcriptions SET state = ?1 WHERE state = ?2")?
        .execute(rusqlite::params![WAITING, FAILED])?;
    Ok(u64::try_from(waiting).unwrap_or(u64::MAX))
}

/// Forget what was transcribed of `document` and, when its empty pages are to be
/// transcribed, record each of `pages` without text as waiting.
fn reset_transcriptions(
    tx: &Transaction<'_>,
    document: DocumentId,
    pages: &[String],
    empty_pages: EmptyPages,
) -> rusqlite::Result<()> {
    tx.prepare_cached("DELETE FROM page_transcriptions WHERE document_id = ?1")?
        .execute(rusqlite::params![document])?;
    if empty_pages == EmptyPages::Final {
        return Ok(());
    }
    let mut wait =
        tx.prepare_cached("INSERT INTO page_transcriptions (document_id, page) VALUES (?1, ?2)")?;
    let empty = pages
        .iter()
        .enumerate()
        .filter(|(_, text)| text.trim().is_empty());
    for (page, _) in empty {
        wait.execute(rusqlite::params![document, page])?;
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

/// Cut the pages stored for `document` into chunks again, keeping each chunk whose
/// text is unchanged, and record the pages' hash. Returns the chunks added.
fn cut_stored_pages(tx: &Transaction<'_>, document: DocumentId) -> rusqlite::Result<Vec<ChunkId>> {
    let stored = StoredPages::load(tx, document)?;
    let pages = &stored.pages;
    let cuts = page_cuts(pages, ChunkingConfig::DOCUMENTS);
    let recut = stored.recut(pages, &cuts);
    recut.apply(tx, document, longest_page(pages))?;
    tx.prepare_cached("UPDATE documents SET content_hash = ?2 WHERE id = ?1")?
        .execute(rusqlite::params![document, pages_hash(pages)])?;
    add_chunks(tx, document, pages, &recut.added)
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

fn longest_page(pages: &[String]) -> usize {
    pages.iter().map(String::len).max().unwrap_or_default()
}

/// Store `pages` as the first pages of `document`, writing only those that differ
/// from the page `stored` holds at the same position.
fn write_changed_pages(
    tx: &Transaction<'_>,
    document: DocumentId,
    stored: &[String],
    pages: &[String],
) -> rusqlite::Result<()> {
    let mut write = tx.prepare_cached(
        "INSERT INTO document_pages (document_id, page, content) VALUES (?1, ?2, ?3)
         ON CONFLICT(document_id, page) DO UPDATE SET content = excluded.content",
    )?;
    let changed = pages
        .iter()
        .enumerate()
        .filter(|(page, content)| stored.get(*page) != Some(*content));
    for (page, content) in changed {
        write.execute(rusqlite::params![document, page, content])?;
    }
    Ok(())
}

/// Store a chunk of `document` for each of `cuts` of its stored `pages`, with the
/// words of its text for keyword search, returning the chunks.
fn add_chunks(
    tx: &Transaction<'_>,
    document: DocumentId,
    pages: &[String],
    cuts: &[&PageCut],
) -> rusqlite::Result<Vec<ChunkId>> {
    let mut insert = tx.prepare_cached(
        "INSERT INTO chunks (document_id, page, byte_start, byte_end)
         VALUES (?1, ?2, ?3, ?4)
         RETURNING id",
    )?;
    let mut index_words =
        tx.prepare_cached("INSERT INTO document_chunk_fts (rowid, text) VALUES (?1, ?2)")?;
    cuts.iter()
        .map(|cut| {
            let chunk: ChunkId = insert.query_row(
                rusqlite::params![document, cut.page, cut.bytes.start, cut.bytes.end],
                |r| r.get(0),
            )?;
            index_words.execute(rusqlite::params![chunk, cut.text_in(pages)])?;
            Ok(chunk)
        })
        .collect()
}

/// Remove the pages of `document` at position `first` and after, with their chunks.
fn remove_pages_from(
    tx: &Transaction<'_>,
    document: DocumentId,
    first: usize,
) -> rusqlite::Result<()> {
    tx.prepare_cached("DELETE FROM document_pages WHERE document_id = ?1 AND page >= ?2")?
        .execute(rusqlite::params![document, first])?;
    Ok(())
}

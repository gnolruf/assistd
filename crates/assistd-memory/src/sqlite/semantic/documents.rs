//! Writes to the document index: passes over index roots, and each document's text and chunks.

use std::collections::HashMap;
use std::ops::Range;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, Transaction};
use sha2::{Digest, Sha256};

use crate::chunking::{ChunkingConfig, chunk_ranges};
use crate::sqlite::timestamp::now_millis;
use crate::{ChunkId, DocumentId, FileStamp, IndexPass};

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

/// Store `content` as the document at `path`, seen by `pass`, returning the chunks
/// added. Text with the stored hash and chunking only has its stamp refreshed.
pub(super) fn store_document(
    tx: &Transaction<'_>,
    pass: IndexPass,
    path: &Path,
    stamp: FileStamp,
    content: &str,
) -> rusqlite::Result<Vec<ChunkId>> {
    let document = DocumentRow {
        pass,
        path: path.as_os_str().as_bytes(),
        stamp,
        content,
        hash: Sha256::digest(content.as_bytes()).into(),
        chunking: ChunkingConfig::DOCUMENTS,
    };
    if document.refresh_stamp_if_same_text(tx)? {
        return Ok(Vec::new());
    }
    let ranges = chunk_ranges(content, document.chunking);
    let stored = StoredChunks::load(tx, document.path)?;
    let recut = stored.recut(content, &ranges);
    recut.apply(tx, document.path, stored.content.len().max(content.len()))?;
    let id = document.upsert(tx)?;
    let mut insert = tx.prepare_cached(
        "INSERT INTO chunks (document_id, byte_start, byte_end)
         VALUES (?1, ?2, ?3)
         RETURNING id",
    )?;
    recut
        .added
        .iter()
        .map(|bytes| insert.query_row(rusqlite::params![id, bytes.start, bytes.end], |r| r.get(0)))
        .collect()
}

/// Remove the documents of `pass`'s root that it did not see, returning how many.
pub(super) fn remove_unseen(conn: &Connection, pass: IndexPass) -> rusqlite::Result<u64> {
    let removed = conn
        .prepare_cached("DELETE FROM documents WHERE root_id = ?1 AND seen_pass < ?2")?
        .execute(rusqlite::params![pass.root, pass.pass])?;
    Ok(u64::try_from(removed).unwrap_or(u64::MAX))
}

/// Remove the document at `path`, returning whether one was stored.
pub(super) fn remove_document(conn: &Connection, path: &Path) -> rusqlite::Result<bool> {
    let removed = conn
        .prepare_cached("DELETE FROM documents WHERE path = ?1")?
        .execute(rusqlite::params![path.as_os_str().as_bytes()])?;
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

/// A `documents` row as one pass read it from disk.
struct DocumentRow<'a> {
    pass: IndexPass,
    path: &'a [u8],
    stamp: FileStamp,
    content: &'a str,
    hash: [u8; 32],
    chunking: ChunkingConfig,
}

impl DocumentRow<'_> {
    /// Record the stamp and pass on the stored row when it already holds this text
    /// cut with this chunking, returning whether it does.
    fn refresh_stamp_if_same_text(&self, tx: &Transaction<'_>) -> rusqlite::Result<bool> {
        let refreshed = tx
            .prepare_cached(
                "UPDATE documents SET root_id = ?1, seen_pass = ?2, mtime_ns = ?3, size = ?4
                 WHERE path = ?5 AND content_hash = ?6
                   AND chunker = ?7 AND chunk_chars = ?8 AND overlap_chars = ?9",
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

    /// Insert the row, or overwrite the one stored for its path.
    fn upsert(&self, tx: &Transaction<'_>) -> rusqlite::Result<DocumentId> {
        tx.prepare_cached(
            "INSERT INTO documents (root_id, seen_pass, mtime_ns, size, path, content_hash,
                                    chunker, chunk_chars, overlap_chars, content, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(path) DO UPDATE
             SET root_id = excluded.root_id, seen_pass = excluded.seen_pass,
                 mtime_ns = excluded.mtime_ns, size = excluded.size,
                 content_hash = excluded.content_hash, chunker = excluded.chunker,
                 chunk_chars = excluded.chunk_chars, overlap_chars = excluded.overlap_chars,
                 content = excluded.content, indexed_at = excluded.indexed_at
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
                self.content,
                now_millis(),
            ],
            |r| r.get(0),
        )
    }
}

/// The text and chunks stored for a document before it is indexed again; empty for
/// a document not stored yet.
#[derive(Default)]
struct StoredChunks {
    content: String,
    chunks: Vec<(ChunkId, Range<usize>)>,
}

impl StoredChunks {
    fn load(tx: &Transaction<'_>, path: &[u8]) -> rusqlite::Result<Self> {
        let stored: Option<(DocumentId, String)> = tx
            .prepare_cached("SELECT id, content FROM documents WHERE path = ?1")?
            .query_row(rusqlite::params![path], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?;
        let Some((document, content)) = stored else {
            return Ok(Self::default());
        };
        let chunks = tx
            .prepare_cached(
                "SELECT id, byte_start, byte_end FROM chunks
                 WHERE document_id = ?1
                 ORDER BY byte_start",
            )?
            .query_map(rusqlite::params![document], |r| {
                Ok((r.get(0)?, r.get(1)?..r.get(2)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(Self { content, chunks })
    }

    /// Match the stored chunks to `ranges` over the new `content` by the text each
    /// covers. Chunks with the same text are matched in order of position.
    fn recut<'r>(&self, content: &str, ranges: &'r [Range<usize>]) -> Recut<'r> {
        let mut recut = Recut::default();
        let mut stored_by_text: HashMap<&str, Vec<ChunkId>> = HashMap::new();
        for (chunk, bytes) in self.chunks.iter().rev() {
            match self.content.get(bytes.clone()) {
                Some(text) => stored_by_text.entry(text).or_default().push(*chunk),
                None => recut.dropped.push(*chunk),
            }
        }
        for bytes in ranges {
            let same_text = stored_by_text
                .get_mut(&content[bytes.clone()])
                .and_then(Vec::pop);
            match same_text {
                Some(chunk) => recut.kept.push((chunk, bytes)),
                None => recut.added.push(bytes),
            }
        }
        recut.dropped.extend(stored_by_text.into_values().flatten());
        recut
    }
}

/// What becomes of a stored document's chunks when its new text is cut.
#[derive(Default)]
struct Recut<'r> {
    /// Stored chunks whose text is unchanged, each with the range that text now has.
    kept: Vec<(ChunkId, &'r Range<usize>)>,
    /// Stored chunks whose text is gone.
    dropped: Vec<ChunkId>,
    /// Ranges whose text no stored chunk covered.
    added: Vec<&'r Range<usize>>,
}

impl Recut<'_> {
    /// Delete the dropped chunks of the document at `path` and give the kept ones
    /// their ranges. Every kept chunk is first shifted up by `clearance`, at least the
    /// length of the old and new texts, so none lands on a start another has yet to leave.
    fn apply(&self, tx: &Transaction<'_>, path: &[u8], clearance: usize) -> rusqlite::Result<()> {
        let mut delete = tx.prepare_cached("DELETE FROM chunks WHERE id = ?1")?;
        for chunk in &self.dropped {
            delete.execute(rusqlite::params![chunk])?;
        }
        tx.prepare_cached(
            "UPDATE chunks SET byte_start = byte_start + ?2, byte_end = byte_end + ?2
             WHERE document_id = (SELECT id FROM documents WHERE path = ?1)",
        )?
        .execute(rusqlite::params![path, clearance])?;
        let mut place =
            tx.prepare_cached("UPDATE chunks SET byte_start = ?2, byte_end = ?3 WHERE id = ?1")?;
        for (chunk, bytes) in &self.kept {
            place.execute(rusqlite::params![chunk, bytes.start, bytes.end])?;
        }
        Ok(())
    }
}

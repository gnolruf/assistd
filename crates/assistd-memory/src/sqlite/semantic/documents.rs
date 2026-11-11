//! Writes to the document index: passes over index roots, and each document's text and chunks.

use std::collections::HashSet;
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
               AND chunk_chars = ?6 AND overlap_chars = ?7",
        )?
        .execute(rusqlite::params![
            pass.root,
            pass.pass,
            path.as_os_str().as_bytes(),
            stamp.mtime_ns,
            stamp.size,
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
    let ranges: Vec<Range<usize>> = chunk_ranges(content, document.chunking).collect();
    let kept_starts = drop_changed_chunks(tx, document.path, content, &ranges)?;
    let id = document.upsert(tx)?;
    let mut insert = tx.prepare_cached(
        "INSERT INTO chunks (document_id, byte_start, byte_end)
         VALUES (?1, ?2, ?3)
         RETURNING id",
    )?;
    ranges
        .iter()
        .filter(|bytes| !kept_starts.contains(&bytes.start))
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
                   AND chunk_chars = ?7 AND overlap_chars = ?8",
            )?
            .execute(rusqlite::params![
                self.pass.root,
                self.pass.pass,
                self.stamp.mtime_ns,
                self.stamp.size,
                self.path,
                self.hash,
                self.chunking.chunk_chars(),
                self.chunking.overlap_chars(),
            ])?;
        Ok(refreshed > 0)
    }

    /// Insert the row, or overwrite the one stored for its path.
    fn upsert(&self, tx: &Transaction<'_>) -> rusqlite::Result<DocumentId> {
        tx.prepare_cached(
            "INSERT INTO documents (root_id, seen_pass, mtime_ns, size, path, content_hash,
                                    chunk_chars, overlap_chars, content, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(path) DO UPDATE
             SET root_id = excluded.root_id, seen_pass = excluded.seen_pass,
                 mtime_ns = excluded.mtime_ns, size = excluded.size,
                 content_hash = excluded.content_hash, chunk_chars = excluded.chunk_chars,
                 overlap_chars = excluded.overlap_chars, content = excluded.content,
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
                self.chunking.chunk_chars(),
                self.chunking.overlap_chars(),
                self.content,
                now_millis(),
            ],
            |r| r.get(0),
        )
    }
}

/// Delete the chunks stored for the document at `path` that are not one of `ranges`
/// over the same bytes in `content`, returning the byte starts of those left.
fn drop_changed_chunks(
    tx: &Transaction<'_>,
    path: &[u8],
    content: &str,
    ranges: &[Range<usize>],
) -> rusqlite::Result<HashSet<usize>> {
    let stored: Option<(DocumentId, String)> = tx
        .prepare_cached("SELECT id, content FROM documents WHERE path = ?1")?
        .query_row(rusqlite::params![path], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?;
    let Some((document, stored_content)) = stored else {
        return Ok(HashSet::new());
    };
    let wanted: HashSet<&Range<usize>> = ranges.iter().collect();
    let stored_chunks: Vec<(ChunkId, Range<usize>)> = tx
        .prepare_cached("SELECT id, byte_start, byte_end FROM chunks WHERE document_id = ?1")?
        .query_map(rusqlite::params![document], |r| {
            Ok((r.get(0)?, r.get(1)?..r.get(2)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut delete = tx.prepare_cached("DELETE FROM chunks WHERE id = ?1")?;
    let mut kept_starts = HashSet::new();
    for (chunk, bytes) in stored_chunks {
        let unchanged = wanted.contains(&bytes)
            && stored_content.as_bytes().get(bytes.clone())
                == content.as_bytes().get(bytes.clone());
        if unchanged {
            kept_starts.insert(bytes.start);
        } else {
            delete.execute(rusqlite::params![chunk])?;
        }
    }
    Ok(kept_starts)
}

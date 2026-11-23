//! The `chunks` table as every store writes it: a chunk with its words, and the
//! vectors of `vectors.db` that go once the last chunk holding their text does.

use std::ops::Range;

use rusqlite::{Connection, Transaction};

use crate::chunking::{TextHash, text_hash};
use crate::{ChunkId, ChunkSource, DocumentId, MemoryId, MessageId};

/// Every kind of chunk, in the order of their `chunks.kind` codes.
pub(super) const CHUNK_SOURCES: [ChunkSource; 3] = [
    ChunkSource::Message,
    ChunkSource::Memory,
    ChunkSource::Document,
];

/// The text a chunk is a byte range of.
#[derive(Debug, Clone, Copy)]
pub(super) enum ChunkOwner {
    /// A message's content.
    Message(MessageId),
    /// A memory's value.
    Memory(MemoryId),
    /// The content of one page of a document, by the page's position from zero.
    Page(DocumentId, usize),
}

impl ChunkOwner {
    fn source(self) -> ChunkSource {
        match self {
            Self::Message(_) => ChunkSource::Message,
            Self::Memory(_) => ChunkSource::Memory,
            Self::Page(..) => ChunkSource::Document,
        }
    }

    /// The `chunks.source` and `chunks.page` of a chunk cut from this text.
    fn source_id_and_page(self) -> (i64, usize) {
        match self {
            Self::Message(message) => (message.0, 0),
            Self::Memory(memory) => (memory.0, 0),
            Self::Page(document, page) => (document.0, page),
        }
    }
}

/// The `chunks.kind` code of the chunks cut from `source`.
pub(super) const fn kind_code(source: ChunkSource) -> i64 {
    match source {
        ChunkSource::Message => 0,
        ChunkSource::Memory => 1,
        ChunkSource::Document => 2,
    }
}

/// The table holding the vector of each text the chunks cut from `source` hold.
pub(super) const fn vector_table(source: ChunkSource) -> &'static str {
    match source {
        ChunkSource::Message => "message_vectors",
        ChunkSource::Memory => "memory_vectors",
        ChunkSource::Document => "document_vectors",
    }
}

/// Store a chunk over `bytes` of `content`, the text of `owner`, with its words for
/// keyword search. Returns its id and the hash its vector is stored under.
pub(super) fn insert_chunk(
    conn: &Connection,
    owner: ChunkOwner,
    content: &str,
    bytes: &Range<usize>,
) -> rusqlite::Result<(ChunkId, TextHash)> {
    let text = &content[bytes.clone()];
    let hash = text_hash(text.as_bytes());
    let (source_id, page) = owner.source_id_and_page();
    let chunk: ChunkId = conn
        .prepare_cached(
            "INSERT INTO chunks (kind, source, page, byte_start, byte_end, hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             RETURNING id",
        )?
        .query_row(
            rusqlite::params![
                owner.source(),
                source_id,
                page,
                bytes.start,
                bytes.end,
                hash
            ],
            |r| r.get(0),
        )?;
    conn.prepare_cached("INSERT INTO chunk_words (rowid, text) VALUES (?1, ?2)")?
        .execute(rusqlite::params![chunk, text])?;
    Ok((chunk, hash))
}

/// Delete the vector of each text the transaction removed a chunk of, unless a chunk
/// of that kind still holds the text, and forget the removals.
pub(super) fn delete_vectors_of_removed_texts(tx: &Transaction<'_>) -> rusqlite::Result<()> {
    let removed_any: bool = tx
        .prepare_cached("SELECT EXISTS (SELECT 1 FROM removed_chunk_texts)")?
        .query_row([], |r| r.get(0))?;
    if !removed_any {
        return Ok(());
    }
    for source in CHUNK_SOURCES {
        let vectors = vector_table(source);
        tx.prepare_cached(&format!(
            "DELETE FROM {vectors}
             WHERE hash IN (SELECT hash FROM removed_chunk_texts WHERE kind = ?1)
               AND NOT EXISTS (SELECT 1 FROM chunks cc
                               WHERE cc.hash = {vectors}.hash AND cc.kind = ?1)"
        ))?
        .execute([source])?;
    }
    tx.prepare_cached("DELETE FROM removed_chunk_texts")?
        .execute([])?;
    Ok(())
}

/// Delete every vector of the texts of `source` that no chunk of that kind holds,
/// returning how many. The chunks of that kind must all be cut: until they are, a
/// vector waits for a chunk that will hold it.
pub(super) fn delete_unheld_vectors(
    conn: &Connection,
    source: ChunkSource,
) -> rusqlite::Result<usize> {
    let vectors = vector_table(source);
    conn.prepare_cached(&format!(
        "DELETE FROM {vectors}
         WHERE NOT EXISTS (SELECT 1 FROM chunks cc
                           WHERE cc.hash = {vectors}.hash AND cc.kind = ?1)"
    ))?
    .execute([source])
}

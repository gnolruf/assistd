//! The schema of `index.db`, which holds only what can be built again from `memory.db`
//! and the files on disk, so a file from another version is replaced, never migrated.

use rusqlite::Connection;

/// The `PRAGMA user_version` of an `index.db` holding [`SCHEMA_SQL`]. Any change to
/// the schema takes the next number, and every older file is then built again.
pub(crate) const VERSION: i64 = 2;

/// Every table of the index. A chunk's `source` names a row of `memory.db` when its
/// kind is a message or a memory, so nothing here can hold it to that row.
const SCHEMA_SQL: &str = r"
-- At most one row: the `memory.db` whose messages and memories are chunked here, by
-- its `database_identity`, and the highest message id whose chunks are cut.
CREATE TABLE chunked_source (
    id              INTEGER PRIMARY KEY CHECK (id = 0),
    memory_identity BLOB NOT NULL CHECK (length(memory_identity) = 16),
    chunked_through INTEGER NOT NULL CHECK (chunked_through >= 0)
) STRICT;

-- `pass` counts the walks over the root. `finished_pass` is the last of them that
-- walked the whole root to its end, 0 while none has since this file was created.
CREATE TABLE index_roots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          BLOB NOT NULL UNIQUE CHECK (length(path) > 0),
    pass          INTEGER NOT NULL CHECK (pass >= 1),
    finished_pass INTEGER NOT NULL DEFAULT 0 CHECK (finished_pass BETWEEN 0 AND pass)
) STRICT;

-- An indexed file: the file's stamp, the hash of its pages in order, the chunking
-- it was cut with, and the root's pass that last saw it. `chunker` is 0 character
-- windows, 1 packed paragraphs.
CREATE TABLE documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id       INTEGER NOT NULL REFERENCES index_roots(id) ON DELETE CASCADE,
    path          BLOB NOT NULL UNIQUE CHECK (length(path) > 0),
    content_hash  BLOB NOT NULL CHECK (length(content_hash) = 32),
    mtime_ns      INTEGER NOT NULL,
    size          INTEGER NOT NULL CHECK (size >= 0),
    chunker       INTEGER NOT NULL,
    chunk_chars   INTEGER NOT NULL CHECK (chunk_chars > 0),
    overlap_chars INTEGER NOT NULL CHECK (overlap_chars >= 0 AND overlap_chars < chunk_chars),
    seen_pass     INTEGER NOT NULL,
    indexed_at    INTEGER NOT NULL
) STRICT;
CREATE INDEX idx_documents_root ON documents(root_id);

-- A file that was read and left nothing to index, with the stamp it had and the
-- root's pass that last saw it; no document is stored at its path. `cause` is
-- 0 unsupported, 1 binary, 2 too large, 3 not decodable, 4 password protected,
-- 5 no text layer, 6 unreadable, 7 extractor failed, 8 timed out.
CREATE TABLE skipped_files (
    path      BLOB PRIMARY KEY CHECK (length(path) > 0),
    root_id   INTEGER NOT NULL REFERENCES index_roots(id) ON DELETE CASCADE,
    mtime_ns  INTEGER NOT NULL,
    size      INTEGER NOT NULL CHECK (size >= 0),
    cause     INTEGER NOT NULL CHECK (cause BETWEEN 0 AND 8),
    seen_pass INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
CREATE INDEX idx_skipped_files_root ON skipped_files(root_id);

-- One page of a document's text as indexed. `page` is its position in the file,
-- from 0 and without gaps; a page with no text has empty `content`.
CREATE TABLE document_pages (
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        INTEGER NOT NULL CHECK (page >= 0),
    content     TEXT NOT NULL,
    PRIMARY KEY (document_id, page)
) STRICT;

-- A page that had no text when its file was read and is an image a model can read.
-- `state` is 0 waiting for the OCR model, 1 read, its page then holding what the model
-- found, 2 failed. Rows last until the file is indexed again.
CREATE TABLE page_transcriptions (
    document_id INTEGER NOT NULL,
    page        INTEGER NOT NULL,
    state       INTEGER NOT NULL DEFAULT 0 CHECK (state BETWEEN 0 AND 2),
    PRIMARY KEY (document_id, page),
    FOREIGN KEY (document_id, page)
        REFERENCES document_pages(document_id, page) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;
CREATE INDEX idx_page_transcriptions_waiting ON page_transcriptions(document_id, page)
    WHERE state = 0;

-- A byte range into the text a chunk is cut from: `kind` is 0 for a message's content,
-- 1 for a memory's value, 2 for a document page's content, and `source` is the id of
-- that message, memory or document. `page` is 0 unless the source is a document.
-- `hash` is the first 16 bytes of the SHA-256 of the bytes the range covers, which is
-- what `vectors.db` stores the text's vector under. The id is also the chunk's rowid
-- in `chunk_words`.
CREATE TABLE chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       INTEGER NOT NULL CHECK (kind BETWEEN 0 AND 2),
    source     INTEGER NOT NULL,
    page       INTEGER NOT NULL CHECK (page >= 0),
    byte_start INTEGER NOT NULL,
    byte_end   INTEGER NOT NULL,
    hash       BLOB NOT NULL CHECK (length(hash) = 16),
    CHECK (byte_start >= 0 AND byte_end > byte_start),
    CHECK (kind = 2 OR page = 0),
    UNIQUE (kind, source, page, byte_start)
) STRICT;
CREATE INDEX idx_chunks_hash ON chunks(hash);

-- Stands in for the foreign key `chunks.source` cannot have.
CREATE TRIGGER document_page_removed AFTER DELETE ON document_pages
BEGIN
    DELETE FROM chunks WHERE kind = 2 AND source = old.document_id AND page = old.page;
END;

-- The words of every chunk, for keyword search, without its text. A row has its
-- chunk's id; the write that cuts a chunk adds it, and the trigger removes it.
CREATE VIRTUAL TABLE chunk_words USING fts5(text, content='', contentless_delete=1);

-- The chunk texts a write removed a chunk of, by `chunks.kind` and `chunks.hash`.
-- The write that fills it empties it before it commits.
CREATE TABLE removed_chunk_texts (
    kind INTEGER NOT NULL CHECK (kind BETWEEN 0 AND 2),
    hash BLOB NOT NULL CHECK (length(hash) = 16),
    PRIMARY KEY (kind, hash)
) STRICT, WITHOUT ROWID;

CREATE TRIGGER chunk_removed AFTER DELETE ON chunks
BEGIN
    DELETE FROM chunk_words WHERE rowid = old.id;
    INSERT INTO removed_chunk_texts (kind, hash)
    VALUES (old.kind, old.hash)
    ON CONFLICT DO NOTHING;
END;
";

/// Create the schema in `conn`'s empty main database and stamp it with [`VERSION`];
/// a no-op on one already stamped.
pub(crate) fn create(conn: &mut Connection) -> rusqlite::Result<()> {
    if stored_version(conn)? == VERSION {
        return Ok(());
    }
    let tx = conn.transaction()?;
    tx.execute_batch(SCHEMA_SQL)?;
    tx.pragma_update(Some("main"), "user_version", VERSION)?;
    tx.commit()
}

/// The schema version stamped on `conn`'s main database; zero for an empty one.
fn stored_version(conn: &Connection) -> rusqlite::Result<i64> {
    conn.pragma_query_value(Some("main"), "user_version", |row| row.get(0))
}

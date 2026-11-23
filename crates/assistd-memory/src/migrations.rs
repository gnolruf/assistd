//! The SQLite schema, applied through `rusqlite_migration`; each `M::up` is one schema version.

use rusqlite::Connection;
use rusqlite_migration::{M, Migrations};

/// The whole schema, versioned by `PRAGMA user_version`; timestamps are Unix ms. Composite
/// foreign keys keep each branch, turn, and message inside its session, and
/// `AUTOINCREMENT` ids never recycle under a queued write or embed job.
const SCHEMA_SQL: &str = r"
CREATE TABLE sessions (
    id                INTEGER PRIMARY KEY,
    uuid              TEXT NOT NULL UNIQUE,
    started_at        INTEGER NOT NULL,
    title             TEXT,
    current_branch_id INTEGER,
    task_run_id       INTEGER REFERENCES task_runs(id) ON DELETE SET NULL,
    FOREIGN KEY (id, current_branch_id) REFERENCES branches(session_id, id)
) STRICT;
CREATE INDEX idx_sessions_task_run ON sessions(task_run_id);

-- At most one row: the session the user's turns are on.
CREATE TABLE active_session (
    id         INTEGER PRIMARY KEY CHECK (id = 0),
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE branches (
    id               INTEGER PRIMARY KEY,
    session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name             TEXT NOT NULL,
    parent_branch_id INTEGER,
    fork_point_seq   INTEGER CHECK (fork_point_seq IS NULL OR parent_branch_id IS NOT NULL),
    next_seq         INTEGER NOT NULL DEFAULT 0 CHECK (next_seq >= 0),
    created_at       INTEGER NOT NULL,
    UNIQUE (name, session_id),
    UNIQUE (session_id, id),
    FOREIGN KEY (session_id, parent_branch_id) REFERENCES branches(session_id, id)
) STRICT;
CREATE INDEX idx_branches_parent ON branches(session_id, parent_branch_id);

-- `incomplete` is set once a message of the turn failed to save.
CREATE TABLE turns (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    duration_ms      INTEGER CHECK (duration_ms >= 0),
    generated_tokens INTEGER CHECK (generated_tokens >= 0),
    incomplete       INTEGER NOT NULL DEFAULT 0 CHECK (incomplete IN (0, 1)),
    UNIQUE (session_id, id),
    CHECK ((duration_ms IS NULL) = (generated_tokens IS NULL))
) STRICT;

-- `role` is 0 user, 1 assistant, 2 tool. `tool_read_host` is set on a result whose call
-- could put data from this machine in it.
CREATE TABLE messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id          INTEGER NOT NULL,
    created_at       INTEGER NOT NULL,
    role             INTEGER NOT NULL CHECK (role BETWEEN 0 AND 2),
    content          TEXT NOT NULL,
    reasoning        TEXT,
    reasoning_ms     INTEGER CHECK (reasoning_ms >= 0),
    tool_calls       TEXT,
    tool_call_id     TEXT,
    tool_name        TEXT,
    tool_exit_code   INTEGER,
    tool_duration_ms INTEGER CHECK (tool_duration_ms >= 0),
    tool_read_host   INTEGER CHECK (tool_read_host IN (0, 1)),
    FOREIGN KEY (session_id, turn_id) REFERENCES turns(session_id, id),
    CHECK (tool_calls IS NULL OR (role = 1 AND json_valid(tool_calls)
                                  AND json_type(tool_calls) = 'array'
                                  AND json_array_length(tool_calls) > 0)),
    CHECK (reasoning IS NULL OR (role = 1 AND reasoning <> '')),
    CHECK ((reasoning IS NULL) = (reasoning_ms IS NULL)),
    CHECK ((role = 2) = (tool_call_id IS NOT NULL)),
    CHECK ((role = 2) = (tool_name IS NOT NULL)),
    CHECK ((role = 2) = (tool_exit_code IS NOT NULL)),
    CHECK ((role = 2) = (tool_duration_ms IS NOT NULL)),
    CHECK ((role = 2) = (tool_read_host IS NOT NULL))
) STRICT;
CREATE INDEX idx_messages_session ON messages(session_id);
CREATE INDEX idx_messages_turn ON messages(turn_id);

CREATE TABLE branch_messages (
    branch_id  INTEGER NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    PRIMARY KEY (branch_id, seq),
    CHECK (seq >= 0)
) STRICT, WITHOUT ROWID;
CREATE INDEX idx_branch_messages_message ON branch_messages(message_id);

CREATE TABLE branch_summaries (
    branch_id            INTEGER PRIMARY KEY REFERENCES branches(id) ON DELETE CASCADE,
    kept_from_seq        INTEGER NOT NULL CHECK (kept_from_seq >= 0),
    steps_folded_through INTEGER CHECK (steps_folded_through > kept_from_seq),
    compacted_after_seq  INTEGER NOT NULL CHECK (compacted_after_seq >= kept_from_seq),
    content              TEXT NOT NULL,
    duration_ms          INTEGER NOT NULL CHECK (duration_ms >= 0)
) STRICT;

-- What a `relay` tool result's sub-agents did. `outcome` is 0 answered, 1 failed,
-- NULL for a step that never ran; `answer_file` is where the whole answer was saved.
CREATE TABLE relay_steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    step        INTEGER NOT NULL CHECK (step >= 1),
    task        TEXT NOT NULL,
    agent       TEXT NOT NULL CHECK (agent <> ''),
    outcome     INTEGER CHECK (outcome BETWEEN 0 AND 1),
    failure     TEXT,
    answer_file BLOB CHECK (length(answer_file) > 0),
    duration_ms INTEGER CHECK (duration_ms >= 0),
    tokens      INTEGER CHECK (tokens >= 0),
    UNIQUE (message_id, step),
    CHECK ((outcome IS 1) = (failure IS NOT NULL)),
    CHECK ((outcome IS 0) = (answer_file IS NOT NULL)),
    CHECK ((outcome IS NULL) = (duration_ms IS NULL)),
    CHECK (outcome IS NOT NULL OR tokens IS NULL)
) STRICT;

-- `kind` is 0 reply text, 1 reasoning, 2 tool call, 3 compaction.
CREATE TABLE relay_entries (
    step_id        INTEGER NOT NULL REFERENCES relay_steps(id) ON DELETE CASCADE,
    seq            INTEGER NOT NULL CHECK (seq >= 0),
    kind           INTEGER NOT NULL CHECK (kind BETWEEN 0 AND 3),
    content        TEXT NOT NULL,
    tool_name      TEXT,
    tool_args      TEXT CHECK (tool_args IS NULL OR json_valid(tool_args)),
    tool_exit_code INTEGER,
    duration_ms    INTEGER CHECK (duration_ms >= 0),
    PRIMARY KEY (step_id, seq),
    CHECK ((kind = 2) = (tool_name IS NOT NULL)),
    CHECK ((kind = 2) = (tool_args IS NOT NULL)),
    CHECK (kind = 2 OR tool_exit_code IS NULL),
    CHECK (kind <> 2 OR (tool_exit_code IS NULL) = (duration_ms IS NULL)),
    CHECK (kind <> 0 OR duration_ms IS NULL),
    CHECK (kind <> 1 OR duration_ms IS NOT NULL),
    CHECK (kind IN (2, 3) OR content <> ''),
    CHECK (kind <> 3 OR content = '')
) STRICT, WITHOUT ROWID;

CREATE TABLE memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL UNIQUE,
    value      TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    CHECK (updated_at >= created_at)
) STRICT;

-- `pass` counts the walks over the root.
CREATE TABLE index_roots (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    path BLOB NOT NULL UNIQUE CHECK (length(path) > 0),
    pass INTEGER NOT NULL CHECK (pass >= 1)
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

-- A byte range into the content of the one message or document page that owns it.
-- `hash` is the first 16 bytes of the SHA-256 of the bytes the range covers.
CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    document_id INTEGER,
    page        INTEGER,
    byte_start  INTEGER NOT NULL,
    byte_end    INTEGER NOT NULL,
    hash        BLOB NOT NULL CHECK (length(hash) = 16),
    CHECK (byte_start >= 0 AND byte_end > byte_start),
    CHECK ((message_id IS NULL) <> (document_id IS NULL)),
    CHECK ((document_id IS NULL) = (page IS NULL)),
    UNIQUE (message_id, byte_start),
    UNIQUE (document_id, page, byte_start),
    FOREIGN KEY (document_id, page)
        REFERENCES document_pages(document_id, page) ON DELETE CASCADE
) STRICT;
CREATE INDEX idx_chunks_hash ON chunks(hash);

-- The words of each document chunk, for keyword search, without its text. A row has
-- its chunk's id; the write that cuts a chunk adds it, and the trigger removes it.
CREATE VIRTUAL TABLE document_chunk_fts USING fts5(text, content='', contentless_delete=1);

CREATE TRIGGER document_chunk_removed AFTER DELETE ON chunks
WHEN old.document_id IS NOT NULL
BEGIN
    DELETE FROM document_chunk_fts WHERE rowid = old.id;
END;

-- The chunk texts a write removed a chunk of, by `chunks.hash`; `of_document` is 1 for
-- a document chunk's. The write that fills it empties it before it commits.
CREATE TABLE removed_chunk_texts (
    of_document INTEGER NOT NULL CHECK (of_document IN (0, 1)),
    hash        BLOB NOT NULL CHECK (length(hash) = 16),
    PRIMARY KEY (of_document, hash)
) STRICT, WITHOUT ROWID;

CREATE TRIGGER chunk_text_removed AFTER DELETE ON chunks
BEGIN
    INSERT INTO removed_chunk_texts (of_document, hash)
    VALUES (old.document_id IS NOT NULL, old.hash)
    ON CONFLICT DO NOTHING;
END;

-- The vector of each distinct text the message chunks hold, by `chunks.hash`, and
-- the same for document chunks. A vector is a little-endian `f32` step, then one
-- `i8` count of steps per dimension. In all three embedding tables a NULL vector
-- marks text the model could not embed, so it is not retried.
CREATE TABLE message_vectors (
    hash   BLOB NOT NULL UNIQUE CHECK (length(hash) = 16),
    vector BLOB CHECK (length(vector) > 4)
) STRICT;

CREATE TABLE document_vectors (
    hash   BLOB NOT NULL UNIQUE CHECK (length(hash) = 16),
    vector BLOB CHECK (length(vector) > 4)
) STRICT;

-- A memory vector is little-endian `f32`s.
CREATE TABLE memory_embeddings (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vector    BLOB CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;

CREATE TRIGGER memory_value_changed AFTER UPDATE OF value ON memories
WHEN old.value IS NOT new.value
BEGIN
    DELETE FROM memory_embeddings WHERE memory_id = new.id;
END;

-- At most one row: the embedding space every stored vector is in.
CREATE TABLE embedding_meta (
    id              INTEGER PRIMARY KEY CHECK (id = 0),
    model           TEXT NOT NULL,
    dim             INTEGER NOT NULL CHECK (dim > 0),
    server_args     TEXT NOT NULL,
    document_prefix TEXT NOT NULL
) STRICT;

-- `next_fire_at` is NULL until the scheduler computes it after a save, or when nothing
-- is due. `mode` is 0 manual, 1 accept edits, 2 auto. `kept_runs` is how many finished
-- runs the task keeps; NULL keeps the configured default.
CREATE TABLE tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL UNIQUE CHECK (name <> ''),
    prompt        TEXT NOT NULL CHECK (prompt <> ''),
    triggers      TEXT NOT NULL CHECK (json_valid(triggers) AND json_type(triggers) = 'array'),
    enabled       INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL CHECK (updated_at >= created_at),
    next_fire_at  INTEGER,
    mode          INTEGER NOT NULL DEFAULT 0 CHECK (mode BETWEEN 0 AND 2),
    allowed_hosts TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(allowed_hosts) AND json_type(allowed_hosts) = 'array'),
    announce      INTEGER NOT NULL DEFAULT 0,
    kept_runs     INTEGER CHECK (kept_runs BETWEEN 1 AND 10000)
) STRICT;

-- `state` codes follow `TaskRunState`'s variant order from 0 (queued), and the detail
-- columns are set only in the states that carry them. Queued runs start in `queue_pos`
-- order, those with equal positions in id order.
CREATE TABLE task_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    trigger        TEXT NOT NULL CHECK (json_valid(trigger)),
    state          INTEGER NOT NULL CHECK (state BETWEEN 0 AND 6),
    suspend_reason INTEGER CHECK (suspend_reason BETWEEN 0 AND 1),
    failure        TEXT,
    queued_at      INTEGER NOT NULL,
    started_at     INTEGER CHECK (started_at >= queued_at),
    ended_at       INTEGER CHECK (ended_at >= queued_at),
    reply          TEXT,
    queue_pos      INTEGER NOT NULL DEFAULT 0,
    CHECK ((state = 2) = (suspend_reason IS NOT NULL)),
    CHECK ((state = 5) = (failure IS NOT NULL)),
    CHECK ((state = 4) = (reply IS NOT NULL)),
    CHECK ((state IN (4, 5, 6)) = (ended_at IS NOT NULL)),
    CHECK (state <> 0 OR started_at IS NULL)
) STRICT;
CREATE INDEX idx_task_runs_task ON task_runs(task_id);
CREATE INDEX idx_task_runs_queue_pos ON task_runs(queue_pos);

-- Whether the task queue is paused; a missing row means it is not.
CREATE TABLE task_queue (
    id     INTEGER PRIMARY KEY CHECK (id = 0),
    paused INTEGER NOT NULL CHECK (paused IN (0, 1))
) STRICT;
";

fn migrations() -> Migrations<'static> {
    Migrations::new(vec![M::up(SCHEMA_SQL)])
}

/// Apply all pending migrations to `conn`; a no-op on an up-to-date database.
pub(crate) fn run(conn: &mut Connection) -> Result<(), rusqlite_migration::Error> {
    migrations().to_latest(conn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn open_in_memory() -> Connection {
        Connection::open_in_memory().expect("open :memory:")
    }

    #[test]
    fn run_is_idempotent() {
        let mut conn = open_in_memory();
        run(&mut conn).expect("first run");
        run(&mut conn).expect("second run no-op");

        let version: i64 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, 1);
    }
}

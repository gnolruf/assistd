//! Schema migrations for the SQLite store; each `M::up` is one schema version.

use rusqlite::Connection;
use rusqlite_migration::{M, Migrations};

/// V1 schema, versioned by `PRAGMA user_version`; timestamps are Unix ms. Composite
/// foreign keys keep each branch, turn, and message inside its session, and
/// `AUTOINCREMENT` ids never recycle under a queued write or embed job.
const V1_SQL: &str = r"
CREATE TABLE sessions (
    id                INTEGER PRIMARY KEY,
    uuid              TEXT NOT NULL UNIQUE,
    started_at        INTEGER NOT NULL,
    ended_at          INTEGER,
    daemon_pid        INTEGER NOT NULL,
    title             TEXT,
    current_branch_id INTEGER,
    FOREIGN KEY (id, current_branch_id) REFERENCES branches(session_id, id)
) STRICT;
CREATE INDEX idx_sessions_resumable ON sessions(started_at)
    WHERE ended_at IS NULL AND current_branch_id IS NOT NULL;

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

CREATE TABLE turns (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    duration_ms      INTEGER CHECK (duration_ms >= 0),
    generated_tokens INTEGER CHECK (generated_tokens >= 0),
    UNIQUE (session_id, id),
    CHECK ((duration_ms IS NULL) = (generated_tokens IS NULL))
) STRICT;

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
    FOREIGN KEY (session_id, turn_id) REFERENCES turns(session_id, id),
    CHECK (tool_calls IS NULL OR (role = 1 AND json_valid(tool_calls)
                                  AND json_type(tool_calls) = 'array'
                                  AND json_array_length(tool_calls) > 0)),
    CHECK (reasoning IS NULL OR (role = 1 AND reasoning <> '')),
    CHECK ((reasoning IS NULL) = (reasoning_ms IS NULL)),
    CHECK ((role = 2) = (tool_call_id IS NOT NULL)),
    CHECK ((role = 2) = (tool_name IS NOT NULL)),
    CHECK ((role = 2) = (tool_exit_code IS NOT NULL)),
    CHECK ((role = 2) = (tool_duration_ms IS NOT NULL))
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

CREATE TABLE memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL UNIQUE,
    value      TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    CHECK (updated_at >= created_at)
) STRICT;

CREATE TABLE message_chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    byte_start INTEGER NOT NULL,
    byte_end   INTEGER NOT NULL,
    CHECK (byte_start >= 0 AND byte_end > byte_start),
    UNIQUE (message_id, byte_start)
) STRICT;

CREATE TABLE embedding_models (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
) STRICT;

CREATE TABLE chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES message_chunks(id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES embedding_models(id),
    vector   BLOB NOT NULL CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;

CREATE TABLE memory_embeddings (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    model_id  INTEGER NOT NULL REFERENCES embedding_models(id),
    vector    BLOB NOT NULL CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;

CREATE TRIGGER memory_value_changed AFTER UPDATE OF value ON memories
WHEN old.value IS NOT new.value
BEGIN
    DELETE FROM memory_embeddings WHERE memory_id = new.id;
END;
";

/// V2: each vector names its model directly instead of through `embedding_models`.
/// Existing vectors are dropped for re-embedding.
const V2_SQL: &str = r"
DROP TRIGGER memory_value_changed;
DROP TABLE chunk_embeddings;
DROP TABLE memory_embeddings;
DROP TABLE embedding_models;

CREATE TABLE chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES message_chunks(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    vector   BLOB NOT NULL CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;

CREATE TABLE memory_embeddings (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    vector    BLOB NOT NULL CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;

CREATE TRIGGER memory_value_changed AFTER UPDATE OF value ON memories
WHEN old.value IS NOT new.value
BEGIN
    DELETE FROM memory_embeddings WHERE memory_id = new.id;
END;
";

/// V3: what each `relay` tool call's sub-agents did, stored beside the call's
/// tool result and deleted with it. A step that never ran has no `outcome`; an
/// entry is reply text, reasoning, a tool call, or a compaction (kinds 0 to 3).
const V3_SQL: &str = r"
CREATE TABLE relay_steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    step        INTEGER NOT NULL CHECK (step >= 1),
    task        TEXT NOT NULL,
    agent       TEXT NOT NULL CHECK (agent <> ''),
    outcome     INTEGER CHECK (outcome BETWEEN 0 AND 1),
    failure     TEXT,
    truncated   INTEGER CHECK (truncated IN (0, 1)),
    duration_ms INTEGER CHECK (duration_ms >= 0),
    tokens      INTEGER CHECK (tokens >= 0),
    UNIQUE (message_id, step),
    CHECK ((outcome IS 1) = (failure IS NOT NULL)),
    CHECK ((outcome IS 0) = (truncated IS NOT NULL)),
    CHECK ((outcome IS NULL) = (duration_ms IS NULL)),
    CHECK (outcome IS NOT NULL OR tokens IS NULL)
) STRICT;

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
";

/// V4: an answered relay step records the file its whole answer was saved to in
/// place of whether its report was cut. Recreates both relay tables, dropping
/// the traces stored so far.
const V4_SQL: &str = r"
DROP TABLE relay_entries;
DROP TABLE relay_steps;

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
";

/// V5: saved tasks and their runs. A run's `state` codes follow `TaskRunState`'s
/// variant order from 0 (queued); its detail columns are set only in the states that carry them.
const V5_SQL: &str = r"
CREATE TABLE tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE CHECK (name <> ''),
    prompt     TEXT NOT NULL CHECK (prompt <> ''),
    triggers   TEXT NOT NULL CHECK (json_valid(triggers) AND json_type(triggers) = 'array'),
    enabled    INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    grants     TEXT NOT NULL CHECK (json_valid(grants) AND json_type(grants) = 'array'),
    plan       TEXT CHECK (plan IS NULL OR json_valid(plan)),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL CHECK (updated_at >= created_at)
) STRICT;

CREATE TABLE task_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    trigger        TEXT NOT NULL CHECK (json_valid(trigger)),
    state          INTEGER NOT NULL CHECK (state BETWEEN 0 AND 6),
    suspend_reason INTEGER CHECK (suspend_reason BETWEEN 0 AND 1),
    failure        TEXT,
    session_uuid   TEXT,
    queued_at      INTEGER NOT NULL,
    started_at     INTEGER CHECK (started_at >= queued_at),
    ended_at       INTEGER CHECK (ended_at >= queued_at),
    reply          TEXT,
    CHECK ((state = 2) = (suspend_reason IS NOT NULL)),
    CHECK ((state = 5) = (failure IS NOT NULL)),
    CHECK ((state = 4) = (reply IS NOT NULL)),
    CHECK ((state IN (4, 5, 6)) = (ended_at IS NOT NULL)),
    CHECK (state <> 0 OR started_at IS NULL)
) STRICT;
CREATE INDEX idx_task_runs_task ON task_runs(task_id);
";

/// V6: a queued run's place in the task queue, where runs with equal positions start in
/// id order, and whether the queue is paused; a missing `task_queue` row means it is not.
const V6_SQL: &str = r"
ALTER TABLE task_runs ADD COLUMN queue_pos INTEGER NOT NULL DEFAULT 0;
CREATE TABLE task_queue (
    id     INTEGER PRIMARY KEY CHECK (id = 0),
    paused INTEGER NOT NULL CHECK (paused IN (0, 1))
) STRICT;
";

/// V7: when a task's triggers next fire, in Unix milliseconds; `NULL` until the
/// scheduler computes it after a save, or when nothing is due.
const V7_SQL: &str = r"
ALTER TABLE tasks ADD COLUMN next_fire_at INTEGER;
";

/// V8: a task's permission mode, coded 0 manual, 1 accept edits, 2 auto, and the hosts its
/// runs fetch from unasked, in place of its grants.
const V8_SQL: &str = r"
ALTER TABLE tasks DROP COLUMN grants;
ALTER TABLE tasks ADD COLUMN mode INTEGER NOT NULL DEFAULT 0 CHECK (mode BETWEEN 0 AND 2);
ALTER TABLE tasks ADD COLUMN allowed_hosts TEXT NOT NULL DEFAULT '[]'
    CHECK (json_valid(allowed_hosts) AND json_type(allowed_hosts) = 'array');
";

/// V9: tasks no longer hold a plan.
const V9_SQL: &str = r"
ALTER TABLE tasks DROP COLUMN plan;
";

const V10_SQL: &str = r"
ALTER TABLE tasks ADD COLUMN announce INTEGER NOT NULL DEFAULT 0;
";

/// V11: chunks are owned by a message or an indexed document, exactly one.
/// `documents` keeps each file's indexed text so a chunk stays a byte range into
/// its source. Existing chunks and vectors are carried over with their ids.
const V11_SQL: &str = r"
CREATE TABLE documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    path       BLOB NOT NULL UNIQUE CHECK (length(path) > 0),
    content    TEXT NOT NULL,
    indexed_at INTEGER NOT NULL
) STRICT;

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    byte_start  INTEGER NOT NULL,
    byte_end    INTEGER NOT NULL,
    CHECK (byte_start >= 0 AND byte_end > byte_start),
    CHECK ((message_id IS NULL) <> (document_id IS NULL)),
    UNIQUE (message_id, byte_start),
    UNIQUE (document_id, byte_start)
) STRICT;
INSERT INTO chunks (id, message_id, byte_start, byte_end)
    SELECT id, message_id, byte_start, byte_end FROM message_chunks;

CREATE TABLE chunk_vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    vector   BLOB NOT NULL CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;
INSERT INTO chunk_vectors (chunk_id, model, vector)
    SELECT chunk_id, model, vector FROM chunk_embeddings;

DROP TABLE chunk_embeddings;
DROP TABLE message_chunks;
ALTER TABLE chunk_vectors RENAME TO chunk_embeddings;
";

/// V12: sessions no longer record the owning daemon's pid.
const V12_SQL: &str = "ALTER TABLE sessions DROP COLUMN daemon_pid;";

/// V13: how many finished runs a task keeps; `NULL` keeps the configured default.
const V13_SQL: &str = r"
ALTER TABLE tasks ADD COLUMN kept_runs INTEGER CHECK (kept_runs BETWEEN 1 AND 10000);
";

/// V14: indexes for looking up a run by its session and for the queue's next position.
const V14_SQL: &str = r"
CREATE INDEX idx_task_runs_session ON task_runs(session_uuid);
CREATE INDEX idx_task_runs_queue_pos ON task_runs(queue_pos);
";

/// V15: a `NULL` vector marks a row its model could not embed, so it is not retried.
const V15_SQL: &str = r"
DROP TRIGGER memory_value_changed;

CREATE TABLE chunk_vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    vector   BLOB CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;
INSERT INTO chunk_vectors (chunk_id, model, vector)
    SELECT chunk_id, model, vector FROM chunk_embeddings;
DROP TABLE chunk_embeddings;
ALTER TABLE chunk_vectors RENAME TO chunk_embeddings;

CREATE TABLE memory_vectors (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    vector    BLOB CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;
INSERT INTO memory_vectors (memory_id, model, vector)
    SELECT memory_id, model, vector FROM memory_embeddings;
DROP TABLE memory_embeddings;
ALTER TABLE memory_vectors RENAME TO memory_embeddings;

CREATE TRIGGER memory_value_changed AFTER UPDATE OF value ON memories
WHEN old.value IS NOT new.value
BEGIN
    DELETE FROM memory_embeddings WHERE memory_id = new.id;
END;
";

/// V16: a turn is flagged once a message of it failed to save.
const V16_SQL: &str = r"
ALTER TABLE turns ADD COLUMN incomplete INTEGER NOT NULL DEFAULT 0 CHECK (incomplete IN (0, 1));
";

/// V17: message chunk and document chunk vectors live in separate tables, so a search
/// of one kind never reads the other's. A document vector is a little-endian `f32`
/// step then one `i8` count of steps per dimension; those stored so far are dropped
/// for re-embedding.
const V17_SQL: &str = r"
CREATE TABLE message_chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    vector   BLOB CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;
INSERT INTO message_chunk_embeddings (chunk_id, model, vector)
    SELECT e.chunk_id, e.model, e.vector
    FROM chunk_embeddings e JOIN chunks cc ON cc.id = e.chunk_id
    WHERE cc.message_id IS NOT NULL;

CREATE TABLE document_chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    vector   BLOB CHECK (length(vector) > 4)
) STRICT;

DROP TABLE chunk_embeddings;
";

/// V18: a document records the index root it was found under, the file's stamp and
/// content hash, the chunking it was cut with, and the root's pass that last saw it.
/// Recreates `documents`, dropping any indexed so far.
const V18_SQL: &str = r"
DELETE FROM chunks WHERE document_id IS NOT NULL;
DROP TABLE documents;

CREATE TABLE index_roots (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    path BLOB NOT NULL UNIQUE CHECK (length(path) > 0),
    pass INTEGER NOT NULL CHECK (pass >= 1)
) STRICT;

CREATE TABLE documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id       INTEGER NOT NULL REFERENCES index_roots(id) ON DELETE CASCADE,
    path          BLOB NOT NULL UNIQUE CHECK (length(path) > 0),
    content       TEXT NOT NULL,
    content_hash  BLOB NOT NULL CHECK (length(content_hash) = 32),
    mtime_ns      INTEGER NOT NULL,
    size          INTEGER NOT NULL CHECK (size >= 0),
    chunk_chars   INTEGER NOT NULL CHECK (chunk_chars > 0),
    overlap_chars INTEGER NOT NULL CHECK (overlap_chars >= 0 AND overlap_chars < chunk_chars),
    seen_pass     INTEGER NOT NULL,
    indexed_at    INTEGER NOT NULL
) STRICT;
CREATE INDEX idx_documents_root ON documents(root_id);
";

/// V19: the session the user's conversation is on is the one `active_session` row, in
/// place of sessions being open or ended, and a task run's session names its run.
const V19_SQL: &str = r"
DROP INDEX idx_sessions_resumable;
ALTER TABLE sessions DROP COLUMN ended_at;
ALTER TABLE sessions ADD COLUMN task_run_id INTEGER REFERENCES task_runs(id) ON DELETE SET NULL;
UPDATE sessions
    SET task_run_id = (SELECT r.id FROM task_runs r WHERE r.session_uuid = sessions.uuid);
CREATE INDEX idx_sessions_task_run ON sessions(task_run_id);
DROP INDEX idx_task_runs_session;
ALTER TABLE task_runs DROP COLUMN session_uuid;

CREATE TABLE active_session (
    id         INTEGER PRIMARY KEY CHECK (id = 0),
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE
) STRICT;
";

/// V20: stored vectors are identified by the one `embedding_meta` row naming the
/// embedding space they are all in, in place of a model per vector. Drops every
/// vector for re-embedding.
const V20_SQL: &str = r"
DROP TABLE message_chunk_embeddings;
DROP TABLE document_chunk_embeddings;
DROP TABLE memory_embeddings;

CREATE TABLE message_chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector   BLOB CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;

CREATE TABLE document_chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector   BLOB CHECK (length(vector) > 4)
) STRICT;

CREATE TABLE memory_embeddings (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vector    BLOB CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;

CREATE TABLE embedding_meta (
    id          INTEGER PRIMARY KEY CHECK (id = 0),
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL CHECK (dim > 0),
    server_args TEXT NOT NULL
) STRICT;
";

/// The full migration set.
fn migrations() -> Migrations<'static> {
    Migrations::new(vec![
        M::up(V1_SQL),
        M::up(V2_SQL),
        M::up(V3_SQL),
        M::up(V4_SQL),
        M::up(V5_SQL),
        M::up(V6_SQL),
        M::up(V7_SQL),
        M::up(V8_SQL),
        M::up(V9_SQL),
        M::up(V10_SQL),
        M::up(V11_SQL),
        M::up(V12_SQL),
        M::up(V13_SQL),
        M::up(V14_SQL),
        M::up(V15_SQL),
        M::up(V16_SQL),
        M::up(V17_SQL),
        M::up(V18_SQL),
        M::up(V19_SQL),
        M::up(V20_SQL),
    ])
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
        assert_eq!(version, 20);
    }
}

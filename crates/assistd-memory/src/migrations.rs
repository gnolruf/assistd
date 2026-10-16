//! Schema migrations for the SQLite store; each `M::up` is one schema version.

use rusqlite::Connection;
use rusqlite_migration::{M, Migrations};

/// V1 schema, versioned by `PRAGMA user_version`; timestamps are Unix ms and
/// every foreign key is indexed. A session's current branch must be one of its own. Message, chunk, and memory ids use
/// `AUTOINCREMENT` so a queued embed job never lands on a row that reused a deleted id.
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

CREATE TABLE branches (
    id               INTEGER PRIMARY KEY,
    session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name             TEXT NOT NULL,
    parent_branch_id INTEGER REFERENCES branches(id) ON DELETE SET NULL,
    fork_point_seq   INTEGER,
    created_at       INTEGER NOT NULL,
    UNIQUE (session_id, name),
    UNIQUE (session_id, id)
) STRICT;
CREATE INDEX idx_branches_parent ON branches(parent_branch_id);

CREATE TABLE turns (
    id         INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE
) STRICT;
CREATE INDEX idx_turns_session ON turns(session_id);

CREATE TABLE messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id          INTEGER REFERENCES turns(id) ON DELETE SET NULL,
    created_at       INTEGER NOT NULL,
    role             INTEGER NOT NULL CHECK (role BETWEEN 0 AND 2),
    content          TEXT NOT NULL,
    reasoning        TEXT,
    tool_calls       TEXT,
    tool_call_id     TEXT,
    tool_name        TEXT,
    tool_exit_code   INTEGER,
    tool_duration_ms INTEGER CHECK (tool_duration_ms >= 0),
    CHECK (tool_calls IS NULL OR (role = 1 AND json_valid(tool_calls))),
    CHECK (reasoning IS NULL OR role = 1),
    CHECK ((role = 2) = (tool_call_id IS NOT NULL)),
    CHECK ((role = 2) = (tool_name IS NOT NULL)),
    CHECK ((role = 2) = (tool_exit_code IS NOT NULL)),
    CHECK ((role = 2) = (tool_duration_ms IS NOT NULL))
) STRICT;
CREATE INDEX idx_messages_turn ON messages(turn_id);
CREATE INDEX idx_messages_session ON messages(session_id);

CREATE TABLE branch_messages (
    branch_id  INTEGER NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    PRIMARY KEY (branch_id, seq),
    CHECK (seq >= 0)
) STRICT, WITHOUT ROWID;
CREATE INDEX idx_branch_messages_message ON branch_messages(message_id);

CREATE TABLE branch_summaries (
    branch_id   INTEGER PRIMARY KEY REFERENCES branches(id) ON DELETE CASCADE,
    through_seq INTEGER NOT NULL,
    content     TEXT NOT NULL
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
CREATE INDEX idx_chunk_embeddings_model ON chunk_embeddings(model_id);

CREATE TABLE memory_embeddings (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    model_id  INTEGER NOT NULL REFERENCES embedding_models(id),
    vector    BLOB NOT NULL CHECK (length(vector) > 0 AND length(vector) % 4 = 0)
) STRICT;
CREATE INDEX idx_memory_embeddings_model ON memory_embeddings(model_id);

CREATE TRIGGER memory_value_changed AFTER UPDATE OF value ON memories
WHEN old.value IS NOT new.value
BEGIN
    DELETE FROM memory_embeddings WHERE memory_id = new.id;
END;
";

/// The full migration set.
fn migrations() -> Migrations<'static> {
    Migrations::new(vec![M::up(V1_SQL)])
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

//! Schema-evolution backbone for the SQLite memory store.
//!
//! A single `Migrations` value is built up from a static `Vec<M>`; each
//! `M::up(SQL)` is one schema version and the runner is idempotent;
//! calling [`run`] against an already-migrated DB is a no-op. New
//! milestones append a new `M::up(...)` entry to [`migrations`] and bump
//! nothing else; the runner records progress in `schema_migrations`.
//!
//! The whole V1 schema lives in one `execute_batch`-style string so we
//! land it as a single atomic step. SQLite executes the statements in
//! order inside an implicit transaction (rusqlite_migration wraps each
//! migration in `BEGIN; ... COMMIT;`), so failures roll back cleanly.

use rusqlite::Connection;
use rusqlite_migration::{M, Migrations};

/// V1: full MVP schema. Tables in order:
/// - `schema_migrations` - version log for future upgrades.
/// - `sessions` - one row per daemon process (uuid PK), carries a
///   nullable `title` (filled asynchronously by an LLM-summarisation
///   task after the first agent response) and a `current_branch_id`
///   pointer into `branches`.
/// - `turns` - logical user-prompt-to-final-assistant grouping inside
///   a session.
/// - `conversations` - one row per `Message`. Tool results are
///   `role='tool'` rows with `tool_call_id` / `tool_name` set.
/// - `conversations_fts` - FTS5 mirror, kept in sync via triggers.
/// - `memories` - flat KV with provenance (source_conversation_id).
/// - `memory_embeddings` - sibling to `embeddings`, indexes the KV
///   rows for semantic recall. `UNIQUE(memory_id)` powers the
///   `INSERT ... ON CONFLICT(memory_id) DO UPDATE` upsert path so
///   re-saving an existing key overwrites the embedding in place and
///   no stale vector survives a value change.
/// - `branches` + `branch_messages` - named branches per session plus
///   a join table mapping `(branch_id, branch-local seq)` to
///   `conversation_id`. Forks share `conversations` rows across
///   branches via separate `branch_messages` entries; no row duplication.
/// - `conversation_chunks` + `embeddings` - chunked conversation text
///   with one embedding per chunk for semantic search.
const V1_SQL: &str = r#"
CREATE TABLE schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE sessions (
    id                 TEXT PRIMARY KEY,
    started_at         TEXT NOT NULL,
    ended_at           TEXT,
    daemon_pid         INTEGER NOT NULL,
    title              TEXT,
    current_branch_id  INTEGER
);

CREATE TABLE turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    user_text   TEXT NOT NULL
);
CREATE INDEX idx_turns_session ON turns(session_id);

CREATE TABLE conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id       INTEGER REFERENCES turns(id) ON DELETE SET NULL,
    seq           INTEGER NOT NULL,
    timestamp     TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('system','user','assistant','tool')),
    content       TEXT NOT NULL,
    tool_calls    TEXT,
    tool_call_id  TEXT,
    tool_name     TEXT
);
CREATE INDEX idx_conversations_session_seq ON conversations(session_id, seq);
CREATE INDEX idx_conversations_turn        ON conversations(turn_id);

CREATE VIRTUAL TABLE conversations_fts USING fts5(
    content,
    content='conversations',
    content_rowid='id'
);
CREATE TRIGGER conv_fts_ai AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER conv_fts_ad AFTER DELETE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
CREATE TRIGGER conv_fts_au AFTER UPDATE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO conversations_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE memories (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    key                    TEXT NOT NULL UNIQUE,
    value                  TEXT NOT NULL,
    source_conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL
);
CREATE INDEX idx_memories_key ON memories(key);

CREATE TABLE memory_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   INTEGER NOT NULL UNIQUE REFERENCES memories(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX idx_memory_embeddings_model ON memory_embeddings(model);

CREATE TABLE branches (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name              TEXT NOT NULL,
    parent_branch_id  INTEGER REFERENCES branches(id) ON DELETE SET NULL,
    fork_point_seq    INTEGER,
    created_at        TEXT NOT NULL,
    UNIQUE(session_id, name)
);
CREATE INDEX idx_branches_session ON branches(session_id);

CREATE TABLE branch_messages (
    branch_id       INTEGER NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    PRIMARY KEY (branch_id, seq)
);
CREATE INDEX idx_branch_messages_branch_seq ON branch_messages(branch_id, seq);
CREATE INDEX idx_branch_messages_conv ON branch_messages(conversation_id);

CREATE TABLE conversation_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    token_count     INTEGER,
    UNIQUE(conversation_id, chunk_index)
);
CREATE INDEX idx_chunks_conv ON conversation_chunks(conversation_id);

CREATE TABLE embeddings (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_chunk_id INTEGER NOT NULL UNIQUE REFERENCES conversation_chunks(id) ON DELETE CASCADE,
    model                 TEXT NOT NULL,
    dim                   INTEGER NOT NULL,
    vector                BLOB NOT NULL,
    created_at            TEXT NOT NULL
);
CREATE INDEX idx_embeddings_model ON embeddings(model);
"#;

/// Build the full migration set. `'static` because the SQL is embedded
/// in the binary; rusqlite_migration just needs read access.
pub fn migrations() -> Migrations<'static> {
    Migrations::new(vec![M::up(V1_SQL)])
}

/// Apply all pending migrations to `conn`. Idempotent: re-running on an
/// already-current DB is a no-op (rusqlite_migration consults the
/// internal `user_version` pragma + our `schema_migrations` table).
pub fn run(conn: &mut Connection) -> Result<(), rusqlite_migration::Error> {
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
    fn migrations_validate() {
        // rusqlite_migration ships a `validate` that re-runs every
        // migration against an in-memory DB and fails loudly on bad
        // SQL; catches schema typos before we ever ship a release.
        migrations().validate().expect("V1 SQL is well-formed");
    }

    #[test]
    fn run_creates_all_expected_objects() {
        let mut conn = open_in_memory();
        run(&mut conn).expect("first migration run");

        let names: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type IN ('table','index','trigger') AND name NOT LIKE 'sqlite_%' ORDER BY name")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();

        for expected in [
            "branch_messages",
            "branches",
            "conv_fts_ad",
            "conv_fts_ai",
            "conv_fts_au",
            "conversation_chunks",
            "conversations",
            "conversations_fts",
            "embeddings",
            "idx_branch_messages_branch_seq",
            "idx_branch_messages_conv",
            "idx_branches_session",
            "idx_chunks_conv",
            "idx_conversations_session_seq",
            "idx_conversations_turn",
            "idx_embeddings_model",
            "idx_memories_key",
            "idx_memory_embeddings_model",
            "idx_turns_session",
            "memories",
            "memory_embeddings",
            "schema_migrations",
            "sessions",
            "turns",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "expected {expected} in {names:?}"
            );
        }
    }

    #[test]
    fn run_is_idempotent() {
        // Running twice against the same DB must not error and must not
        // recreate tables. This is the contract every restart relies on.
        let mut conn = open_in_memory();
        run(&mut conn).expect("first run");
        run(&mut conn).expect("second run no-op");

        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='conversations'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
}

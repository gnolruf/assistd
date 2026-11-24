//! The schema of `memory.db`, applied through `rusqlite_migration`; each `M::up` is one
//! schema version.

use rusqlite::Connection;
use rusqlite_migration::{M, Migrations};

/// The whole schema of what exists nowhere else: transcripts, memories and tasks,
/// versioned by `PRAGMA user_version`; timestamps are Unix ms. Composite
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

-- What a tool result's call touched. `action` is 0 read, 1 wrote, 2 fetched, 3 recalled,
-- 4 saved; `target` is a file's absolute path, a URL, a session's uuid or a memory's key,
-- and `label` a fetched page's title.
CREATE TABLE accesses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    action     INTEGER NOT NULL CHECK (action BETWEEN 0 AND 4),
    target     BLOB NOT NULL CHECK (length(target) > 0),
    label      TEXT,
    CHECK (label IS NULL OR action = 2)
) STRICT;
CREATE INDEX idx_accesses_message ON accesses(message_id);
CREATE INDEX idx_accesses_target ON accesses(target);

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

//! The schema of `vectors.db`, which holds what the embedding model computed and
//! names no row of another file: a vector is found by the hash of its text.

use rusqlite::Connection;

/// The `PRAGMA user_version` of a `vectors.db` holding [`SCHEMA_SQL`]. Any change to
/// the schema takes the next number, and every older file is then replaced by an
/// empty one.
pub(crate) const VERSION: i64 = 1;

const SCHEMA_SQL: &str = r"
-- The vector of each distinct text the chunks of one kind hold, by the first 16 bytes
-- of the SHA-256 of that text. A vector is a little-endian `f32` step, then one `i8`
-- count of steps per dimension. A NULL vector marks text the model could not embed,
-- so it is not retried.
CREATE TABLE message_vectors (
    hash   BLOB NOT NULL UNIQUE CHECK (length(hash) = 16),
    vector BLOB CHECK (length(vector) > 4)
) STRICT;

CREATE TABLE memory_vectors (
    hash   BLOB NOT NULL UNIQUE CHECK (length(hash) = 16),
    vector BLOB CHECK (length(vector) > 4)
) STRICT;

CREATE TABLE document_vectors (
    hash   BLOB NOT NULL UNIQUE CHECK (length(hash) = 16),
    vector BLOB CHECK (length(vector) > 4)
) STRICT;

-- At most one row: the embedding space every stored vector is in.
CREATE TABLE embedding_meta (
    id              INTEGER PRIMARY KEY CHECK (id = 0),
    model           TEXT NOT NULL,
    dim             INTEGER NOT NULL CHECK (dim > 0),
    server_args     TEXT NOT NULL,
    document_prefix TEXT NOT NULL
) STRICT;
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

//! What ties the message and memory chunks of `index.db` to the rows of `memory.db`
//! they are cut from, which no key or trigger reaches across the two files. Every
//! statement runs on the index writer, which has `memory.db` attached to read.

use rusqlite::{Connection, OptionalExtension, Transaction};

use crate::chunking::{ChunkingConfig, TextHash, chunk_ranges, text_hash};
use crate::{ChunkSource, MemoryId, MessageId, PersistedRole};

use super::chunks::{ChunkOwner, delete_unheld_vectors, insert_chunk};

/// The kinds of chunk cut from a table of `memory.db`, each with that table.
const MEMORY_DATABASE_SOURCES: [(ChunkSource, &str); 2] = [
    (ChunkSource::Message, "messages"),
    (ChunkSource::Memory, "memories"),
];

/// The random value a `memory.db` is told from every other by.
type DatabaseIdentity = [u8; 16];

/// What bringing `index.db` in line with `memory.db` changed.
#[derive(Debug, Clone, Copy)]
pub(super) struct Reconciled {
    /// Whether the chunks were cut from another `memory.db` and dropped for it.
    pub built_from_another: bool,
    /// Chunks removed because their message or memory is gone.
    pub orphans_removed: usize,
    pub memories: RecutMemories,
    /// Memory vectors removed because no memory's chunk holds their text.
    pub unheld_vectors_removed: usize,
}

impl Reconciled {
    /// Whether any chunk or vector was removed.
    pub(super) fn erased_rows(&self) -> bool {
        self.built_from_another
            || self.orphans_removed > 0
            || self.memories.replaced > 0
            || self.unheld_vectors_removed > 0
    }
}

/// The memory chunks cut again because they no longer covered their memory's value.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RecutMemories {
    /// Chunks removed for a value that changed.
    pub replaced: usize,
    /// Chunks cut for a value that had none or another.
    pub cut: usize,
}

/// Bring the message and memory chunks in line with the attached `memory.db`: drop
/// them all when they were cut from another, remove those whose source is gone, and
/// cut again each memory's chunk that does not cover its value. Every memory then
/// has its chunk, so the memory vectors none of them holds are removed.
pub(super) fn reconcile(tx: &Transaction<'_>) -> rusqlite::Result<Reconciled> {
    let built_from_another = adopt_memory_database(tx)?;
    let orphans_removed = remove_orphaned_chunks(tx)?;
    lower_mark_to_stored_messages(tx)?;
    let memories = recut_changed_memories(tx, None)?;
    let unheld_vectors_removed = delete_unheld_vectors(tx, ChunkSource::Memory)?;
    Ok(Reconciled {
        built_from_another,
        orphans_removed,
        memories,
        unheld_vectors_removed,
    })
}

/// Remove the chunks whose message or memory `memory.db` no longer holds, returning
/// how many.
pub(super) fn remove_orphaned_chunks(conn: &Connection) -> rusqlite::Result<usize> {
    MEMORY_DATABASE_SOURCES
        .iter()
        .map(|(source, table)| {
            conn.prepare_cached(&format!(
                "DELETE FROM chunks
                 WHERE kind = ?1
                   AND NOT EXISTS (SELECT 1 FROM {table} src WHERE src.id = chunks.source)"
            ))?
            .execute([source])
        })
        .sum()
}

/// Cut again the chunk of each memory, or of `only`, whose chunk does not cover the
/// value `memory.db` holds for it; a value of whitespace alone has no chunk.
pub(super) fn recut_changed_memories(
    tx: &Transaction<'_>,
    only: Option<MemoryId>,
) -> rusqlite::Result<RecutMemories> {
    let memories: Vec<(MemoryId, String, Option<TextHash>)> = tx
        .prepare_cached(
            "SELECT m.id, m.value, cc.hash
             FROM memories m LEFT JOIN chunks cc ON cc.kind = ?1 AND cc.source = m.id
             WHERE ?2 IS NULL OR m.id = ?2",
        )?
        .query_map(rusqlite::params![ChunkSource::Memory, only], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut recut = RecutMemories::default();
    for (memory, value, chunk_hash) in memories {
        let value_hash = (!value.trim().is_empty()).then(|| text_hash(value.as_bytes()));
        if chunk_hash == value_hash {
            continue;
        }
        recut.replaced += tx
            .prepare_cached("DELETE FROM chunks WHERE kind = ?1 AND source = ?2")?
            .execute(rusqlite::params![ChunkSource::Memory, memory])?;
        if value_hash.is_some() {
            insert_chunk(tx, ChunkOwner::Memory(memory), &value, &(0..value.len()))?;
            recut.cut += 1;
        }
    }
    Ok(recut)
}

/// Up to `limit` user and assistant messages saved after both `after` and the highest
/// message chunked, in id order, each with its content.
pub(super) fn unchunked_messages(
    conn: &Connection,
    after: Option<MessageId>,
    limit: usize,
) -> rusqlite::Result<Vec<(MessageId, String)>> {
    conn.prepare_cached(
        "SELECT m.id, m.content FROM messages m
         WHERE m.id > max(ifnull(?1, 0), (SELECT chunked_through FROM chunked_source))
           AND m.role IN (?2, ?3)
         ORDER BY m.id
         LIMIT ?4",
    )?
    .query_map(
        rusqlite::params![
            after,
            PersistedRole::User,
            PersistedRole::Assistant,
            i64::try_from(limit).unwrap_or(i64::MAX)
        ],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?
    .collect()
}

/// Cut the chunks of each of `messages`, listed in id order with their content, that
/// is still stored and above the highest message chunked, then record the last of
/// them as that highest. Returns how many chunks were added.
pub(super) fn chunk_messages(
    tx: &Transaction<'_>,
    messages: &[(MessageId, String)],
) -> rusqlite::Result<u64> {
    let Some((last, _)) = messages.last() else {
        return Ok(0);
    };
    let mut stored_and_unchunked = tx.prepare_cached(
        "SELECT EXISTS (SELECT 1 FROM messages m
                        WHERE m.id = ?1
                          AND m.id > (SELECT chunked_through FROM chunked_source))",
    )?;
    let mut added = 0u64;
    for (message, content) in messages {
        if !stored_and_unchunked.query_row([message], |row| row.get::<_, bool>(0))? {
            continue;
        }
        for bytes in chunk_ranges(content, ChunkingConfig::MESSAGES) {
            insert_chunk(tx, ChunkOwner::Message(*message), content, &bytes)?;
            added += 1;
        }
    }
    tx.prepare_cached("UPDATE chunked_source SET chunked_through = max(chunked_through, ?1)")?
        .execute([last])?;
    Ok(added)
}

/// Remove the message vectors whose text no message chunk holds, returning how many;
/// `None`, removing nothing, while a saved message waits to be chunked.
pub(super) fn remove_unheld_message_vectors(conn: &Connection) -> rusqlite::Result<Option<usize>> {
    if !unchunked_messages(conn, None, 1)?.is_empty() {
        return Ok(None);
    }
    delete_unheld_vectors(conn, ChunkSource::Message).map(Some)
}

/// Record the attached `memory.db` as the one whose messages and memories are chunked
/// here. When another was recorded, drop every chunk cut from its rows and have every
/// message chunked again; returns whether one was.
fn adopt_memory_database(tx: &Transaction<'_>) -> rusqlite::Result<bool> {
    let attached: DatabaseIdentity =
        tx.query_row("SELECT identity FROM database_identity", [], |row| {
            row.get(0)
        })?;
    let recorded: Option<DatabaseIdentity> = tx
        .query_row("SELECT memory_identity FROM chunked_source", [], |row| {
            row.get(0)
        })
        .optional()?;
    if recorded == Some(attached) {
        return Ok(false);
    }
    tx.execute(
        "DELETE FROM chunks WHERE kind IN (?1, ?2)",
        [ChunkSource::Message, ChunkSource::Memory],
    )?;
    tx.execute(
        "INSERT INTO chunked_source (id, memory_identity, chunked_through) VALUES (0, ?1, 0)
         ON CONFLICT(id) DO UPDATE
         SET memory_identity = excluded.memory_identity, chunked_through = 0",
        [attached],
    )?;
    Ok(recorded.is_some())
}

/// Lower the highest message chunked to the highest stored, so that a `memory.db` put
/// back from an older copy, whose next messages take ids the mark has passed, still
/// has them chunked.
fn lower_mark_to_stored_messages(conn: &Connection) -> rusqlite::Result<usize> {
    conn.execute(
        "UPDATE chunked_source
         SET chunked_through = min(chunked_through,
                                   (SELECT ifnull(max(id), 0) FROM messages))",
        [],
    )
}

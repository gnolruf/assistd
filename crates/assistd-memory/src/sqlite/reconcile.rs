//! What ties the message, memory and access chunks of `index.db` to the rows of `memory.db`
//! they are cut from, which no key or trigger reaches across the two files. Every
//! statement runs on the index writer, which has `memory.db` attached to read.

use rusqlite::{Connection, OptionalExtension, Transaction};

use crate::chunking::{ChunkingConfig, TextHash, chunk_ranges, text_hash};
use crate::{AccessId, ChunkSource, MemoryId, MessageId, PersistedRole};

use super::chunks::{
    ACCESS_TEXT, ChunkOwner, TOUCHED_FILE_OR_PAGE, delete_unheld_vectors, insert_chunk,
};

/// The kinds of chunk cut from a table of `memory.db`, each with that table.
const MEMORY_DATABASE_SOURCES: [(ChunkSource, &str); 3] = [
    (ChunkSource::Message, "messages"),
    (ChunkSource::Memory, "memories"),
    (ChunkSource::Access, "accesses"),
];

/// What bringing `index.db` in line with `memory.db` changed.
#[derive(Debug, Clone, Copy)]
pub(super) struct Reconciled {
    /// Chunks removed because their message, memory or access is gone.
    pub orphans_removed: usize,
    /// Whether the message and access chunks were cut from other messages than those
    /// stored and dropped for it.
    pub cut_from_other_messages: bool,
    pub memories: RecutMemories,
    /// Memory vectors removed because no memory's chunk holds their text.
    pub unheld_vectors_removed: usize,
}

impl Reconciled {
    /// Whether any chunk or vector was removed.
    pub(super) fn erased_rows(&self) -> bool {
        self.orphans_removed > 0
            || self.cut_from_other_messages
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

/// Bring the message, memory and access chunks in line with the attached `memory.db`:
/// remove those whose source is gone, drop every message and access chunk when they
/// were cut from other messages, and cut again each memory's chunk that does not cover
/// its value. Every memory then has its chunk, so the memory vectors none of them holds
/// are removed.
pub(super) fn reconcile(tx: &Transaction<'_>) -> rusqlite::Result<Reconciled> {
    let orphans_removed = remove_orphaned_chunks(tx)?;
    lower_mark_to_stored_messages(tx)?;
    let cut_from_other_messages = drop_chunks_cut_from_other_messages(tx)?;
    let memories = recut_changed_memories(tx, None)?;
    let unheld_vectors_removed = delete_unheld_vectors(tx, ChunkSource::Memory)?;
    Ok(Reconciled {
        orphans_removed,
        cut_from_other_messages,
        memories,
        unheld_vectors_removed,
    })
}

/// Remove the chunks whose message, memory or access `memory.db` no longer holds,
/// returning how many.
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

/// Up to `limit` messages saved after both `after` and the highest message chunked
/// that have chunks to cut, in id order: a user or assistant message with its content,
/// and a tool result stored with an access to a file or a page with none.
pub(super) fn unchunked_messages(
    conn: &Connection,
    after: Option<MessageId>,
    limit: usize,
) -> rusqlite::Result<Vec<(MessageId, Option<String>)>> {
    conn.prepare_cached(&format!(
        "SELECT m.id, CASE WHEN m.role IN (?2, ?3) THEN m.content END FROM messages m
         WHERE m.id > max(ifnull(?1, 0), (SELECT chunked_through FROM chunked_source))
           AND (m.role IN (?2, ?3)
                OR EXISTS (SELECT 1 FROM accesses src
                           WHERE src.message_id = m.id AND {TOUCHED_FILE_OR_PAGE}))
         ORDER BY m.id
         LIMIT ?4"
    ))?
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

/// Cut the chunks of each of `messages`, listed in id order with the content of those
/// that have any to cut, and of the accesses stored with it, when it is still stored
/// and above the highest message chunked; then record the last of them as that
/// highest. Returns how many chunks were added.
pub(super) fn chunk_messages(
    tx: &Transaction<'_>,
    messages: &[(MessageId, Option<String>)],
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
        if let Some(content) = content {
            for bytes in chunk_ranges(content, ChunkingConfig::MESSAGES) {
                insert_chunk(tx, ChunkOwner::Message(*message), content, &bytes)?;
                added += 1;
            }
        }
        added += chunk_accesses(tx, *message)?;
    }
    tx.prepare_cached("UPDATE chunked_source SET chunked_through = max(chunked_through, ?1)")?
        .execute([last])?;
    Ok(added)
}

/// Remove the message and access vectors whose text no chunk of their kind holds,
/// returning how many; `None`, removing nothing, while a saved message waits to be
/// chunked.
pub(super) fn remove_unheld_transcript_vectors(
    conn: &Connection,
) -> rusqlite::Result<Option<usize>> {
    if !unchunked_messages(conn, None, 1)?.is_empty() {
        return Ok(None);
    }
    let removed = delete_unheld_vectors(conn, ChunkSource::Message)?
        + delete_unheld_vectors(conn, ChunkSource::Access)?;
    Ok(Some(removed))
}

/// Cut one chunk for each access to a file or a page stored with `message`, over its
/// whole text, returning how many. An access whose target is not UTF-8 gets none.
fn chunk_accesses(tx: &Transaction<'_>, message: MessageId) -> rusqlite::Result<u64> {
    let accesses: Vec<(AccessId, Vec<u8>)> = tx
        .prepare_cached(&format!(
            "SELECT src.id, CAST({ACCESS_TEXT} AS BLOB) FROM accesses src
             WHERE src.message_id = ?1 AND {TOUCHED_FILE_OR_PAGE}
             ORDER BY src.id"
        ))?
        .query_map([message], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let mut added = 0u64;
    for (access, text) in accesses {
        let Ok(text) = String::from_utf8(text) else {
            continue;
        };
        insert_chunk(tx, ChunkOwner::Access(access), &text, &(0..text.len()))?;
        added += 1;
    }
    Ok(added)
}

/// Drop every message and access chunk and have every message chunked again when the
/// chunk of the highest message that has one does not cover that message's text. A
/// message is never rewritten, so one check tells whether the chunks below it were cut
/// from the messages stored. The vectors stay for the chunks cut again. Returns whether
/// the chunks were dropped.
fn drop_chunks_cut_from_other_messages(tx: &Transaction<'_>) -> rusqlite::Result<bool> {
    let newest: Option<(TextHash, Option<Vec<u8>>)> = tx
        .query_row(
            "SELECT cc.hash,
                    (SELECT substr(CAST(m.content AS BLOB),
                                   cc.byte_start + 1, cc.byte_end - cc.byte_start)
                     FROM messages m WHERE m.id = cc.source)
             FROM chunks cc
             WHERE cc.kind = ?1
             ORDER BY cc.source DESC
             LIMIT 1",
            [ChunkSource::Message],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((chunk_hash, covered)) = newest else {
        return Ok(false);
    };
    if covered.is_some_and(|bytes| text_hash(&bytes) == chunk_hash) {
        return Ok(false);
    }
    let transcript_kinds = [ChunkSource::Message, ChunkSource::Access];
    tx.execute(
        "DELETE FROM chunks WHERE kind IN (?1, ?2)",
        transcript_kinds,
    )?;
    tx.execute(
        "DELETE FROM removed_chunk_texts WHERE kind IN (?1, ?2)",
        transcript_kinds,
    )?;
    tx.execute("UPDATE chunked_source SET chunked_through = 0", [])?;
    Ok(true)
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

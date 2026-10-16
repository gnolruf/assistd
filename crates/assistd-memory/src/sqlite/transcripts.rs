//! SQLite-backed [`TranscriptStore`] over sessions, turns, branches, and messages.

use std::num::NonZeroUsize;
use std::ops::Range;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::types::Type;
use rusqlite::{ErrorCode, OptionalExtension, Row, Transaction};

use crate::chunking::{ChunkingConfig, chunk_ranges};
use crate::{
    AppendedMessage, BranchId, BranchInfo, BranchSummary, ChunkId, HistoryRow, MemoryError,
    MessageId, MessageKind, OpenedTurn, PersistedMessage, PersistedRole, PersistedToolCall, Result,
    ResumeCandidate, Seq, SessionId, ToolOutcome, TranscriptStore, TurnId, UndoOutcome,
};

use super::connection::{SqliteHandle, truncate_wal};
use super::timestamp::{now_millis, optional_timestamp_at, timestamp_at};

/// Columns [`branch_info`] reads, over `branches b JOIN sessions s`.
const BRANCH_INFO_SELECT: &str = "
    SELECT  b.id,
            s.uuid,
            s.started_at,
            s.ended_at,
            s.title,
            b.name,
            b.parent_branch_id,
            (SELECT name FROM branches p WHERE p.id = b.parent_branch_id),
            b.fork_point_seq,
            b.created_at,
            (SELECT COUNT(*) FROM branch_messages bm WHERE bm.branch_id = b.id),
            (b.id IS s.current_branch_id) AS is_current,
            MAX(b.created_at, COALESCE(
                (SELECT m.created_at
                 FROM branch_messages bm JOIN messages m ON m.id = bm.message_id
                 WHERE bm.branch_id = b.id ORDER BY bm.seq DESC LIMIT 1),
                b.created_at)) AS last_active_at
    FROM branches b JOIN sessions s ON s.id = b.session_id";

/// SQLite-backed [`TranscriptStore`].
#[derive(Debug, Clone)]
pub struct SqliteTranscriptStore {
    handle: SqliteHandle,
}

impl SqliteTranscriptStore {
    /// Store over the shared database handle.
    pub fn new(handle: SqliteHandle) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl TranscriptStore for SqliteTranscriptStore {
    async fn end_session(&self, session: SessionId) -> Result<()> {
        let ended = now_millis();
        let updated = self
            .handle
            .writer()
            .call(move |c| {
                c.execute(
                    "UPDATE sessions SET ended_at = ?1 WHERE uuid = ?2",
                    rusqlite::params![ended, session],
                )
            })
            .await
            .map_err(MemoryError::sqlite("end_session"))?;
        session_updated(updated, session)
    }

    async fn begin_turn(&self, branch: BranchId, prompt: PersistedMessage) -> Result<OpenedTurn> {
        let row = MessageRow::new(prompt, now_millis());
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let slot = claim_next_seq(&tx, branch)?;
                let turn = tx
                    .prepare_cached("INSERT INTO turns (session_id) VALUES (?1) RETURNING id")?
                    .query_row(rusqlite::params![slot.session_rowid], |r| r.get(0))?;
                let message = store_message(&tx, branch, slot, turn, &row)?;
                tx.commit()?;
                Ok(OpenedTurn { turn, message })
            })
            .await
            .map_err(MemoryError::sqlite_or_missing(
                "begin_turn",
                MemoryError::BranchNotFound(branch),
            ))
    }

    async fn begin_session_with_main_branch(
        &self,
        session: SessionId,
        daemon_pid: u32,
    ) -> Result<BranchId> {
        let started = now_millis();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let session_rowid: i64 = tx.query_row(
                    "INSERT INTO sessions (uuid, started_at, daemon_pid) VALUES (?1, ?2, ?3)
                     RETURNING id",
                    rusqlite::params![session, started, daemon_pid],
                    |r| r.get(0),
                )?;
                let branch_id: BranchId = tx.query_row(
                    "INSERT INTO branches (session_id, name, created_at) VALUES (?1, 'main', ?2)
                     RETURNING id",
                    rusqlite::params![session_rowid, started],
                    |r| r.get(0),
                )?;
                tx.execute(
                    "UPDATE sessions SET current_branch_id = ?1 WHERE id = ?2",
                    rusqlite::params![branch_id, session_rowid],
                )?;
                tx.commit()?;
                Ok(branch_id)
            })
            .await
            .map_err(MemoryError::sqlite("begin_session_with_main_branch"))
    }

    async fn set_current_branch(&self, session: SessionId, branch: BranchId) -> Result<()> {
        let updated = self
            .handle
            .writer()
            .call(move |c| {
                c.execute(
                    "UPDATE sessions SET current_branch_id = ?1 WHERE uuid = ?2",
                    rusqlite::params![branch, session],
                )
            })
            .await
            .map_err(|e| set_current_branch_error(e, branch))?;
        session_updated(updated, session)
    }

    async fn current_branch(&self, session: SessionId) -> Result<Option<BranchId>> {
        self.handle
            .reader()
            .call(move |c| {
                c.query_row(
                    "SELECT current_branch_id FROM sessions WHERE uuid = ?1",
                    rusqlite::params![session],
                    |r| r.get::<_, Option<BranchId>>(0),
                )
                .optional()
                .map(Option::flatten)
            })
            .await
            .map_err(MemoryError::sqlite("current_branch"))
    }

    async fn append_message_to_branch(
        &self,
        branch: BranchId,
        turn: TurnId,
        msg: PersistedMessage,
    ) -> Result<AppendedMessage> {
        let row = MessageRow::new(msg, now_millis());
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let slot = claim_next_seq(&tx, branch)?;
                let appended = store_message(&tx, branch, slot, turn, &row)?;
                tx.commit()?;
                Ok(appended)
            })
            .await
            .map_err(MemoryError::sqlite_or_missing(
                "append_message_to_branch",
                MemoryError::BranchNotFound(branch),
            ))
    }

    async fn list_branches(&self) -> Result<Vec<BranchInfo>> {
        self.handle
            .reader()
            .call(|c| -> rusqlite::Result<_> {
                let sql = format!("{BRANCH_INFO_SELECT} ORDER BY last_active_at DESC, b.id ASC");
                let mut stmt = c.prepare_cached(&sql)?;
                let rows: Vec<BranchInfo> = stmt
                    .query_map([], branch_info)?
                    .collect::<std::result::Result<_, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("list_branches"))
    }

    async fn branch_info(&self, branch: BranchId) -> Result<Option<BranchInfo>> {
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let sql = format!("{BRANCH_INFO_SELECT} WHERE b.id = ?1");
                c.prepare_cached(&sql)?
                    .query_row(rusqlite::params![branch], branch_info)
                    .optional()
            })
            .await
            .map_err(MemoryError::sqlite("branch_info"))
    }

    async fn resolve_branch(
        &self,
        target: &str,
        prefer_session: Option<SessionId>,
    ) -> Result<Option<(SessionId, BranchId)>> {
        let (session_pattern, name) = match target.split_once('/') {
            Some((prefix, name)) => (like_prefix_pattern(prefix), name.to_string()),
            None => ("%".to_string(), target.to_string()),
        };
        self.handle
            .reader()
            .call(move |c| {
                c.prepare_cached(
                    "SELECT s.uuid, b.id
                     FROM branches b JOIN sessions s ON s.id = b.session_id
                     WHERE b.name = ?1 AND s.uuid LIKE ?2 ESCAPE '\\'
                     ORDER BY s.uuid IS ?3 DESC, s.started_at DESC LIMIT 1",
                )?
                .query_row(
                    rusqlite::params![name, session_pattern, prefer_session],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()
            })
            .await
            .map_err(MemoryError::sqlite("resolve_branch"))
    }

    async fn fork_and_switch(&self, src: BranchId, new_name: &str) -> Result<BranchId> {
        let name = new_name.to_string();
        let created = now_millis();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let new_branch: BranchId = tx.query_row(
                    "INSERT INTO branches
                        (session_id, name, parent_branch_id, fork_point_seq, next_seq, created_at)
                     SELECT session_id, ?2, id,
                            (SELECT MAX(seq) FROM branch_messages WHERE branch_id = ?1),
                            next_seq, ?3
                     FROM branches WHERE id = ?1
                     RETURNING id",
                    rusqlite::params![src, name, created],
                    |r| r.get(0),
                )?;
                tx.execute(
                    "INSERT INTO branch_messages (branch_id, seq, message_id)
                     SELECT ?1, seq, message_id FROM branch_messages WHERE branch_id = ?2",
                    rusqlite::params![new_branch, src],
                )?;
                tx.execute(
                    "INSERT INTO branch_summaries (branch_id, through_seq, content)
                     SELECT ?1, through_seq, content FROM branch_summaries WHERE branch_id = ?2",
                    rusqlite::params![new_branch, src],
                )?;
                tx.execute(
                    "UPDATE sessions SET current_branch_id = ?1
                     WHERE id = (SELECT session_id FROM branches WHERE id = ?1)",
                    rusqlite::params![new_branch],
                )?;
                tx.commit()?;
                Ok(new_branch)
            })
            .await
            .map_err(|e| fork_error(e, src, new_name))
    }

    async fn load_branch_history(&self, branch: BranchId) -> Result<Vec<HistoryRow>> {
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare_cached(
                    "SELECT  m.id,
                             bm.seq,
                             m.role,
                             m.content,
                             m.reasoning,
                             m.tool_calls,
                             m.tool_call_id,
                             m.tool_name,
                             m.tool_exit_code,
                             m.tool_duration_ms
                     FROM branch_messages bm JOIN messages m ON m.id = bm.message_id
                     WHERE bm.branch_id = ?1
                     ORDER BY bm.seq ASC",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![branch], history_row)?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("load_branch_history"))
    }

    async fn record_branch_summary(
        &self,
        branch: BranchId,
        summary: String,
        kept_turns: NonZeroUsize,
    ) -> Result<()> {
        let offset = kept_turns.get() - 1;
        self.handle
            .writer()
            .call(move |c| {
                c.execute(
                    "INSERT OR REPLACE INTO branch_summaries (branch_id, through_seq, content)
                     SELECT ?1, through_seq, ?2
                     FROM (
                         SELECT (SELECT MAX(prev.seq) FROM branch_messages prev
                                 WHERE prev.branch_id = ?1 AND prev.seq < bm.seq) AS through_seq
                         FROM branch_messages bm JOIN messages m ON m.id = bm.message_id
                         WHERE bm.branch_id = ?1 AND m.role = ?3
                         ORDER BY bm.seq DESC LIMIT 1 OFFSET ?4
                     )
                     WHERE through_seq IS NOT NULL",
                    rusqlite::params![branch, summary, PersistedRole::User, offset],
                )
            })
            .await
            .map_err(MemoryError::sqlite("record_branch_summary"))?;
        Ok(())
    }

    async fn load_branch_summary(&self, branch: BranchId) -> Result<Option<BranchSummary>> {
        self.handle
            .reader()
            .call(move |c| {
                c.query_row(
                    "SELECT through_seq, content FROM branch_summaries WHERE branch_id = ?1",
                    rusqlite::params![branch],
                    |r| {
                        Ok(BranchSummary {
                            through_seq: r.get(0)?,
                            content: r.get(1)?,
                        })
                    },
                )
                .optional()
            })
            .await
            .map_err(MemoryError::sqlite("load_branch_summary"))
    }

    async fn latest_branch_activity(&self, branch: BranchId) -> Result<Option<DateTime<Utc>>> {
        self.handle
            .reader()
            .call(move |c| {
                c.query_row(
                    "SELECT m.created_at
                     FROM branch_messages bm
                     JOIN messages m ON m.id = bm.message_id
                     WHERE bm.branch_id = ?1
                     ORDER BY bm.seq DESC LIMIT 1",
                    rusqlite::params![branch],
                    |r| timestamp_at(r, 0),
                )
                .optional()
            })
            .await
            .map_err(MemoryError::sqlite("latest_branch_activity"))
    }

    async fn undo_last_turn(&self, branch: BranchId) -> Result<UndoOutcome> {
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let Some(turn) = latest_turn_on_branch(&tx, branch)? else {
                    return Ok(UndoOutcome::default());
                };
                let last_user_text = first_user_text_of_turn(&tx, turn)?;
                let dropped_summaries = drop_summary_covering_turn(&tx, branch, turn)?;
                let removed = tx.execute(
                    "DELETE FROM branch_messages
                     WHERE branch_id = ?1
                       AND message_id IN (SELECT id FROM messages WHERE turn_id = ?2)",
                    rusqlite::params![branch, turn],
                )?;
                let deleted_messages = delete_unreferenced_turn(&tx, turn)?;
                tx.commit()?;
                if dropped_summaries + deleted_messages > 0 {
                    truncate_wal(c)?;
                }
                Ok(UndoOutcome {
                    removed_messages: removed,
                    last_user_text,
                })
            })
            .await
            .map_err(MemoryError::sqlite("undo_last_turn"))
    }

    async fn find_resumable_session(&self) -> Result<Option<ResumeCandidate>> {
        self.handle
            .reader()
            .call(|c| {
                c.query_row(
                    "SELECT uuid, current_branch_id, daemon_pid, started_at
                     FROM sessions
                     WHERE ended_at IS NULL AND current_branch_id IS NOT NULL
                     ORDER BY started_at DESC LIMIT 1",
                    [],
                    |r| {
                        Ok(ResumeCandidate {
                            session_id: r.get(0)?,
                            current_branch_id: r.get(1)?,
                            daemon_pid: r.get(2)?,
                            started_at: timestamp_at(r, 3)?,
                        })
                    },
                )
                .optional()
            })
            .await
            .map_err(MemoryError::sqlite("find_resumable_session"))
    }

    async fn session_title(&self, session: SessionId) -> Result<Option<String>> {
        self.handle
            .reader()
            .call(move |c| {
                c.query_row(
                    "SELECT title FROM sessions WHERE uuid = ?1",
                    rusqlite::params![session],
                    |r| r.get::<_, Option<String>>(0),
                )
                .optional()
                .map(Option::flatten)
            })
            .await
            .map_err(MemoryError::sqlite("session_title"))
    }

    async fn set_session_title(&self, session: SessionId, title: &str) -> Result<()> {
        let title = title.to_string();
        self.handle
            .writer()
            .call(move |c| {
                c.execute(
                    "UPDATE sessions SET title = ?1 WHERE uuid = ?2",
                    rusqlite::params![title, session],
                )
            })
            .await
            .map_err(MemoryError::sqlite("set_session_title"))?;
        Ok(())
    }
}

/// One `messages` row ready to insert, with the byte ranges of the chunks
/// embedded from it; tool results get none.
struct MessageRow {
    created_at: i64,
    role: PersistedRole,
    content: String,
    columns: MessageColumns,
    chunk_bytes: Vec<Range<usize>>,
}

impl MessageRow {
    fn new(msg: PersistedMessage, created_at: i64) -> Self {
        let role = msg.role();
        let chunk_bytes = match role {
            PersistedRole::User | PersistedRole::Assistant => {
                chunk_ranges(&msg.content, &ChunkingConfig::MESSAGES)
            }
            PersistedRole::Tool => Vec::new(),
        };
        Self {
            created_at,
            role,
            content: msg.content,
            columns: MessageColumns::from_kind(msg.kind),
            chunk_bytes,
        }
    }
}

/// A claimed position on a branch: the branch's session rowid and the taken `seq`.
#[derive(Clone, Copy)]
struct BranchSlot {
    session_rowid: i64,
    seq: Seq,
}

/// The role-specific `messages` columns of a [`MessageKind`], `NULL` where the role has
/// none; empty reasoning and an empty call list are stored as `NULL` too.
#[derive(Default)]
struct MessageColumns {
    reasoning: Option<String>,
    tool_calls: Vec<PersistedToolCall>,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
    tool_exit_code: Option<i64>,
    tool_duration_ms: Option<u64>,
}

impl MessageColumns {
    fn from_kind(kind: MessageKind) -> Self {
        match kind {
            MessageKind::User => Self::default(),
            MessageKind::Assistant {
                reasoning,
                tool_calls,
            } => Self {
                reasoning: Some(reasoning).filter(|r| !r.is_empty()),
                tool_calls,
                ..Self::default()
            },
            MessageKind::Tool {
                call_id,
                tool_name,
                outcome,
            } => Self {
                tool_call_id: Some(call_id),
                tool_name: Some(tool_name),
                tool_exit_code: Some(outcome.exit_code),
                tool_duration_ms: Some(outcome.duration_ms),
                ..Self::default()
            },
        }
    }
}

/// Take `branch`'s next `seq`.
fn claim_next_seq(tx: &Transaction<'_>, branch: BranchId) -> rusqlite::Result<BranchSlot> {
    tx.prepare_cached(
        "UPDATE branches SET next_seq = next_seq + 1 WHERE id = ?1
         RETURNING session_id, next_seq - 1",
    )?
    .query_row(rusqlite::params![branch], |r| {
        Ok(BranchSlot {
            session_rowid: r.get(0)?,
            seq: r.get(1)?,
        })
    })
}

/// Insert `row` as part of `turn` at `slot` on `branch`, along with its chunks.
fn store_message(
    tx: &Transaction<'_>,
    branch: BranchId,
    slot: BranchSlot,
    turn: TurnId,
    row: &MessageRow,
) -> rusqlite::Result<AppendedMessage> {
    let id = insert_message(tx, slot.session_rowid, turn, row)?;
    tx.prepare_cached(
        "INSERT INTO branch_messages (branch_id, seq, message_id) VALUES (?1, ?2, ?3)",
    )?
    .execute(rusqlite::params![branch, slot.seq, id])?;
    let chunks = insert_chunks(tx, id, row)?;
    Ok(AppendedMessage { id, chunks })
}

/// Insert `row` into the session with rowid `session_rowid`, returning its id.
fn insert_message(
    tx: &Transaction<'_>,
    session_rowid: i64,
    turn: TurnId,
    row: &MessageRow,
) -> rusqlite::Result<MessageId> {
    let columns = &row.columns;
    tx.prepare_cached(
        "INSERT INTO messages
            (session_id, turn_id, created_at, role, content, reasoning, tool_calls,
             tool_call_id, tool_name, tool_exit_code, tool_duration_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
         RETURNING id",
    )?
    .query_row(
        rusqlite::params![
            session_rowid,
            turn,
            row.created_at,
            row.role,
            row.content,
            columns.reasoning,
            tool_calls_json(&columns.tool_calls)?,
            columns.tool_call_id,
            columns.tool_name,
            columns.tool_exit_code,
            columns.tool_duration_ms,
        ],
        |r| r.get(0),
    )
}

/// Store `row`'s chunks under `message`, returning each as `(id, chunk text)` in order.
fn insert_chunks(
    tx: &Transaction<'_>,
    message: MessageId,
    row: &MessageRow,
) -> rusqlite::Result<Vec<(ChunkId, String)>> {
    let mut insert = tx.prepare_cached(
        "INSERT INTO message_chunks (message_id, byte_start, byte_end)
         VALUES (?1, ?2, ?3)
         RETURNING id",
    )?;
    row.chunk_bytes
        .iter()
        .map(|bytes| {
            let id = insert.query_row(rusqlite::params![message, bytes.start, bytes.end], |r| {
                r.get(0)
            })?;
            Ok((id, row.content[bytes.clone()].to_string()))
        })
        .collect()
}

/// `calls` as the `messages.tool_calls` JSON array, or `None` when there are none.
fn tool_calls_json(calls: &[PersistedToolCall]) -> rusqlite::Result<Option<String>> {
    if calls.is_empty() {
        return Ok(None);
    }
    serde_json::to_string(calls)
        .map(Some)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
}

/// Map a failed fork of `src` into `name` to its typed error.
fn fork_error(error: tokio_rusqlite::Error, src: BranchId, name: &str) -> MemoryError {
    match &error {
        tokio_rusqlite::Error::Error(rusqlite::Error::QueryReturnedNoRows) => {
            MemoryError::BranchNotFound(src)
        }
        tokio_rusqlite::Error::Error(rusqlite::Error::SqliteFailure(failure, _))
            if failure.code == ErrorCode::ConstraintViolation
                && failure.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE =>
        {
            MemoryError::BranchNameTaken(name.to_string())
        }
        _ => MemoryError::sqlite("fork_and_switch")(error),
    }
}

/// `Ok` when an update by session matched a row.
fn session_updated(updated: usize, session: SessionId) -> Result<()> {
    if updated == 0 {
        return Err(MemoryError::SessionNotFound(session));
    }
    Ok(())
}

/// Map a failed switch to `branch` to its typed error; the foreign key rejects a
/// branch outside the session.
fn set_current_branch_error(error: tokio_rusqlite::Error, branch: BranchId) -> MemoryError {
    match &error {
        tokio_rusqlite::Error::Error(rusqlite::Error::SqliteFailure(failure, _))
            if failure.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_FOREIGNKEY =>
        {
            MemoryError::BranchNotFound(branch)
        }
        _ => MemoryError::sqlite("set_current_branch")(error),
    }
}

fn latest_turn_on_branch(
    tx: &Transaction<'_>,
    branch: BranchId,
) -> rusqlite::Result<Option<TurnId>> {
    tx.query_row(
        "SELECT m.turn_id
         FROM branch_messages bm JOIN messages m ON m.id = bm.message_id
         WHERE bm.branch_id = ?1
         ORDER BY bm.seq DESC LIMIT 1",
        rusqlite::params![branch],
        |r| r.get(0),
    )
    .optional()
}

fn first_user_text_of_turn(tx: &Transaction<'_>, turn: TurnId) -> rusqlite::Result<Option<String>> {
    tx.query_row(
        "SELECT content FROM messages WHERE turn_id = ?1 AND role = ?2 ORDER BY id LIMIT 1",
        rusqlite::params![turn, PersistedRole::User],
        |r| r.get(0),
    )
    .optional()
}

/// Drop the branch's summary once it stands in for any message of `turn`,
/// returning how many summaries were dropped.
fn drop_summary_covering_turn(
    tx: &Transaction<'_>,
    branch: BranchId,
    turn: TurnId,
) -> rusqlite::Result<usize> {
    tx.execute(
        "DELETE FROM branch_summaries
         WHERE branch_id = ?1
           AND through_seq >= (
               SELECT MIN(bm.seq)
               FROM branch_messages bm JOIN messages m ON m.id = bm.message_id
               WHERE bm.branch_id = ?1 AND m.turn_id = ?2
           )",
        rusqlite::params![branch, turn],
    )
}

/// Delete the turn's messages that no branch references any more, then the
/// turn itself once no message points at it; returns how many messages were deleted.
fn delete_unreferenced_turn(tx: &Transaction<'_>, turn: TurnId) -> rusqlite::Result<usize> {
    let deleted = tx.execute(
        "DELETE FROM messages
         WHERE turn_id = ?1
           AND NOT EXISTS (SELECT 1 FROM branch_messages bm WHERE bm.message_id = messages.id)",
        rusqlite::params![turn],
    )?;
    tx.execute(
        "DELETE FROM turns
         WHERE id = ?1
           AND NOT EXISTS (SELECT 1 FROM messages WHERE turn_id = ?1)",
        rusqlite::params![turn],
    )?;
    Ok(deleted)
}

/// A `LIKE` pattern (escape char `\`) matching values that start with `prefix` literally,
/// ignoring ASCII case.
fn like_prefix_pattern(prefix: &str) -> String {
    let escaped = prefix
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_");
    format!("{escaped}%")
}

fn branch_info(row: &Row<'_>) -> rusqlite::Result<BranchInfo> {
    Ok(BranchInfo {
        branch_id: row.get(0)?,
        session_id: row.get(1)?,
        session_started_at: timestamp_at(row, 2)?,
        session_ended_at: optional_timestamp_at(row, 3)?,
        session_title: row.get(4)?,
        name: row.get(5)?,
        parent_branch_id: row.get(6)?,
        parent_branch_name: row.get(7)?,
        fork_point_seq: row.get(8)?,
        created_at: timestamp_at(row, 9)?,
        message_count: row.get(10)?,
        is_current_in_session: row.get(11)?,
        last_active_at: timestamp_at(row, 12)?,
    })
}

fn history_row(row: &Row<'_>) -> rusqlite::Result<HistoryRow> {
    let kind = match row.get(2)? {
        PersistedRole::User => MessageKind::User,
        PersistedRole::Assistant => MessageKind::Assistant {
            reasoning: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            tool_calls: tool_calls_at(row, 5)?,
        },
        PersistedRole::Tool => MessageKind::Tool {
            call_id: row.get(6)?,
            tool_name: row.get(7)?,
            outcome: ToolOutcome {
                exit_code: row.get(8)?,
                duration_ms: row.get(9)?,
            },
        },
    };
    Ok(HistoryRow {
        message_id: row.get(0)?,
        seq: row.get(1)?,
        message: PersistedMessage {
            content: row.get(3)?,
            kind,
        },
    })
}

fn tool_calls_at(row: &Row<'_>, idx: usize) -> rusqlite::Result<Vec<PersistedToolCall>> {
    row.get::<_, Option<String>>(idx)?
        .map_or(Ok(Vec::new()), |json| {
            serde_json::from_str(&json).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(idx, Type::Text, Box::new(e))
            })
        })
}

#[cfg(test)]
mod tests;

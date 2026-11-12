//! SQLite-backed [`TranscriptStore`] over sessions, turns, branches, and messages.

use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::types::Type;
use rusqlite::{ErrorCode, OptionalExtension, Row, Transaction};

use assistd_ipc::{TaskId, TaskName, TaskRunId, TaskRunSummary};

use crate::chunking::{ChunkingConfig, chunk_ranges};
use crate::{
    AppendedMessage, BranchHistory, BranchId, BranchInfo, BranchSummary, ChunkId, HistoryRow,
    HistoryScope, MemoryError, MessageId, MessageKind, OpenedTurn, PersistedMessage,
    PersistedRelayStep, PersistedRole, PersistedToolCall, Reasoning, Result, ResumeCandidate, Seq,
    SessionId, ToolOutcome, TranscriptStore, TurnId, TurnStats, UndoOutcome,
};

use super::connection::{Erasing, SqliteHandle};
use super::tasks::run_state_at;
use super::timestamp::{now_millis, optional_timestamp_at, timestamp_at};

mod relays;

pub(super) use relays::session_answer_files;
use relays::{all_answer_files, insert_relay_steps, relay_traces, unreferenced_answer_files};

/// Columns [`branch_info`] reads, over `branches b JOIN sessions s` and the task run, if any,
/// the session holds.
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
                b.created_at)) AS last_active_at,
            r.task_id,
            r.id,
            t.name,
            r.state,
            r.suspend_reason,
            r.failure
    FROM branches b JOIN sessions s ON s.id = b.session_id
    LEFT JOIN task_runs r ON r.session_uuid = s.uuid
    LEFT JOIN tasks t ON t.id = r.task_id";

/// SQLite-backed [`TranscriptStore`].
#[derive(Debug, Clone)]
pub struct SqliteTranscriptStore {
    handle: SqliteHandle,
    chunk_messages: bool,
}

impl SqliteTranscriptStore {
    /// Store over the shared database handle, cutting each stored message into chunks
    /// for embedding.
    pub fn new(handle: SqliteHandle) -> Self {
        Self {
            handle,
            chunk_messages: true,
        }
    }

    /// Store messages without cutting chunks, for a run that embeds nothing.
    pub fn without_message_chunks(mut self) -> Self {
        self.chunk_messages = false;
        self
    }

    /// Cut the chunks of every user and assistant message stored without any,
    /// returning how many chunks were added.
    pub async fn chunk_unchunked_messages(&self) -> Result<u64> {
        self.handle
            .write("chunk_unchunked_messages", cut_missing_chunks)
            .await
    }

    /// Every saved relay answer a stored relay step names.
    pub async fn relay_answer_files(&self) -> Result<Vec<PathBuf>> {
        self.handle
            .read("relay_answer_files", all_answer_files)
            .await
    }

    fn message_row(&self, msg: PersistedMessage) -> MessageRow {
        MessageRow::new(msg, now_millis(), self.chunk_messages)
    }
}

#[async_trait]
impl TranscriptStore for SqliteTranscriptStore {
    async fn end_session(&self, session: SessionId) -> Result<()> {
        let ended = now_millis();
        let updated = self
            .handle
            .write("end_session", move |tx| {
                tx.execute(
                    "UPDATE sessions SET ended_at = ?1 WHERE uuid = ?2",
                    rusqlite::params![ended, session],
                )
            })
            .await?;
        session_updated(updated, session)
    }

    async fn begin_turn(&self, branch: BranchId, prompt: PersistedMessage) -> Result<OpenedTurn> {
        let row = self.message_row(prompt);
        self.handle
            .try_write(move |tx| {
                let slot = claim_next_seq(tx, branch)?;
                let turn = tx
                    .prepare_cached("INSERT INTO turns (session_id) VALUES (?1) RETURNING id")?
                    .query_row(rusqlite::params![slot.session_rowid], |r| r.get(0))?;
                let message = store_message(tx, branch, slot, turn, &row)?;
                Ok(OpenedTurn { turn, message })
            })
            .await
            .map_err(MemoryError::sqlite_or_missing(
                "begin_turn",
                MemoryError::BranchNotFound(branch),
            ))
    }

    async fn begin_session_with_main_branch(&self, session: SessionId) -> Result<BranchId> {
        let started = now_millis();
        self.handle
            .write("begin_session_with_main_branch", move |tx| {
                let session_rowid: i64 = tx.query_row(
                    "INSERT INTO sessions (uuid, started_at) VALUES (?1, ?2) RETURNING id",
                    rusqlite::params![session, started],
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
                Ok(branch_id)
            })
            .await
    }

    async fn set_current_branch(&self, session: SessionId, branch: BranchId) -> Result<()> {
        let updated = self
            .handle
            .try_write(move |tx| {
                tx.execute(
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
            .read("current_branch", move |c| {
                c.query_row(
                    "SELECT current_branch_id FROM sessions WHERE uuid = ?1",
                    rusqlite::params![session],
                    |r| r.get::<_, Option<BranchId>>(0),
                )
                .optional()
                .map(Option::flatten)
            })
            .await
    }

    async fn append_message_to_branch(
        &self,
        branch: BranchId,
        turn: TurnId,
        msg: PersistedMessage,
    ) -> Result<AppendedMessage> {
        let row = self.message_row(msg);
        self.handle
            .try_write(move |tx| {
                let slot = claim_next_seq(tx, branch)?;
                store_message(tx, branch, slot, turn, &row)
            })
            .await
            .map_err(MemoryError::sqlite_or_missing(
                "append_message_to_branch",
                MemoryError::BranchNotFound(branch),
            ))
    }

    async fn append_relay_result(
        &self,
        branch: BranchId,
        turn: TurnId,
        msg: PersistedMessage,
        steps: Vec<PersistedRelayStep>,
    ) -> Result<AppendedMessage> {
        let row = self.message_row(msg);
        self.handle
            .try_write(move |tx| {
                let slot = claim_next_seq(tx, branch)?;
                let appended = store_message(tx, branch, slot, turn, &row)?;
                insert_relay_steps(tx, appended.id, &steps)?;
                Ok(appended)
            })
            .await
            .map_err(MemoryError::sqlite_or_missing(
                "append_relay_result",
                MemoryError::BranchNotFound(branch),
            ))
    }

    async fn load_relay_traces(
        &self,
        branch: BranchId,
    ) -> Result<HashMap<MessageId, Vec<PersistedRelayStep>>> {
        self.handle
            .read_snapshot("load_relay_traces", move |tx| relay_traces(tx, branch))
            .await
    }

    async fn list_branches(&self) -> Result<Vec<BranchInfo>> {
        self.handle
            .read("list_branches", |c| {
                let sql = format!("{BRANCH_INFO_SELECT} ORDER BY last_active_at DESC, b.id ASC");
                c.prepare_cached(&sql)?
                    .query_map([], branch_info)?
                    .collect()
            })
            .await
    }

    async fn branch_info(&self, branch: BranchId) -> Result<Option<BranchInfo>> {
        self.handle
            .read("branch_info", move |c| {
                let sql = format!("{BRANCH_INFO_SELECT} WHERE b.id = ?1");
                c.prepare_cached(&sql)?
                    .query_row(rusqlite::params![branch], branch_info)
                    .optional()
            })
            .await
    }

    async fn resolve_branch(
        &self,
        target: &str,
        prefer_session: Option<SessionId>,
    ) -> Result<Option<(SessionId, BranchId)>> {
        let (prefix, name) = target.split_once('/').unwrap_or(("", target));
        let (uuid_from, uuid_before) = uuid_prefix_bounds(prefix);
        let name = name.to_string();
        self.handle
            .read("resolve_branch", move |c| {
                c.prepare_cached(
                    "SELECT s.uuid, b.id
                     FROM branches b JOIN sessions s ON s.id = b.session_id
                     WHERE b.name = ?1 AND s.uuid >= ?2 AND s.uuid < ?3
                     ORDER BY s.uuid IS ?4 DESC, s.started_at DESC LIMIT 1",
                )?
                .query_row(
                    rusqlite::params![name, uuid_from, uuid_before, prefer_session],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()
            })
            .await
    }

    async fn resolve_session(&self, prefix: &str) -> Result<Option<SessionId>> {
        let (uuid_from, uuid_before) = uuid_prefix_bounds(prefix);
        self.handle
            .read("resolve_session", move |c| {
                c.prepare_cached(
                    "SELECT uuid FROM sessions WHERE uuid >= ?1 AND uuid < ?2
                     ORDER BY started_at DESC LIMIT 1",
                )?
                .query_row(rusqlite::params![uuid_from, uuid_before], |r| r.get(0))
                .optional()
            })
            .await
    }

    async fn fork_and_switch(&self, src: BranchId, new_name: &str) -> Result<BranchId> {
        let name = new_name.to_string();
        let created = now_millis();
        self.handle
            .try_write(move |tx| {
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
                    "INSERT INTO branch_summaries
                         (branch_id, kept_from_seq, steps_folded_through, compacted_after_seq,
                          content, duration_ms)
                     SELECT ?1, kept_from_seq, steps_folded_through, compacted_after_seq,
                            content, duration_ms
                     FROM branch_summaries WHERE branch_id = ?2",
                    rusqlite::params![new_branch, src],
                )?;
                tx.execute(
                    "UPDATE sessions SET current_branch_id = ?1
                     WHERE id = (SELECT session_id FROM branches WHERE id = ?1)",
                    rusqlite::params![new_branch],
                )?;
                Ok(new_branch)
            })
            .await
            .map_err(|e| fork_error(e, src, new_name))
    }

    async fn load_branch(&self, branch: BranchId, scope: HistoryScope) -> Result<BranchHistory> {
        self.handle
            .read_snapshot("load_branch", move |tx| {
                let summary = branch_summary(tx, branch)?;
                let from = match (scope, &summary) {
                    (HistoryScope::Unsummarized, Some(summary)) => summary.kept_from_seq,
                    _ => Seq(0),
                };
                let rows = history_rows(tx, branch, from)?;
                Ok(BranchHistory { summary, rows })
            })
            .await
    }

    async fn record_branch_summary(&self, branch: BranchId, summary: BranchSummary) -> Result<()> {
        self.handle
            .write("record_branch_summary", move |tx| {
                tx.execute(
                    "INSERT OR REPLACE INTO branch_summaries
                         (branch_id, kept_from_seq, steps_folded_through, compacted_after_seq,
                          content, duration_ms)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![
                        branch,
                        summary.kept_from_seq,
                        summary.steps_folded_through,
                        summary.compacted_after_seq,
                        summary.content,
                        summary.duration_ms
                    ],
                )
            })
            .await
            .map(drop)
    }

    async fn record_turn_stats(&self, turn: TurnId, stats: TurnStats) -> Result<()> {
        let updated = self
            .handle
            .write("record_turn_stats", move |tx| {
                tx.execute(
                    "UPDATE turns SET duration_ms = ?2, generated_tokens = ?3 WHERE id = ?1",
                    rusqlite::params![turn, stats.duration_ms, stats.generated_tokens],
                )
            })
            .await?;
        turn_updated(updated, turn)
    }

    async fn mark_turn_incomplete(&self, turn: TurnId) -> Result<()> {
        let updated = self
            .handle
            .write("mark_turn_incomplete", move |tx| {
                tx.execute(
                    "UPDATE turns SET incomplete = 1 WHERE id = ?1",
                    rusqlite::params![turn],
                )
            })
            .await?;
        turn_updated(updated, turn)
    }

    async fn latest_branch_activity(&self, branch: BranchId) -> Result<Option<DateTime<Utc>>> {
        self.handle
            .read("latest_branch_activity", move |c| {
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
    }

    async fn undo_last_turn(&self, branch: BranchId) -> Result<UndoOutcome> {
        self.handle
            .write_erasing("undo_last_turn", move |tx| {
                let Some(turn) = latest_turn_on_branch(tx, branch)? else {
                    return Ok(Erasing {
                        value: UndoOutcome::default(),
                        erased: false,
                    });
                };
                let last_user_text = first_user_text_of_turn(tx, turn)?;
                let dropped_summaries = drop_summary_covering_turn(tx, branch, turn)?;
                let removed = tx.execute(
                    "DELETE FROM branch_messages
                     WHERE branch_id = ?1
                       AND message_id IN (SELECT id FROM messages WHERE turn_id = ?2)",
                    rusqlite::params![branch, turn],
                )?;
                let released_answer_files = unreferenced_answer_files(tx, turn)?;
                let deleted_messages = delete_unreferenced_turn(tx, turn)?;
                Ok(Erasing {
                    value: UndoOutcome {
                        removed_messages: removed as u64,
                        last_user_text,
                        released_answer_files,
                    },
                    erased: dropped_summaries + deleted_messages > 0,
                })
            })
            .await
    }

    async fn find_resumable_session(&self) -> Result<Option<ResumeCandidate>> {
        self.handle
            .read("find_resumable_session", |c| {
                c.query_row(
                    "SELECT uuid, current_branch_id, started_at
                     FROM sessions s
                     WHERE ended_at IS NULL AND current_branch_id IS NOT NULL
                       AND NOT EXISTS (SELECT 1 FROM task_runs r WHERE r.session_uuid = s.uuid)
                     ORDER BY started_at DESC LIMIT 1",
                    [],
                    |r| {
                        Ok(ResumeCandidate {
                            session_id: r.get(0)?,
                            current_branch_id: r.get(1)?,
                            started_at: timestamp_at(r, 2)?,
                        })
                    },
                )
                .optional()
            })
            .await
    }

    async fn session_title(&self, session: SessionId) -> Result<Option<String>> {
        self.handle
            .read("session_title", move |c| {
                c.query_row(
                    "SELECT title FROM sessions WHERE uuid = ?1",
                    rusqlite::params![session],
                    |r| r.get::<_, Option<String>>(0),
                )
                .optional()
                .map(Option::flatten)
            })
            .await
    }

    async fn set_session_title(&self, session: SessionId, title: &str) -> Result<()> {
        let title = title.to_string();
        let updated = self
            .handle
            .write("set_session_title", move |tx| {
                tx.execute(
                    "UPDATE sessions SET title = ?1 WHERE uuid = ?2",
                    rusqlite::params![title, session],
                )
            })
            .await?;
        session_updated(updated, session)
    }
}

/// One `messages` row ready to insert, with the byte ranges of the chunks
/// embedded from it; tool results get none, as does every row of a store that cuts none.
struct MessageRow {
    created_at: i64,
    role: PersistedRole,
    content: String,
    columns: MessageColumns,
    chunk_bytes: Vec<Range<usize>>,
}

impl MessageRow {
    fn new(msg: PersistedMessage, created_at: i64, chunk_messages: bool) -> Self {
        let role = msg.role();
        let chunk_bytes = match role {
            PersistedRole::User | PersistedRole::Assistant if chunk_messages => {
                chunk_ranges(&msg.content, ChunkingConfig::MESSAGES).collect()
            }
            PersistedRole::User | PersistedRole::Assistant | PersistedRole::Tool => Vec::new(),
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
    reasoning_ms: Option<u64>,
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
            } => {
                let reasoning = reasoning.filter(|r| !r.text.is_empty());
                Self {
                    reasoning_ms: reasoning.as_ref().map(|r| r.duration_ms),
                    reasoning: reasoning.map(|r| r.text),
                    tool_calls,
                    ..Self::default()
                }
            }
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
    let chunks = insert_chunks(tx, id, &row.chunk_bytes)?;
    Ok(AppendedMessage {
        id,
        seq: slot.seq,
        chunks,
    })
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
            (session_id, turn_id, created_at, role, content, reasoning, reasoning_ms,
             tool_calls, tool_call_id, tool_name, tool_exit_code, tool_duration_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
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
            columns.reasoning_ms,
            tool_calls_json(&columns.tool_calls)?,
            columns.tool_call_id,
            columns.tool_name,
            columns.tool_exit_code,
            columns.tool_duration_ms,
        ],
        |r| r.get(0),
    )
}

/// Store chunks over `chunk_bytes` of `message`'s content, returning their ids in order.
fn insert_chunks(
    tx: &Transaction<'_>,
    message: MessageId,
    chunk_bytes: &[Range<usize>],
) -> rusqlite::Result<Vec<ChunkId>> {
    let mut insert = tx.prepare_cached(
        "INSERT INTO chunks (message_id, byte_start, byte_end)
         VALUES (?1, ?2, ?3)
         RETURNING id",
    )?;
    chunk_bytes
        .iter()
        .map(|bytes| {
            insert.query_row(rusqlite::params![message, bytes.start, bytes.end], |r| {
                r.get(0)
            })
        })
        .collect()
}

/// Cut and store the chunks of every user and assistant message that has none,
/// returning how many chunks were added.
fn cut_missing_chunks(tx: &Transaction<'_>) -> rusqlite::Result<u64> {
    let unchunked: Vec<(MessageId, String)> = tx
        .prepare(
            "SELECT m.id, m.content FROM messages m
             WHERE m.role IN (?1, ?2) AND m.content != ''
               AND NOT EXISTS (SELECT 1 FROM chunks cc WHERE cc.message_id = m.id)",
        )?
        .query_map(
            rusqlite::params![PersistedRole::User, PersistedRole::Assistant],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?
        .collect::<rusqlite::Result<_>>()?;
    let mut added = 0;
    for (message, content) in unchunked {
        let chunk_bytes: Vec<Range<usize>> =
            chunk_ranges(&content, ChunkingConfig::MESSAGES).collect();
        added += insert_chunks(tx, message, &chunk_bytes)?.len() as u64;
    }
    Ok(added)
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

/// `Ok` when an update by turn matched a row.
fn turn_updated(updated: usize, turn: TurnId) -> Result<()> {
    if updated == 0 {
        return Err(MemoryError::TurnNotFound(turn));
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
           AND COALESCE(steps_folded_through, kept_from_seq - 1) >= (
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

/// The half-open range of hyphenated lowercase UUIDs starting with `prefix`, ignoring
/// ASCII case; every character such a UUID holds sorts below `~`.
fn uuid_prefix_bounds(prefix: &str) -> (String, String) {
    let from = prefix.to_ascii_lowercase();
    let before = format!("{from}~");
    (from, before)
}

/// The branch's latest compaction summary, if one is recorded.
fn branch_summary(
    tx: &Transaction<'_>,
    branch: BranchId,
) -> rusqlite::Result<Option<BranchSummary>> {
    tx.query_row(
        "SELECT kept_from_seq, steps_folded_through, compacted_after_seq, content, duration_ms
         FROM branch_summaries WHERE branch_id = ?1",
        rusqlite::params![branch],
        |r| {
            Ok(BranchSummary {
                kept_from_seq: r.get(0)?,
                steps_folded_through: r.get(1)?,
                compacted_after_seq: r.get(2)?,
                content: r.get(3)?,
                duration_ms: r.get(4)?,
            })
        },
    )
    .optional()
}

/// The messages on `branch` from seq `from` on, ordered by seq.
fn history_rows(
    tx: &Transaction<'_>,
    branch: BranchId,
    from: Seq,
) -> rusqlite::Result<Vec<HistoryRow>> {
    tx.prepare_cached(
        "SELECT  m.id,
                 bm.seq,
                 m.role,
                 m.content,
                 m.reasoning,
                 m.reasoning_ms,
                 m.tool_calls,
                 m.tool_call_id,
                 m.tool_name,
                 m.tool_exit_code,
                 m.tool_duration_ms,
                 LEAD(m.turn_id) OVER (ORDER BY bm.seq) IS NOT m.turn_id,
                 t.duration_ms,
                 t.generated_tokens
         FROM branch_messages bm
         JOIN messages m ON m.id = bm.message_id
         JOIN turns t ON t.id = m.turn_id
         WHERE bm.branch_id = ?1 AND bm.seq >= ?2
         ORDER BY bm.seq ASC",
    )?
    .query_map(rusqlite::params![branch, from], history_row)?
    .collect()
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
        task_run: task_run_at(row, 13)?,
    })
}

/// The task run in columns `idx..idx + 6` of a [`BRANCH_INFO_SELECT`] row, if the session
/// holds one.
fn task_run_at(row: &Row<'_>, idx: usize) -> rusqlite::Result<Option<TaskRunSummary>> {
    let Some(task) = row.get::<_, Option<i64>>(idx)? else {
        return Ok(None);
    };
    let task_name = TaskName::try_from(row.get::<_, String>(idx + 2)?)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(idx + 2, Type::Text, Box::new(e)))?;
    Ok(Some(TaskRunSummary {
        task: TaskId(task),
        run: TaskRunId(row.get(idx + 1)?),
        task_name,
        state: run_state_at(row, idx + 3)?,
    }))
}

fn history_row(row: &Row<'_>) -> rusqlite::Result<HistoryRow> {
    let kind = match row.get(2)? {
        PersistedRole::User => MessageKind::User,
        PersistedRole::Assistant => MessageKind::Assistant {
            reasoning: reasoning_at(row, 4)?,
            tool_calls: tool_calls_at(row, 6)?,
        },
        PersistedRole::Tool => MessageKind::Tool {
            call_id: row.get(7)?,
            tool_name: row.get(8)?,
            outcome: ToolOutcome {
                exit_code: row.get(9)?,
                duration_ms: row.get(10)?,
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
        turn_stats: turn_stats_at(row, 11)?,
    })
}

/// The turn's stats when the flag at `idx` marks the row as its turn's last,
/// read from the two columns after it.
fn turn_stats_at(row: &Row<'_>, idx: usize) -> rusqlite::Result<Option<TurnStats>> {
    let ends_turn: bool = row.get(idx)?;
    let duration_ms: Option<u64> = row.get(idx + 1)?;
    let generated_tokens: Option<u64> = row.get(idx + 2)?;
    Ok(duration_ms.zip(generated_tokens).filter(|_| ends_turn).map(
        |(duration_ms, generated_tokens)| TurnStats {
            duration_ms,
            generated_tokens,
        },
    ))
}

/// The reasoning text at `idx` and its duration at `idx + 1`; the schema sets both or neither.
fn reasoning_at(row: &Row<'_>, idx: usize) -> rusqlite::Result<Option<Reasoning>> {
    let text: Option<String> = row.get(idx)?;
    let duration_ms: Option<u64> = row.get(idx + 1)?;
    Ok(text
        .zip(duration_ms)
        .map(|(text, duration_ms)| Reasoning { text, duration_ms }))
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

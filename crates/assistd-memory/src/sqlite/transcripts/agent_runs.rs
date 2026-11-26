//! Sub-agent runs in `agent_runs`, their histories in the `messages` rows that name
//! them, and their summaries in `agent_run_summaries`.

use std::collections::HashMap;
use std::ffi::OsString;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::PathBuf;

use rusqlite::types::{FromSqlError, Type};
use rusqlite::{Connection, Row, Transaction};

use assistd_ipc::Access;

use crate::{
    AgentRunEnd, AgentRunRecord, BranchId, BranchSummary, MessageId, PersistedAgentRun, RunMessage,
    SessionId, TurnId,
};

use super::{
    MESSAGE_COLUMN_COUNT, MESSAGE_COLUMNS, MessageRow, insert_accesses, insert_message,
    stored_message, summary_at,
};

const OUTCOME_ANSWERED: i64 = 0;
const OUTCOME_FAILED: i64 = 1;

/// Joins each message `bm` of a branch to the runs `r` stored with it.
const RUNS_ON_BRANCH: &str = "branch_messages bm JOIN agent_runs r ON r.message_id = bm.message_id";

/// The tool result a call's runs are stored with, and what their messages share with it.
#[derive(Clone, Copy)]
pub(super) struct RunOwner {
    pub(super) message: MessageId,
    pub(super) session_rowid: i64,
    pub(super) turn: TurnId,
    pub(super) created_at: i64,
}

/// The `agent_runs` columns that say how a run ended, `NULL` where it has none.
#[derive(Default)]
struct EndColumns<'a> {
    outcome: Option<i64>,
    failure: Option<&'a str>,
    answer_file: Option<&'a [u8]>,
    duration_ms: Option<u64>,
    tokens: Option<u32>,
}

impl<'a> EndColumns<'a> {
    fn of(record: Option<&'a AgentRunRecord>) -> Self {
        let Some(record) = record else {
            return Self::default();
        };
        let (outcome, failure, answer_file) = match &record.end {
            AgentRunEnd::Answered { answer_file } => (
                OUTCOME_ANSWERED,
                None,
                Some(answer_file.as_os_str().as_bytes()),
            ),
            AgentRunEnd::Failed { message } => (OUTCOME_FAILED, Some(message.as_str()), None),
        };
        Self {
            outcome: Some(outcome),
            failure,
            answer_file,
            duration_ms: Some(record.duration_ms),
            tokens: record.tokens,
        }
    }
}

/// Store `runs` under `owner`'s tool result, numbering them from 1 in plan order, each
/// with its history and summary.
pub(super) fn insert_agent_runs(
    tx: &Transaction<'_>,
    owner: RunOwner,
    runs: Vec<PersistedAgentRun>,
) -> rusqlite::Result<()> {
    for (index, run) in runs.into_iter().enumerate() {
        let columns = EndColumns::of(run.record.as_ref());
        let run_id: i64 = tx
            .prepare_cached(
                "INSERT INTO agent_runs
                    (message_id, step, task, agent, outcome, failure, duration_ms, tokens,
                     answer_file)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 RETURNING id",
            )?
            .query_row(
                rusqlite::params![
                    owner.message,
                    index + 1,
                    run.task,
                    run.agent,
                    columns.outcome,
                    columns.failure,
                    columns.duration_ms,
                    columns.tokens,
                    columns.answer_file,
                ],
                |r| r.get(0),
            )?;
        if let Some(record) = run.record {
            insert_history(tx, owner, run_id, record.messages)?;
            insert_summary(tx, run_id, record.summary.as_ref())?;
        }
    }
    Ok(())
}

fn insert_history(
    tx: &Transaction<'_>,
    owner: RunOwner,
    run_id: i64,
    messages: Vec<RunMessage>,
) -> rusqlite::Result<()> {
    for RunMessage { message, accesses } in messages {
        let row = MessageRow::new(message, owner.created_at);
        let id = insert_message(tx, owner.session_rowid, owner.turn, Some(run_id), &row)?;
        insert_accesses(tx, id, &accesses)?;
    }
    Ok(())
}

fn insert_summary(
    tx: &Transaction<'_>,
    run_id: i64,
    summary: Option<&BranchSummary>,
) -> rusqlite::Result<()> {
    let Some(summary) = summary else {
        return Ok(());
    };
    tx.prepare_cached(
        "INSERT INTO agent_run_summaries
             (run_id, kept_from_seq, steps_folded_through, compacted_after_seq, content,
              duration_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?
    .execute(rusqlite::params![
        run_id,
        summary.kept_from_seq,
        summary.steps_folded_through,
        summary.compacted_after_seq,
        summary.content,
        summary.duration_ms
    ])?;
    Ok(())
}

/// The runs stored with each message on `branch`, in plan order.
pub(super) fn branch_agent_runs(
    tx: &Transaction<'_>,
    branch: BranchId,
) -> rusqlite::Result<HashMap<MessageId, Vec<PersistedAgentRun>>> {
    let mut histories = run_histories(tx, branch)?;
    let mut summaries = run_summaries(tx, branch)?;
    let mut runs: HashMap<MessageId, Vec<PersistedAgentRun>> = HashMap::new();
    let mut select = tx.prepare_cached(&format!(
        "SELECT r.message_id, r.id, r.task, r.agent, r.outcome, r.failure, r.duration_ms,
                r.tokens, r.answer_file
         FROM {RUNS_ON_BRANCH}
         WHERE bm.branch_id = ?1
         ORDER BY r.message_id, r.step"
    ))?;
    for row in select.query_map(rusqlite::params![branch], run_row)? {
        let (message, run_id, mut run) = row?;
        if let Some(record) = run.record.as_mut() {
            record.messages = histories.remove(&run_id).unwrap_or_default();
            record.summary = summaries.remove(&run_id);
        }
        runs.entry(message).or_default().push(run);
    }
    Ok(runs)
}

/// The history of every run on `branch`, keyed by run id, in order.
fn run_histories(
    tx: &Transaction<'_>,
    branch: BranchId,
) -> rusqlite::Result<HashMap<i64, Vec<RunMessage>>> {
    let mut accesses = run_accesses(tx, branch)?;
    let mut histories: HashMap<i64, Vec<RunMessage>> = HashMap::new();
    let mut select = tx.prepare_cached(&format!(
        "SELECT {MESSAGE_COLUMNS}, m.run_id
         FROM {RUNS_ON_BRANCH} JOIN messages m ON m.run_id = r.id
         WHERE bm.branch_id = ?1
         ORDER BY m.id"
    ))?;
    let rows = select.query_map(rusqlite::params![branch], |row| {
        Ok((
            row.get::<_, i64>(MESSAGE_COLUMN_COUNT)?,
            stored_message(row)?,
        ))
    })?;
    for row in rows {
        let (run_id, (id, message)) = row?;
        histories.entry(run_id).or_default().push(RunMessage {
            message,
            accesses: accesses.remove(&id).unwrap_or_default(),
        });
    }
    Ok(histories)
}

/// What each message of every run on `branch` is stored as having touched.
fn run_accesses(
    tx: &Transaction<'_>,
    branch: BranchId,
) -> rusqlite::Result<HashMap<MessageId, Vec<Access>>> {
    let mut accesses: HashMap<MessageId, Vec<Access>> = HashMap::new();
    let mut select = tx.prepare_cached(&format!(
        "SELECT a.message_id, a.action, a.target, a.label
         FROM {RUNS_ON_BRANCH}
         JOIN messages m ON m.run_id = r.id
         JOIN accesses a ON a.message_id = m.id
         WHERE bm.branch_id = ?1
         ORDER BY a.id"
    ))?;
    let rows = select.query_map(rusqlite::params![branch], |row| {
        Ok((row.get(0)?, access_at(row, 1)?))
    })?;
    for row in rows {
        let (message, access) = row?;
        accesses.entry(message).or_default().push(access);
    }
    Ok(accesses)
}

/// The summary of every run on `branch` that has one, keyed by run id.
fn run_summaries(
    tx: &Transaction<'_>,
    branch: BranchId,
) -> rusqlite::Result<HashMap<i64, BranchSummary>> {
    tx.prepare_cached(&format!(
        "SELECT s.run_id, s.kept_from_seq, s.steps_folded_through, s.compacted_after_seq,
                s.content, s.duration_ms
         FROM {RUNS_ON_BRANCH} JOIN agent_run_summaries s ON s.run_id = r.id
         WHERE bm.branch_id = ?1"
    ))?
    .query_map(rusqlite::params![branch], |row| {
        Ok((row.get(0)?, summary_at(row, 1)?))
    })?
    .collect()
}

/// A run's message, id and contents, with no history yet.
fn run_row(row: &Row<'_>) -> rusqlite::Result<(MessageId, i64, PersistedAgentRun)> {
    let end = match row.get::<_, Option<i64>>(4)? {
        None => None,
        Some(OUTCOME_ANSWERED) => Some(AgentRunEnd::Answered {
            answer_file: path_from_bytes(row.get(8)?),
        }),
        Some(OUTCOME_FAILED) => Some(AgentRunEnd::Failed {
            message: row.get(5)?,
        }),
        Some(code) => return Err(out_of_range(4, code)),
    };
    let record = end
        .map(|end| -> rusqlite::Result<_> {
            Ok(AgentRunRecord {
                end,
                duration_ms: row.get(6)?,
                tokens: row.get(7)?,
                messages: Vec::new(),
                summary: None,
            })
        })
        .transpose()?;
    let run = PersistedAgentRun {
        task: row.get(2)?,
        agent: row.get(3)?,
        record,
    };
    Ok((row.get(0)?, row.get(1)?, run))
}

/// The access in the `action`, `target` and `label` columns from `idx` on.
fn access_at(row: &Row<'_>, idx: usize) -> rusqlite::Result<Access> {
    let target: Vec<u8> = row.get(idx + 1)?;
    let text = |target: Vec<u8>| {
        String::from_utf8(target).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(idx + 1, Type::Blob, Box::new(e))
        })
    };
    Ok(match row.get(idx)? {
        0 => Access::Read(path_from_bytes(target)),
        1 => Access::Wrote(path_from_bytes(target)),
        2 => Access::Fetched {
            url: text(target)?,
            title: row.get(idx + 2)?,
        },
        3 => Access::Recalled(text(target)?),
        4 => Access::Saved(text(target)?),
        code => return Err(out_of_range(idx, code)),
    })
}

/// The answer files of the runs stored with `turn`'s messages that no branch references.
pub(super) fn unreferenced_answer_files(
    tx: &Transaction<'_>,
    turn: TurnId,
) -> rusqlite::Result<Vec<PathBuf>> {
    let mut select = tx.prepare_cached(
        "SELECT r.answer_file
         FROM messages m JOIN agent_runs r ON r.message_id = m.id
         WHERE m.turn_id = ?1 AND r.answer_file IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM branch_messages bm WHERE bm.message_id = m.id)",
    )?;
    select
        .query_map(rusqlite::params![turn], |r| r.get(0).map(path_from_bytes))?
        .collect()
}

/// The answer files of every run in the session with uuid `session`.
pub(in crate::sqlite) fn session_answer_files(
    tx: &Transaction<'_>,
    session: SessionId,
) -> rusqlite::Result<Vec<PathBuf>> {
    let mut select = tx.prepare_cached(
        "SELECT r.answer_file
         FROM sessions s
         JOIN messages m ON m.session_id = s.id
         JOIN agent_runs r ON r.message_id = m.id
         WHERE s.uuid = ?1 AND r.answer_file IS NOT NULL",
    )?;
    select
        .query_map(rusqlite::params![session], |r| {
            r.get(0).map(path_from_bytes)
        })?
        .collect()
}

/// Every answer file a stored run names.
pub(super) fn all_answer_files(conn: &Connection) -> rusqlite::Result<Vec<PathBuf>> {
    let mut select =
        conn.prepare_cached("SELECT answer_file FROM agent_runs WHERE answer_file IS NOT NULL")?;
    select
        .query_map([], |r| r.get(0).map(path_from_bytes))?
        .collect()
}

fn path_from_bytes(bytes: Vec<u8>) -> PathBuf {
    PathBuf::from(OsString::from_vec(bytes))
}

fn out_of_range(idx: usize, code: i64) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        idx,
        Type::Integer,
        Box::new(FromSqlError::OutOfRange(code)),
    )
}

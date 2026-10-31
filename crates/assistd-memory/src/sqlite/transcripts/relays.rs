//! A relay's sub-agent steps in `relay_steps` and their entries in `relay_entries`.

use std::collections::HashMap;

use rusqlite::types::{FromSqlError, Type};
use rusqlite::{Row, Transaction};

use crate::{
    BranchId, MessageId, PersistedRelayEnd, PersistedRelayEntry, PersistedRelayProfile,
    PersistedRelayRun, PersistedRelayStep, ToolOutcome,
};

const PROFILE_READ: i64 = 0;
const PROFILE_FETCH: i64 = 1;
const OUTCOME_ANSWERED: i64 = 0;
const OUTCOME_FAILED: i64 = 1;
const KIND_TEXT: i64 = 0;
const KIND_REASONING: i64 = 1;
const KIND_TOOL: i64 = 2;
const KIND_COMPACTION: i64 = 3;

/// The `relay_steps` columns that say how a step ended, `NULL` where it has none.
#[derive(Default)]
struct StepColumns<'a> {
    outcome: Option<i64>,
    failure: Option<&'a str>,
    truncated: Option<bool>,
    duration_ms: Option<u64>,
    tokens: Option<u32>,
}

impl<'a> StepColumns<'a> {
    fn of(run: Option<&'a PersistedRelayRun>) -> Self {
        let Some(run) = run else {
            return Self::default();
        };
        let (outcome, failure, truncated) = match &run.end {
            PersistedRelayEnd::Answered { truncated } => (OUTCOME_ANSWERED, None, Some(*truncated)),
            PersistedRelayEnd::Failed { message } => (OUTCOME_FAILED, Some(message.as_str()), None),
        };
        Self {
            outcome: Some(outcome),
            failure,
            truncated,
            duration_ms: Some(run.duration_ms),
            tokens: run.tokens,
        }
    }
}

/// One `relay_entries` row's columns, `NULL` where its kind has none.
struct EntryColumns<'a> {
    kind: i64,
    content: &'a str,
    tool_name: Option<&'a str>,
    tool_args: Option<String>,
    tool_exit_code: Option<i64>,
    duration_ms: Option<u64>,
}

impl<'a> EntryColumns<'a> {
    fn of(entry: &'a PersistedRelayEntry) -> Self {
        let blank = Self {
            kind: KIND_TEXT,
            content: "",
            tool_name: None,
            tool_args: None,
            tool_exit_code: None,
            duration_ms: None,
        };
        match entry {
            PersistedRelayEntry::Text(text) => Self {
                content: text,
                ..blank
            },
            PersistedRelayEntry::Reasoning { text, duration_ms } => Self {
                kind: KIND_REASONING,
                content: text,
                duration_ms: Some(*duration_ms),
                ..blank
            },
            PersistedRelayEntry::Tool {
                name,
                arguments,
                output,
                outcome,
            } => Self {
                kind: KIND_TOOL,
                content: output,
                tool_name: Some(name),
                tool_args: Some(arguments.to_string()),
                tool_exit_code: outcome.map(|o| o.exit_code),
                duration_ms: outcome.map(|o| o.duration_ms),
            },
            PersistedRelayEntry::Compaction { duration_ms } => Self {
                kind: KIND_COMPACTION,
                duration_ms: *duration_ms,
                ..blank
            },
        }
    }
}

/// Store `steps` under `message`, numbering them from 1 in plan order.
pub(super) fn insert_relay_steps(
    tx: &Transaction<'_>,
    message: MessageId,
    steps: &[PersistedRelayStep],
) -> rusqlite::Result<()> {
    let mut insert = tx.prepare_cached(
        "INSERT INTO relay_steps
            (message_id, step, task, profile, outcome, failure, truncated, duration_ms, tokens)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
         RETURNING id",
    )?;
    for (index, step) in steps.iter().enumerate() {
        let columns = StepColumns::of(step.run.as_ref());
        let step_id: i64 = insert.query_row(
            rusqlite::params![
                message,
                index + 1,
                step.task,
                profile_code(step.profile),
                columns.outcome,
                columns.failure,
                columns.truncated,
                columns.duration_ms,
                columns.tokens,
            ],
            |r| r.get(0),
        )?;
        if let Some(run) = &step.run {
            insert_entries(tx, step_id, &run.entries)?;
        }
    }
    Ok(())
}

fn insert_entries(
    tx: &Transaction<'_>,
    step_id: i64,
    entries: &[PersistedRelayEntry],
) -> rusqlite::Result<()> {
    let mut insert = tx.prepare_cached(
        "INSERT INTO relay_entries
            (step_id, seq, kind, content, tool_name, tool_args, tool_exit_code, duration_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?;
    for (seq, entry) in entries.iter().enumerate() {
        let columns = EntryColumns::of(entry);
        insert.execute(rusqlite::params![
            step_id,
            seq,
            columns.kind,
            columns.content,
            columns.tool_name,
            columns.tool_args,
            columns.tool_exit_code,
            columns.duration_ms,
        ])?;
    }
    Ok(())
}

/// The steps stored with each message on `branch`, in plan order.
pub(super) fn relay_traces(
    tx: &Transaction<'_>,
    branch: BranchId,
) -> rusqlite::Result<HashMap<MessageId, Vec<PersistedRelayStep>>> {
    let mut entries = relay_entries(tx, branch)?;
    let mut traces: HashMap<MessageId, Vec<PersistedRelayStep>> = HashMap::new();
    let mut select = tx.prepare_cached(
        "SELECT rs.message_id, rs.id, rs.task, rs.profile, rs.outcome, rs.failure,
                rs.truncated, rs.duration_ms, rs.tokens
         FROM branch_messages bm JOIN relay_steps rs ON rs.message_id = bm.message_id
         WHERE bm.branch_id = ?1
         ORDER BY rs.message_id, rs.step",
    )?;
    for row in select.query_map(rusqlite::params![branch], step_row)? {
        let (message, step_id, mut step) = row?;
        if let Some(run) = step.run.as_mut() {
            run.entries = entries.remove(&step_id).unwrap_or_default();
        }
        traces.entry(message).or_default().push(step);
    }
    Ok(traces)
}

/// The entries of every relay step on `branch`, keyed by step id, in order.
fn relay_entries(
    tx: &Transaction<'_>,
    branch: BranchId,
) -> rusqlite::Result<HashMap<i64, Vec<PersistedRelayEntry>>> {
    let mut entries: HashMap<i64, Vec<PersistedRelayEntry>> = HashMap::new();
    let mut select = tx.prepare_cached(
        "SELECT re.step_id, re.kind, re.content, re.tool_name, re.tool_args,
                re.tool_exit_code, re.duration_ms
         FROM branch_messages bm
         JOIN relay_steps rs ON rs.message_id = bm.message_id
         JOIN relay_entries re ON re.step_id = rs.id
         WHERE bm.branch_id = ?1
         ORDER BY re.step_id, re.seq",
    )?;
    for row in select.query_map(rusqlite::params![branch], |r| {
        Ok((r.get::<_, i64>(0)?, entry_row(r)?))
    })? {
        let (step_id, entry) = row?;
        entries.entry(step_id).or_default().push(entry);
    }
    Ok(entries)
}

/// A step's message, id and contents, with no entries yet.
fn step_row(row: &Row<'_>) -> rusqlite::Result<(MessageId, i64, PersistedRelayStep)> {
    let end = match row.get::<_, Option<i64>>(4)? {
        None => None,
        Some(OUTCOME_ANSWERED) => Some(PersistedRelayEnd::Answered {
            truncated: row.get(6)?,
        }),
        Some(OUTCOME_FAILED) => Some(PersistedRelayEnd::Failed {
            message: row.get(5)?,
        }),
        Some(code) => return Err(out_of_range(4, code)),
    };
    let run = end
        .map(|end| -> rusqlite::Result<_> {
            Ok(PersistedRelayRun {
                end,
                duration_ms: row.get(7)?,
                tokens: row.get(8)?,
                entries: Vec::new(),
            })
        })
        .transpose()?;
    let profile = match row.get(3)? {
        PROFILE_READ => PersistedRelayProfile::Read,
        PROFILE_FETCH => PersistedRelayProfile::Fetch,
        code => return Err(out_of_range(3, code)),
    };
    let step = PersistedRelayStep {
        task: row.get(2)?,
        profile,
        run,
    };
    Ok((row.get(0)?, row.get(1)?, step))
}

/// An entry from columns 1 to 6: kind, content, tool name, tool arguments,
/// exit code, and duration.
fn entry_row(row: &Row<'_>) -> rusqlite::Result<PersistedRelayEntry> {
    let content: String = row.get(2)?;
    Ok(match row.get(1)? {
        KIND_TEXT => PersistedRelayEntry::Text(content),
        KIND_REASONING => PersistedRelayEntry::Reasoning {
            text: content,
            duration_ms: row.get(6)?,
        },
        KIND_TOOL => {
            let arguments: String = row.get(4)?;
            let exit_code: Option<i64> = row.get(5)?;
            let duration_ms: Option<u64> = row.get(6)?;
            PersistedRelayEntry::Tool {
                name: row.get(3)?,
                arguments: serde_json::from_str(&arguments).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(4, Type::Text, Box::new(e))
                })?,
                output: content,
                outcome: exit_code
                    .zip(duration_ms)
                    .map(|(exit_code, duration_ms)| ToolOutcome {
                        exit_code,
                        duration_ms,
                    }),
            }
        }
        KIND_COMPACTION => PersistedRelayEntry::Compaction {
            duration_ms: row.get(6)?,
        },
        code => return Err(out_of_range(1, code)),
    })
}

fn profile_code(profile: PersistedRelayProfile) -> i64 {
    match profile {
        PersistedRelayProfile::Read => PROFILE_READ,
        PersistedRelayProfile::Fetch => PROFILE_FETCH,
    }
}

fn out_of_range(idx: usize, code: i64) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        idx,
        Type::Integer,
        Box::new(FromSqlError::OutOfRange(code)),
    )
}

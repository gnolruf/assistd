//! SQLite-backed [`TaskStore`] over the `tasks` and `task_runs` tables.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::types::Type;
use rusqlite::{ErrorCode, OptionalExtension, Row, Transaction};
use serde::Serialize;
use serde::de::DeserializeOwned;

use assistd_ipc::{SuspendReason, TaskId, TaskRunId, TaskRunState, TaskSpec, TaskTrigger};

use crate::{
    MemoryError, RecoveredQueue, Result, SessionId, TaskRecord, TaskRunEnd, TaskStore, WaitingRun,
};

use super::connection::{SqliteHandle, truncate_wal};
use super::timestamp::{now_millis, optional_timestamp_at};

const TASK_COLUMNS: &str = "id, name, prompt, triggers, enabled, grants, plan, next_fire_at";

const RUN_QUEUED: i64 = 0;
const RUN_RUNNING: i64 = 1;
const RUN_SUSPENDED: i64 = 2;
const RUN_NEEDS_INPUT: i64 = 3;
const RUN_SUCCEEDED: i64 = 4;
const RUN_FAILED: i64 = 5;
const RUN_CANCELLED: i64 = 6;

/// SQLite-backed [`TaskStore`].
#[derive(Debug, Clone)]
pub struct SqliteTaskStore {
    handle: SqliteHandle,
}

impl SqliteTaskStore {
    /// Store over the shared database handle.
    pub fn new(handle: SqliteHandle) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl TaskStore for SqliteTaskStore {
    async fn list(&self) -> Result<Vec<TaskRecord>> {
        self.handle
            .reader()
            .call(|c| -> rusqlite::Result<_> {
                let mut stmt =
                    c.prepare_cached(&format!("SELECT {TASK_COLUMNS} FROM tasks ORDER BY name"))?;
                let rows = stmt
                    .query_map([], task_record)?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("task list"))
    }

    async fn get(&self, id: TaskId) -> Result<TaskRecord> {
        self.handle
            .reader()
            .call(move |c| {
                c.query_row(
                    &format!("SELECT {TASK_COLUMNS} FROM tasks WHERE id = ?1"),
                    [id.0],
                    task_record,
                )
            })
            .await
            .map_err(MemoryError::sqlite_or_missing(
                "task get",
                MemoryError::TaskNotFound(id),
            ))
    }

    async fn save(&self, id: Option<TaskId>, spec: TaskSpec) -> Result<TaskRecord> {
        let name = spec.name.to_string();
        let now = now_millis();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let record = match id {
                    None => insert_task(&tx, spec, now)?,
                    Some(id) => replace_task(&tx, id, spec, now)?,
                };
                tx.commit()?;
                Ok(record)
            })
            .await
            .map_err(|error| task_save_error(error, id, name))
    }

    /// Truncates the WAL so the deleted prompt leaves no trace.
    async fn delete(&self, id: TaskId) -> Result<()> {
        let deleted = self
            .handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let deleted = c.execute("DELETE FROM tasks WHERE id = ?1", [id.0])?;
                if deleted > 0 {
                    truncate_wal(c);
                }
                Ok(deleted)
            })
            .await
            .map_err(MemoryError::sqlite("task delete"))?;
        if deleted == 0 {
            return Err(MemoryError::TaskNotFound(id));
        }
        Ok(())
    }

    async fn set_enabled(&self, id: TaskId, enabled: bool) -> Result<TaskRecord> {
        let now = now_millis();
        self.handle
            .writer()
            .call(move |c| {
                c.query_row(
                    &format!(
                        "UPDATE tasks SET enabled = ?2, next_fire_at = NULL, \
                         updated_at = MAX(?3, created_at) WHERE id = ?1 RETURNING {TASK_COLUMNS}"
                    ),
                    rusqlite::params![id.0, enabled, now],
                    task_record,
                )
            })
            .await
            .map_err(MemoryError::sqlite_or_missing(
                "task set_enabled",
                MemoryError::TaskNotFound(id),
            ))
    }

    async fn set_next_fire(&self, id: TaskId, next_fire: Option<DateTime<Utc>>) -> Result<()> {
        let updated = self
            .handle
            .writer()
            .call(move |c| update_next_fire(c, id, next_fire))
            .await
            .map_err(MemoryError::sqlite("task set_next_fire"))?;
        if updated == 0 {
            return Err(MemoryError::TaskNotFound(id));
        }
        Ok(())
    }

    async fn fire(
        &self,
        task: TaskId,
        trigger: TaskTrigger,
        next_fire: Option<DateTime<Utc>>,
    ) -> Result<Option<TaskRunId>> {
        let now = now_millis();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                if update_next_fire(&tx, task, next_fire)? == 0 {
                    return Err(rusqlite::Error::QueryReturnedNoRows);
                }
                let queued = queue_coalesced(&tx, task, &trigger, now)?;
                tx.commit()?;
                Ok(queued)
            })
            .await
            .map_err(MemoryError::sqlite_or_missing(
                "task fire",
                MemoryError::TaskNotFound(task),
            ))
    }

    async fn queue_unless_unfinished(
        &self,
        task: TaskId,
        trigger: TaskTrigger,
    ) -> Result<Option<TaskRunId>> {
        let now = now_millis();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let queued = queue_coalesced(&tx, task, &trigger, now)?;
                tx.commit()?;
                Ok(queued)
            })
            .await
            .map_err(missing_task_or("task queue_unless_unfinished", task))
    }

    async fn queue_run(&self, task: TaskId, trigger: TaskTrigger) -> Result<TaskRunId> {
        let now = now_millis();
        self.handle
            .writer()
            .call(move |c| insert_queued_run(c, task, &trigger, now))
            .await
            .map_err(missing_task_or("task queue_run", task))
    }

    async fn start_run(&self, run: TaskRunId, session: SessionId) -> Result<()> {
        let now = now_millis();
        let updated = self
            .handle
            .writer()
            .call(move |c| {
                c.execute(
                    "UPDATE task_runs
                     SET state = ?2, suspend_reason = NULL, started_at = MAX(?3, queued_at),
                         session_uuid = ?4
                     WHERE id = ?1 AND state IN (?5, ?6)",
                    rusqlite::params![run.0, RUN_RUNNING, now, session, RUN_QUEUED, RUN_SUSPENDED],
                )
            })
            .await
            .map_err(MemoryError::sqlite("task start_run"))?;
        run_updated(updated, run)
    }

    async fn set_run_waiting(&self, run: TaskRunId, reason: Option<SuspendReason>) -> Result<()> {
        let (state, reason_code) = match reason {
            Some(reason) => (RUN_SUSPENDED, Some(suspend_reason_code(reason))),
            None => (RUN_QUEUED, None),
        };
        let updated = self
            .handle
            .writer()
            .call(move |c| {
                c.execute(
                    "UPDATE task_runs SET state = ?2, suspend_reason = ?3
                     WHERE id = ?1 AND state IN (?4, ?5) AND started_at IS NULL",
                    rusqlite::params![run.0, state, reason_code, RUN_QUEUED, RUN_SUSPENDED],
                )
            })
            .await
            .map_err(MemoryError::sqlite("task set_run_waiting"))?;
        run_updated(updated, run)
    }

    async fn reorder_queue(&self, order: Vec<TaskRunId>) -> Result<()> {
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                {
                    let mut stmt = tx.prepare_cached(
                        "UPDATE task_runs SET queue_pos = ?2 WHERE id = ?1 AND state IN (?3, ?4)",
                    )?;
                    for (pos, run) in (0_i64..).zip(&order) {
                        stmt.execute(rusqlite::params![run.0, pos, RUN_QUEUED, RUN_SUSPENDED])?;
                    }
                }
                tx.commit()
            })
            .await
            .map_err(MemoryError::sqlite("task reorder_queue"))
    }

    async fn set_queue_paused(&self, paused: bool) -> Result<()> {
        self.handle
            .writer()
            .call(move |c| {
                c.execute(
                    "INSERT INTO task_queue (id, paused) VALUES (0, ?1)
                     ON CONFLICT (id) DO UPDATE SET paused = excluded.paused",
                    [paused],
                )
            })
            .await
            .map_err(MemoryError::sqlite("task set_queue_paused"))?;
        Ok(())
    }

    async fn recover_queue(&self, reason: String) -> Result<RecoveredQueue> {
        let now = now_millis();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                fail_interrupted_runs(&tx, &reason, now)?;
                tx.execute(
                    "UPDATE task_runs SET state = ?1, suspend_reason = NULL WHERE state = ?2",
                    rusqlite::params![RUN_QUEUED, RUN_SUSPENDED],
                )?;
                let recovered = RecoveredQueue {
                    waiting: waiting_runs(&tx)?,
                    paused: tx
                        .query_row("SELECT paused FROM task_queue WHERE id = 0", [], |r| {
                            r.get(0)
                        })
                        .optional()?
                        .unwrap_or(false),
                };
                tx.commit()?;
                Ok(recovered)
            })
            .await
            .map_err(MemoryError::sqlite("task recover_queue"))
    }

    async fn finish_run(&self, run: TaskRunId, end: TaskRunEnd) -> Result<()> {
        let now = now_millis();
        let (state, failure, reply) = match end {
            TaskRunEnd::Succeeded { reply } => (RUN_SUCCEEDED, None, Some(reply)),
            TaskRunEnd::Failed { reason } => (RUN_FAILED, Some(reason), None),
            TaskRunEnd::Cancelled => (RUN_CANCELLED, None, None),
        };
        let updated = self
            .handle
            .writer()
            .call(move |c| {
                c.execute(
                    "UPDATE task_runs
                     SET state = ?2, suspend_reason = NULL, failure = ?3, reply = ?4,
                         ended_at = MAX(?5, queued_at)
                     WHERE id = ?1 AND state NOT IN (?6, ?7, ?8)",
                    rusqlite::params![
                        run.0,
                        state,
                        failure,
                        reply,
                        now,
                        RUN_SUCCEEDED,
                        RUN_FAILED,
                        RUN_CANCELLED,
                    ],
                )
            })
            .await
            .map_err(MemoryError::sqlite("task finish_run"))?;
        run_updated(updated, run)
    }
}

/// The run state stored in columns `idx` (state code), `idx + 1` (suspend reason) and
/// `idx + 2` (failure) of `row`.
pub(super) fn run_state_at(row: &Row<'_>, idx: usize) -> rusqlite::Result<TaskRunState> {
    let code: i64 = row.get(idx)?;
    let out_of_range = || {
        rusqlite::Error::FromSqlConversionFailure(
            idx,
            Type::Integer,
            Box::new(rusqlite::types::FromSqlError::OutOfRange(code)),
        )
    };
    Ok(match code {
        RUN_QUEUED => TaskRunState::Queued,
        RUN_RUNNING => TaskRunState::Running,
        RUN_SUSPENDED => TaskRunState::Suspended {
            reason: match row.get::<_, i64>(idx + 1)? {
                0 => SuspendReason::UserConversation,
                1 => SuspendReason::GpuBusy,
                _ => return Err(out_of_range()),
            },
        },
        RUN_NEEDS_INPUT => TaskRunState::NeedsInput,
        RUN_SUCCEEDED => TaskRunState::Succeeded,
        RUN_FAILED => TaskRunState::Failed {
            reason: row.get(idx + 2)?,
        },
        RUN_CANCELLED => TaskRunState::Cancelled,
        _ => return Err(out_of_range()),
    })
}

/// Record `next_fire` on task `task`; returns how many rows changed.
fn update_next_fire(
    c: &rusqlite::Connection,
    task: TaskId,
    next_fire: Option<DateTime<Utc>>,
) -> rusqlite::Result<usize> {
    c.execute(
        "UPDATE tasks SET next_fire_at = ?2 WHERE id = ?1",
        rusqlite::params![task.0, next_fire.map(|at| at.timestamp_millis())],
    )
}

/// Map a failed insert of a run of `task` to [`MemoryError::TaskNotFound`] when no such
/// task exists, else to the SQLite error of `op`.
fn missing_task_or(
    op: &'static str,
    task: TaskId,
) -> impl FnOnce(tokio_rusqlite::Error) -> MemoryError {
    move |error| match &error {
        tokio_rusqlite::Error::Error(rusqlite::Error::SqliteFailure(failure, _))
            if failure.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_FOREIGNKEY =>
        {
            MemoryError::TaskNotFound(task)
        }
        _ => MemoryError::sqlite(op)(error),
    }
}

/// Queue a run of `task` that `trigger` started unless one is unfinished.
fn queue_coalesced(
    tx: &Transaction<'_>,
    task: TaskId,
    trigger: &TaskTrigger,
    now: i64,
) -> rusqlite::Result<Option<TaskRunId>> {
    if has_unfinished_run(tx, task)? {
        return Ok(None);
    }
    insert_queued_run(tx, task, trigger, now).map(Some)
}

/// Queue a run of `task` that `trigger` started, last in the queue.
fn insert_queued_run(
    c: &rusqlite::Connection,
    task: TaskId,
    trigger: &TaskTrigger,
    now: i64,
) -> rusqlite::Result<TaskRunId> {
    c.query_row(
        "INSERT INTO task_runs (task_id, trigger, state, queued_at, queue_pos)
         VALUES (?1, ?2, ?3, ?4, (SELECT COALESCE(MAX(queue_pos), -1) + 1 FROM task_runs))
         RETURNING id",
        rusqlite::params![task.0, to_json(trigger)?, RUN_QUEUED, now],
        |r| r.get(0).map(TaskRunId),
    )
}

/// Whether a run of `task` is queued, running, suspended or awaiting input.
fn has_unfinished_run(tx: &Transaction<'_>, task: TaskId) -> rusqlite::Result<bool> {
    tx.query_row(
        "SELECT EXISTS (SELECT 1 FROM task_runs WHERE task_id = ?1 AND state IN (?2, ?3, ?4, ?5))",
        rusqlite::params![
            task.0,
            RUN_QUEUED,
            RUN_RUNNING,
            RUN_SUSPENDED,
            RUN_NEEDS_INPUT
        ],
        |r| r.get(0),
    )
}

/// The queued runs in queue order.
fn waiting_runs(tx: &Transaction<'_>) -> rusqlite::Result<Vec<WaitingRun>> {
    let mut stmt =
        tx.prepare("SELECT id, task_id FROM task_runs WHERE state = ?1 ORDER BY queue_pos, id")?;
    stmt.query_map([RUN_QUEUED], |r| {
        Ok(WaitingRun {
            run: TaskRunId(r.get(0)?),
            task: TaskId(r.get(1)?),
        })
    })?
    .collect()
}

/// Fail with `reason` every run that started and has not finished.
fn fail_interrupted_runs(tx: &Transaction<'_>, reason: &str, now: i64) -> rusqlite::Result<()> {
    tx.execute(
        "UPDATE task_runs
         SET state = ?1, suspend_reason = NULL, failure = ?2, ended_at = MAX(?3, queued_at)
         WHERE state IN (?4, ?5) OR (state = ?6 AND started_at IS NOT NULL)",
        rusqlite::params![
            RUN_FAILED,
            reason,
            now,
            RUN_RUNNING,
            RUN_NEEDS_INPUT,
            RUN_SUSPENDED
        ],
    )?;
    Ok(())
}

fn suspend_reason_code(reason: SuspendReason) -> i64 {
    match reason {
        SuspendReason::UserConversation => 0,
        SuspendReason::GpuBusy => 1,
    }
}

fn run_updated(updated: usize, run: TaskRunId) -> Result<()> {
    if updated == 0 {
        return Err(MemoryError::TaskRunNotFound(run));
    }
    Ok(())
}

fn insert_task(tx: &Transaction<'_>, spec: TaskSpec, now: i64) -> rusqlite::Result<TaskRecord> {
    let id = tx.query_row(
        "INSERT INTO tasks (name, prompt, triggers, enabled, grants, plan, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)
         RETURNING id",
        rusqlite::params![
            spec.name.as_str(),
            spec.prompt.as_str(),
            to_json(&spec.triggers)?,
            spec.enabled,
            to_json(&spec.grants)?,
            spec.plan.as_ref().map(to_json).transpose()?,
            now,
        ],
        |r| r.get(0),
    )?;
    Ok(TaskRecord {
        id: TaskId(id),
        spec,
        next_fire: None,
    })
}

/// Overwrite task `id` with `spec`, dropping `spec`'s plan and grants when its prompt
/// differs from the stored one.
fn replace_task(
    tx: &Transaction<'_>,
    id: TaskId,
    mut spec: TaskSpec,
    now: i64,
) -> rusqlite::Result<TaskRecord> {
    let stored_prompt: String =
        tx.query_row("SELECT prompt FROM tasks WHERE id = ?1", [id.0], |r| {
            r.get(0)
        })?;
    if stored_prompt != spec.prompt.as_str() {
        spec.grants.clear();
        spec.plan = None;
    }
    tx.execute(
        "UPDATE tasks SET name = ?2, prompt = ?3, triggers = ?4, enabled = ?5, grants = ?6,
                          plan = ?7, next_fire_at = NULL, updated_at = MAX(?8, created_at)
         WHERE id = ?1",
        rusqlite::params![
            id.0,
            spec.name.as_str(),
            spec.prompt.as_str(),
            to_json(&spec.triggers)?,
            spec.enabled,
            to_json(&spec.grants)?,
            spec.plan.as_ref().map(to_json).transpose()?,
            now,
        ],
    )?;
    Ok(TaskRecord {
        id,
        spec,
        next_fire: None,
    })
}

/// Map a failed save of the task named `name` to its typed error.
fn task_save_error(error: tokio_rusqlite::Error, id: Option<TaskId>, name: String) -> MemoryError {
    match (&error, id) {
        (tokio_rusqlite::Error::Error(rusqlite::Error::QueryReturnedNoRows), Some(id)) => {
            MemoryError::TaskNotFound(id)
        }
        (tokio_rusqlite::Error::Error(rusqlite::Error::SqliteFailure(failure, _)), _)
            if failure.code == ErrorCode::ConstraintViolation
                && failure.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE =>
        {
            MemoryError::TaskNameTaken(name)
        }
        _ => MemoryError::sqlite("task save")(error),
    }
}

fn task_record(row: &Row<'_>) -> rusqlite::Result<TaskRecord> {
    Ok(TaskRecord {
        id: TaskId(row.get(0)?),
        spec: TaskSpec {
            name: validated_text_at(row, 1)?,
            prompt: validated_text_at(row, 2)?,
            triggers: json_at(row, 3)?,
            enabled: row.get(4)?,
            grants: json_at(row, 5)?,
            plan: row
                .get::<_, Option<String>>(6)?
                .map(|plan| from_json(6, &plan))
                .transpose()?,
        },
        next_fire: optional_timestamp_at(row, 7)?,
    })
}

/// The text in column `idx`, checked by `T`'s own conversion.
fn validated_text_at<T>(row: &Row<'_>, idx: usize) -> rusqlite::Result<T>
where
    T: TryFrom<String>,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    T::try_from(row.get(idx)?)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(idx, Type::Text, Box::new(e)))
}

fn json_at<T: DeserializeOwned>(row: &Row<'_>, idx: usize) -> rusqlite::Result<T> {
    from_json(idx, &row.get::<_, String>(idx)?)
}

fn from_json<T: DeserializeOwned>(idx: usize, json: &str) -> rusqlite::Result<T> {
    serde_json::from_str(json)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(idx, Type::Text, Box::new(e)))
}

fn to_json<T: Serialize>(value: &T) -> rusqlite::Result<String> {
    serde_json::to_string(value).map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
}

#[cfg(test)]
mod tests;

//! SQLite-backed [`TaskStore`] over the `tasks` table.

use async_trait::async_trait;
use rusqlite::types::Type;
use rusqlite::{ErrorCode, Row, Transaction};
use serde::Serialize;
use serde::de::DeserializeOwned;

use assistd_ipc::{TaskId, TaskSpec};

use crate::{MemoryError, Result, TaskRecord, TaskStore};

use super::connection::{SqliteHandle, truncate_wal};
use super::timestamp::now_millis;

const SELECT_TASK: &str =
    "SELECT id, name, prompt, triggers, enabled, grants, plan FROM tasks WHERE id = ?1";

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
                let mut stmt = c.prepare_cached(
                    "SELECT id, name, prompt, triggers, enabled, grants, plan FROM tasks \
                     ORDER BY name",
                )?;
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
            .call(move |c| c.query_row(SELECT_TASK, [id.0], task_record))
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
                    "UPDATE tasks SET enabled = ?2, updated_at = MAX(?3, created_at) \
                     WHERE id = ?1 \
                     RETURNING id, name, prompt, triggers, enabled, grants, plan",
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
                          plan = ?7, updated_at = MAX(?8, created_at)
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
    Ok(TaskRecord { id, spec })
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

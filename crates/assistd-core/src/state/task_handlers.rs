//! Handlers for the `Task*` variants of `Request`.

use std::sync::Arc;

use chrono::{DateTime, Local, Utc};
use tokio::sync::mpsc;

use assistd_ipc::{Event, TaskId, TaskSpec};
use assistd_memory::TaskRecord;

use super::fire_times::{check_triggers, next_fire};
use super::{AppState, DispatchError, send_error};

impl AppState {
    pub(super) async fn handle_task_list(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let tasks = self.memory.tasks.list().await.map_err(Into::into);
        send_tasks(id, "task list", tasks, &tx).await
    }

    pub(super) async fn handle_task_get(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let record = self.memory.tasks.get(task).await;
        send_tasks(
            id,
            "task get",
            record.map(|r| vec![r]).map_err(Into::into),
            &tx,
        )
        .await
    }

    pub(super) async fn handle_task_save(
        self: Arc<Self>,
        id: String,
        task: Option<TaskId>,
        spec: TaskSpec,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let record = {
            let _pass = self.runtime.task_scheduler.hold().await;
            self.save_task(task, spec).await
        };
        self.runtime.task_scheduler.reschedule();
        send_tasks(id, "task save", record.map(|r| vec![r]), &tx).await
    }

    pub(super) async fn handle_task_delete(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let deleted = {
            let _pass = self.runtime.task_scheduler.hold().await;
            self.memory.tasks.delete(task).await
        };
        if deleted.is_ok() {
            self.runtime.task_queue.forget_task(task);
        }
        let deleted = deleted.map(|()| Vec::new()).map_err(Into::into);
        send_tasks(id, "task delete", deleted, &tx).await
    }

    pub(super) async fn handle_task_set_enabled(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        enabled: bool,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let record = {
            let _pass = self.runtime.task_scheduler.hold().await;
            self.memory.tasks.set_enabled(task, enabled).await
        };
        self.runtime.task_scheduler.reschedule();
        let record = record.map(|r| vec![r]).map_err(Into::into);
        send_tasks(id, "task set_enabled", record, &tx).await
    }

    /// Save `spec` as task `task`, or as a new task, once its triggers pass
    /// [`check_triggers`] against those it replaces.
    async fn save_task(
        &self,
        task: Option<TaskId>,
        spec: TaskSpec,
    ) -> Result<TaskRecord, DispatchError> {
        let stored = match task {
            Some(task) => self.memory.tasks.get(task).await?.spec.triggers,
            None => Vec::new(),
        };
        check_triggers(&spec.triggers, &stored, &Local::now())?;
        Ok(self.memory.tasks.save(task, spec).await?)
    }
}

/// When `record`'s triggers next fire: as recorded, or computed now when nothing is
/// recorded yet; `None` while it is disabled.
fn reported_next_fire(record: &TaskRecord) -> Option<DateTime<Utc>> {
    if !record.spec.enabled {
        return None;
    }
    record
        .next_fire
        .or_else(|| next_fire(&record.spec.triggers, &Local::now()))
}

/// Send each record as an `Event::Task` then `Done`, or the failure as `Error`.
async fn send_tasks(
    id: String,
    op: &str,
    records: Result<Vec<TaskRecord>, DispatchError>,
    tx: &mpsc::Sender<Event>,
) -> Result<(), DispatchError> {
    match records {
        Ok(records) => {
            for record in records {
                let next_fire = reported_next_fire(&record);
                let _ = tx
                    .send(Event::Task {
                        id: id.clone(),
                        task_id: record.id,
                        spec: record.spec,
                        next_fire,
                    })
                    .await;
            }
            let _ = tx.send(Event::Done { id }).await;
            Ok(())
        }
        Err(e) => {
            send_error(tx, id, format!("{op} failed: {e}")).await;
            Err(e)
        }
    }
}

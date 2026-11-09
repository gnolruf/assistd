//! Handlers for the `Task*` variants of `Request`.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Local, Utc};
use tokio::sync::mpsc;
use tracing::warn;

use assistd_ipc::{Event, MAX_LISTED_RUNS, PermissionMode, TaskId, TaskSpec, TaskTrigger};
use assistd_memory::TaskRecord;

use super::event_triggers::{check_chain, resolve_watched_paths};
use super::fire_times::{check_triggers, next_fire, upcoming_fires};
use super::{AppState, DispatchError, send_error};

/// Fire times a `TaskCheckTrigger` reports.
const CHECKED_FIRES: usize = 3;

impl AppState {
    pub(super) async fn handle_task_list(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let tasks = self.memory.tasks.list().await.map_err(Into::into);
        self.send_tasks(id, "task list", tasks, &tx).await
    }

    pub(super) async fn handle_task_get(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let record = self.memory.tasks.get(task).await;
        self.send_tasks(
            id,
            "task get",
            record.map(|r| vec![r]).map_err(Into::into),
            &tx,
        )
        .await
    }

    pub(super) async fn handle_task_runs(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let records = match self.memory.tasks.runs(task, MAX_LISTED_RUNS).await {
            Ok(records) => records,
            Err(e) => {
                send_error(&tx, id, format!("task runs failed: {e}")).await;
                return Err(e.into());
            }
        };
        for record in records {
            let _ = tx
                .send(Event::TaskRunRecord {
                    id: id.clone(),
                    record,
                })
                .await;
        }
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
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
        self.runtime.task_events.rewatch();
        self.send_tasks(id, "task save", record.map(|r| vec![r]), &tx)
            .await
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
            self.runtime.task_events.rewatch();
        }
        let deleted = deleted.map(|()| Vec::new()).map_err(Into::into);
        self.send_tasks(id, "task delete", deleted, &tx).await
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
        self.runtime.task_events.rewatch();
        let record = record.map(|r| vec![r]).map_err(Into::into);
        self.send_tasks(id, "task set_enabled", record, &tx).await
    }

    pub(super) async fn handle_task_check_trigger(
        self: Arc<Self>,
        id: String,
        task: Option<TaskId>,
        trigger: TaskTrigger,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match self.check_trigger(task, trigger).await {
            Ok(fires) => {
                let _ = tx
                    .send(Event::TriggerFires {
                        id: id.clone(),
                        fires,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                send_error(&tx, id, e.to_string()).await;
                Err(e)
            }
        }
    }

    /// Run the checks [`Self::save_task`] makes on one trigger of task `task`, returning its
    /// next [`CHECKED_FIRES`] fire times.
    async fn check_trigger(
        &self,
        task: Option<TaskId>,
        trigger: TaskTrigger,
    ) -> Result<Vec<DateTime<Utc>>, DispatchError> {
        let stored = match task {
            Some(task) => self.memory.tasks.get(task).await?.spec.triggers,
            None => Vec::new(),
        };
        let now = Local::now();
        let mut triggers = [trigger];
        check_triggers(&triggers, &stored, &now)?;
        resolve_watched_paths(&mut triggers).await?;
        check_chain(task, &triggers, &self.memory.tasks.list().await?)?;
        Ok(upcoming_fires(&triggers[0], &now, CHECKED_FIRES))
    }

    /// Send each record as an `Event::Task`, with its last finished run, then `Done`, or
    /// the failure as `Error`.
    pub(super) async fn send_tasks(
        &self,
        id: String,
        op: &str,
        records: Result<Vec<TaskRecord>, DispatchError>,
        tx: &mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let records = match records {
            Ok(records) => records,
            Err(e) => {
                send_error(tx, id, format!("{op} failed: {e}")).await;
                return Err(e);
            }
        };
        let mut last_runs = self
            .memory
            .tasks
            .last_finished_runs()
            .await
            .unwrap_or_else(|e| {
                warn!(target: "assistd::tasks", "reading last task runs failed: {e}");
                HashMap::new()
            });
        for record in records {
            let next_fire = reported_next_fire(&record);
            let _ = tx
                .send(Event::Task {
                    id: id.clone(),
                    task_id: record.id,
                    last_run: last_runs.remove(&record.id),
                    trigger_failure: self.runtime.task_events.watch_failure(record.id),
                    spec: record.spec,
                    next_fire,
                })
                .await;
        }
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// Save `spec` as task `task`, or as a new task, once its mode is one the daemon allows and
    /// its triggers pass
    /// [`check_triggers`] against those it replaces and [`check_chain`], with its
    /// watched paths tilde-expanded.
    async fn save_task(
        &self,
        task: Option<TaskId>,
        mut spec: TaskSpec,
    ) -> Result<TaskRecord, DispatchError> {
        if spec.mode == PermissionMode::Auto && !self.config.tools.allow_auto_mode {
            return Err(DispatchError::AutoModeOff);
        }
        let stored = match task {
            Some(task) => self.memory.tasks.get(task).await?.spec.triggers,
            None => Vec::new(),
        };
        check_triggers(&spec.triggers, &stored, &Local::now())?;
        resolve_watched_paths(&mut spec.triggers).await?;
        check_chain(task, &spec.triggers, &self.memory.tasks.list().await?)?;
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

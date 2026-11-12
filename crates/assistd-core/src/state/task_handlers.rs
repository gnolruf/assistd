//! Handlers for the `Task*` variants of `Request`.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Local, Utc};
use tracing::warn;

use assistd_ipc::{
    Event, EventBody, MAX_LISTED_RUNS, PermissionMode, TaskId, TaskRequest, TaskSpec, TaskTrigger,
};
use assistd_memory::TaskRecord;

use super::event_triggers::{check_chain, resolve_watched_paths};
use super::fire_times::{check_triggers, next_fire, upcoming_fires};
use super::task_service::TaskService;
use super::{DispatchError, FailedStep};
use crate::bus::EventSink;

/// Fire times a `TaskCheckTrigger` reports.
const CHECKED_FIRES: usize = 3;

impl TaskService {
    /// Route one `Task` request to its handler.
    pub(super) async fn dispatch(
        self: &Arc<Self>,
        id: String,
        req: TaskRequest,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        match req {
            TaskRequest::List => self.handle_task_list(id, tx).await,
            TaskRequest::Save { task, spec } => self.handle_task_save(id, task, spec, tx).await,
            TaskRequest::Delete { task } => self.handle_task_delete(id, task, tx).await,
            TaskRequest::SetEnabled { task, enabled } => {
                self.handle_task_set_enabled(id, task, enabled, tx).await
            }
            TaskRequest::Run { task } => self.handle_task_run(id, task, tx).await,
            TaskRequest::Cancel { run } => self.handle_task_cancel(id, run, tx).await,
            TaskRequest::Queue { op } => self.handle_task_queue(id, op, tx).await,
            TaskRequest::Confirm {
                run,
                confirm_id,
                allow,
                always,
            } => {
                let approval = super::task_confirms::approval(allow, always);
                self.handle_task_confirm(id, run, confirm_id, approval, tx)
                    .await
            }
            TaskRequest::Runs { task } => self.handle_task_runs(id, task, tx).await,
            TaskRequest::Refine {
                prompt,
                mode,
                allowed_hosts,
                feedback,
            } => {
                let ask = super::task_plans::RefineAsk {
                    prompt,
                    mode,
                    allowed_hosts,
                    feedback,
                };
                self.handle_task_refine(id, ask, tx).await
            }
            TaskRequest::CheckTrigger { task, trigger } => {
                self.handle_task_check_trigger(id, task, trigger, tx).await
            }
        }
    }

    pub(super) async fn handle_task_list(
        &self,
        id: String,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let tasks = self.deps.store.list().await.context("task list failed")?;
        self.send_tasks(id, tasks, &tx).await;
        Ok(())
    }

    pub(super) async fn handle_task_runs(
        &self,
        id: String,
        task: TaskId,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let records = self
            .deps
            .store
            .runs(task, MAX_LISTED_RUNS)
            .await
            .context("task runs failed")?;
        for record in records {
            let _ = tx
                .send(Event::new(id.clone(), EventBody::TaskRunRecord { record }))
                .await;
        }
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    pub(super) async fn handle_task_save(
        &self,
        id: String,
        task: Option<TaskId>,
        spec: TaskSpec,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let record = {
            let _pass = self.scheduler.hold().await;
            self.save_task(task, spec).await
        };
        self.scheduler.reschedule();
        self.events.rewatch();
        let record = record.context("task save failed")?;
        self.send_tasks(id, vec![record], &tx).await;
        Ok(())
    }

    pub(super) async fn handle_task_delete(
        &self,
        id: String,
        task: TaskId,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let deleted = {
            let _pass = self.scheduler.hold().await;
            self.deps.store.delete(task).await
        };
        if deleted.is_ok() {
            self.deps.admission.task_queue.forget_task(task);
            self.events.rewatch();
        }
        deleted.context("task delete failed")?;
        self.send_tasks(id, Vec::new(), &tx).await;
        Ok(())
    }

    pub(super) async fn handle_task_set_enabled(
        &self,
        id: String,
        task: TaskId,
        enabled: bool,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let record = {
            let _pass = self.scheduler.hold().await;
            self.deps.store.set_enabled(task, enabled).await
        };
        self.scheduler.reschedule();
        self.events.rewatch();
        let record = record.context("task set_enabled failed")?;
        self.send_tasks(id, vec![record], &tx).await;
        Ok(())
    }

    pub(super) async fn handle_task_check_trigger(
        &self,
        id: String,
        task: Option<TaskId>,
        trigger: TaskTrigger,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let fires = self.check_trigger(task, trigger).await?;
        let _ = tx
            .send(Event::new(id.clone(), EventBody::TriggerFires { fires }))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    /// Run the checks [`Self::save_task`] makes on one trigger of task `task`, returning its
    /// next [`CHECKED_FIRES`] fire times.
    async fn check_trigger(
        &self,
        task: Option<TaskId>,
        trigger: TaskTrigger,
    ) -> Result<Vec<DateTime<Utc>>, DispatchError> {
        let stored = match task {
            Some(task) => self.deps.store.get(task).await?.spec.triggers,
            None => Vec::new(),
        };
        let now = Local::now();
        let mut triggers = [trigger];
        check_triggers(&triggers, &stored, &now)?;
        resolve_watched_paths(&mut triggers).await?;
        check_chain(task, &triggers, &self.deps.store.list().await?)?;
        Ok(upcoming_fires(&triggers[0], &now, CHECKED_FIRES))
    }

    /// Send each record as an `EventBody::Task`, with its last finished run, then `Done`.
    pub(super) async fn send_tasks(&self, id: String, records: Vec<TaskRecord>, tx: &EventSink) {
        let mut last_runs = self
            .deps
            .store
            .last_finished_runs()
            .await
            .unwrap_or_else(|e| {
                warn!(target: "assistd::tasks", "reading last task runs failed: {e}");
                HashMap::new()
            });
        for record in records {
            let next_fire = reported_next_fire(&record);
            let _ = tx
                .send(Event::new(
                    id.clone(),
                    EventBody::Task {
                        task_id: record.id,
                        last_run: last_runs.remove(&record.id),
                        trigger_failure: self.events.watch_failure(record.id),
                        spec: record.spec,
                        next_fire,
                    },
                ))
                .await;
        }
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
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
        if spec.mode == PermissionMode::Auto && !self.deps.config.tools.allow_auto_mode {
            return Err(DispatchError::AutoModeOff);
        }
        let stored = match task {
            Some(task) => self.deps.store.get(task).await?.spec.triggers,
            None => Vec::new(),
        };
        check_triggers(&spec.triggers, &stored, &Local::now())?;
        resolve_watched_paths(&mut spec.triggers).await?;
        check_chain(task, &spec.triggers, &self.deps.store.list().await?)?;
        Ok(self.deps.store.save(task, spec).await?)
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

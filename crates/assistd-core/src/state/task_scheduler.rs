//! `TaskScheduler`: queues a run of each enabled task when one of its time
//! triggers fires, recording every task's next fire time as it goes.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Local, Utc};
use tokio::sync::{Mutex, MutexGuard, Notify};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use assistd_ipc::TaskId;
use assistd_memory::TaskRecord;

use super::AppState;
use super::fire_times::{fired_trigger, next_fire};
use super::task_dispatch::unrequested_run_id;

/// Longest the scheduler sleeps before checking the wall clock again, which tokio's
/// clock does not follow through a suspend or a clock change.
const MAX_SLEEP: Duration = Duration::from_secs(60);

/// Wakes the scheduler when a task changes, and keeps its passes apart from task writes.
#[derive(Debug, Default)]
pub(in crate::state) struct TaskScheduler {
    changed: Notify,
    passes: Mutex<()>,
}

impl TaskScheduler {
    /// Hold off scheduler passes until the guard drops, so none records a fire time
    /// computed from triggers that a write made meanwhile replaced.
    pub(in crate::state) async fn hold(&self) -> MutexGuard<'_, ()> {
        self.passes.lock().await
    }

    /// Have the scheduler recompute fire times now.
    pub(in crate::state) fn reschedule(&self) {
        self.changed.notify_one();
    }
}

impl AppState {
    /// Fire due tasks, then sleep until the next fire time, a task change, or
    /// [`MAX_SLEEP`], until `stopping` fires.
    pub(super) async fn schedule_task_fires(self: Arc<Self>, stopping: CancellationToken) {
        loop {
            let now = Local::now();
            let next = self.fire_due_tasks(&now).await;
            let sleep = next.map_or(MAX_SLEEP, |at| {
                (at - now.to_utc())
                    .to_std()
                    .map_or(MAX_SLEEP, |until| until.min(MAX_SLEEP))
            });
            tokio::select! {
                biased;
                () = stopping.cancelled() => return,
                () = self.runtime.task_scheduler.changed.notified() => {}
                () = tokio::time::sleep(sleep) => {}
            }
        }
    }

    /// Fire every enabled task whose recorded fire time has passed at `now` and record
    /// fire times not yet computed; returns the earliest one still ahead.
    async fn fire_due_tasks(&self, now: &DateTime<Local>) -> Option<DateTime<Utc>> {
        let _pass = self.runtime.task_scheduler.hold().await;
        let tasks = match self.memory.tasks.list().await {
            Ok(tasks) => tasks,
            Err(e) => {
                warn!(target: "assistd::tasks", error = %e, "listing tasks to schedule failed");
                return None;
            }
        };
        let mut earliest = None;
        for record in tasks.into_iter().filter(|record| record.spec.enabled) {
            let next = self.advance_task(record, now).await;
            earliest = earliest.into_iter().chain(next).min();
        }
        earliest
    }

    /// `record`'s next fire time after firing it if it is due at `now`, or after
    /// computing it if none is recorded.
    async fn advance_task(
        &self,
        record: TaskRecord,
        now: &DateTime<Local>,
    ) -> Option<DateTime<Utc>> {
        match record.next_fire {
            Some(due) if due > now.to_utc() => Some(due),
            Some(due) => {
                self.fire_task(record, &due.with_timezone(&Local), now)
                    .await
            }
            None => {
                let next = next_fire(&record.spec.triggers, now);
                if next.is_some() {
                    self.record_next_fire(record.id, next).await;
                }
                next
            }
        }
    }

    /// Queue a run of `record`, whose fire time `due` passed by `now`, unless every
    /// missed fire is skipped or a run of it is unfinished; returns its next fire time.
    async fn fire_task(
        &self,
        record: TaskRecord,
        due: &DateTime<Local>,
        now: &DateTime<Local>,
    ) -> Option<DateTime<Utc>> {
        let next = next_fire(&record.spec.triggers, now);
        let Some(trigger) = fired_trigger(&record.spec.triggers, due, now).cloned() else {
            info!(target: "assistd::tasks", task = %record.id, "skipping missed fire times");
            self.record_next_fire(record.id, next).await;
            return next;
        };
        match self.memory.tasks.fire(record.id, trigger, next).await {
            Ok(Some(run)) => {
                info!(target: "assistd::tasks", task = %record.id, %run, "trigger fired");
                self.runtime
                    .task_queue
                    .push(run, record, unrequested_run_id(run));
            }
            Ok(None) => debug!(
                target: "assistd::tasks",
                task = %record.id,
                "trigger fired into the task's unfinished run"
            ),
            Err(e) => warn!(
                target: "assistd::tasks",
                task = %record.id,
                error = %e,
                "queueing the fired task failed"
            ),
        }
        next
    }

    async fn record_next_fire(&self, task: TaskId, next: Option<DateTime<Utc>>) {
        if let Err(e) = self.memory.tasks.set_next_fire(task, next).await {
            warn!(
                target: "assistd::tasks",
                %task,
                error = %e,
                "recording the task's next fire time failed"
            );
        }
    }
}

//! `TaskEvents`: queues a run of each enabled task whose event trigger fires,
//! and keeps a watch running for each enabled trigger that needs one.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex as StdMutex;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info, warn};

use assistd_ipc::{PresenceState, TaskEvent, TaskId, TaskTrigger};
use assistd_memory::TaskRecord;

use super::event_triggers::{Occurrence, event_trigger, fires_on};
use super::task_dispatch::unrequested_run_id;
use super::task_service::TaskService;

/// Wakes the event watcher when a task changes, and holds when each task's event
/// triggers last fired and why each trigger whose watch is retrying last stopped.
#[derive(Debug, Default)]
pub(in crate::state) struct TaskEvents {
    changed: Notify,
    last_fired: StdMutex<HashMap<TaskId, Instant>>,
    watch_failures: StdMutex<HashMap<(TaskId, TaskTrigger), String>>,
}

/// The watches running for enabled event triggers.
#[derive(Default)]
struct EventWatches {
    session_unlocks: Option<AbortOnDropHandle<()>>,
    paths: HashMap<(TaskId, TaskTrigger), AbortOnDropHandle<()>>,
}

impl TaskEvents {
    /// Have the event watcher match its watches to the enabled triggers now.
    pub(in crate::state) fn rewatch(&self) {
        self.changed.notify_one();
    }

    /// Whether `task` may fire now, `cooldown` after it last did; marks it fired if so.
    fn try_fire(&self, task: TaskId, cooldown: Duration) -> bool {
        let now = Instant::now();
        let mut last_fired = self.last_fired.lock();
        if last_fired
            .get(&task)
            .is_some_and(|at| now.duration_since(*at) < cooldown)
        {
            return false;
        }
        last_fired.insert(task, now);
        true
    }

    /// Why a watch for one of `task`'s triggers last stopped, while it is retrying.
    pub(in crate::state) fn watch_failure(&self, task: TaskId) -> Option<String> {
        self.watch_failures
            .lock()
            .iter()
            .find(|((failed, _), _)| *failed == task)
            .map(|(_, reason)| reason.clone())
    }

    /// Note that the watch for `trigger` of `task` stopped because of `reason`.
    pub(in crate::state) fn record_watch_failure(
        &self,
        task: TaskId,
        trigger: &TaskTrigger,
        reason: String,
    ) {
        self.watch_failures
            .lock()
            .insert((task, trigger.clone()), reason);
    }

    /// Note that the watch for `trigger` of `task` is running.
    pub(in crate::state) fn clear_watch_failure(&self, task: TaskId, trigger: &TaskTrigger) {
        self.watch_failures.lock().remove(&(task, trigger.clone()));
    }

    /// Forget the failures of watches no longer in `watched`.
    fn keep_watch_failures(&self, watched: &HashSet<(TaskId, TaskTrigger)>) {
        self.watch_failures
            .lock()
            .retain(|failed, _| watched.contains(failed));
    }
}

impl TaskService {
    /// Keep a watch running for every enabled trigger that needs one, re-matching them
    /// after each task change, until `stopping` fires.
    pub(super) async fn watch_task_events(self: Arc<Self>, stopping: CancellationToken) {
        let mut watches = EventWatches::default();
        loop {
            self.update_event_watches(&mut watches).await;
            tokio::select! {
                biased;
                () = stopping.cancelled() => return,
                () = self.events.changed.notified() => {}
            }
        }
    }

    /// Fire every `DaemonStarted` trigger once the model first becomes ready.
    pub(super) async fn fire_on_daemon_started(self: Arc<Self>, stopping: CancellationToken) {
        let mut presence = self.deps.presence.subscribe();
        let ready = tokio::select! {
            biased;
            () = stopping.cancelled() => false,
            active = presence.wait_for(|state| *state == PresenceState::Active) => active.is_ok(),
        };
        if ready {
            self.fire_event_triggers(Occurrence::DaemonStarted).await;
        }
    }

    /// Queue a run of every enabled task with an event trigger that fires on `occurrence`.
    pub(in crate::state) async fn fire_event_triggers(&self, occurrence: Occurrence) {
        for record in self.enabled_tasks().await.into_iter().flatten() {
            let fired = record.spec.triggers.iter().find_map(|trigger| {
                let (event, cooldown) = event_trigger(trigger)?;
                fires_on(event, occurrence).then(|| (trigger.clone(), cooldown))
            });
            if let Some((trigger, cooldown)) = fired {
                self.queue_event_run(record, trigger, cooldown).await;
            }
        }
    }

    /// Queue a run of task `task` for its watched `trigger`, if it is still enabled
    /// and still has it.
    pub(in crate::state) async fn fire_watched_trigger(&self, task: TaskId, trigger: &TaskTrigger) {
        let Some((_, cooldown)) = event_trigger(trigger) else {
            return;
        };
        match self.deps.store.get(task).await {
            Ok(record) if record.spec.enabled && record.spec.triggers.contains(trigger) => {
                self.queue_event_run(record, trigger.clone(), cooldown)
                    .await;
            }
            Ok(_) => {}
            Err(e) => warn!(
                target: "assistd::tasks",
                %task,
                error = %e,
                "reading the watched task failed"
            ),
        }
    }

    /// Start the watches the enabled triggers need that are not running, and stop
    /// those no enabled trigger needs any more.
    async fn update_event_watches(self: &Arc<Self>, watches: &mut EventWatches) {
        let Some(tasks) = self.enabled_tasks().await else {
            return;
        };
        let wants_unlocks = tasks.iter().any(|record| {
            record.spec.triggers.iter().any(|trigger| {
                matches!(
                    event_trigger(trigger),
                    Some((TaskEvent::SessionUnlocked, _))
                )
            })
        });
        watches.session_unlocks = wants_unlocks.then(|| {
            watches
                .session_unlocks
                .take()
                .filter(|watch| !watch.is_finished())
                .unwrap_or_else(|| {
                    AbortOnDropHandle::new(tokio::spawn(Arc::clone(self).watch_session_unlocks()))
                })
        });
        let wanted: HashSet<(TaskId, TaskTrigger)> = tasks
            .iter()
            .flat_map(|record| {
                record
                    .spec
                    .triggers
                    .iter()
                    .filter(|trigger| {
                        matches!(
                            event_trigger(trigger),
                            Some((TaskEvent::PathChanged { .. }, _))
                        )
                    })
                    .map(|trigger| (record.id, trigger.clone()))
            })
            .collect();
        watches
            .paths
            .retain(|watched, watch| wanted.contains(watched) && !watch.is_finished());
        self.events.keep_watch_failures(&wanted);
        for watched in wanted {
            if let Entry::Vacant(slot) = watches.paths.entry(watched) {
                let (task, trigger) = slot.key().clone();
                let watch = tokio::spawn(Arc::clone(self).watch_path(task, trigger));
                slot.insert(AbortOnDropHandle::new(watch));
            }
        }
    }

    /// Queue a run of `record` that `trigger` started, unless the task fired within
    /// `cooldown` or has an unfinished run.
    async fn queue_event_run(&self, record: TaskRecord, trigger: TaskTrigger, cooldown: Duration) {
        if !self.events.try_fire(record.id, cooldown) {
            debug!(target: "assistd::tasks", task = %record.id, "event trigger cooling down");
            return;
        }
        match self
            .deps
            .store
            .queue_unless_unfinished(record.id, trigger)
            .await
        {
            Ok(Some(run)) => {
                info!(target: "assistd::tasks", task = %record.id, %run, "event trigger fired");
                self.deps
                    .admission
                    .task_queue
                    .push(run, record, unrequested_run_id(run));
            }
            Ok(None) => debug!(
                target: "assistd::tasks",
                task = %record.id,
                "event trigger fired into the task's unfinished run"
            ),
            Err(e) => warn!(
                target: "assistd::tasks",
                task = %record.id,
                error = %e,
                "queueing the fired task failed"
            ),
        }
    }

    /// The enabled tasks; `None` when they could not be listed.
    async fn enabled_tasks(&self) -> Option<Vec<TaskRecord>> {
        match self.deps.store.list().await {
            Ok(tasks) => Some(
                tasks
                    .into_iter()
                    .filter(|record| record.spec.enabled)
                    .collect(),
            ),
            Err(e) => {
                warn!(target: "assistd::tasks", error = %e, "listing tasks for event triggers failed");
                None
            }
        }
    }
}

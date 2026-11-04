//! The [`TaskStore`] of saved tasks and its no-op fallback.

use std::fmt;

use async_trait::async_trait;

use assistd_ipc::{SuspendReason, TaskId, TaskRunId, TaskSpec, TaskTrigger};

use crate::{MemoryError, Result, SessionId};

/// One row from the `tasks` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskRecord {
    pub id: TaskId,
    pub spec: TaskSpec,
}

/// How a task run finished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskRunEnd {
    /// The model answered; `reply` is its final reply.
    Succeeded {
        reply: String,
    },
    Failed {
        reason: String,
    },
    Cancelled,
}

/// A run waiting in the task queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitingRun {
    pub run: TaskRunId,
    pub task: TaskId,
}

/// The task queue as the daemon left it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveredQueue {
    /// The runs waiting to start, in queue order.
    pub waiting: Vec<WaitingRun>,
    pub paused: bool,
}

/// Persistent saved tasks.
#[async_trait]
pub trait TaskStore: fmt::Debug + Send + Sync + 'static {
    /// Every task, sorted by name.
    async fn list(&self) -> Result<Vec<TaskRecord>>;

    /// The task with `id`; [`MemoryError::TaskNotFound`] when there is none.
    async fn get(&self, id: TaskId) -> Result<TaskRecord>;

    /// Insert `spec` as a new task, or replace task `id` with it. When `id`'s stored
    /// prompt differs from `spec`'s, its plan and grants are cleared in the same write.
    /// Fails with [`MemoryError::TaskNameTaken`] or [`MemoryError::TaskNotFound`].
    async fn save(&self, id: Option<TaskId>, spec: TaskSpec) -> Result<TaskRecord>;

    /// Remove task `id` and its runs; [`MemoryError::TaskNotFound`] when there is none.
    async fn delete(&self, id: TaskId) -> Result<()>;

    /// Set whether task `id`'s triggers fire; [`MemoryError::TaskNotFound`] when there is none.
    async fn set_enabled(&self, id: TaskId, enabled: bool) -> Result<TaskRecord>;

    /// Record a queued run of task `task` that `trigger` started, last in the queue;
    /// [`MemoryError::TaskNotFound`] when there is no such task.
    async fn queue_run(&self, task: TaskId, trigger: TaskTrigger) -> Result<TaskRunId>;

    /// Mark waiting run `run` as running in transcript session `session`;
    /// [`MemoryError::TaskRunNotFound`] unless it is queued or suspended.
    async fn start_run(&self, run: TaskRunId, session: SessionId) -> Result<()>;

    /// Mark waiting run `run` as suspended for `reason`, or as queued again when `None`;
    /// [`MemoryError::TaskRunNotFound`] unless it is queued or suspended.
    async fn set_run_waiting(&self, run: TaskRunId, reason: Option<SuspendReason>) -> Result<()>;

    /// Place the waiting runs in `order` in the queue in that order.
    async fn reorder_queue(&self, order: Vec<TaskRunId>) -> Result<()>;

    /// Record whether the task queue is paused.
    async fn set_queue_paused(&self, paused: bool) -> Result<()>;

    /// Fail every run left running or awaiting input with `reason`, requeue every
    /// suspended run, and return the queue as it was left.
    async fn recover_queue(&self, reason: String) -> Result<RecoveredQueue>;

    /// Record how unfinished run `run` ended; [`MemoryError::TaskRunNotFound`] when it has
    /// already finished or does not exist.
    async fn finish_run(&self, run: TaskRunId, end: TaskRunEnd) -> Result<()>;
}

/// Store with no database: lists nothing and refuses every write.
#[derive(Debug)]
pub struct NoTaskStore;

#[async_trait]
impl TaskStore for NoTaskStore {
    async fn list(&self) -> Result<Vec<TaskRecord>> {
        Ok(Vec::new())
    }

    async fn get(&self, id: TaskId) -> Result<TaskRecord> {
        Err(MemoryError::TaskNotFound(id))
    }

    async fn save(&self, _id: Option<TaskId>, _spec: TaskSpec) -> Result<TaskRecord> {
        Err(MemoryError::Disabled)
    }

    async fn delete(&self, id: TaskId) -> Result<()> {
        Err(MemoryError::TaskNotFound(id))
    }

    async fn set_enabled(&self, id: TaskId, _enabled: bool) -> Result<TaskRecord> {
        Err(MemoryError::TaskNotFound(id))
    }

    async fn queue_run(&self, task: TaskId, _trigger: TaskTrigger) -> Result<TaskRunId> {
        Err(MemoryError::TaskNotFound(task))
    }

    async fn start_run(&self, run: TaskRunId, _session: SessionId) -> Result<()> {
        Err(MemoryError::TaskRunNotFound(run))
    }

    async fn set_run_waiting(&self, run: TaskRunId, _reason: Option<SuspendReason>) -> Result<()> {
        Err(MemoryError::TaskRunNotFound(run))
    }

    async fn reorder_queue(&self, _order: Vec<TaskRunId>) -> Result<()> {
        Ok(())
    }

    async fn set_queue_paused(&self, _paused: bool) -> Result<()> {
        Ok(())
    }

    async fn recover_queue(&self, _reason: String) -> Result<RecoveredQueue> {
        Ok(RecoveredQueue::default())
    }

    async fn finish_run(&self, run: TaskRunId, _end: TaskRunEnd) -> Result<()> {
        Err(MemoryError::TaskRunNotFound(run))
    }
}

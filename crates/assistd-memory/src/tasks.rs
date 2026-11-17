//! Rows and results of the [`TaskStore`](crate::TaskStore).

use std::path::PathBuf;

use chrono::{DateTime, Utc};

use assistd_ipc::{TaskId, TaskRunId, TaskSpec};

/// One row from the `tasks` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskRecord {
    pub id: TaskId,
    pub spec: TaskSpec,
    /// When its triggers next fire, as last recorded; `None` after every save or
    /// enable until it is recorded again, and when nothing is due.
    pub next_fire: Option<DateTime<Utc>>,
}

/// A row of the `tasks` table that no longer decodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndecodableTask {
    pub id: TaskId,
    /// Its name as stored, unchecked.
    pub name: String,
    /// What about the row failed to decode.
    pub reason: String,
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

/// What [`TaskStore::prune_runs`](crate::TaskStore::prune_runs) deleted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrunedRuns {
    pub runs: u64,
    /// Saved relay answers the deleted sessions named, for the caller to delete.
    pub released_answer_files: Vec<PathBuf>,
}

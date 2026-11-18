//! Which event triggers an occurrence fires, and the checks `TaskSave` runs on
//! event triggers.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use assistd_ipc::{FinishOutcome, TaskEvent, TaskId, TaskTrigger};
use assistd_memory::TaskRecord;
use assistd_utils::path::expand_tilde_from_env;

use super::fire_times::TriggerError;

/// Something that happened which every task's event triggers are checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Occurrence {
    DaemonStarted,
    SessionUnlocked,
    /// A run of `task` finished, not by being cancelled.
    TaskFinished {
        task: TaskId,
        succeeded: bool,
    },
}

/// `trigger`'s event and cool-down when it is an event trigger.
pub(super) fn event_trigger(trigger: &TaskTrigger) -> Option<(&TaskEvent, Duration)> {
    match trigger {
        TaskTrigger::Event {
            event,
            cooldown_secs,
        } => Some((event, Duration::from_secs(u64::from(*cooldown_secs)))),
        TaskTrigger::Manual | TaskTrigger::At { .. } | TaskTrigger::Schedule { .. } => None,
    }
}

/// Whether `event` fires on `occurrence`.
pub(super) fn fires_on(event: &TaskEvent, occurrence: Occurrence) -> bool {
    match (event, occurrence) {
        (TaskEvent::DaemonStarted, Occurrence::DaemonStarted)
        | (TaskEvent::SessionUnlocked, Occurrence::SessionUnlocked) => true,
        (
            TaskEvent::TaskFinished { task, outcome },
            Occurrence::TaskFinished {
                task: finished,
                succeeded,
            },
        ) => {
            *task == finished
                && match outcome {
                    FinishOutcome::Succeeded => succeeded,
                    FinishOutcome::Failed => !succeeded,
                    FinishOutcome::Either => true,
                }
        }
        _ => false,
    }
}

/// Tilde-expand every watched path in `triggers`, refusing one that is relative or
/// cannot be read.
pub(super) async fn resolve_watched_paths(
    triggers: &mut [TaskTrigger],
) -> Result<(), TriggerError> {
    for trigger in triggers {
        let TaskTrigger::Event {
            event: TaskEvent::PathChanged { path, .. },
            ..
        } = trigger
        else {
            continue;
        };
        *path = tilde_expanded(path);
        if !path.is_absolute() {
            return Err(TriggerError::RelativePath(path.clone()));
        }
        tokio::fs::metadata(&*path)
            .await
            .map_err(|source| TriggerError::MissingPath {
                path: path.clone(),
                source,
            })?;
    }
    Ok(())
}

/// Refuse a `TaskFinished` trigger in `triggers`, the new triggers of task `saving`
/// (`None` for a task not saved yet), that waits on a task missing from `tasks` or on
/// a chain leading back to `saving`.
pub(super) fn check_chain(
    saving: Option<TaskId>,
    triggers: &[TaskTrigger],
    tasks: &[TaskRecord],
) -> Result<(), TriggerError> {
    for upstream in upstreams(triggers) {
        if !tasks.iter().any(|record| record.id == upstream) {
            return Err(TriggerError::UnknownUpstream(upstream));
        }
        if saving.is_some_and(|saving| chain_reaches(upstream, saving, tasks)) {
            return Err(TriggerError::ChainCycle(upstream));
        }
    }
    Ok(())
}

/// Whether `target` is `start` or a task `start` waits on, directly or along a chain.
fn chain_reaches(start: TaskId, target: TaskId, tasks: &[TaskRecord]) -> bool {
    let mut seen = HashSet::new();
    let mut pending = vec![start];
    while let Some(task) = pending.pop() {
        if task == target {
            return true;
        }
        if !seen.insert(task) {
            continue;
        }
        if let Some(record) = tasks.iter().find(|record| record.id == task) {
            pending.extend(upstreams(&record.spec.triggers));
        }
    }
    false
}

/// The tasks whose runs finishing fire one of `triggers`.
fn upstreams(triggers: &[TaskTrigger]) -> impl Iterator<Item = TaskId> + '_ {
    triggers
        .iter()
        .filter_map(|trigger| match event_trigger(trigger) {
            Some((TaskEvent::TaskFinished { task, .. }, _)) => Some(*task),
            _ => None,
        })
}

fn tilde_expanded(path: &Path) -> PathBuf {
    path.to_str()
        .map_or_else(|| path.to_path_buf(), expand_tilde_from_env)
}

#[cfg(test)]
mod tests {
    use assistd_ipc::{TaskName, TaskPrompt, TaskSpec};

    use super::*;

    fn after(task: i64) -> TaskTrigger {
        TaskTrigger::Event {
            event: TaskEvent::TaskFinished {
                task: TaskId(task),
                outcome: FinishOutcome::Either,
            },
            cooldown_secs: 60,
        }
    }

    fn task(id: i64, triggers: Vec<TaskTrigger>) -> TaskRecord {
        TaskRecord {
            id: TaskId(id),
            spec: TaskSpec {
                name: TaskName::try_from(format!("task {id}")).unwrap(),
                prompt: TaskPrompt::try_from("work".to_string()).unwrap(),
                triggers,
                enabled: true,
                mode: assistd_ipc::PermissionMode::Manual,
                allowed_hosts: Vec::new(),
                announce: false,
                kept_runs: None,
            },
            next_fire: None,
        }
    }

    #[test]
    fn a_task_finished_chain_that_would_loop_is_refused() {
        let tasks = [
            task(1, vec![after(3)]),
            task(2, vec![after(1)]),
            task(3, Vec::new()),
        ];
        assert!(matches!(
            check_chain(Some(TaskId(3)), &[after(2)], &tasks),
            Err(TriggerError::ChainCycle(TaskId(2)))
        ));
        assert!(matches!(
            check_chain(Some(TaskId(3)), &[after(3)], &tasks),
            Err(TriggerError::ChainCycle(TaskId(3)))
        ));
        assert!(check_chain(Some(TaskId(1)), &[after(3)], &tasks).is_ok());
        assert!(check_chain(None, &[after(2)], &tasks).is_ok());
    }
}

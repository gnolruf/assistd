//! Which event triggers an occurrence fires, and the checks `TaskSave` runs on
//! event triggers.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use assistd_config::ToolsScratchConfig;
use assistd_ipc::{FinishOutcome, TaskEvent, TaskId, TaskTrigger};
use assistd_memory::TaskRecord;
use assistd_tools::policy::on_blocking_pool;
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

/// A directory every turn writes in, which no task may watch, alone or as part of a
/// larger tree, since a watched path is writable only by that task's runs.
#[derive(Debug)]
pub(super) struct KeptWritable {
    /// Canonical.
    dir: PathBuf,
    what: &'static str,
}

impl KeptWritable {
    /// `$HOME`, the daemon's working directory and the scratch directory of `scratch`,
    /// leaving out any that cannot be resolved.
    pub(super) async fn dirs(scratch: &ToolsScratchConfig) -> Vec<Self> {
        let scratch = scratch.resolved_dir().ok();
        on_blocking_pool(move || {
            [
                (
                    std::env::var_os("HOME").map(PathBuf::from),
                    "your home folder",
                ),
                (
                    std::env::current_dir().ok(),
                    "the daemon's working directory",
                ),
                (scratch, "the scratch directory"),
            ]
            .into_iter()
            .filter_map(|(dir, what)| {
                let dir = dir?.canonicalize().ok()?;
                Some(Self { dir, what })
            })
            .collect()
        })
        .await
    }
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

/// Tilde-expand every watched path in `triggers`, refusing one that is relative, cannot be
/// read, or holds one of `kept_writable`.
pub(super) async fn resolve_watched_paths(
    triggers: &mut [TaskTrigger],
    kept_writable: &[KeptWritable],
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
        let watched =
            tokio::fs::canonicalize(&*path)
                .await
                .map_err(|source| TriggerError::MissingPath {
                    path: path.clone(),
                    source,
                })?;
        if let Some(kept) = kept_writable
            .iter()
            .find(|kept| kept.dir.starts_with(&watched))
        {
            return Err(TriggerError::LocksTooMuch {
                path: path.clone(),
                kept: kept.what,
            });
        }
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

    #[tokio::test]
    async fn a_watch_over_a_directory_every_turn_writes_in_is_refused() {
        let home_dir = tempfile::tempdir().unwrap();
        let home = home_dir.path().canonicalize().unwrap();
        let inbox = home.join("inbox");
        std::fs::create_dir(&inbox).unwrap();
        let kept_writable = [KeptWritable {
            dir: home.clone(),
            what: "your home folder",
        }];
        let parent = home.parent().unwrap().to_path_buf();
        for (path, refused) in [(home, true), (parent, true), (inbox, false)] {
            let mut triggers = [TaskTrigger::Event {
                event: TaskEvent::PathChanged {
                    path: path.clone(),
                    debounce_secs: 2,
                },
                cooldown_secs: 60,
            }];
            let checked = resolve_watched_paths(&mut triggers, &kept_writable).await;
            assert_eq!(
                matches!(checked, Err(TriggerError::LocksTooMuch { .. })),
                refused,
                "{path:?}: {checked:?}"
            );
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

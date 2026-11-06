//! One-off notices about task runs: a run that finished or failed, and a run newly paused on
//! a prompt nobody has answered.

use std::collections::HashSet;

use assistd_ipc::{Event, PausedPrompt, QueueItem, TaskRunId, TaskRunState};

use super::activity::{flatten_whitespace, truncate_chars_from_start};

/// Characters of a finished run's reply a notice shows, from its start.
const REPLY_START_CHARS: usize = 200;

/// A notice about one task run, shown beside the turn notification rather than replacing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TaskNotice {
    pub run: TaskRunId,
    pub task_name: String,
    pub outcome: NoticeOutcome,
}

impl TaskNotice {
    /// Shown even while the chat has focus.
    pub(super) fn ignores_focus(&self) -> bool {
        matches!(self.outcome, NoticeOutcome::NeedsInput { .. })
    }
}

/// What a [`TaskNotice`] reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum NoticeOutcome {
    /// The start of the run's final reply.
    Finished {
        reply_start: String,
    },
    Failed {
        reason: String,
    },
    /// Why the prompt the run waits on needs confirming.
    NeedsInput {
        reason: String,
    },
}

/// Turns task events into [`TaskNotice`]s, each paused prompt noticed once.
#[derive(Debug, Default)]
pub(super) struct TaskNoticeTracker {
    /// Confirmation ids of the prompts paused runs wait on, as last broadcast.
    paused: HashSet<String>,
}

impl TaskNoticeTracker {
    /// The notices `ev` raises; a cancelled run raises none.
    pub(super) fn notices_for(&mut self, ev: &Event) -> Vec<TaskNotice> {
        match ev {
            Event::TaskRunEnded {
                run,
                task_name,
                state,
                reply,
                ..
            } => ended_outcome(state, reply.as_deref())
                .map(|outcome| TaskNotice {
                    run: *run,
                    task_name: task_name.to_string(),
                    outcome,
                })
                .into_iter()
                .collect(),
            Event::TaskQueue { snapshot, .. } => self.newly_paused(&snapshot.items),
            _ => Vec::new(),
        }
    }

    fn newly_paused(&mut self, items: &[QueueItem]) -> Vec<TaskNotice> {
        let paused: Vec<(TaskRunId, &str, &PausedPrompt)> = items
            .iter()
            .filter_map(|item| match item {
                QueueItem::Run {
                    run,
                    task_name,
                    prompt: Some(prompt),
                    ..
                } => Some((*run, task_name.as_str(), prompt)),
                _ => None,
            })
            .collect();
        let notices = paused
            .iter()
            .filter(|(_, _, prompt)| !self.paused.contains(&prompt.confirm_id))
            .map(|(run, task_name, prompt)| TaskNotice {
                run: *run,
                task_name: (*task_name).to_string(),
                outcome: NoticeOutcome::NeedsInput {
                    reason: prompt.matched_pattern.clone(),
                },
            })
            .collect();
        self.paused = paused
            .into_iter()
            .map(|(_, _, prompt)| prompt.confirm_id.clone())
            .collect();
        notices
    }
}

fn ended_outcome(state: &TaskRunState, reply: Option<&str>) -> Option<NoticeOutcome> {
    match state {
        TaskRunState::Succeeded => Some(NoticeOutcome::Finished {
            reply_start: truncate_chars_from_start(
                &flatten_whitespace(reply.unwrap_or_default()),
                REPLY_START_CHARS,
            ),
        }),
        TaskRunState::Failed { reason } => Some(NoticeOutcome::Failed {
            reason: reason.clone(),
        }),
        TaskRunState::Queued
        | TaskRunState::Running
        | TaskRunState::Suspended { .. }
        | TaskRunState::NeedsInput
        | TaskRunState::Cancelled => None,
    }
}

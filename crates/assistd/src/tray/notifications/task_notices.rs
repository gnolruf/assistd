//! One-off notices about task runs: a run newly paused on a prompt nobody has answered, and
//! the final reply of a task that announces its replies.

use std::collections::HashSet;

use assistd_ipc::{Event, QueueItem, TaskQueueSnapshot, TaskRunId};

use super::activity::truncate_chars_from_start;

/// Characters of an announced reply a notice shows, from its start.
const REPLY_CHARS: usize = 300;

/// A notice about one task run, shown beside the turn notification rather than replacing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TaskNotice {
    pub run: TaskRunId,
    pub task_name: String,
    pub kind: NoticeKind,
}

impl TaskNotice {
    /// Shown even while the chat has focus.
    pub(super) fn ignores_focus(&self) -> bool {
        matches!(self.kind, NoticeKind::NeedsInput { .. })
    }
}

/// What a [`TaskNotice`] reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum NoticeKind {
    /// Why the prompt the run waits on needs confirming.
    NeedsInput { reason: String },
    /// The start of the run's final reply.
    Reply { text: String },
}

/// Turns task events into [`TaskNotice`]s, each paused prompt noticed once.
#[derive(Debug, Default)]
pub(super) struct TaskNoticeTracker {
    /// Confirmation ids of the prompts paused runs wait on, as last broadcast.
    paused: HashSet<String>,
}

impl TaskNoticeTracker {
    /// The notices `ev` raises.
    pub(super) fn notices_for(&mut self, ev: &Event) -> Vec<TaskNotice> {
        match ev {
            Event::TaskQueue { snapshot, .. } => self.newly_paused(snapshot),
            Event::TaskReply {
                run,
                task_name,
                reply,
                ..
            } => vec![TaskNotice {
                run: *run,
                task_name: task_name.to_string(),
                kind: NoticeKind::Reply {
                    text: truncate_chars_from_start(reply.trim(), REPLY_CHARS),
                },
            }],
            _ => Vec::new(),
        }
    }

    /// A notice for every prompt in `snapshot` not in the previous one.
    fn newly_paused(&mut self, snapshot: &TaskQueueSnapshot) -> Vec<TaskNotice> {
        let mut paused = HashSet::new();
        let mut notices = Vec::new();
        for item in &snapshot.items {
            let QueueItem::Run {
                run,
                task_name,
                prompt: Some(prompt),
                ..
            } = item
            else {
                continue;
            };
            if !self.paused.contains(&prompt.confirm_id) {
                notices.push(TaskNotice {
                    run: *run,
                    task_name: task_name.to_string(),
                    kind: NoticeKind::NeedsInput {
                        reason: prompt.matched_pattern.clone(),
                    },
                });
            }
            paused.insert(prompt.confirm_id.clone());
        }
        self.paused = paused;
        notices
    }
}

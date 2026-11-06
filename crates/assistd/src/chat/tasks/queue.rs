//! The Queue tab: the task queue as the daemon last broadcast it, and the keys that
//! reorder, cancel and pause it or answer a paused run.

use assistd_ipc::{
    PausedPrompt, QueueItem, TaskQueueOp, TaskQueueSnapshot, TaskRunId, TaskRunState,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// What a key on the Queue tab asks for.
pub(super) enum QueueAction {
    Stay,
    Close,
    Change(TaskQueueOp),
    /// Show the prompt paused run `run` waits on.
    Answer(TaskRunId, PausedPrompt),
}

/// Which queue item is selected, kept across snapshots that move it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ItemKey {
    Run(TaskRunId),
    Turn(String),
}

impl ItemKey {
    fn of(item: &QueueItem) -> Self {
        match item {
            QueueItem::Run { run, .. } => Self::Run(*run),
            QueueItem::UserTurn { id, .. } => Self::Turn(id.clone()),
        }
    }
}

/// The queue as last broadcast and the selected item.
#[derive(Default)]
pub(in crate::chat) struct QueueView {
    snapshot: Option<TaskQueueSnapshot>,
    selected: usize,
}

impl QueueView {
    /// `None` until the first broadcast arrives.
    pub(in crate::chat) fn snapshot(&self) -> Option<&TaskQueueSnapshot> {
        self.snapshot.as_ref()
    }

    pub(in crate::chat) fn selected(&self) -> usize {
        self.selected
    }

    /// Show `snapshot`, keeping the selected item selected when it is still queued.
    pub(super) fn set_snapshot(&mut self, snapshot: TaskQueueSnapshot) {
        let kept = self.selected_item().map(ItemKey::of).and_then(|key| {
            snapshot
                .items
                .iter()
                .position(|item| ItemKey::of(item) == key)
        });
        self.selected = kept
            .unwrap_or(self.selected)
            .min(snapshot.items.len().saturating_sub(1));
        self.snapshot = Some(snapshot);
    }

    /// ↑/↓ select, shift+↑/↓ move a waiting run, Enter answers a paused run, `x` cancels,
    /// `p` pauses or resumes the queue, and Esc closes the panel.
    pub(super) fn on_key(&mut self, ev: KeyEvent) -> QueueAction {
        let shift = ev.modifiers.contains(KeyModifiers::SHIFT);
        match ev.code {
            KeyCode::Esc => return QueueAction::Close,
            KeyCode::Up if shift => return self.move_selected(-1),
            KeyCode::Down if shift => return self.move_selected(1),
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.items().len().saturating_sub(1));
            }
            KeyCode::Enter => return self.answer_selected(),
            KeyCode::Char('x') | KeyCode::Delete => return self.cancel_selected(),
            KeyCode::Char('p') => return self.toggle_paused(),
            _ => {}
        }
        QueueAction::Stay
    }

    fn items(&self) -> &[QueueItem] {
        self.snapshot
            .as_ref()
            .map_or(&[], |snapshot| snapshot.items.as_slice())
    }

    fn selected_item(&self) -> Option<&QueueItem> {
        self.items().get(self.selected)
    }

    /// Index of the first waiting item: the running runs come first.
    fn waiting_start(&self) -> usize {
        self.items()
            .iter()
            .take_while(|item| {
                matches!(
                    item,
                    QueueItem::Run {
                        state: TaskRunState::Running | TaskRunState::NeedsInput,
                        ..
                    }
                )
            })
            .count()
    }

    /// Move the selected waiting run `by` places among the waiting items.
    fn move_selected(&mut self, by: isize) -> QueueAction {
        let Some(QueueItem::Run { run, .. }) = self.selected_item() else {
            return QueueAction::Stay;
        };
        let run = *run;
        let waiting_start = self.waiting_start();
        let Some(from) = self.selected.checked_sub(waiting_start) else {
            return QueueAction::Stay;
        };
        let last = self.items().len() - waiting_start - 1;
        let Some(to) = from.checked_add_signed(by).filter(|to| *to <= last) else {
            return QueueAction::Stay;
        };
        self.selected = waiting_start + to;
        QueueAction::Change(TaskQueueOp::Move {
            run,
            to: u32::try_from(to).unwrap_or(u32::MAX),
        })
    }

    fn answer_selected(&self) -> QueueAction {
        match self.selected_item() {
            Some(QueueItem::Run {
                run,
                prompt: Some(prompt),
                ..
            }) => QueueAction::Answer(*run, prompt.clone()),
            _ => QueueAction::Stay,
        }
    }

    fn cancel_selected(&self) -> QueueAction {
        let op = match self.selected_item() {
            Some(QueueItem::Run { run, .. }) => TaskQueueOp::Cancel { run: *run },
            Some(QueueItem::UserTurn { id, .. }) => TaskQueueOp::CancelTurn { id: id.clone() },
            None => return QueueAction::Stay,
        };
        QueueAction::Change(op)
    }

    fn toggle_paused(&self) -> QueueAction {
        match &self.snapshot {
            Some(snapshot) if snapshot.paused => QueueAction::Change(TaskQueueOp::Resume),
            Some(_) => QueueAction::Change(TaskQueueOp::Pause),
            None => QueueAction::Stay,
        }
    }
}

/// Every run in `snapshot` paused on a prompt, with that prompt.
pub(in crate::chat) fn paused_prompts(
    snapshot: &TaskQueueSnapshot,
) -> impl Iterator<Item = (TaskRunId, &PausedPrompt)> {
    snapshot.items.iter().filter_map(|item| match item {
        QueueItem::Run {
            run,
            prompt: Some(prompt),
            ..
        } => Some((*run, prompt)),
        _ => None,
    })
}

/// Whether paused run `run` waits on confirmation `confirm_id` in `snapshot`.
pub(in crate::chat) fn waits_on(
    snapshot: &TaskQueueSnapshot,
    run: TaskRunId,
    confirm_id: &str,
) -> bool {
    paused_prompts(snapshot)
        .any(|(paused, prompt)| paused == run && prompt.confirm_id == confirm_id)
}

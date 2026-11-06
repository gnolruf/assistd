//! The Runs tab: one task's run history, newest first, as `TaskRuns` lists it.

use assistd_ipc::{TaskId, TaskRunRecord};
use crossterm::event::{KeyCode, KeyEvent};

/// The branch a task run's session holds its turn on.
const TASK_RUN_BRANCH: &str = "main";

/// What a key on the Runs tab asks for.
pub(super) enum RunsAction {
    Stay,
    Close,
    /// Show the runs of the task `by` places along the task list.
    ShowTask(isize),
    /// Switch the chat to this session-qualified branch.
    Resume(String),
}

/// The runs of the task shown, and the listing that will replace them.
#[derive(Default)]
pub(in crate::chat) struct RunsView {
    task: Option<TaskId>,
    runs: Vec<TaskRunRecord>,
    /// The request id of the listing in flight, and its rows so far.
    listing: Option<(String, Vec<TaskRunRecord>)>,
    loaded: bool,
    selected: usize,
}

impl RunsView {
    /// The task whose runs are shown or asked for.
    pub(super) fn task(&self) -> Option<TaskId> {
        self.task
    }

    pub(in crate::chat) fn runs(&self) -> &[TaskRunRecord] {
        &self.runs
    }

    pub(in crate::chat) fn loaded(&self) -> bool {
        self.loaded
    }

    pub(in crate::chat) fn selected(&self) -> usize {
        self.selected
    }

    /// Show task `task`'s runs once the listing `request` arrives, clearing them first
    /// when it is another task.
    pub(super) fn listing(&mut self, task: TaskId, request: String) {
        if self.task != Some(task) {
            self.runs.clear();
            self.loaded = false;
            self.selected = 0;
        }
        self.task = Some(task);
        self.listing = Some((request, Vec::new()));
    }

    /// Show no task's runs.
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }

    /// Add a row of listing `request`; rows of an older listing are dropped.
    pub(super) fn push(&mut self, request: &str, record: TaskRunRecord) {
        if let Some((current, rows)) = self.listing.as_mut()
            && current == request
        {
            rows.push(record);
        }
    }

    /// Show the rows of listing `request` once it is done.
    pub(super) fn finish(&mut self, request: &str) {
        if self
            .listing
            .as_ref()
            .is_some_and(|(current, _)| current == request)
            && let Some((_, rows)) = self.listing.take()
        {
            self.runs = rows;
            self.loaded = true;
            self.selected = self.selected.min(self.runs.len().saturating_sub(1));
        }
    }

    /// Listing `request` failed; the runs shown stay.
    pub(super) fn fail(&mut self, request: &str) {
        if self
            .listing
            .as_ref()
            .is_some_and(|(current, _)| current == request)
        {
            self.listing = None;
            self.loaded = true;
        }
    }

    /// ↑/↓ select, ←/→ show the previous or next task, Enter resumes the run's session,
    /// and Esc closes the panel.
    pub(super) fn on_key(&mut self, ev: KeyEvent) -> RunsAction {
        match ev.code {
            KeyCode::Esc => return RunsAction::Close,
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.runs.len().saturating_sub(1));
            }
            KeyCode::Left => return RunsAction::ShowTask(-1),
            KeyCode::Right => return RunsAction::ShowTask(1),
            KeyCode::Enter => {
                if let Some(session) = self
                    .runs
                    .get(self.selected)
                    .and_then(|run| run.session_id.as_deref())
                {
                    return RunsAction::Resume(format!("{session}/{TASK_RUN_BRANCH}"));
                }
            }
            _ => {}
        }
        RunsAction::Stay
    }
}

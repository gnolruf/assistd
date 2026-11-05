//! The task editor: a task's name, prompt, triggers and enabled flag, saved through
//! the daemon, which checks each trigger as it is written.

use assistd_ipc::{Request, TaskId, TaskName, TaskPrompt, TaskSpec, TaskTrigger};
use chrono::{DateTime, Utc};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::trigger_form::{FormAction, TriggerForm, UpstreamChoice};
use super::{Outbox, PendingOp};
use crate::chat::settings::{EditAction, EditorShape, TextEditor};

/// One row of the editor, in display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat) enum EditorRow {
    Name,
    Prompt,
    Enabled,
    Trigger(usize),
    AddTrigger,
    Plan,
    Grants,
}

/// What the daemon said of one trigger.
pub(in crate::chat) enum TriggerCheck {
    /// Waiting on the `TaskCheckTrigger` with this request id.
    Checking { request: String },
    /// Its next fire times; empty for a trigger with none.
    Fires(Vec<DateTime<Utc>>),
    /// Why a save would refuse it.
    Refused(String),
}

pub(in crate::chat) struct TriggerRow {
    pub trigger: TaskTrigger,
    pub check: TriggerCheck,
}

/// What the editor is doing with the keys.
pub(in crate::chat) enum EditorMode {
    Browse,
    Name(TextEditor),
    Prompt(TextEditor),
    Trigger(Box<TriggerForm>),
    /// Asking whether to drop unsaved changes.
    Discard,
}

/// What an editor key asks of the panel.
pub(super) enum EditorAction {
    Stay,
    Close,
    OpenPlan(TaskId),
    OpenGrants,
}

/// A task as edited; its grants and plan are only ever the daemon's.
pub(in crate::chat) struct TaskEditor {
    task: Option<TaskId>,
    /// The task as last saved; `None` until a new task is first saved.
    stored: Option<TaskSpec>,
    name: String,
    prompt: String,
    enabled: bool,
    triggers: Vec<TriggerRow>,
    /// Index into [`rows`](Self::rows).
    selected: usize,
    mode: EditorMode,
    /// Why the last save or step failed.
    error: Option<String>,
}

impl TaskEditor {
    /// An empty, disabled task.
    pub(super) fn new_task() -> Self {
        Self {
            task: None,
            stored: None,
            name: String::new(),
            prompt: String::new(),
            enabled: false,
            triggers: Vec::new(),
            selected: 0,
            mode: EditorMode::Browse,
            error: None,
        }
    }

    /// Saved task `task`, with each of its triggers sent to be checked.
    pub(super) fn existing(task: TaskId, spec: TaskSpec, outbox: &mut Outbox) -> Self {
        let mut editor = Self {
            task: Some(task),
            ..Self::new_task()
        };
        editor.load(spec, outbox);
        editor
    }

    pub(in crate::chat) fn task(&self) -> Option<TaskId> {
        self.task
    }

    pub(in crate::chat) fn stored(&self) -> Option<&TaskSpec> {
        self.stored.as_ref()
    }

    pub(in crate::chat) fn name(&self) -> &str {
        &self.name
    }

    pub(in crate::chat) fn prompt(&self) -> &str {
        &self.prompt
    }

    pub(in crate::chat) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(in crate::chat) fn triggers(&self) -> &[TriggerRow] {
        &self.triggers
    }

    pub(in crate::chat) fn mode(&self) -> &EditorMode {
        &self.mode
    }

    pub(in crate::chat) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub(in crate::chat) fn selected_row(&self) -> EditorRow {
        self.rows()
            .get(self.selected)
            .copied()
            .unwrap_or(EditorRow::Name)
    }

    /// Every row in display order.
    pub(in crate::chat) fn rows(&self) -> Vec<EditorRow> {
        [EditorRow::Name, EditorRow::Prompt, EditorRow::Enabled]
            .into_iter()
            .chain((0..self.triggers.len()).map(EditorRow::Trigger))
            .chain([EditorRow::AddTrigger, EditorRow::Plan, EditorRow::Grants])
            .collect()
    }

    /// Whether the name, prompt, enabled flag or triggers differ from the saved task.
    pub(in crate::chat) fn is_modified(&self) -> bool {
        let triggers = self.triggers.iter().map(|row| &row.trigger);
        match &self.stored {
            Some(spec) => {
                self.name != spec.name.as_str()
                    || self.prompt != spec.prompt.as_str()
                    || self.enabled != spec.enabled
                    || !triggers.eq(&spec.triggers)
            }
            None => !self.name.is_empty() || !self.prompt.is_empty() || !self.triggers.is_empty(),
        }
    }

    /// Take the daemon's copy of the task as saved by this editor, reloading every field.
    pub(super) fn adopt_saved(&mut self, task: TaskId, spec: TaskSpec, outbox: &mut Outbox) {
        self.task = Some(task);
        self.error = None;
        self.load(spec, outbox);
    }

    /// Take the daemon's copy after a change to its grants or plan, keeping unsaved edits.
    pub(super) fn refresh_stored(&mut self, spec: TaskSpec) {
        self.stored = Some(spec);
    }

    pub(super) fn set_error(&mut self, message: String) {
        self.error = Some(message);
    }

    /// Record the daemon's answer to the trigger check sent as `request`.
    pub(super) fn record_check(&mut self, request: &str, check: TriggerCheck) {
        if let Some(row) = self.triggers.iter_mut().find(|row| {
            matches!(&row.check, TriggerCheck::Checking { request: pending } if pending == request)
        }) {
            row.check = check;
        }
    }

    /// Record the columns the open prompt box wraps at.
    pub(in crate::chat) fn set_prompt_width(&mut self, width: u16) {
        if let EditorMode::Prompt(editor) = &mut self.mode {
            editor.set_wrap_width(width);
        }
    }

    fn load(&mut self, spec: TaskSpec, outbox: &mut Outbox) {
        self.name = spec.name.as_str().to_string();
        self.prompt = spec.prompt.as_str().to_string();
        self.enabled = spec.enabled;
        let task = self.task;
        self.triggers = spec
            .triggers
            .iter()
            .map(|trigger| checked_row(task, trigger.clone(), outbox))
            .collect();
        self.selected = self.selected.min(self.rows().len() - 1);
        self.stored = Some(spec);
    }

    /// Route a key to the open field, form or prompt, else to the rows.
    pub(super) fn on_key(
        &mut self,
        ev: KeyEvent,
        upstreams: &[UpstreamChoice],
        outbox: &mut Outbox,
    ) -> EditorAction {
        match &mut self.mode {
            EditorMode::Browse => return self.on_browse_key(ev, outbox),
            EditorMode::Name(editor) => match editor.on_key(ev) {
                EditAction::Apply => {
                    self.name = editor.text().trim().to_string();
                    self.mode = EditorMode::Browse;
                }
                EditAction::Cancel => self.mode = EditorMode::Browse,
                EditAction::Pending => {}
            },
            EditorMode::Prompt(editor) => match editor.on_key(ev) {
                EditAction::Apply => {
                    self.prompt = editor.text().to_string();
                    self.mode = EditorMode::Browse;
                }
                EditAction::Cancel => self.mode = EditorMode::Browse,
                EditAction::Pending => {}
            },
            EditorMode::Trigger(form) => match form.on_key(ev, upstreams) {
                FormAction::Apply(trigger) => {
                    let replaces = form.replaces;
                    self.mode = EditorMode::Browse;
                    self.place_trigger(replaces, trigger, outbox);
                }
                FormAction::Cancel => self.mode = EditorMode::Browse,
                FormAction::Pending => {}
            },
            EditorMode::Discard => {
                self.mode = EditorMode::Browse;
                if matches!(ev.code, KeyCode::Char('y') | KeyCode::Enter) {
                    return EditorAction::Close;
                }
            }
        }
        EditorAction::Stay
    }

    /// ↑/↓ move, Enter or Space opens the row, Delete drops a trigger, Ctrl+S saves and
    /// Esc leaves, asking first when something is unsaved.
    fn on_browse_key(&mut self, ev: KeyEvent, outbox: &mut Outbox) -> EditorAction {
        let control = ev.modifiers.contains(KeyModifiers::CONTROL);
        match ev.code {
            KeyCode::Char('s') if control => self.save(outbox),
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(self.rows().len() - 1),
            KeyCode::Esc if self.is_modified() => self.mode = EditorMode::Discard,
            KeyCode::Esc => return EditorAction::Close,
            KeyCode::Enter | KeyCode::Char(' ') => return self.open_row(),
            KeyCode::Delete | KeyCode::Backspace | KeyCode::Char('d') => {
                if let EditorRow::Trigger(index) = self.selected_row() {
                    self.triggers.remove(index);
                }
            }
            _ => {}
        }
        EditorAction::Stay
    }

    fn open_row(&mut self) -> EditorAction {
        match self.selected_row() {
            EditorRow::Name => {
                self.mode = EditorMode::Name(TextEditor::new(self.name.clone(), EditorShape::Line));
            }
            EditorRow::Prompt => {
                let shape = EditorShape::Box { wrap_width: 0 };
                self.mode = EditorMode::Prompt(TextEditor::new(self.prompt.clone(), shape));
            }
            EditorRow::Enabled => self.enabled = !self.enabled,
            EditorRow::Trigger(index) => {
                let form = TriggerForm::editing(index, &self.triggers[index].trigger);
                self.mode = EditorMode::Trigger(Box::new(form));
            }
            EditorRow::AddTrigger => self.mode = EditorMode::Trigger(Box::new(TriggerForm::new())),
            EditorRow::Plan => match self.saved_task() {
                Some(task) => return EditorAction::OpenPlan(task),
                None => self.error = Some("save the task (ctrl+s) before planning it".into()),
            },
            EditorRow::Grants => match self.saved_task() {
                Some(_) => return EditorAction::OpenGrants,
                None => self.error = Some("save the task (ctrl+s) first".into()),
            },
        }
        EditorAction::Stay
    }

    /// The task's id when it is saved with nothing left unsaved.
    fn saved_task(&self) -> Option<TaskId> {
        self.task.filter(|_| !self.is_modified())
    }

    /// Put `trigger` in row `replaces`, or after the others, and send it to be checked.
    fn place_trigger(
        &mut self,
        replaces: Option<usize>,
        trigger: TaskTrigger,
        outbox: &mut Outbox,
    ) {
        let row = checked_row(self.task, trigger, outbox);
        match replaces.and_then(|index| self.triggers.get_mut(index)) {
            Some(slot) => *slot = row,
            None => self.triggers.push(row),
        }
    }

    fn save(&mut self, outbox: &mut Outbox) {
        match self.draft_spec() {
            Ok(spec) => {
                self.error = None;
                let task = self.task;
                outbox.send(PendingOp::Save, |id| Request::TaskSave { id, task, spec });
            }
            Err(message) => self.error = Some(message),
        }
    }

    /// The task to save: the edited fields with the saved grants and plan.
    fn draft_spec(&self) -> Result<TaskSpec, String> {
        let name = TaskName::try_from(self.name.clone()).map_err(|e| e.to_string())?;
        let prompt = TaskPrompt::try_from(self.prompt.clone()).map_err(|e| e.to_string())?;
        let (grants, plan) = self
            .stored
            .as_ref()
            .map(|spec| (spec.grants.clone(), spec.plan.clone()))
            .unwrap_or_default();
        Ok(TaskSpec {
            name,
            prompt,
            triggers: self
                .triggers
                .iter()
                .map(|row| row.trigger.clone())
                .collect(),
            enabled: self.enabled,
            grants,
            plan,
        })
    }
}

/// A row for `trigger` of task `task`, its check sent through `outbox`.
fn checked_row(task: Option<TaskId>, trigger: TaskTrigger, outbox: &mut Outbox) -> TriggerRow {
    let check = trigger.clone();
    let request = outbox.send(PendingOp::CheckTrigger, |id| Request::TaskCheckTrigger {
        id,
        task,
        trigger: check,
    });
    TriggerRow {
        trigger,
        check: TriggerCheck::Checking { request },
    }
}

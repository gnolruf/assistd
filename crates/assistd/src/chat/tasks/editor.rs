//! The task editor: a task's name, prompt, permission mode, allowed hosts, announce flag,
//! kept runs, triggers and enabled flag, saved through the daemon, which checks each trigger as written.

use assistd_ipc::{
    HostName, KeptRuns, ModeRecommendation, PermissionMode, Plan, Request, TaskId, TaskName,
    TaskPrompt, TaskSpec, TaskTrigger,
};
use chrono::{DateTime, Utc};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::prompt_box::{PromptBox, RunPermissions};
use super::trigger_form::{FormAction, TriggerForm, UpstreamChoice, cycled};
use super::{Outbox, PendingOp};
use crate::chat::settings::{
    EditAction, EditorShape, ListAction, ListEditor, ListItem, TextEditor,
};

/// One row of the editor, in display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat) enum EditorRow {
    Name,
    Prompt,
    Mode,
    Hosts,
    Announce,
    KeptRuns,
    Enabled,
    Trigger(usize),
    AddTrigger,
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
    KeptRuns(TextEditor),
    Prompt(PromptBox),
    Hosts(ListEditor),
    Trigger(Box<TriggerForm>),
    /// Leaving with unsaved changes: asking whether to save or drop them.
    Unsaved,
    /// Leaving once the `TaskSave` with this request id is done.
    SavingToClose {
        request: String,
    },
    /// Asking whether to delete the saved task.
    ConfirmDelete,
    /// Asking whether to remove the trigger in this row of the triggers.
    ConfirmRemoveTrigger(usize),
}

/// What an editor key asks of the panel.
pub(super) enum EditorAction {
    Stay,
    Close,
    Delete(TaskId),
}

/// A task as edited.
pub(in crate::chat) struct TaskEditor {
    task: Option<TaskId>,
    /// The task as last saved; `None` until a new task is first saved.
    stored: Option<TaskSpec>,
    name: String,
    prompt: String,
    permission: PermissionMode,
    hosts: Vec<HostName>,
    /// The daemon allows Auto mode, so the mode row offers it.
    auto_offered: bool,
    announce: bool,
    /// `None` keeps the daemon's configured default.
    kept_runs: Option<KeptRuns>,
    enabled: bool,
    triggers: Vec<TriggerRow>,
    /// Index into [`rows`](Self::rows).
    selected: usize,
    mode: EditorMode,
    /// Why the last save or step failed.
    error: Option<String>,
}

impl TaskEditor {
    /// An empty, disabled task in Manual mode; Auto is offered when `auto_offered`.
    pub(super) fn new_task(auto_offered: bool) -> Self {
        Self {
            task: None,
            stored: None,
            name: String::new(),
            prompt: String::new(),
            permission: PermissionMode::Manual,
            hosts: Vec::new(),
            auto_offered,
            announce: false,
            kept_runs: None,
            enabled: false,
            triggers: Vec::new(),
            selected: 0,
            mode: EditorMode::Browse,
            error: None,
        }
    }

    /// Saved task `task`, with each of its triggers sent to be checked.
    pub(super) fn existing(
        task: TaskId,
        spec: TaskSpec,
        auto_offered: bool,
        outbox: &mut Outbox,
    ) -> Self {
        let mut editor = Self {
            task: Some(task),
            ..Self::new_task(auto_offered)
        };
        editor.load(spec, outbox);
        editor
    }

    pub(in crate::chat) fn task(&self) -> Option<TaskId> {
        self.task
    }

    pub(in crate::chat) fn name(&self) -> &str {
        &self.name
    }

    pub(in crate::chat) fn prompt(&self) -> &str {
        &self.prompt
    }

    pub(in crate::chat) fn permission(&self) -> PermissionMode {
        self.permission
    }

    pub(in crate::chat) fn hosts(&self) -> &[HostName] {
        &self.hosts
    }

    pub(in crate::chat) fn announce(&self) -> bool {
        self.announce
    }

    pub(in crate::chat) fn kept_runs(&self) -> Option<KeptRuns> {
        self.kept_runs
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
        [
            EditorRow::Name,
            EditorRow::Prompt,
            EditorRow::Mode,
            EditorRow::Hosts,
            EditorRow::Announce,
            EditorRow::KeptRuns,
            EditorRow::Enabled,
        ]
        .into_iter()
        .chain((0..self.triggers.len()).map(EditorRow::Trigger))
        .chain([EditorRow::AddTrigger])
        .collect()
    }

    /// Whether any edited field differs from the saved task.
    pub(in crate::chat) fn is_modified(&self) -> bool {
        let triggers = self.triggers.iter().map(|row| &row.trigger);
        match &self.stored {
            Some(spec) => {
                self.name != spec.name.as_str()
                    || self.prompt != spec.prompt.as_str()
                    || self.permission != spec.mode
                    || self.hosts != spec.allowed_hosts
                    || self.announce != spec.announce
                    || self.kept_runs != spec.kept_runs
                    || self.enabled != spec.enabled
                    || !triggers.eq(&spec.triggers)
            }
            None => {
                !self.name.is_empty()
                    || !self.prompt.is_empty()
                    || self.permission != PermissionMode::Manual
                    || !self.hosts.is_empty()
                    || self.announce
                    || self.kept_runs.is_some()
                    || !self.triggers.is_empty()
            }
        }
    }

    /// Take the daemon's copy of the task as saved by this editor, reloading every field.
    pub(super) fn adopt_saved(&mut self, task: TaskId, spec: TaskSpec, outbox: &mut Outbox) {
        self.task = Some(task);
        self.error = None;
        self.load(spec, outbox);
    }

    /// Take the daemon's copy after a change to its enabled flag, keeping unsaved edits.
    pub(super) fn refresh_stored(&mut self, spec: TaskSpec) {
        self.stored = Some(spec);
    }

    /// Show why a save failed, staying open if it was to close the editor.
    pub(super) fn set_error(&mut self, message: String) {
        self.leave_saving();
        self.error = Some(message);
    }

    /// Whether the editor closes once save `request` is done.
    pub(super) fn closes_after(&self, request: &str) -> bool {
        matches!(&self.mode, EditorMode::SavingToClose { request: pending } if pending == request)
    }

    fn leave_saving(&mut self) {
        if matches!(self.mode, EditorMode::SavingToClose { .. }) {
            self.mode = EditorMode::Browse;
        }
    }

    /// Record the daemon's answer to the trigger check sent as `request`.
    pub(super) fn record_check(&mut self, request: &str, check: TriggerCheck) {
        if let Some(row) = self.triggers.iter_mut().find(|row| {
            matches!(&row.check, TriggerCheck::Checking { request: pending } if pending == request)
        }) {
            row.check = check;
        }
    }

    fn prompt_box(&mut self) -> Option<&mut PromptBox> {
        match &mut self.mode {
            EditorMode::Prompt(prompt) => Some(prompt),
            _ => None,
        }
    }

    /// Whether the open prompt box waits on a refine.
    pub(super) fn refining(&self) -> bool {
        matches!(&self.mode, EditorMode::Prompt(prompt) if prompt.refining())
    }

    pub(super) fn advance_refine_dots(&mut self) {
        if let Some(prompt) = self.prompt_box() {
            prompt.advance_refine_dots();
        }
    }

    /// Stop the prompt box's refine in flight.
    pub(super) fn cancel_refine(&mut self, outbox: &mut Outbox) {
        if let Some(prompt) = self.prompt_box() {
            prompt.cancel_refine(outbox);
        }
    }

    /// Put the rewrite answering `request` in the prompt box, if it still waits on it.
    pub(super) fn on_refined(
        &mut self,
        request: &str,
        prompt: TaskPrompt,
        plan: &Plan,
        recommended: &ModeRecommendation,
    ) {
        if let Some(prompt_box) = self.prompt_box() {
            prompt_box.on_refined(request, prompt, plan, recommended);
        }
    }

    /// Show why refine `request` failed, if the prompt box still waits on it.
    pub(super) fn on_refine_failed(&mut self, request: &str, message: &str) {
        if self
            .prompt_box()
            .is_some_and(|prompt| prompt.on_refine_ended(request))
        {
            self.error = Some(format!("refine failed: {message}"));
        }
    }

    /// Stop waiting on a refine or save whose answer was lost with the connection.
    pub(super) fn on_connection_lost(&mut self) {
        self.leave_saving();
        if let Some(prompt) = self.prompt_box() {
            prompt.abandon_refine();
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
        self.permission = spec.mode;
        self.hosts.clone_from(&spec.allowed_hosts);
        self.announce = spec.announce;
        self.kept_runs = spec.kept_runs;
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
            EditorMode::KeptRuns(editor) => match editor.on_key(ev) {
                EditAction::Apply => match parsed_kept_runs(editor.text()) {
                    Ok(kept_runs) => {
                        self.kept_runs = kept_runs;
                        self.error = None;
                        self.mode = EditorMode::Browse;
                    }
                    Err(reason) => self.error = Some(reason),
                },
                EditAction::Cancel => {
                    self.error = None;
                    self.mode = EditorMode::Browse;
                }
                EditAction::Pending => {}
            },
            EditorMode::Prompt(prompt) => {
                let permissions = RunPermissions {
                    mode: self.permission,
                    allowed_hosts: &self.hosts,
                };
                match prompt.on_key(ev, permissions, outbox) {
                    Ok(EditAction::Apply) => {
                        self.prompt = prompt.text().text().to_string();
                        self.mode = EditorMode::Browse;
                    }
                    Ok(EditAction::Cancel) => self.mode = EditorMode::Browse,
                    Ok(EditAction::Pending) => self.error = None,
                    Err(reason) => self.error = Some(reason),
                }
            }
            EditorMode::Hosts(list) => {
                let entries = host_values(&self.hosts);
                let action = list.on_key(ev, &entries);
                self.apply_host_action(action);
            }
            EditorMode::Trigger(form) => match form.on_key(ev, upstreams) {
                FormAction::Apply(trigger) => {
                    let replaces = form.replaces;
                    self.mode = EditorMode::Browse;
                    self.place_trigger(replaces, trigger, outbox);
                }
                FormAction::Cancel => self.mode = EditorMode::Browse,
                FormAction::Pending => {}
            },
            EditorMode::Unsaved => return self.on_unsaved_key(ev, outbox),
            EditorMode::SavingToClose { .. } => {
                if ev.code == KeyCode::Esc {
                    self.mode = EditorMode::Browse;
                }
            }
            EditorMode::ConfirmDelete => {
                if let Some(confirmed) = confirmation(ev) {
                    self.mode = EditorMode::Browse;
                    if let Some(task) = self.task.filter(|_| confirmed) {
                        return EditorAction::Delete(task);
                    }
                }
            }
            EditorMode::ConfirmRemoveTrigger(index) => {
                let index = *index;
                if let Some(confirmed) = confirmation(ev) {
                    self.mode = EditorMode::Browse;
                    if confirmed {
                        self.triggers.remove(index);
                    }
                }
            }
        }
        EditorAction::Stay
    }

    /// Enter saves then closes, `d` closes without saving, Esc keeps editing.
    fn on_unsaved_key(&mut self, ev: KeyEvent, outbox: &mut Outbox) -> EditorAction {
        match ev.code {
            KeyCode::Enter => {
                self.mode = match self.save(outbox) {
                    Some(request) => EditorMode::SavingToClose { request },
                    None => EditorMode::Browse,
                };
            }
            KeyCode::Char('d') => return EditorAction::Close,
            KeyCode::Esc => self.mode = EditorMode::Browse,
            _ => {}
        }
        EditorAction::Stay
    }

    /// ↑/↓ move, Enter or Space opens the row, ←/→ change the mode, Delete or Backspace
    /// removes a trigger and Ctrl+D deletes the saved task, each after asking, Ctrl+S saves,
    /// and Esc leaves, asking first when something is unsaved.
    fn on_browse_key(&mut self, ev: KeyEvent, outbox: &mut Outbox) -> EditorAction {
        let control = ev.modifiers.contains(KeyModifiers::CONTROL);
        match ev.code {
            KeyCode::Char('s') if control => {
                self.save(outbox);
            }
            KeyCode::Char('d') if control && self.task.is_some() => {
                self.mode = EditorMode::ConfirmDelete;
            }
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(self.rows().len() - 1),
            KeyCode::Esc if self.is_modified() => self.mode = EditorMode::Unsaved,
            KeyCode::Esc => return EditorAction::Close,
            KeyCode::Left | KeyCode::Right if self.selected_row() == EditorRow::Mode => {
                self.cycle_permission(ev.code == KeyCode::Right);
            }
            KeyCode::Enter | KeyCode::Char(' ') => return self.open_row(),
            KeyCode::Delete | KeyCode::Backspace => {
                if let EditorRow::Trigger(index) = self.selected_row() {
                    self.mode = EditorMode::ConfirmRemoveTrigger(index);
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
                self.mode = EditorMode::Prompt(PromptBox::new(self.prompt.clone()));
            }
            EditorRow::Mode => {}
            EditorRow::Hosts => self.mode = EditorMode::Hosts(ListEditor::new(ListItem::Text)),
            EditorRow::Announce => self.announce = !self.announce,
            EditorRow::KeptRuns => {
                let shown = self.kept_runs.map(|count| count.to_string());
                self.mode = EditorMode::KeptRuns(TextEditor::new(
                    shown.unwrap_or_default(),
                    EditorShape::Line,
                ));
            }
            EditorRow::Enabled => self.enabled = !self.enabled,
            EditorRow::Trigger(index) => {
                let form = TriggerForm::editing(index, &self.triggers[index].trigger);
                self.mode = EditorMode::Trigger(Box::new(form));
            }
            EditorRow::AddTrigger => self.mode = EditorMode::Trigger(Box::new(TriggerForm::new())),
        }
        EditorAction::Stay
    }

    /// Step to the next or previous mode the daemon allows.
    fn cycle_permission(&mut self, forward: bool) {
        let modes = [
            PermissionMode::Manual,
            PermissionMode::AcceptEdits,
            PermissionMode::Auto,
        ];
        let offered = if self.auto_offered { 3 } else { 2 };
        self.permission = cycled(&modes[..offered], self.permission, forward);
    }

    /// The allowed hosts as the list box's entries.
    pub(in crate::chat) fn host_entries(&self) -> Vec<toml::Value> {
        host_values(&self.hosts)
    }

    fn apply_host_action(&mut self, action: ListAction) {
        match action {
            ListAction::Pending => {}
            ListAction::Close => {
                self.mode = EditorMode::Browse;
                self.error = None;
            }
            ListAction::Store { index, value } => match self.place_host(index, &value) {
                Ok(()) => {
                    self.error = None;
                    if let EditorMode::Hosts(list) = &mut self.mode {
                        list.close_entry();
                    }
                }
                Err(reason) => self.error = Some(reason),
            },
            ListAction::Delete(index) => {
                self.hosts.remove(index);
            }
            ListAction::Refused(reason) => self.error = Some(reason),
        }
    }

    /// Put the host `value` names at `index`, appending when `index` is the host count;
    /// refuses a value that names no host or one listed elsewhere.
    fn place_host(&mut self, index: usize, value: &toml::Value) -> Result<(), String> {
        let text = value.as_str().unwrap_or_default().to_string();
        let host = HostName::try_from(text).map_err(|e| e.to_string())?;
        let listed_elsewhere = self
            .hosts
            .iter()
            .enumerate()
            .any(|(at, listed)| at != index && *listed == host);
        if listed_elsewhere {
            return Err(format!("{} is already allowed", host.as_str()));
        }
        match self.hosts.get_mut(index) {
            Some(slot) => *slot = host,
            None => self.hosts.push(host),
        }
        Ok(())
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

    /// Send the edited task to be saved, returning the request id; `None`, with the error
    /// shown, when the fields make no task.
    fn save(&mut self, outbox: &mut Outbox) -> Option<String> {
        match self.draft_spec() {
            Ok(spec) => {
                self.error = None;
                let task = self.task;
                Some(outbox.send(PendingOp::Save, |id| Request::TaskSave { id, task, spec }))
            }
            Err(message) => {
                self.error = Some(message);
                None
            }
        }
    }

    fn draft_spec(&self) -> Result<TaskSpec, String> {
        let name = TaskName::try_from(self.name.clone()).map_err(|e| e.to_string())?;
        let prompt = TaskPrompt::try_from(self.prompt.clone()).map_err(|e| e.to_string())?;
        Ok(TaskSpec {
            name,
            prompt,
            triggers: self
                .triggers
                .iter()
                .map(|row| row.trigger.clone())
                .collect(),
            enabled: self.enabled,
            mode: self.permission,
            allowed_hosts: self.hosts.clone(),
            announce: self.announce,
            kept_runs: self.kept_runs,
        })
    }
}

/// The kept-runs count written as `text`; blank means the configured default.
fn parsed_kept_runs(text: &str) -> Result<Option<KeptRuns>, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let count: u32 = trimmed
        .parse()
        .map_err(|_| format!("{trimmed:?} is not a whole number"))?;
    KeptRuns::try_from(count)
        .map(Some)
        .map_err(|e| e.to_string())
}

/// `Some(true)` for Enter, `Some(false)` for Esc, and `None` for any other key.
fn confirmation(ev: KeyEvent) -> Option<bool> {
    match ev.code {
        KeyCode::Enter => Some(true),
        KeyCode::Esc => Some(false),
        _ => None,
    }
}

fn host_values(hosts: &[HostName]) -> Vec<toml::Value> {
    hosts
        .iter()
        .map(|host| toml::Value::from(host.as_str()))
        .collect()
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

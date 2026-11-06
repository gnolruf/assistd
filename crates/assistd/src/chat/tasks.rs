//! The `/tasks` panel: saved tasks listed, edited, and their prompts refined through the
//! daemon, which alone stores them.

use std::collections::HashMap;
use std::ops::ControlFlow;

use assistd_ipc::{Event, Request, TaskId, TaskRunState, TaskSpec};
use chrono::{DateTime, Utc};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

mod editor;
mod prompt_box;
mod trigger_form;

use editor::EditorAction;
pub(super) use editor::{EditorMode, EditorRow, TaskEditor, TriggerCheck, TriggerRow};
pub(super) use prompt_box::RefineNote;
pub(super) use trigger_form::{AT_FORMAT, TriggerForm, UpstreamChoice};

/// A task as the daemon last listed it.
pub(super) struct ListedTask {
    pub id: TaskId,
    pub spec: TaskSpec,
    pub next_fire: Option<DateTime<Utc>>,
    pub last_run: Option<TaskRunState>,
}

/// What a request the panel sent is for, so its events land in the right place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingOp {
    List,
    /// A save from the editor, whose result the editor adopts.
    Save,
    /// A change to a task's enabled flag from the list.
    Update,
    Delete(TaskId),
    CheckTrigger,
    Refine,
    Interrupt,
}

/// Requests waiting to be sent, and the op each request id still in flight is for.
#[derive(Default)]
struct Outbox {
    requests: Vec<Request>,
    pending: HashMap<String, PendingOp>,
}

impl Outbox {
    /// Queue the request `build` makes from a fresh id, returning the id.
    fn send(&mut self, op: PendingOp, build: impl FnOnce(String) -> Request) -> String {
        let id = Uuid::new_v4().to_string();
        self.pending.insert(id.clone(), op);
        self.requests.push(build(id.clone()));
        id
    }
}

/// One row of the task list: a task, or the row that makes a new one after them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ListRow {
    Task(usize),
    NewTask,
}

/// A line under the panel saying how the last step went.
pub(super) enum Message {
    Info(String),
    Error(String),
}

/// The tasks panel: the list, and the editor on top of it.
pub(super) struct TasksPanel {
    tasks: Vec<ListedTask>,
    /// Rows of the `TaskList` in flight.
    listing: Vec<ListedTask>,
    loaded: bool,
    /// Index into the list rows: the tasks, then the new-task row.
    selected: usize,
    editor: Option<TaskEditor>,
    outbox: Outbox,
    message: Option<Message>,
    /// The daemon allows Auto mode, so the editor offers it.
    auto_offered: bool,
}

impl TasksPanel {
    /// An empty panel with its task list requested; its editor offers Auto mode when
    /// `auto_offered`.
    pub(super) fn new(auto_offered: bool) -> Self {
        let mut panel = Self {
            tasks: Vec::new(),
            listing: Vec::new(),
            loaded: false,
            selected: 0,
            editor: None,
            outbox: Outbox::default(),
            message: None,
            auto_offered,
        };
        panel
            .outbox
            .send(PendingOp::List, |id| Request::TaskList { id });
        panel
    }

    pub(super) fn tasks(&self) -> &[ListedTask] {
        &self.tasks
    }

    pub(super) fn loaded(&self) -> bool {
        self.loaded
    }

    pub(super) fn selected_row(&self) -> ListRow {
        if self.selected < self.tasks.len() {
            ListRow::Task(self.selected)
        } else {
            ListRow::NewTask
        }
    }

    pub(super) fn editor(&self) -> Option<&TaskEditor> {
        self.editor.as_ref()
    }

    pub(super) fn editor_mut(&mut self) -> Option<&mut TaskEditor> {
        self.editor.as_mut()
    }

    pub(super) fn message(&self) -> Option<&Message> {
        self.message.as_ref()
    }

    /// Whether the editor's prompt box waits on a refine, whose status animates.
    pub(super) fn refining(&self) -> bool {
        self.editor.as_ref().is_some_and(TaskEditor::refining)
    }

    pub(super) fn advance_refine_dots(&mut self) {
        if let Some(editor) = self.editor.as_mut() {
            editor.advance_refine_dots();
        }
    }

    /// The tasks another task's trigger may wait on: all but the one being edited.
    pub(super) fn upstream_choices(&self) -> Vec<UpstreamChoice> {
        let editing = self.editor.as_ref().and_then(TaskEditor::task);
        self.tasks
            .iter()
            .filter(|task| Some(task.id) != editing)
            .map(|task| UpstreamChoice {
                id: task.id,
                name: task.spec.name.as_str().to_string(),
            })
            .collect()
    }

    /// The requests queued since the last call.
    pub(super) fn take_requests(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.outbox.requests)
    }

    /// Ctrl+C closes the panel from any view, stopping a refine; other keys go to the view
    /// on top. `Break` closes the panel.
    pub(super) fn on_key(&mut self, ev: KeyEvent) -> ControlFlow<()> {
        if ev.code == KeyCode::Char('c') && ev.modifiers.contains(KeyModifiers::CONTROL) {
            if let Some(editor) = self.editor.as_mut() {
                editor.cancel_refine(&mut self.outbox);
            }
            return ControlFlow::Break(());
        }
        self.message = None;
        if self.editor.is_some() {
            self.on_editor_key(ev);
        } else {
            return self.on_list_key(ev);
        }
        ControlFlow::Continue(())
    }

    /// ↑/↓ select, Enter opens the task or makes a new one, Space turns the task on or off,
    /// and Esc closes the panel.
    fn on_list_key(&mut self, ev: KeyEvent) -> ControlFlow<()> {
        match ev.code {
            KeyCode::Esc => return ControlFlow::Break(()),
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(self.tasks.len()),
            KeyCode::Enter => self.open_selected(),
            KeyCode::Char(' ') => self.toggle_selected(),
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn open_selected(&mut self) {
        let editor = match self.selected_row() {
            ListRow::NewTask => TaskEditor::new_task(self.auto_offered),
            ListRow::Task(index) => {
                let task = &self.tasks[index];
                let spec = task.spec.clone();
                TaskEditor::existing(task.id, spec, self.auto_offered, &mut self.outbox)
            }
        };
        self.editor = Some(editor);
    }

    fn toggle_selected(&mut self) {
        if let ListRow::Task(index) = self.selected_row() {
            let task = &self.tasks[index];
            let (task, enabled) = (task.id, !task.spec.enabled);
            self.outbox
                .send(PendingOp::Update, |id| Request::TaskSetEnabled {
                    id,
                    task,
                    enabled,
                });
        }
    }

    fn delete(&mut self, task: TaskId) {
        self.outbox
            .send(PendingOp::Delete(task), |id| Request::TaskDelete {
                id,
                task,
            });
    }

    fn on_editor_key(&mut self, ev: KeyEvent) {
        let upstreams = self.upstream_choices();
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        match editor.on_key(ev, &upstreams, &mut self.outbox) {
            EditorAction::Stay => {}
            EditorAction::Close => self.editor = None,
            EditorAction::Delete(task) => {
                self.editor = None;
                self.delete(task);
            }
        }
    }

    /// Apply an event answering one of the panel's requests; others are ignored.
    pub(super) fn on_event(&mut self, ev: Event) {
        let id = ev.id().to_string();
        let Some(op) = self.outbox.pending.get(&id).copied() else {
            return;
        };
        if ev.is_terminal() {
            self.outbox.pending.remove(&id);
        }
        match ev {
            Event::Task {
                task_id,
                spec,
                next_fire,
                last_run,
                ..
            } => self.on_task(
                op,
                ListedTask {
                    id: task_id,
                    spec,
                    next_fire,
                    last_run,
                },
            ),
            Event::TriggerFires { fires, .. } => self.record_check(&id, TriggerCheck::Fires(fires)),
            Event::TaskRefined {
                prompt,
                plan,
                recommended,
                ..
            } => {
                if let Some(editor) = self.editor.as_mut() {
                    editor.on_refined(&id, prompt, (&plan, &recommended));
                }
            }
            Event::Done { .. } => self.on_done(op, &id),
            Event::Error { message, .. } => self.on_error(op, &id, message),
            _ => {}
        }
    }

    /// The connection of a request failed without a terminal event.
    pub(super) fn on_connection_lost(&mut self, message: &str) {
        if let Some(editor) = self.editor.as_mut() {
            editor.on_connection_lost();
        }
        self.message = Some(Message::Error(message.to_string()));
    }

    fn on_task(&mut self, op: PendingOp, task: ListedTask) {
        if op == PendingOp::List {
            self.listing.push(task);
            return;
        }
        if let Some(editor) = self.editor.as_mut() {
            if op == PendingOp::Save && editor.task().is_none_or(|editing| editing == task.id) {
                editor.adopt_saved(task.id, task.spec.clone(), &mut self.outbox);
            } else if editor.task() == Some(task.id) {
                editor.refresh_stored(task.spec.clone());
            }
        }
        self.upsert(task);
    }

    /// Put `task` in the list in name order, replacing its old row.
    fn upsert(&mut self, task: ListedTask) {
        self.tasks.retain(|listed| listed.id != task.id);
        let index = self
            .tasks
            .partition_point(|listed| listed.spec.name.as_str() < task.spec.name.as_str());
        self.tasks.insert(index, task);
    }

    fn record_check(&mut self, request: &str, check: TriggerCheck) {
        if let Some(editor) = self.editor.as_mut() {
            editor.record_check(request, check);
        }
    }

    /// Finish `op`, sent as `request`; a save the editor was leaving after closes it.
    fn on_done(&mut self, op: PendingOp, request: &str) {
        match op {
            PendingOp::List => {
                self.tasks = std::mem::take(&mut self.listing);
                self.loaded = true;
                self.selected = self.selected.min(self.tasks.len());
            }
            PendingOp::Save => {
                if self
                    .editor
                    .as_ref()
                    .is_some_and(|editor| editor.closes_after(request))
                {
                    self.editor = None;
                }
                self.message = Some(Message::Info("saved".into()));
            }
            PendingOp::Delete(task) => {
                self.tasks.retain(|listed| listed.id != task);
                self.selected = self.selected.min(self.tasks.len());
                self.message = Some(Message::Info("deleted".into()));
            }
            PendingOp::Update
            | PendingOp::CheckTrigger
            | PendingOp::Refine
            | PendingOp::Interrupt => {}
        }
    }

    fn on_error(&mut self, op: PendingOp, request: &str, message: String) {
        match op {
            PendingOp::CheckTrigger => self.record_check(request, TriggerCheck::Refused(message)),
            PendingOp::Save => match self.editor.as_mut() {
                Some(editor) => editor.set_error(message),
                None => self.message = Some(Message::Error(message)),
            },
            PendingOp::Refine => {
                if let Some(editor) = self.editor.as_mut() {
                    editor.on_refine_failed(request, &message);
                }
            }
            PendingOp::List => {
                self.loaded = true;
                self.listing.clear();
                self.message = Some(Message::Error(message));
            }
            PendingOp::Update | PendingOp::Delete(_) | PendingOp::Interrupt => {
                self.message = Some(Message::Error(message));
            }
        }
    }
}

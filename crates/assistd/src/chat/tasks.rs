//! The `/tasks` panel: saved tasks listed, edited, run and their prompts refined, the task
//! queue, and each task's run history, all through the daemon, which alone stores them.

use std::collections::HashMap;

use assistd_ipc::{
    Event, PausedPrompt, Request, TaskId, TaskQueueOp, TaskQueueSnapshot, TaskRunId, TaskRunState,
    TaskSpec,
};
use assistd_tools::Approval;
use chrono::{DateTime, Utc};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

use super::selection::Selection;

mod editor;
mod prompt_box;
mod queue;
mod runs;
mod trigger_form;

use editor::EditorAction;
pub(super) use editor::{EditorMode, EditorRow, TaskEditor, TriggerCheck, TriggerRow};
pub(super) use prompt_box::RefineNote;
use queue::QueueAction;
pub(super) use queue::{QueueView, paused_prompts, waits_on};
use runs::RunsAction;
pub(super) use runs::RunsView;
pub(super) use trigger_form::{AT_FORMAT, TriggerForm, UpstreamChoice};

/// A task as the daemon last listed it.
pub(super) struct ListedTask {
    pub id: TaskId,
    pub spec: TaskSpec,
    pub next_fire: Option<DateTime<Utc>>,
    pub last_run: Option<TaskRunState>,
    pub trigger_failure: Option<String>,
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
    RunNow,
    QueueChange,
    Runs,
    /// An answer to a paused run's prompt.
    Confirm,
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

/// What a key on the panel asks the chat to do.
pub(super) enum PanelAction {
    Stay,
    Close,
    /// Close the panel and switch to this session-qualified branch.
    Resume(String),
    /// Show the prompt paused run `run` waits on in the chat's confirmation dialog.
    Answer(TaskRunId, PausedPrompt),
}

/// The panel's tabs, in the order Tab steps through them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PanelTab {
    Tasks,
    Queue,
    Runs,
}

impl PanelTab {
    pub(super) const ALL: [Self; 3] = [Self::Tasks, Self::Queue, Self::Runs];

    fn next(self) -> Self {
        match self {
            Self::Tasks => Self::Queue,
            Self::Queue => Self::Runs,
            Self::Runs => Self::Tasks,
        }
    }

    fn previous(self) -> Self {
        match self {
            Self::Tasks => Self::Runs,
            Self::Queue => Self::Tasks,
            Self::Runs => Self::Queue,
        }
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

/// The tasks panel: its tabs, and the editor on top of the task list.
pub(super) struct TasksPanel {
    tab: PanelTab,
    tasks: Vec<ListedTask>,
    /// Rows of the `TaskList` in flight.
    listing: Vec<ListedTask>,
    loaded: bool,
    /// Index into the list rows: the tasks, then the new-task row.
    selected: Selection,
    editor: Option<TaskEditor>,
    outbox: Outbox,
    message: Option<Message>,
    /// The daemon allows Auto mode, so the editor offers it.
    auto_offered: bool,
    queue: QueueView,
    runs: RunsView,
}

impl TasksPanel {
    /// An empty panel with its task list requested, showing `queue` until the next
    /// broadcast; its editor offers Auto mode when `auto_offered`.
    pub(super) fn new(auto_offered: bool, queue: Option<TaskQueueSnapshot>) -> Self {
        let mut panel = Self {
            tab: PanelTab::Tasks,
            tasks: Vec::new(),
            listing: Vec::new(),
            loaded: false,
            selected: Selection::default(),
            editor: None,
            outbox: Outbox::default(),
            message: None,
            auto_offered,
            queue: QueueView::default(),
            runs: RunsView::default(),
        };
        if let Some(snapshot) = queue {
            panel.queue.set_snapshot(snapshot);
        }
        panel
            .outbox
            .send(PendingOp::List, |id| Request::TaskList { id });
        panel
    }

    pub(super) fn tab(&self) -> PanelTab {
        self.tab
    }

    pub(super) fn queue(&self) -> &QueueView {
        &self.queue
    }

    pub(super) fn runs(&self) -> &RunsView {
        &self.runs
    }

    /// The task the Runs tab lists, as last listed.
    pub(super) fn runs_task(&self) -> Option<&ListedTask> {
        let task = self.runs.task()?;
        self.tasks.iter().find(|listed| listed.id == task)
    }

    /// Show the queue as just broadcast, and list the shown task's runs again since one
    /// may have ended.
    pub(super) fn set_queue(&mut self, snapshot: TaskQueueSnapshot) {
        self.queue.set_snapshot(snapshot);
        if self.tab == PanelTab::Runs {
            self.request_runs();
        }
    }

    /// Answer confirmation `confirm_id` of paused run `run` with `approval`.
    pub(super) fn answer_paused(&mut self, run: TaskRunId, confirm_id: String, approval: Approval) {
        self.outbox.send(PendingOp::Confirm, |id| {
            task_confirm(id, run, confirm_id, approval)
        });
    }

    pub(super) fn tasks(&self) -> &[ListedTask] {
        &self.tasks
    }

    pub(super) fn loaded(&self) -> bool {
        self.loaded
    }

    pub(super) fn selected_row(&self) -> ListRow {
        if self.selected.index() < self.tasks.len() {
            ListRow::Task(self.selected.index())
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

    /// Ctrl+C closes the panel from any view, stopping a refine; the editor takes every
    /// other key while open, else Tab and shift+Tab change tabs and the rest go to the tab.
    pub(super) fn on_key(&mut self, ev: KeyEvent) -> PanelAction {
        if ev.code == KeyCode::Char('c') && ev.modifiers.contains(KeyModifiers::CONTROL) {
            if let Some(editor) = self.editor.as_mut() {
                editor.cancel_refine(&mut self.outbox);
            }
            return PanelAction::Close;
        }
        self.message = None;
        if self.editor.is_some() {
            self.on_editor_key(ev);
            return PanelAction::Stay;
        }
        match ev.code {
            KeyCode::Tab => self.show_tab(self.tab.next()),
            KeyCode::BackTab => self.show_tab(self.tab.previous()),
            _ => return self.on_tab_key(ev),
        }
        PanelAction::Stay
    }

    fn on_tab_key(&mut self, ev: KeyEvent) -> PanelAction {
        match self.tab {
            PanelTab::Tasks => self.on_list_key(ev),
            PanelTab::Queue => match self.queue.on_key(ev) {
                QueueAction::Stay => PanelAction::Stay,
                QueueAction::Close => PanelAction::Close,
                QueueAction::Change(op) => {
                    self.change_queue(op);
                    PanelAction::Stay
                }
                QueueAction::Answer(run, prompt) => PanelAction::Answer(run, prompt),
            },
            PanelTab::Runs => match self.runs.on_key(ev) {
                RunsAction::Stay => PanelAction::Stay,
                RunsAction::Close => PanelAction::Close,
                RunsAction::ShowTask(by) => {
                    self.step_runs_task(by);
                    PanelAction::Stay
                }
                RunsAction::Resume(target) => PanelAction::Resume(target),
            },
        }
    }

    fn show_tab(&mut self, tab: PanelTab) {
        self.tab = tab;
        if tab == PanelTab::Runs {
            self.request_runs();
        }
    }

    /// ↑/↓ select, Enter opens the task or makes a new one, Space turns the task on or off,
    /// `r` queues a run of it, and Esc closes the panel.
    fn on_list_key(&mut self, ev: KeyEvent) -> PanelAction {
        match ev.code {
            KeyCode::Esc => return PanelAction::Close,
            KeyCode::Up => self.selected.previous(),
            KeyCode::Down => self.selected.next(self.tasks.len() + 1),
            KeyCode::Enter => self.open_selected(),
            KeyCode::Char(' ') => self.toggle_selected(),
            KeyCode::Char('r') => self.run_selected(),
            _ => {}
        }
        PanelAction::Stay
    }

    fn run_selected(&mut self) {
        if let ListRow::Task(index) = self.selected_row() {
            let task = self.tasks[index].id;
            self.outbox
                .send(PendingOp::RunNow, |id| Request::TaskRun { id, task });
        }
    }

    fn change_queue(&mut self, op: TaskQueueOp) {
        self.outbox
            .send(PendingOp::QueueChange, |id| Request::TaskQueue { id, op });
    }

    /// List the runs of the task selected in the task list, or show none when the new-task
    /// row is selected.
    fn request_runs(&mut self) {
        let ListRow::Task(index) = self.selected_row() else {
            self.runs.clear();
            return;
        };
        let task = self.tasks[index].id;
        let request = self
            .outbox
            .send(PendingOp::Runs, |id| Request::TaskRuns { id, task });
        self.runs.listing(task, request);
    }

    /// Select the task `by` places along the task list and list its runs.
    fn step_runs_task(&mut self, by: isize) {
        let Some(last) = self.tasks.len().checked_sub(1) else {
            return;
        };
        let at = self.selected.index().min(last);
        self.selected = Selection::at(at.saturating_add_signed(by).min(last));
        if self.selected.index() != at || self.runs.task().is_none() {
            self.request_runs();
        }
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
                trigger_failure,
                ..
            } => self.on_task(
                op,
                ListedTask {
                    id: task_id,
                    spec,
                    next_fire,
                    last_run,
                    trigger_failure,
                },
            ),
            Event::TriggerFires { fires, .. } => self.record_check(&id, TriggerCheck::Fires(fires)),
            Event::TaskRunRecord { record, .. } => self.runs.push(&id, record),
            Event::TaskRunQueued { run, .. } => {
                self.message = Some(Message::Info(format!("queued run {run}")));
            }
            Event::TaskRefined {
                prompt,
                plan,
                recommended,
                ..
            } => {
                if let Some(editor) = self.editor.as_mut() {
                    editor.on_refined(&id, prompt, &plan, &recommended);
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
        match op {
            PendingOp::List => {
                self.listing.push(task);
                return;
            }
            PendingOp::Confirm => {
                self.message = Some(Message::Info("host allowed for the task".into()));
            }
            _ => {}
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
                self.selected.clamp(self.tasks.len() + 1);
                if self.tab == PanelTab::Runs {
                    self.request_runs();
                }
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
                self.selected.clamp(self.tasks.len() + 1);
                self.message = Some(Message::Info("deleted".into()));
            }
            PendingOp::Runs => self.runs.finish(request),
            PendingOp::Update
            | PendingOp::CheckTrigger
            | PendingOp::Refine
            | PendingOp::Interrupt
            | PendingOp::RunNow
            | PendingOp::QueueChange
            | PendingOp::Confirm => {}
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
            PendingOp::Runs => {
                self.runs.fail(request);
                self.message = Some(Message::Error(message));
            }
            PendingOp::Update
            | PendingOp::Delete(_)
            | PendingOp::Interrupt
            | PendingOp::RunNow
            | PendingOp::QueueChange
            | PendingOp::Confirm => {
                self.message = Some(Message::Error(message));
            }
        }
    }
}

/// The `TaskConfirm` answering confirmation `confirm_id` of paused run `run` with `approval`.
pub(super) fn task_confirm(
    id: String,
    run: TaskRunId,
    confirm_id: String,
    approval: Approval,
) -> Request {
    Request::TaskConfirm {
        id,
        run,
        confirm_id,
        allow: approval != Approval::Deny,
        always: approval == Approval::Always,
    }
}

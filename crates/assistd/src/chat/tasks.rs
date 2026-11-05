//! The `/tasks` panel: saved tasks listed, edited, planned and approved through the
//! daemon, which alone stores them.

use std::collections::HashMap;
use std::ops::ControlFlow;

use assistd_ipc::{Event, Plan, PlanId, RelayEvent, Request, TaskId, TaskRunState, TaskSpec};
use chrono::{DateTime, Utc};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

use super::output::{OutputPane, tool_block_label, tool_outcome, tool_output};
use super::plan_review::PlanReview;
use super::settings::{EditAction, EditorShape, TextEditor};

mod editor;
mod trigger_form;

use editor::EditorAction;
pub(super) use editor::{EditorMode, EditorRow, TaskEditor, TriggerCheck, TriggerRow};
pub(super) use trigger_form::{AT_FORMAT, TriggerForm, UpstreamChoice};

/// Lines PgUp/PgDn scroll a plan under review.
const REVIEW_PAGE_LINES: usize = 10;

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
    Plan,
    Approve,
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

/// Planning a task, and the plan once made, drawn over the editor.
pub(super) struct PlanSession {
    task: TaskId,
    /// The `TaskPlan` request whose events this shows.
    request: String,
    pub state: PlanState,
}

pub(super) enum PlanState {
    /// The planning turn so far.
    Planning(Box<OutputPane>),
    Review {
        review: PlanReview,
        /// Feedback being typed, sent to revise the plan.
        feedback: Option<TextEditor>,
    },
    Failed(String),
}

/// A line under the panel saying how the last step went.
pub(super) enum Message {
    Info(String),
    Error(String),
}

/// The tasks panel: the list, the editor on top of it, and the plan view on top of the
/// editor.
pub(super) struct TasksPanel {
    tasks: Vec<ListedTask>,
    /// Rows of the `TaskList` in flight.
    listing: Vec<ListedTask>,
    loaded: bool,
    /// Index into `tasks`.
    selected: usize,
    /// The selected task is about to be deleted unless the next key says no.
    confirm_delete: bool,
    editor: Option<TaskEditor>,
    plan_view: Option<PlanSession>,
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
            confirm_delete: false,
            editor: None,
            plan_view: None,
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

    pub(super) fn selected(&self) -> usize {
        self.selected
    }

    pub(super) fn confirm_delete(&self) -> bool {
        self.confirm_delete
    }

    pub(super) fn editor(&self) -> Option<&TaskEditor> {
        self.editor.as_ref()
    }

    pub(super) fn editor_mut(&mut self) -> Option<&mut TaskEditor> {
        self.editor.as_mut()
    }

    pub(super) fn plan_view(&self) -> Option<&PlanSession> {
        self.plan_view.as_ref()
    }

    pub(super) fn plan_view_mut(&mut self) -> Option<&mut PlanSession> {
        self.plan_view.as_mut()
    }

    pub(super) fn message(&self) -> Option<&Message> {
        self.message.as_ref()
    }

    /// The planning turn on screen.
    pub(super) fn planning_pane(&self) -> Option<&OutputPane> {
        match &self.plan_view {
            Some(PlanSession {
                state: PlanState::Planning(pane),
                ..
            }) => Some(pane),
            _ => None,
        }
    }

    pub(super) fn planning_pane_mut(&mut self) -> Option<&mut OutputPane> {
        match &mut self.plan_view {
            Some(PlanSession {
                state: PlanState::Planning(pane),
                ..
            }) => Some(pane),
            _ => None,
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

    /// Ctrl+C closes the panel from any view, stopping a planning turn; other keys go to
    /// the view on top. `Break` closes the panel.
    pub(super) fn on_key(&mut self, ev: KeyEvent) -> ControlFlow<()> {
        if ev.code == KeyCode::Char('c') && ev.modifiers.contains(KeyModifiers::CONTROL) {
            if self.planning_pane().is_some() {
                self.interrupt_planning();
            }
            return ControlFlow::Break(());
        }
        self.message = None;
        if self.plan_view.is_some() {
            self.on_plan_key(ev);
        } else if self.editor.is_some() {
            self.on_editor_key(ev);
        } else {
            return self.on_list_key(ev);
        }
        ControlFlow::Continue(())
    }

    /// ↑/↓ select, Enter edits, `n` makes a task, Space turns it on or off, `d` deletes
    /// after a `y`, and Esc closes the panel.
    fn on_list_key(&mut self, ev: KeyEvent) -> ControlFlow<()> {
        if std::mem::take(&mut self.confirm_delete) {
            if matches!(ev.code, KeyCode::Char('y')) {
                self.delete_selected();
            }
            return ControlFlow::Continue(());
        }
        match ev.code {
            KeyCode::Esc => return ControlFlow::Break(()),
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.tasks.len().saturating_sub(1));
            }
            KeyCode::Char('n') => self.editor = Some(TaskEditor::new_task(self.auto_offered)),
            KeyCode::Enter => self.edit_selected(),
            KeyCode::Char(' ') => self.toggle_selected(),
            KeyCode::Char('d') | KeyCode::Delete if !self.tasks.is_empty() => {
                self.confirm_delete = true;
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn edit_selected(&mut self) {
        if let Some(task) = self.tasks.get(self.selected) {
            let spec = task.spec.clone();
            let editor = TaskEditor::existing(task.id, spec, self.auto_offered, &mut self.outbox);
            self.editor = Some(editor);
        }
    }

    fn toggle_selected(&mut self) {
        if let Some(task) = self.tasks.get(self.selected) {
            let (task, enabled) = (task.id, !task.spec.enabled);
            self.outbox
                .send(PendingOp::Update, |id| Request::TaskSetEnabled {
                    id,
                    task,
                    enabled,
                });
        }
    }

    fn delete_selected(&mut self) {
        if let Some(task) = self.tasks.get(self.selected).map(|task| task.id) {
            self.outbox
                .send(PendingOp::Delete(task), |id| Request::TaskDelete {
                    id,
                    task,
                });
        }
    }

    fn on_editor_key(&mut self, ev: KeyEvent) {
        let upstreams = self.upstream_choices();
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        match editor.on_key(ev, &upstreams, &mut self.outbox) {
            EditorAction::Stay => {}
            EditorAction::Close => self.editor = None,
            EditorAction::OpenPlan(task) => self.start_plan(task, None),
        }
    }

    /// Ask the daemon to plan task `task`, or with `feedback` to revise its last plan.
    fn start_plan(&mut self, task: TaskId, feedback: Option<String>) {
        let mut pane = OutputPane::new();
        let asked = feedback.clone().or_else(|| {
            self.editor
                .as_ref()
                .map(|editor| editor.prompt().to_string())
        });
        pane.push_user(&asked.unwrap_or_default());
        pane.begin_assistant();
        let request = self.outbox.send(PendingOp::Plan, |id| Request::TaskPlan {
            id,
            task,
            feedback,
        });
        self.plan_view = Some(PlanSession {
            task,
            request,
            state: PlanState::Planning(Box::new(pane)),
        });
    }

    /// Stop the planning turn on screen, whose failure then goes unreported.
    fn interrupt_planning(&mut self) {
        if let Some(session) = &self.plan_view {
            self.outbox.pending.remove(&session.request);
        }
        self.outbox
            .send(PendingOp::Interrupt, |id| Request::InterruptTurn { id });
    }

    /// Esc stops a planning turn or leaves the plan. In review, ↑/↓ and PgUp/PgDn scroll,
    /// `f` writes feedback to revise the plan and `a` approves it.
    fn on_plan_key(&mut self, ev: KeyEvent) {
        let Some(session) = self.plan_view.as_mut() else {
            return;
        };
        let task = session.task;
        match plan_key_outcome(&mut session.state, ev) {
            PlanKeyOutcome::Stay => {}
            PlanKeyOutcome::Leave => self.plan_view = None,
            PlanKeyOutcome::Interrupt => {
                self.interrupt_planning();
                self.plan_view = None;
            }
            PlanKeyOutcome::Replan(feedback) => self.start_plan(task, feedback),
            PlanKeyOutcome::Approve(plan) => {
                self.outbox
                    .send(PendingOp::Approve, |id| Request::TaskPlanApprove {
                        id,
                        task,
                        plan,
                    });
            }
        }
    }

    /// Record the columns the feedback box wraps at.
    pub(super) fn set_feedback_width(&mut self, width: u16) {
        if let Some(PlanSession {
            state:
                PlanState::Review {
                    feedback: Some(editor),
                    ..
                },
            ..
        }) = self.plan_view.as_mut()
        {
            editor.set_wrap_width(width);
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
            Event::TaskPlan { event, .. } => self.on_planning_event(&id, *event),
            Event::Plan { plan, .. } => self.on_plan(&id, plan),
            Event::Done { .. } => self.on_done(op),
            Event::Error { message, .. } => self.on_error(op, &id, message),
            _ => {}
        }
    }

    /// The connection of a request failed without a terminal event.
    pub(super) fn on_connection_lost(&mut self, message: &str) {
        if self.planning_pane().is_some()
            && let Some(session) = self.plan_view.as_mut()
        {
            session.state = PlanState::Failed(message.to_string());
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

    /// The plan session for `request`, when it is the one on screen.
    fn plan_session_mut(&mut self, request: &str) -> Option<&mut PlanSession> {
        match &mut self.plan_view {
            Some(session) if session.request == request => Some(session),
            _ => None,
        }
    }

    fn on_planning_event(&mut self, request: &str, ev: Event) {
        if let Some(PlanSession {
            state: PlanState::Planning(pane),
            ..
        }) = self.plan_session_mut(request)
        {
            apply_planning_event(pane, ev);
        }
    }

    fn on_plan(&mut self, request: &str, plan: Plan) {
        if let Some(session) = self.plan_session_mut(request) {
            session.state = PlanState::Review {
                review: PlanReview::new(plan),
                feedback: None,
            };
        }
    }

    fn on_done(&mut self, op: PendingOp) {
        match op {
            PendingOp::List => {
                self.tasks = std::mem::take(&mut self.listing);
                self.loaded = true;
                self.selected = self.selected.min(self.tasks.len().saturating_sub(1));
            }
            PendingOp::Save => self.message = Some(Message::Info("saved".into())),
            PendingOp::Delete(task) => {
                self.tasks.retain(|listed| listed.id != task);
                self.selected = self.selected.min(self.tasks.len().saturating_sub(1));
                self.message = Some(Message::Info("deleted".into()));
            }
            PendingOp::Approve => {
                self.plan_view = None;
                self.message = Some(Message::Info("plan approved".into()));
            }
            PendingOp::Update
            | PendingOp::CheckTrigger
            | PendingOp::Plan
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
            PendingOp::Plan => match self.plan_session_mut(request) {
                Some(session) => session.state = PlanState::Failed(message),
                None => self.message = Some(Message::Error(message)),
            },
            PendingOp::List => {
                self.loaded = true;
                self.listing.clear();
                self.message = Some(Message::Error(message));
            }
            PendingOp::Update
            | PendingOp::Delete(_)
            | PendingOp::Approve
            | PendingOp::Interrupt => {
                self.message = Some(Message::Error(message));
            }
        }
    }
}

/// What a key in the plan view asks of the panel.
enum PlanKeyOutcome {
    Stay,
    Leave,
    Interrupt,
    /// Plan afresh, or with feedback revise the plan.
    Replan(Option<String>),
    Approve(PlanId),
}

/// Apply `ev` to the plan view's `state`, saying what else it asks for.
fn plan_key_outcome(state: &mut PlanState, ev: KeyEvent) -> PlanKeyOutcome {
    match state {
        PlanState::Planning(_) if ev.code == KeyCode::Esc => PlanKeyOutcome::Interrupt,
        PlanState::Planning(_) => PlanKeyOutcome::Stay,
        PlanState::Failed(_) => match ev.code {
            KeyCode::Esc => PlanKeyOutcome::Leave,
            KeyCode::Enter | KeyCode::Char('r') => PlanKeyOutcome::Replan(None),
            _ => PlanKeyOutcome::Stay,
        },
        PlanState::Review { review, feedback } => match feedback {
            Some(editor) => match editor.on_key(ev) {
                EditAction::Apply => {
                    let text = editor.text().trim().to_string();
                    if text.is_empty() {
                        PlanKeyOutcome::Stay
                    } else {
                        PlanKeyOutcome::Replan(Some(text))
                    }
                }
                EditAction::Cancel => {
                    *feedback = None;
                    PlanKeyOutcome::Stay
                }
                EditAction::Pending => PlanKeyOutcome::Stay,
            },
            None => review_key_outcome(review, feedback, ev),
        },
    }
}

/// Keys of a plan under review with no feedback being typed.
fn review_key_outcome(
    review: &mut PlanReview,
    feedback: &mut Option<TextEditor>,
    ev: KeyEvent,
) -> PlanKeyOutcome {
    match ev.code {
        KeyCode::Esc => return PlanKeyOutcome::Leave,
        KeyCode::Up => review.scroll_up(1),
        KeyCode::Down => review.scroll_down(1),
        KeyCode::PageUp => review.scroll_up(REVIEW_PAGE_LINES),
        KeyCode::PageDown => review.scroll_down(REVIEW_PAGE_LINES),
        KeyCode::Char('f') => {
            *feedback = Some(TextEditor::new(
                String::new(),
                EditorShape::Box { wrap_width: 0 },
            ));
        }
        KeyCode::Char('a') | KeyCode::Enter => {
            return PlanKeyOutcome::Approve(review.plan().id.clone());
        }
        _ => {}
    }
    PlanKeyOutcome::Stay
}

/// Draw one event of a planning turn in `pane`, as the chat draws its own turns.
fn apply_planning_event(pane: &mut OutputPane, ev: Event) {
    match ev {
        Event::Delta { text, .. } => {
            pane.finish_thinking();
            pane.append_assistant(&text);
        }
        Event::ReasoningDelta { text, .. } => pane.append_thinking(&text),
        Event::ToolCall { name, args, .. } => {
            pane.finish_thinking();
            pane.begin_tool_block(tool_block_label(name, Some(&args)));
        }
        Event::ToolResult { result, .. } => {
            let (exit_code, duration_ms) = tool_outcome(&result);
            pane.finish_tool_block(tool_output(&result), exit_code, duration_ms);
        }
        Event::Relay {
            event: RelayEvent::Planned { steps },
            ..
        } => pane.begin_relay(steps),
        Event::Relay { event, .. } => pane.apply_relay(event),
        _ => {}
    }
}

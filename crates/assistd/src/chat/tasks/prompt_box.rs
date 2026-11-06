//! The task editor's prompt box, whose text the daemon can rewrite in place for unattended
//! runs while the box waits.

use assistd_ipc::{
    HostName, ModeRecommendation, PermissionMode, Plan, PlannedAction, Request, TaskPrompt,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::{Outbox, PendingOp};
use crate::chat::settings::{EditAction, EditorShape, TextEditor};

/// Animation ticks each step of the refining status's dots lasts.
const TICKS_PER_DOT: usize = 3;

/// The prompt being written, and the refine of it in flight or the note on the last one.
pub(in crate::chat) struct PromptBox {
    text: TextEditor,
    /// The refine rewriting the text; the box takes no key but Esc meanwhile.
    refining: Option<Refining>,
    note: Option<RefineNote>,
}

/// A refine in flight: its `TaskRefine` request, and the animation ticks since it began.
struct Refining {
    request: String,
    ticks: usize,
}

/// The mode the last rewrite's runs are best given, and briefly why.
pub(in crate::chat) struct RefineNote {
    pub mode: PermissionMode,
    pub reason: String,
    /// Runs would still stop to ask in that mode, or an action looks like it will fail.
    pub warns: bool,
}

/// The permissions the task's runs would have, which a rewrite's plan is checked against.
#[derive(Clone, Copy)]
pub(super) struct RunPermissions<'a> {
    pub mode: PermissionMode,
    pub allowed_hosts: &'a [HostName],
}

impl PromptBox {
    pub(super) fn new(text: String) -> Self {
        Self {
            text: TextEditor::new(text, EditorShape::Box { wrap_width: 0 }),
            refining: None,
            note: None,
        }
    }

    pub(in crate::chat) fn text(&self) -> &TextEditor {
        &self.text
    }

    pub(in crate::chat) fn refining(&self) -> bool {
        self.refining.is_some()
    }

    /// How many dots the refining status shows now, 0 to 3; `None` when no refine runs.
    pub(in crate::chat) fn refine_dots(&self) -> Option<usize> {
        self.refining
            .as_ref()
            .map(|refining| refining.ticks / TICKS_PER_DOT % 4)
    }

    pub(super) fn advance_refine_dots(&mut self) {
        if let Some(refining) = &mut self.refining {
            refining.ticks = refining.ticks.wrapping_add(1);
        }
    }

    pub(in crate::chat) fn note(&self) -> Option<&RefineNote> {
        self.note.as_ref()
    }

    pub(super) fn set_wrap_width(&mut self, width: u16) {
        self.text.set_wrap_width(width);
    }

    /// While a refine runs, Esc cancels it and other keys do nothing. Otherwise Ctrl+R sends
    /// the text to be refined under `permissions`, and other keys edit it. Errors with why
    /// the text cannot be refined.
    pub(super) fn on_key(
        &mut self,
        ev: KeyEvent,
        permissions: RunPermissions<'_>,
        outbox: &mut Outbox,
    ) -> Result<EditAction, String> {
        if self.refining.is_some() {
            if ev.code == KeyCode::Esc {
                self.cancel_refine(outbox);
            }
            return Ok(EditAction::Pending);
        }
        if ev.code == KeyCode::Char('r') && ev.modifiers.contains(KeyModifiers::CONTROL) {
            self.start_refine(permissions, outbox)?;
            return Ok(EditAction::Pending);
        }
        Ok(self.text.on_key(ev))
    }

    fn start_refine(
        &mut self,
        permissions: RunPermissions<'_>,
        outbox: &mut Outbox,
    ) -> Result<(), String> {
        let prompt =
            TaskPrompt::try_from(self.text.text().to_string()).map_err(|e| e.to_string())?;
        let request = outbox.send(PendingOp::Refine, |id| Request::TaskRefine {
            id,
            prompt,
            mode: permissions.mode,
            allowed_hosts: permissions.allowed_hosts.to_vec(),
            feedback: None,
        });
        self.refining = Some(Refining { request, ticks: 0 });
        self.note = None;
        Ok(())
    }

    /// Stop the refine in flight, whose failure then goes unreported.
    pub(super) fn cancel_refine(&mut self, outbox: &mut Outbox) {
        if let Some(Refining { request, .. }) = self.refining.take() {
            outbox.pending.remove(&request);
            outbox.send(PendingOp::Interrupt, |id| Request::InterruptTurn { id });
        }
    }

    /// Put the rewrite `prompt` that answered `request` in place of the text, noting the
    /// `recommended` mode and any problem its `plan` found, when `request` is the refine in
    /// flight.
    pub(super) fn on_refined(
        &mut self,
        request: &str,
        prompt: TaskPrompt,
        plan: &Plan,
        recommended: &ModeRecommendation,
    ) {
        if self.on_refine_ended(request) {
            self.text = TextEditor::new(prompt.into(), EditorShape::Box { wrap_width: 0 });
            self.note = Some(RefineNote::of(plan, recommended));
        }
    }

    /// End the refine in flight when `request` was it, saying whether it was.
    pub(super) fn on_refine_ended(&mut self, request: &str) -> bool {
        let ours = self.refines(request);
        if ours {
            self.refining = None;
        }
        ours
    }

    fn refines(&self, request: &str) -> bool {
        self.refining
            .as_ref()
            .is_some_and(|refining| refining.request == request)
    }

    /// End the refine in flight, whose answer is lost.
    pub(super) fn abandon_refine(&mut self) {
        self.refining = None;
    }
}

impl RefineNote {
    /// `recommended`'s mode, with the actions that decide it and the first problem `plan`
    /// found as its reason.
    fn of(plan: &Plan, recommended: &ModeRecommendation) -> Self {
        let mut reason = match (recommended.covers.is_empty(), recommended.mode) {
            (true, _) if recommended.still_asks.is_empty() => "nothing needs approval".to_string(),
            (true, _) => format!(
                "{} would pause runs in any mode",
                brief_list(&recommended.still_asks)
            ),
            (false, PermissionMode::Auto) => format!(
                "the safety judge can approve {}",
                brief_list(&recommended.covers)
            ),
            (false, _) => format!("lets it {}", brief_list(&recommended.covers)),
        };
        if !recommended.covers.is_empty() && !recommended.still_asks.is_empty() {
            reason.push_str(&format!(
                "; {} would still pause runs",
                brief_list(&recommended.still_asks)
            ));
        }
        let problem = plan
            .steps
            .iter()
            .flat_map(|step| &step.actions)
            .find_map(|anticipated| anticipated.problem.as_deref());
        if let Some(problem) = problem {
            reason.push_str(&format!(" · ⚠ {problem}"));
        }
        Self {
            mode: recommended.mode,
            reason,
            warns: !recommended.still_asks.is_empty() || problem.is_some(),
        }
    }
}

/// The first two of `actions` in a few words each, then how many more there are.
fn brief_list(actions: &[PlannedAction]) -> String {
    let mut list = actions
        .iter()
        .take(2)
        .map(action_label)
        .collect::<Vec<_>>()
        .join(", ");
    if actions.len() > 2 {
        list.push_str(&format!(" +{} more", actions.len() - 2));
    }
    list
}

/// What `action` does, in a few words.
fn action_label(action: &PlannedAction) -> String {
    match action {
        PlannedAction::Program {
            program,
            subcommand,
        } => match subcommand {
            Some(subcommand) => format!("run {program} {subcommand}"),
            None => format!("run {program}"),
        },
        PlannedAction::WebHost { host } => format!("reach {host}"),
        PlannedAction::McpTool { name } => format!("call {name}"),
        PlannedAction::App { desktop_entry } => format!("launch {desktop_entry}"),
        PlannedAction::Write { under } => format!("write under {}", under.display()),
    }
}

//! Plan mode: Shift+Tab or the model's ask puts the chat in it, and the plan the model hands
//! over is accepted under a permission mode or sent back with what to change.

use std::time::Instant;

use assistd_ipc::{PermissionMode, PlanVerdict, Request, RequestBody};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

use super::App;
use crate::chat::selection::Selection;

/// What this chat's messages do; Shift+Tab cycles it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat) enum ChatMode {
    /// Run agent turns whose prompts are answered under the mode.
    Act(PermissionMode),
    /// Run turns that change nothing until the user accepts their plan.
    Plan,
}

/// An answer the plan review offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat) enum PlanChoice {
    Accept(PermissionMode),
    /// Send the typed feedback back, or with none, wait for the user's next message.
    KeepPlanning,
}

/// The plan the turn waits on the user to accept or send back.
pub(in crate::chat) struct PlanReview {
    review_id: String,
    pub choices: Vec<PlanChoice>,
    /// Index into `choices`.
    pub selected: Selection,
    /// What to change, typed on the keep-planning row.
    pub feedback: String,
}

impl PlanReview {
    fn new(review_id: String, auto_offered: bool) -> Self {
        let mut choices = Vec::with_capacity(4);
        if auto_offered {
            choices.push(PlanChoice::Accept(PermissionMode::Auto));
        }
        choices.extend([
            PlanChoice::Accept(PermissionMode::AcceptEdits),
            PlanChoice::Accept(PermissionMode::Manual),
            PlanChoice::KeepPlanning,
        ]);
        Self {
            review_id,
            choices,
            selected: Selection::default(),
            feedback: String::new(),
        }
    }

    fn select_feedback_row(&mut self) {
        self.selected = Selection::at(self.choices.len() - 1);
    }

    /// The answer `choice` gives, with the feedback typed when it keeps planning.
    fn verdict(&self, choice: PlanChoice) -> PlanVerdict {
        match choice {
            PlanChoice::Accept(mode) => PlanVerdict::Accept { mode },
            PlanChoice::KeepPlanning => PlanVerdict::KeepPlanning {
                feedback: Some(self.feedback.trim())
                    .filter(|feedback| !feedback.is_empty())
                    .map(str::to_string),
            },
        }
    }
}

impl App {
    /// Cycle manual, accept-edits, auto when offered, then plan mode, applying it to the open
    /// turn too.
    pub(super) fn toggle_mode(&mut self) {
        let next = match self.mode {
            ChatMode::Act(PermissionMode::Manual) => ChatMode::Act(PermissionMode::AcceptEdits),
            ChatMode::Act(PermissionMode::AcceptEdits) if self.auto_mode_offered => {
                ChatMode::Act(PermissionMode::Auto)
            }
            ChatMode::Act(_) => ChatMode::Plan,
            ChatMode::Plan => ChatMode::Act(PermissionMode::Manual),
        };
        self.set_mode(next);
    }

    /// Record whether the daemon honors Auto, leaving Auto for Manual when it no longer does.
    pub(super) fn offer_auto_mode(&mut self, offered: bool) {
        self.auto_mode_offered = offered;
        if !offered && self.mode == ChatMode::Act(PermissionMode::Auto) {
            self.set_mode(ChatMode::Act(PermissionMode::Manual));
        }
    }

    /// Switch to `mode`, applying it to the open turn too.
    fn set_mode(&mut self, mode: ChatMode) {
        self.mode = mode;
        let (mode, plan) = match mode {
            ChatMode::Act(mode) => (mode, false),
            ChatMode::Plan => (PermissionMode::Manual, true),
        };
        self.send_on_reply(Request::new(
            Uuid::new_v4().to_string(),
            RequestBody::SetPermissionMode { mode, plan },
        ));
    }

    /// Show the plan the turn hands over in the transcript and ask whether to carry it out.
    pub(super) fn open_plan_review(&mut self, review_id: String, plan: &str) {
        self.mode = ChatMode::Plan;
        self.output.append_assistant(plan);
        self.output.finish_assistant();
        self.plan_review = Some(PlanReview::new(review_id, self.auto_mode_offered));
        self.type_ahead.arm(Instant::now());
    }

    /// Arrows move the selection, Enter or a row's number answers, typing writes what to
    /// change on the keep-planning row, and Esc keeps planning without feedback.
    pub(super) fn on_plan_review_key(&mut self, ev: KeyEvent) {
        let Some(review) = self.plan_review.as_mut() else {
            return;
        };
        let on_feedback_row = review.selected.index() + 1 == review.choices.len();
        let answer = match ev.code {
            KeyCode::Esc => Some(PlanChoice::KeepPlanning),
            KeyCode::Char('c') if ev.modifiers.contains(KeyModifiers::CONTROL) => {
                Some(PlanChoice::KeepPlanning)
            }
            KeyCode::Enter => review.choices.get(review.selected.index()).copied(),
            KeyCode::Up => {
                review.selected.previous();
                None
            }
            KeyCode::Down => {
                review.selected.next(review.choices.len());
                None
            }
            KeyCode::Backspace if on_feedback_row => {
                review.feedback.pop();
                None
            }
            KeyCode::Char(digit) if review.feedback.is_empty() && digit.is_ascii_digit() => {
                numbered_choice(&review.choices, digit)
            }
            KeyCode::Char(typed) => {
                review.select_feedback_row();
                review.feedback.push(typed);
                None
            }
            KeyCode::PageUp => {
                self.output.scroll_page_up(self.last_output_height);
                None
            }
            KeyCode::PageDown => {
                self.output.scroll_page_down(self.last_output_height);
                None
            }
            _ => None,
        };
        if let Some(choice) = answer {
            self.resolve_plan_review(choice);
        }
    }

    /// Send the answer `choice` gives, following an accepted plan into its mode.
    fn resolve_plan_review(&mut self, choice: PlanChoice) {
        let Some(review) = self.plan_review.take() else {
            return;
        };
        let verdict = review.verdict(choice);
        match &verdict {
            PlanVerdict::Accept { mode } => self.mode = ChatMode::Act(*mode),
            PlanVerdict::KeepPlanning {
                feedback: Some(feedback),
            } => self.output.push_info(&format!("Keep planning: {feedback}")),
            PlanVerdict::KeepPlanning { feedback: None } => {}
        }
        self.send_on_reply(Request::new(
            Uuid::new_v4().to_string(),
            RequestBody::PlanVerdict {
                review_id: review.review_id,
                verdict,
            },
        ));
    }

    /// Send `req` on the active reply's query connection, if this chat owns it.
    fn send_on_reply(&mut self, req: Request) {
        let Some(writer) = self
            .active_reply
            .as_ref()
            .and_then(|reply| reply.writer.clone())
        else {
            return;
        };
        self.tasks.spawn(async move {
            if let Err(e) = writer.send(req).await {
                tracing::warn!("mid-turn request send failed: {e}");
            }
        });
    }
}

/// The choice shown as `digit`, counting from 1.
fn numbered_choice(choices: &[PlanChoice], digit: char) -> Option<PlanChoice> {
    let number = usize::try_from(digit.to_digit(10)?).ok()?;
    choices.get(number.checked_sub(1)?).copied()
}

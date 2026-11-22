//! Plan mode: Shift+Tab or the model's ask puts the chat in it, and the plan the model hands
//! over is accepted under a permission mode or sent back with what to change.

use std::time::Instant;

use assistd_ipc::{PermissionMode, PlanVerdict, Request, RequestBody};
use crossterm::event::{KeyCode, KeyEvent};
use uuid::Uuid;

use super::App;
use crate::choice_list::{ChoiceKey, ChoiceList};

/// What this chat's messages do; Shift+Tab cycles it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatMode {
    /// Run agent turns whose prompts are answered under the mode.
    Act(PermissionMode),
    /// Run turns that change nothing until the user accepts their plan.
    Plan,
}

/// An answer the plan review offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanChoice {
    Accept(PermissionMode),
    /// Send the typed feedback back, or with none, wait for the user's next message.
    KeepPlanning,
}

/// The plan the turn waits on the user to accept or send back.
pub(crate) struct PlanReview {
    review_id: String,
    /// The answers on offer; what to change is typed on the keep-planning row.
    pub choices: ChoiceList<PlanChoice>,
}

impl PlanReview {
    fn new(review_id: String, auto_offered: bool) -> Self {
        let auto = auto_offered.then_some(PlanChoice::Accept(PermissionMode::Auto));
        let choices = auto
            .into_iter()
            .chain([
                PlanChoice::Accept(PermissionMode::AcceptEdits),
                PlanChoice::Accept(PermissionMode::Manual),
                PlanChoice::KeepPlanning,
            ])
            .collect();
        Self {
            review_id,
            choices: ChoiceList::new(choices, PlanChoice::KeepPlanning),
        }
    }

    /// The answer `choice` gives, with the feedback typed when it keeps planning.
    fn verdict(&self, choice: PlanChoice) -> PlanVerdict {
        match choice {
            PlanChoice::Accept(mode) => PlanVerdict::Accept { mode },
            PlanChoice::KeepPlanning => PlanVerdict::KeepPlanning {
                feedback: self.choices.message(),
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

    /// The choice list takes the keys that pick an answer or type what to change, Esc
    /// keeping the plan open; PgUp/PgDn scroll the plan.
    pub(super) fn on_plan_review_key(&mut self, ev: KeyEvent) {
        let Some(review) = self.plan_review.as_mut() else {
            return;
        };
        match (review.choices.on_key(ev), ev.code) {
            (ChoiceKey::Answered(choice), _) => self.resolve_plan_review(choice),
            (ChoiceKey::Passed, KeyCode::PageUp) => {
                self.output.scroll_page_up(self.last_output_height);
            }
            (ChoiceKey::Passed, KeyCode::PageDown) => {
                self.output.scroll_page_down(self.last_output_height);
            }
            (ChoiceKey::Taken | ChoiceKey::Passed, _) => {}
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

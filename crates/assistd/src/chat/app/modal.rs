//! The inline command-confirmation prompt.

use std::time::{Duration, Instant};

use assistd_ipc::{PausedPrompt, Request, TaskRunId};
use assistd_tools::{Approval, ConfirmationRequest};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

use super::{AnswerTo, App, ConfirmChoice, ConfirmationPrompt};

/// Script rows one PgUp/PgDn moves an expanded prompt.
const PROMPT_SCROLL_STEP: usize = 10;

/// How long typing must pause after a prompt opens before a key answers it.
const TYPE_AHEAD_GRACE: Duration = Duration::from_millis(500);

/// Holds keys back from a prompt that just opened, so what the user was typing when it
/// appeared lands in the input line instead of answering it.
#[derive(Debug, Default)]
pub(in crate::chat) struct TypeAheadGuard {
    /// Keys go to the input line until then; each one held pushes it back.
    deaf_until: Option<Instant>,
}

impl TypeAheadGuard {
    /// A prompt opened at `now`.
    pub(super) fn arm(&mut self, now: Instant) {
        self.deaf_until = Some(now + TYPE_AHEAD_GRACE);
    }

    /// Whether a key at `now` is held from the prompt, which keeps holding until typing
    /// pauses for [`TYPE_AHEAD_GRACE`].
    fn holds(&mut self, now: Instant) -> bool {
        let held = self.deaf_until.is_some_and(|until| now < until);
        self.deaf_until = held.then(|| now + TYPE_AHEAD_GRACE);
        held
    }
}

impl App {
    /// Show a reply's command-confirmation prompt in place of an open task run's, which
    /// stays paused; one arriving while another reply prompt is open is denied immediately.
    pub(super) fn open_confirmation(&mut self, confirm_id: String, request: ConfirmationRequest) {
        if self
            .confirmation
            .as_ref()
            .is_some_and(|prompt| prompt.answer_to == AnswerTo::Reply)
        {
            self.send_confirm_response(&confirm_id, Approval::Deny);
            return;
        }
        self.confirmation = Some(ConfirmationPrompt::new(confirm_id, request));
        self.type_ahead.arm(Instant::now());
    }

    /// Show the prompt paused run `run` waits on, unless another prompt is open.
    pub(super) fn open_task_run_confirmation(&mut self, run: TaskRunId, prompt: PausedPrompt) {
        if self.confirmation.is_some() {
            self.set_notice("answer the open prompt first");
            return;
        }
        self.confirmation = Some(ConfirmationPrompt::for_task_run(run, prompt));
        self.type_ahead.arm(Instant::now());
    }

    /// Send `ev` to the input line instead of an open prompt when the prompt opened while the
    /// user was typing; Enter is dropped rather than submitted. Esc and Ctrl+C, which only
    /// decline, always reach the prompt. Returns whether `ev` was held.
    pub(super) fn held_from_prompt(&mut self, ev: KeyEvent) -> bool {
        let declines = ev.code == KeyCode::Esc
            || (ev.code == KeyCode::Char('c') && ev.modifiers.contains(KeyModifiers::CONTROL));
        if declines || !self.type_ahead.holds(Instant::now()) {
            return false;
        }
        if ev.code != KeyCode::Enter {
            self.on_input_key(ev);
        }
        true
    }

    pub(super) fn resolve_confirmation(&mut self, approval: Approval) {
        let Some(prompt) = self.confirmation.take() else {
            return;
        };
        match prompt.answer_to {
            AnswerTo::Reply => self.send_confirm_response(&prompt.confirm_id, approval),
            AnswerTo::TaskRun(run) => self.answer_task_run(run, prompt.confirm_id, approval),
        }
    }

    fn send_confirm_response(&mut self, confirm_id: &str, approval: Approval) {
        let Some(writer) = self
            .active_reply
            .as_ref()
            .and_then(|reply| reply.writer.clone())
        else {
            tracing::warn!(
                confirm_id,
                "no active query writer to forward ConfirmResponse"
            );
            return;
        };
        let req = Request::ConfirmResponse {
            id: Uuid::new_v4().to_string(),
            confirm_id: confirm_id.to_string(),
            allow: approval != Approval::Deny,
            always: approval == Approval::Always,
        };
        self.tasks.spawn(async move {
            if let Err(e) = writer.send(req).await {
                tracing::warn!("ConfirmResponse send failed: {e}");
            }
        });
    }

    /// Arrows move the selection, Enter or a choice's number answers, Esc
    /// or Ctrl+C denies, and Tab and PgUp/PgDn expand and scroll the script.
    pub(super) fn on_confirmation_key(&mut self, ev: KeyEvent) {
        let Some(prompt) = self.confirmation.as_mut() else {
            return;
        };
        let choices = prompt.choices();
        let answer = match ev.code {
            KeyCode::Esc => Some(ConfirmChoice::Deny),
            KeyCode::Char('c') if ev.modifiers.contains(KeyModifiers::CONTROL) => {
                Some(ConfirmChoice::Deny)
            }
            KeyCode::Enter => choices.get(prompt.selected).copied(),
            KeyCode::Char(digit) => numbered_choice(choices, digit),
            KeyCode::Up => {
                prompt.selected = prompt.selected.saturating_sub(1);
                None
            }
            KeyCode::Down => {
                prompt.selected = (prompt.selected + 1).min(choices.len() - 1);
                None
            }
            KeyCode::Tab => {
                prompt.expanded = !prompt.expanded;
                prompt.scroll = 0;
                None
            }
            KeyCode::PageUp => {
                prompt.scroll = prompt.scroll.saturating_sub(PROMPT_SCROLL_STEP);
                None
            }
            KeyCode::PageDown => {
                prompt.scroll = prompt.scroll.saturating_add(PROMPT_SCROLL_STEP);
                None
            }
            _ => None,
        };
        if let Some(choice) = answer {
            self.resolve_confirmation(choice.approval());
        }
    }
}

/// The choice shown as `digit`, counting from 1.
fn numbered_choice(choices: &[ConfirmChoice], digit: char) -> Option<ConfirmChoice> {
    let number = usize::try_from(digit.to_digit(10)?).ok()?;
    choices.get(number.checked_sub(1)?).copied()
}

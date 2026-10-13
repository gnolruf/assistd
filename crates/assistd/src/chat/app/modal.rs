//! The inline command-confirmation prompt.

use assistd_ipc::Request;
use assistd_tools::{Approval, ConfirmationRequest};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

use super::{App, ConfirmChoice, ConfirmationPrompt};

/// Script rows one PgUp/PgDn moves an expanded prompt.
const PROMPT_SCROLL_STEP: usize = 10;

impl App {
    /// Show a command-confirmation prompt. A second prompt arriving while
    /// one is open is denied immediately.
    pub(super) fn open_confirmation(&mut self, confirm_id: String, request: ConfirmationRequest) {
        if self.confirmation.is_some() {
            self.send_confirm_response(&confirm_id, Approval::Deny);
            return;
        }
        self.confirmation = Some(ConfirmationPrompt::new(confirm_id, request));
    }

    pub(super) fn resolve_confirmation(&mut self, approval: Approval) {
        if let Some(prompt) = self.confirmation.take() {
            self.send_confirm_response(&prompt.confirm_id, approval);
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

//! The inline command-confirmation prompt and the branch-picker overlay.

use assistd_ipc::Request;
use assistd_tools::{Approval, ConfirmationRequest};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use uuid::Uuid;

use super::{App, BranchPickerModal, ConfirmChoice, ConfirmationPrompt};

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

    /// Open the picker on the buffered `/resume` rows, preselecting the
    /// active branch.
    pub(super) fn open_branch_picker(&mut self) {
        let entries = std::mem::take(&mut self.branches_buffer);
        if entries.is_empty() {
            self.set_notice("no branches to resume");
            return;
        }
        let selected = entries
            .iter()
            .position(|e| e.is_active_session && e.is_current_in_session)
            .unwrap_or(0);
        self.picker_modal = Some(BranchPickerModal { entries, selected });
    }

    pub(super) fn on_picker_key(&mut self, ev: KeyEvent) {
        let Some(picker) = self.picker_modal.as_mut() else {
            return;
        };
        let len = picker.entries.len();
        match ev.code {
            KeyCode::Up if picker.selected > 0 => picker.selected -= 1,
            KeyCode::Down if picker.selected + 1 < len => picker.selected += 1,
            KeyCode::Home => picker.selected = 0,
            KeyCode::End => picker.selected = len.saturating_sub(1),
            KeyCode::Enter => self.switch_to_picked_branch(),
            KeyCode::Esc => self.picker_modal = None,
            _ => {}
        }
    }

    fn switch_to_picked_branch(&mut self) {
        let target = self
            .picker_modal
            .take()
            .and_then(|picker| picker.current_target());
        if let Some(target) = target {
            self.switch_branch(target);
        }
    }
}

/// The choice shown as `digit`, counting from 1.
fn numbered_choice(choices: &[ConfirmChoice], digit: char) -> Option<ConfirmChoice> {
    let number = usize::try_from(digit.to_digit(10)?).ok()?;
    choices.get(number.checked_sub(1)?).copied()
}

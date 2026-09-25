//! The command-confirmation and branch-picker overlays.

use std::time::Instant;

use assistd_ipc::Request;
use assistd_tools::{Approval, ConfirmationRequest};
use crossterm::event::{KeyCode, KeyEvent};
use uuid::Uuid;

use super::{App, BranchPickerModal, ConfirmationModal};

impl App {
    /// Show a command-confirmation prompt. A second prompt arriving while
    /// one is open is denied immediately.
    pub(super) fn open_confirmation_modal(
        &mut self,
        confirm_id: String,
        request: ConfirmationRequest,
    ) {
        if self.modal.is_some() {
            self.send_confirm_response(&confirm_id, Approval::Deny);
            return;
        }
        self.modal = Some(ConfirmationModal {
            request,
            confirm_id,
            opened_at: Instant::now(),
        });
    }

    pub(super) fn resolve_modal(&mut self, approval: Approval) {
        if let Some(modal) = self.modal.take() {
            self.send_confirm_response(&modal.confirm_id, approval);
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

    pub(super) fn on_confirmation_key(&mut self, ev: KeyEvent) {
        let armed = self.modal.as_ref().is_some_and(ConfirmationModal::armed);
        let offers_always = self
            .modal
            .as_ref()
            .is_some_and(|m| !m.request.always_allow.is_empty());
        match ev.code {
            KeyCode::Char('y') | KeyCode::Char('Y') if armed => {
                self.resolve_modal(Approval::Once);
            }
            KeyCode::Char('a') | KeyCode::Char('A') if armed && offers_always => {
                self.resolve_modal(Approval::Always);
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.resolve_modal(Approval::Deny);
            }
            _ => {}
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

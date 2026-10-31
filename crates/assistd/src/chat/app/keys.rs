//! Keyboard and mouse input: scrolling, toggles and the slash-command popup.

use assistd_ipc::{PermissionMode, Request};
use assistd_tools::Approval;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use uuid::Uuid;

use super::{App, WireStream};
use crate::chat::input::InputAction;

const MOUSE_WHEEL_STEP: u16 = 3;

/// `(command, usage_hint)` pairs for the autocomplete popup.
const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/agents", ""),
    ("/attach", "<path>"),
    ("/exit", ""),
    ("/fork", "<name>"),
    ("/new", ""),
    ("/resume", ""),
    ("/settings", ""),
    ("/undo", ""),
    ("/wakeword", ""),
];

impl App {
    pub(crate) fn on_mouse(&mut self, ev: MouseEvent) {
        match ev.kind {
            MouseEventKind::ScrollUp => {
                self.scroll_shown_pane(|pane| pane.scroll_lines_up(MOUSE_WHEEL_STEP));
            }
            MouseEventKind::ScrollDown => {
                self.scroll_shown_pane(|pane| pane.scroll_lines_down(MOUSE_WHEEL_STEP));
            }
            _ => {}
        }
    }

    pub(crate) fn on_key(&mut self, ev: KeyEvent) {
        if self.shown_resume_picker().is_some() {
            self.on_resume_picker_key(ev);
        } else if self.shown_wake_word_panel().is_some() {
            self.on_wake_word_key(ev);
        } else if self.shown_settings_panel().is_some() {
            self.on_settings_key(ev);
        } else if self.shown_agent_panel().is_some() {
            self.on_agent_panel_key(ev);
        } else if ev.code == KeyCode::BackTab {
            self.toggle_permission_mode();
        } else if self.confirmation.is_some() {
            self.on_confirmation_key(ev);
        } else if !self.on_shortcut_key(ev) {
            self.on_input_key(ev);
        }
    }

    /// Keys consumed before the input line sees them. Returns whether `ev`
    /// was consumed.
    fn on_shortcut_key(&mut self, ev: KeyEvent) -> bool {
        let slash_active = !self.slash_suggestions().is_empty();
        match ev.code {
            KeyCode::PageUp => self.output.scroll_page_up(self.last_output_height),
            KeyCode::PageDown => self.output.scroll_page_down(self.last_output_height),
            KeyCode::Tab => self.on_tab(slash_active),
            KeyCode::Char('o') if ev.modifiers.contains(KeyModifiers::CONTROL) => {
                self.toggle_verbose();
            }
            KeyCode::Up if slash_active => {
                self.slash_selected = self.slash_selected.saturating_sub(1);
            }
            KeyCode::Down if slash_active => {
                if self.slash_selected + 1 < self.slash_suggestions().len() {
                    self.slash_selected += 1;
                }
            }
            KeyCode::Esc if slash_active => self.slash_dismissed = true,
            KeyCode::Esc if self.generating() => self.interrupt_turn(),
            _ => return false,
        }
        true
    }

    fn on_tab(&mut self, slash_active: bool) {
        if slash_active {
            self.accept_slash_selection();
            self.refresh_slash_state();
        } else if !self.try_complete_attach_path() {
            self.output.toggle_last_expandable();
        }
    }

    fn on_settings_key(&mut self, ev: KeyEvent) {
        if let Some(panel) = self.settings_panel.as_mut()
            && panel.on_key(ev).is_break()
        {
            self.settings_panel = None;
        }
    }

    pub(super) fn toggle_verbose(&mut self) {
        self.verbose = !self.verbose;
        self.output.set_verbose(self.verbose);
        self.set_notice(if self.verbose {
            "verbose mode: on"
        } else {
            "verbose mode: off"
        });
    }

    /// Flip between manual and accept-commands mode, applying it to the open
    /// turn too.
    fn toggle_permission_mode(&mut self) {
        self.permission_mode = match self.permission_mode {
            PermissionMode::Manual => PermissionMode::AcceptCommands,
            PermissionMode::AcceptCommands => PermissionMode::Manual,
        };
        if let Some(writer) = self
            .active_reply
            .as_ref()
            .and_then(|reply| reply.writer.clone())
        {
            let req = Request::SetPermissionMode {
                id: Uuid::new_v4().to_string(),
                mode: self.permission_mode,
            };
            self.tasks.spawn(async move {
                if let Err(e) = writer.send(req).await {
                    tracing::warn!("SetPermissionMode send failed: {e}");
                }
            });
        }
    }

    fn on_input_key(&mut self, ev: KeyEvent) {
        let action = self.input.on_key(ev);
        self.refresh_slash_state();
        match action {
            InputAction::None => {}
            InputAction::Submit(text) => self.submit_typed(text),
            InputAction::Quit => self.quit(),
        }
    }

    /// Deny any open confirmation and end the event loop.
    pub(super) fn quit(&mut self) {
        self.resolve_confirmation(Approval::Deny);
        self.quitting = true;
    }

    fn interrupt_turn(&mut self) {
        self.set_notice("interrupting…");
        let req = Request::InterruptTurn {
            id: Uuid::new_v4().to_string(),
        };
        self.spawn_one_shot(req, WireStream::Status, "interrupt");
    }

    /// Empty when the popup should be hidden.
    pub(crate) fn slash_suggestions(&self) -> Vec<&'static (&'static str, &'static str)> {
        if self.slash_dismissed {
            return Vec::new();
        }
        let buf = self.input.buffer();
        if !buf.starts_with('/') || buf.chars().any(char::is_whitespace) {
            return Vec::new();
        }
        SLASH_COMMANDS
            .iter()
            .filter(|(cmd, _)| cmd.starts_with(buf) && *cmd != buf)
            .collect()
    }

    pub(crate) fn slash_selected(&self) -> usize {
        self.slash_selected
    }

    fn accept_slash_selection(&mut self) {
        let suggestions = self.slash_suggestions();
        if let Some((cmd, _)) = suggestions.get(self.slash_selected).copied() {
            self.input.set_buffer(cmd.to_string());
        }
    }

    fn refresh_slash_state(&mut self) {
        if !self.input.buffer().starts_with('/') {
            self.slash_dismissed = false;
            self.slash_selected = 0;
            return;
        }
        let count = self.slash_suggestions().len();
        self.slash_selected = self.slash_selected.min(count.saturating_sub(1));
    }
}

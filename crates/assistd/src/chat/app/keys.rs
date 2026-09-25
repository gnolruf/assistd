//! Keyboard and mouse input: scrolling, toggles, the slash-command popup
//! and the F2 presence cycle.

use assistd_core::PresenceState;
use assistd_ipc::Request;
use assistd_tools::Approval;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use uuid::Uuid;

use super::{App, WireStream};
use crate::chat::input::InputAction;

pub(super) const MOUSE_WHEEL_STEP: u16 = 3;

/// `(command, usage_hint)` pairs for the autocomplete popup.
pub(super) const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/attach", "<path>"),
    ("/fork", "<name>"),
    ("/new", ""),
    ("/resume", ""),
    ("/switch", "<target>"),
    ("/undo", ""),
];

impl App {
    pub fn on_mouse(&mut self, ev: MouseEvent) {
        match ev.kind {
            MouseEventKind::ScrollUp => {
                self.touch_activity();
                self.output.scroll_lines_up(MOUSE_WHEEL_STEP);
            }
            MouseEventKind::ScrollDown => {
                self.touch_activity();
                self.output.scroll_lines_down(MOUSE_WHEEL_STEP);
            }
            _ => {}
        }
    }

    pub fn on_key(&mut self, ev: KeyEvent) {
        self.touch_activity();
        if self.modal.is_some() {
            self.on_confirmation_key(ev);
        } else if self.picker_modal.is_some() {
            self.on_picker_key(ev);
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
            KeyCode::F(2) => self.on_cycle_key(),
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

    fn toggle_verbose(&mut self) {
        self.verbose = !self.verbose;
        self.output.set_verbose(self.verbose);
        self.set_notice(if self.verbose {
            "verbose mode: on"
        } else {
            "verbose mode: off"
        });
    }

    fn on_input_key(&mut self, ev: KeyEvent) {
        let action = self.input.on_key(ev);
        self.refresh_slash_state();
        match action {
            InputAction::None => {}
            InputAction::Submit(text) => self.submit_typed(text),
            InputAction::Quit => {
                self.resolve_modal(Approval::Deny);
                self.quitting = true;
            }
        }
    }

    fn on_cycle_key(&mut self) {
        let target = self
            .presence_state
            .map(|s| s.next())
            .unwrap_or(PresenceState::Active);
        self.set_notice(&format!("cycling → {}", presence_label(target)));
        let req = Request::Cycle {
            id: Uuid::new_v4().to_string(),
        };
        self.spawn_one_shot(req, WireStream::Status, "cycle");
    }

    /// Empty when the popup should be hidden.
    pub fn slash_suggestions(&self) -> Vec<&'static (&'static str, &'static str)> {
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

    pub fn slash_selected(&self) -> usize {
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

fn presence_label(s: PresenceState) -> &'static str {
    match s {
        PresenceState::Active => "active",
        PresenceState::Drowsy => "drowsy",
        PresenceState::Sleeping => "sleeping",
    }
}

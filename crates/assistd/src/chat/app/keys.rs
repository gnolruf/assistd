//! Keyboard and mouse input: scrolling, toggles and the slash-command popup.

use assistd_ipc::{Request, RequestBody};
use assistd_tools::Approval;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use uuid::Uuid;

use super::commands::runs_offline;
use super::{App, Panel, WireStream};
use crate::chat::input::{InputAction, paste_as_keys};
use crate::chat::selection::Selection;

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
    ("/tasks", ""),
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

    /// Pasted text goes into the input line, never answering a prompt; an open
    /// panel receives it as typed characters until the paste closes it.
    pub(crate) fn on_paste(&mut self, text: &str) {
        if !self.panel_shown() {
            self.input.insert_paste(text);
            self.refresh_slash_state();
            return;
        }
        for key in paste_as_keys(text) {
            if !self.panel_shown() {
                break;
            }
            self.on_key(key);
        }
    }

    pub(super) fn panel_shown(&self) -> bool {
        self.shown_panel().is_some()
    }

    pub(crate) fn on_key(&mut self, ev: KeyEvent) {
        match self.shown_panel() {
            Some(Panel::Resume(_)) => self.on_resume_picker_key(ev),
            Some(Panel::WakeWord(_)) => self.on_wake_word_key(ev),
            Some(Panel::Settings(_)) => self.on_settings_key(ev),
            Some(Panel::Agents(_)) => self.on_agent_panel_key(ev),
            Some(Panel::Tasks(_)) => self.on_tasks_key(ev),
            None => self.on_chat_key(ev),
        }
    }

    /// A key pressed with no panel on screen: an open prompt answers first, then
    /// shortcuts, then the input line.
    fn on_chat_key(&mut self, ev: KeyEvent) {
        if self.plan_review.is_some() && self.confirmation.is_none() {
            if !self.held_from_prompt(ev) {
                self.on_plan_review_key(ev);
            }
        } else if ev.code == KeyCode::BackTab {
            self.toggle_mode();
        } else if self.confirmation.is_some() {
            if !self.held_from_prompt(ev) {
                self.on_confirmation_key(ev);
            }
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
            KeyCode::Char('t') if ev.modifiers.contains(KeyModifiers::CONTROL) => {
                self.open_tasks_panel();
            }
            KeyCode::Up if slash_active => {
                self.slash_selected.previous();
            }
            KeyCode::Down if slash_active => {
                let count = self.slash_suggestions().len();
                self.slash_selected.next(count);
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

    /// Closing the panel keeps its theme only when nothing is left unsaved.
    fn on_settings_key(&mut self, ev: KeyEvent) {
        if let Some(Panel::Settings(panel)) = self.panel.as_mut()
            && panel.on_key(ev).is_break()
        {
            if !panel.document().is_modified() {
                self.theme = panel.theme();
            }
            self.panel = None;
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

    pub(super) fn on_input_key(&mut self, ev: KeyEvent) {
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
        let req = Request::new(Uuid::new_v4().to_string(), RequestBody::InterruptTurn);
        self.spawn_one_shot(req, WireStream::Status, "interrupt");
    }

    /// Empty when the popup should be hidden; only offline-capable commands while the daemon
    /// is offline.
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
            .filter(|(cmd, _)| self.daemon_reachable || runs_offline(cmd))
            .collect()
    }

    pub(crate) fn slash_selected(&self) -> usize {
        self.slash_selected.index()
    }

    fn accept_slash_selection(&mut self) {
        let suggestions = self.slash_suggestions();
        if let Some((cmd, _)) = suggestions.get(self.slash_selected.index()).copied() {
            self.input.set_buffer(cmd.to_string());
        }
    }

    fn refresh_slash_state(&mut self) {
        if !self.input.buffer().starts_with('/') {
            self.slash_dismissed = false;
            self.slash_selected = Selection::default();
            return;
        }
        let count = self.slash_suggestions().len();
        self.slash_selected.clamp(count);
    }
}

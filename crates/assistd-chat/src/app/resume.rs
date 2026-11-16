//! The `/resume` session list and its search.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::{App, Panel, ResumePicker};

impl App {
    /// Open the list on the buffered `/resume` rows, unless a panel opened while
    /// they were on their way.
    pub(super) fn open_resume_picker(&mut self) {
        let entries = std::mem::take(&mut self.branches_buffer);
        if entries.is_empty() {
            self.set_notice("nothing to resume");
        } else if self.panel.is_none() {
            self.panel = Some(Panel::Resume(ResumePicker::new(entries)));
        }
    }

    /// Typing edits the search, ↑/↓ move the selection, Tab switches between
    /// the Sessions and Tasks tabs, Enter resumes, Ctrl+U clears the search,
    /// and Esc or Ctrl+C closes the list.
    pub(super) fn on_resume_picker_key(&mut self, ev: KeyEvent) {
        let Some(Panel::Resume(picker)) = self.panel.as_mut() else {
            return;
        };
        let control = ev.modifiers.contains(KeyModifiers::CONTROL);
        match ev.code {
            KeyCode::Esc => self.panel = None,
            KeyCode::Char('c') if control => self.panel = None,
            KeyCode::Tab => picker.switch_tab(),
            KeyCode::Up => picker.select_previous(),
            KeyCode::Down => picker.select_next(),
            KeyCode::Enter => self.resume_picked_branch(),
            _ => picker.edit_query(ev),
        }
    }

    fn resume_picked_branch(&mut self) {
        let Some(Panel::Resume(picker)) = &self.panel else {
            return;
        };
        if let Some(target) = picker.current_target() {
            self.panel = None;
            self.switch_branch(target);
        }
    }
}

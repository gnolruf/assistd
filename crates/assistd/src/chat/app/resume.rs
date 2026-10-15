//! The `/resume` session list and its search.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::{App, ResumePicker};

impl App {
    /// Open the list on the buffered `/resume` rows.
    pub(super) fn open_resume_picker(&mut self) {
        let entries = std::mem::take(&mut self.branches_buffer);
        if entries.is_empty() {
            self.set_notice("nothing to resume");
            return;
        }
        self.resume_picker = Some(ResumePicker::new(entries));
    }

    /// Typing searches, arrows move the selection, Enter resumes, Ctrl+U
    /// clears the search, and Esc or Ctrl+C closes the list.
    pub(super) fn on_resume_picker_key(&mut self, ev: KeyEvent) {
        let Some(picker) = self.resume_picker.as_mut() else {
            return;
        };
        let control = ev.modifiers.contains(KeyModifiers::CONTROL);
        match ev.code {
            KeyCode::Esc => self.resume_picker = None,
            KeyCode::Char('c') if control => self.resume_picker = None,
            KeyCode::Char('u') if control => picker.clear_query(),
            KeyCode::Char(c) if !control => picker.push_query(c),
            KeyCode::Backspace => picker.pop_query(),
            KeyCode::Up => picker.select_previous(),
            KeyCode::Down => picker.select_next(),
            KeyCode::Enter => self.resume_picked_branch(),
            _ => {}
        }
    }

    fn resume_picked_branch(&mut self) {
        let target = self
            .resume_picker
            .as_ref()
            .and_then(ResumePicker::current_target);
        if let Some(target) = target {
            self.resume_picker = None;
            self.switch_branch(target);
        }
    }
}

//! Input buffer with readline keybindings and history. Shift+Enter or
//! Alt+Enter inserts a line break; Enter submits; a paste never submits.

use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::text_editor::{EditAction, EditorShape, TextEditor};

const HISTORY_CAP: usize = 500;

/// A [`TextEditor`] box with submitted-text history; ↑/↓ move by row inside
/// multi-line text and recall history from its first and last rows.
#[derive(Debug)]
pub(super) struct InputLine {
    editor: TextEditor,
    history: VecDeque<String>,
    history_idx: Option<usize>,
    draft: Option<String>,
}

/// Action returned by [`InputLine::on_key`].
#[derive(Debug, PartialEq, Eq)]
pub(super) enum InputAction {
    None,
    /// Enter on a non-blank buffer.
    Submit(String),
    /// Ctrl+C or Ctrl+D on an empty buffer.
    Quit,
}

impl Default for InputLine {
    fn default() -> Self {
        Self::new()
    }
}

impl InputLine {
    pub(super) fn new() -> Self {
        Self {
            editor: TextEditor::new(String::new(), EditorShape::Box { wrap_width: 0 }),
            history: VecDeque::new(),
            history_idx: None,
            draft: None,
        }
    }

    pub(super) fn buffer(&self) -> &str {
        self.editor.text()
    }

    /// Replace the buffer and put the cursor at the end.
    pub(super) fn set_buffer(&mut self, text: String) {
        self.editor.set_text(text);
    }

    /// Cursor position in characters, not bytes, counting line breaks.
    pub(super) fn cursor_chars(&self) -> usize {
        self.editor.cursor_chars()
    }

    /// Records the columns the buffer wraps at, so ↑/↓ move by the rows on screen.
    pub(super) fn set_wrap_width(&mut self, width: u16) {
        self.editor.set_wrap_width(width);
    }

    /// Insert pasted text at the cursor, line breaks included.
    pub(super) fn insert_paste(&mut self, text: &str) {
        self.editor.insert_str(&normalize_line_breaks(text));
    }

    pub(super) fn on_key(&mut self, ev: KeyEvent) -> InputAction {
        if !matches!(ev.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return InputAction::None;
        }
        match ev.code {
            KeyCode::Char(c) if ev.modifiers.contains(KeyModifiers::CONTROL) => {
                return self.on_ctrl_char(c);
            }
            KeyCode::Up => {
                if !self.editor.move_row(false) {
                    self.history_prev();
                }
            }
            KeyCode::Down => {
                if !self.editor.move_row(true) {
                    self.history_next();
                }
            }
            _ => match self.editor.on_key(ev) {
                EditAction::Pending => {}
                EditAction::Apply => return self.submit(),
                EditAction::Cancel => self.clear(),
            },
        }
        InputAction::None
    }

    fn on_ctrl_char(&mut self, c: char) -> InputAction {
        match c.to_ascii_lowercase() {
            'c' | 'd' if self.buffer().is_empty() => return InputAction::Quit,
            'c' => self.clear(),
            'd' => self.editor.delete(),
            'a' => self.editor.move_to_start(),
            'e' => self.editor.move_to_end(),
            'k' => self.editor.kill_to_end(),
            'u' => self.editor.kill_to_start(),
            'w' => self.editor.kill_word_back(),
            'b' => self.editor.move_left(),
            'f' => self.editor.move_right(),
            'h' => self.editor.backspace(),
            _ => {}
        }
        InputAction::None
    }

    fn clear(&mut self) {
        self.editor.take_text();
        self.history_idx = None;
        self.draft = None;
    }

    fn submit(&mut self) -> InputAction {
        let text = self.editor.take_text();
        self.history_idx = None;
        self.draft = None;
        if text.trim().is_empty() {
            return InputAction::None;
        }
        if self.history.back().map(String::as_str) != Some(text.as_str()) {
            self.history.push_back(text.clone());
            while self.history.len() > HISTORY_CAP {
                self.history.pop_front();
            }
        }
        InputAction::Submit(text)
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let new_idx = match self.history_idx {
            None => {
                self.draft = Some(self.buffer().to_string());
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_idx = Some(new_idx);
        self.editor.set_text(self.history[new_idx].clone());
    }

    fn history_next(&mut self) {
        match self.history_idx {
            None => {}
            Some(i) if i + 1 < self.history.len() => {
                self.history_idx = Some(i + 1);
                self.editor.set_text(self.history[i + 1].clone());
            }
            Some(_) => {
                self.history_idx = None;
                let draft = self.draft.take().unwrap_or_default();
                self.editor.set_text(draft);
            }
        }
    }
}

/// Pasted text as key presses for single-line fields, with line breaks and
/// other control characters typed as spaces so a paste cannot press Enter.
pub(super) fn paste_as_keys(text: &str) -> Vec<KeyEvent> {
    normalize_line_breaks(text)
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .map(|c| KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
        .collect()
}

fn normalize_line_breaks(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyEventState;

    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn typed(c: char) -> KeyEvent {
        key(KeyCode::Char(c))
    }

    fn type_str(input: &mut InputLine, s: &str) {
        for c in s.chars() {
            input.on_key(typed(c));
        }
    }

    #[test]
    fn ctrl_w_kills_previous_word() {
        for (typed, expected) in [
            ("hello  world", "hello  "),
            ("foo bar ", "foo "),
            ("   ", ""),
        ] {
            let mut i = InputLine::new();
            type_str(&mut i, typed);
            i.on_key(ctrl('w'));
            assert_eq!(i.buffer(), expected, "{typed:?}");
        }
    }

    #[test]
    fn ctrl_c_and_ctrl_d_on_empty_quit() {
        for c in ['c', 'd'] {
            let mut i = InputLine::new();
            assert_eq!(i.on_key(ctrl(c)), InputAction::Quit, "ctrl-{c}");
        }
    }

    #[test]
    fn up_recalls_last_history_entries_in_order() {
        let mut i = InputLine::new();
        type_str(&mut i, "one");
        i.on_key(key(KeyCode::Enter));
        type_str(&mut i, "two");
        i.on_key(key(KeyCode::Enter));
        i.on_key(key(KeyCode::Up));
        assert_eq!(i.buffer(), "two");
        i.on_key(key(KeyCode::Up));
        assert_eq!(i.buffer(), "one");
    }

    #[test]
    fn down_past_newest_restores_draft() {
        let mut i = InputLine::new();
        type_str(&mut i, "old");
        i.on_key(key(KeyCode::Enter));
        type_str(&mut i, "drafted");
        i.on_key(key(KeyCode::Up));
        assert_eq!(i.buffer(), "old");
        i.on_key(key(KeyCode::Down));
        assert_eq!(i.buffer(), "drafted");
    }

    #[test]
    fn up_and_down_move_by_row_before_recalling_history() {
        let mut i = InputLine::new();
        i.set_wrap_width(20);
        type_str(&mut i, "old");
        i.on_key(key(KeyCode::Enter));
        i.insert_paste("ab\ncd");
        i.on_key(key(KeyCode::Up));
        i.on_key(typed('X'));
        assert_eq!(i.buffer(), "abX\ncd");
        i.on_key(key(KeyCode::Up));
        assert_eq!(i.buffer(), "old");
        i.on_key(key(KeyCode::Down));
        assert_eq!(i.buffer(), "abX\ncd");
    }

    #[test]
    fn utf8_cursor_motion_and_backspace() {
        let mut i = InputLine::new();
        type_str(&mut i, "héllo");
        for _ in 0..4 {
            i.on_key(key(KeyCode::Left));
        }
        i.on_key(key(KeyCode::Backspace));
        assert_eq!(i.buffer(), "éllo");
    }

    #[test]
    fn utf8_insert_keeps_boundaries() {
        let mut i = InputLine::new();
        type_str(&mut i, "ab");
        i.on_key(key(KeyCode::Left));
        i.on_key(typed('é'));
        assert_eq!(i.buffer(), "aéb");
    }

    #[test]
    fn modified_enter_breaks_the_line_and_plain_enter_submits() {
        let mut i = InputLine::new();
        type_str(&mut i, "one");
        for modifier in [KeyModifiers::SHIFT, KeyModifiers::ALT] {
            assert_eq!(
                i.on_key(KeyEvent::new(KeyCode::Enter, modifier)),
                InputAction::None
            );
        }
        type_str(&mut i, "two");
        assert_eq!(
            i.on_key(key(KeyCode::Enter)),
            InputAction::Submit("one\n\ntwo".to_string())
        );
    }

    #[test]
    fn paste_keeps_line_breaks_without_submitting() {
        let mut input = InputLine::new();
        type_str(&mut input, "see: ");
        input.insert_paste("line one\r\nline two\rline three\n");
        assert_eq!(input.buffer(), "see: line one\nline two\nline three\n");
        assert_eq!(
            input.on_key(key(KeyCode::Enter)),
            InputAction::Submit("see: line one\nline two\nline three\n".to_string())
        );
        let replayed: String = paste_as_keys("a\r\nb\tc")
            .into_iter()
            .filter_map(|ev| ev.code.as_char())
            .collect();
        assert_eq!(replayed, "a b c");
    }

    #[test]
    fn release_events_are_ignored() {
        let mut i = InputLine::new();
        let release = KeyEvent {
            code: KeyCode::Char('x'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: KeyEventState::NONE,
        };
        i.on_key(release);
        assert_eq!(i.buffer(), "");
    }
}

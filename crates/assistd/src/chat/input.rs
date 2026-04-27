//! Single-line input buffer with readline-style keybindings and in-memory
//! history. All state is plain Rust and fully unit-testable without a
//! terminal.

use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

pub const DEFAULT_HISTORY_CAP: usize = 500;

#[derive(Debug)]
pub struct InputLine {
    buffer: String,
    cursor: usize,
    history: VecDeque<String>,
    history_idx: Option<usize>,
    draft: Option<String>,
    history_cap: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum InputAction {
    None,
    Submit(String),
    Quit,
}

impl Default for InputLine {
    fn default() -> Self {
        Self::new()
    }
}

impl InputLine {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
            history: VecDeque::new(),
            history_idx: None,
            draft: None,
            history_cap: DEFAULT_HISTORY_CAP,
        }
    }

    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Replace the entire buffer with `text` and place the cursor at
    /// the end. Used by tab completion to extend a `/attach <path>`
    /// prefix in place without rebuilding the InputLine.
    pub fn set_buffer(&mut self, text: String) {
        self.cursor = text.len();
        self.buffer = text;
    }

    /// Cursor column in characters (not bytes). Suitable for placing the
    /// terminal cursor when rendering the input line.
    pub fn cursor_col(&self) -> u16 {
        self.buffer[..self.cursor].chars().count() as u16
    }

    pub fn on_key(&mut self, ev: KeyEvent) -> InputAction {
        if !matches!(ev.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return InputAction::None;
        }
        let ctrl = ev.modifiers.contains(KeyModifiers::CONTROL);
        let alt = ev.modifiers.contains(KeyModifiers::ALT);

        match ev.code {
            KeyCode::Char(c) if ctrl => self.on_ctrl_char(c),
            KeyCode::Char(c) if !alt => {
                self.insert_char(c);
                InputAction::None
            }
            KeyCode::Backspace => {
                self.backspace();
                InputAction::None
            }
            KeyCode::Delete => {
                self.delete_at_cursor();
                InputAction::None
            }
            KeyCode::Left => {
                self.move_left();
                InputAction::None
            }
            KeyCode::Right => {
                self.move_right();
                InputAction::None
            }
            KeyCode::Home => {
                self.cursor = 0;
                InputAction::None
            }
            KeyCode::End => {
                self.cursor = self.buffer.len();
                InputAction::None
            }
            KeyCode::Up => {
                self.history_prev();
                InputAction::None
            }
            KeyCode::Down => {
                self.history_next();
                InputAction::None
            }
            KeyCode::Enter => self.submit(),
            KeyCode::Esc => {
                self.clear();
                InputAction::None
            }
            _ => InputAction::None,
        }
    }

    fn on_ctrl_char(&mut self, c: char) -> InputAction {
        match c.to_ascii_lowercase() {
            'c' => {
                if self.buffer.is_empty() {
                    InputAction::Quit
                } else {
                    self.clear();
                    InputAction::None
                }
            }
            'd' => {
                if self.buffer.is_empty() {
                    InputAction::Quit
                } else {
                    self.delete_at_cursor();
                    InputAction::None
                }
            }
            'a' => {
                self.cursor = 0;
                InputAction::None
            }
            'e' => {
                self.cursor = self.buffer.len();
                InputAction::None
            }
            'k' => {
                self.buffer.truncate(self.cursor);
                InputAction::None
            }
            'u' => {
                self.buffer.drain(..self.cursor);
                self.cursor = 0;
                InputAction::None
            }
            'w' => {
                self.kill_word_back();
                InputAction::None
            }
            'b' => {
                self.move_left();
                InputAction::None
            }
            'f' => {
                self.move_right();
                InputAction::None
            }
            'h' => {
                self.backspace();
                InputAction::None
            }
            _ => InputAction::None,
        }
    }

    fn insert_char(&mut self, c: char) {
        self.buffer.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.buffer[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.buffer.drain(prev..self.cursor);
        self.cursor = prev;
    }

    fn delete_at_cursor(&mut self) {
        if self.cursor == self.buffer.len() {
            return;
        }
        if let Some((_, ch)) = self.buffer[self.cursor..].char_indices().next() {
            let end = self.cursor + ch.len_utf8();
            self.buffer.drain(self.cursor..end);
        }
    }

    fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor = self.buffer[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    fn move_right(&mut self) {
        if self.cursor == self.buffer.len() {
            return;
        }
        if let Some((_, ch)) = self.buffer[self.cursor..].char_indices().next() {
            self.cursor += ch.len_utf8();
        }
    }

    fn kill_word_back(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let chars: Vec<(usize, char)> = self.buffer[..self.cursor].char_indices().collect();
        if chars.is_empty() {
            return;
        }
        let mut n = chars.len();
        while n > 0 && chars[n - 1].1.is_whitespace() {
            n -= 1;
        }
        while n > 0 && !chars[n - 1].1.is_whitespace() {
            n -= 1;
        }
        let new_cursor = if n == chars.len() {
            return;
        } else {
            chars[n].0
        };
        self.buffer.drain(new_cursor..self.cursor);
        self.cursor = new_cursor;
    }

    fn clear(&mut self) {
        self.buffer.clear();
        self.cursor = 0;
        self.history_idx = None;
        self.draft = None;
    }

    fn submit(&mut self) -> InputAction {
        let text = std::mem::take(&mut self.buffer);
        self.cursor = 0;
        self.history_idx = None;
        self.draft = None;
        if text.trim().is_empty() {
            return InputAction::None;
        }
        if self.history.back().map(String::as_str) != Some(text.as_str()) {
            self.history.push_back(text.clone());
            while self.history.len() > self.history_cap {
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
                self.draft = Some(self.buffer.clone());
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_idx = Some(new_idx);
        self.buffer = self.history[new_idx].clone();
        self.cursor = self.buffer.len();
    }

    fn history_next(&mut self) {
        match self.history_idx {
            None => {}
            Some(i) if i + 1 < self.history.len() => {
                let new_idx = i + 1;
                self.history_idx = Some(new_idx);
                self.buffer = self.history[new_idx].clone();
                self.cursor = self.buffer.len();
            }
            Some(_) => {
                self.history_idx = None;
                self.buffer = self.draft.take().unwrap_or_default();
                self.cursor = self.buffer.len();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

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
    fn inserts_typed_chars() {
        let mut i = InputLine::new();
        type_str(&mut i, "hello");
        assert_eq!(i.buffer(), "hello");
        assert_eq!(i.cursor_col(), 5);
    }

    #[test]
    fn backspace_removes_previous_char() {
        let mut i = InputLine::new();
        type_str(&mut i, "foo");
        i.on_key(key(KeyCode::Backspace));
        assert_eq!(i.buffer(), "fo");
        assert_eq!(i.cursor_col(), 2);
    }

    #[test]
    fn cursor_and_delete() {
        let mut i = InputLine::new();
        type_str(&mut i, "hello");
        i.on_key(key(KeyCode::Home));
        assert_eq!(i.cursor_col(), 0);
        i.on_key(key(KeyCode::Delete));
        assert_eq!(i.buffer(), "ello");
        i.on_key(key(KeyCode::End));
        assert_eq!(i.cursor_col(), 4);
    }

    #[test]
    fn ctrl_a_and_ctrl_e_move_cursor() {
        let mut i = InputLine::new();
        type_str(&mut i, "abc");
        i.on_key(ctrl('a'));
        assert_eq!(i.cursor_col(), 0);
        i.on_key(ctrl('e'));
        assert_eq!(i.cursor_col(), 3);
    }

    #[test]
    fn ctrl_k_kills_to_end() {
        let mut i = InputLine::new();
        type_str(&mut i, "hello world");
        for _ in 0..5 {
            i.on_key(key(KeyCode::Left));
        }
        i.on_key(ctrl('k'));
        assert_eq!(i.buffer(), "hello ");
    }

    #[test]
    fn ctrl_u_kills_to_start() {
        let mut i = InputLine::new();
        type_str(&mut i, "hello world");
        for _ in 0..5 {
            i.on_key(key(KeyCode::Left));
        }
        i.on_key(ctrl('u'));
        assert_eq!(i.buffer(), "world");
        assert_eq!(i.cursor_col(), 0);
    }

    #[test]
    fn ctrl_w_kills_previous_word() {
        let mut i = InputLine::new();
        type_str(&mut i, "hello  world");
        i.on_key(ctrl('w'));
        assert_eq!(i.buffer(), "hello  ");
    }

    #[test]
    fn ctrl_w_skips_trailing_whitespace_then_word() {
        let mut i = InputLine::new();
        type_str(&mut i, "foo bar ");
        i.on_key(ctrl('w'));
        assert_eq!(i.buffer(), "foo ");
    }

    #[test]
    fn ctrl_w_only_whitespace_clears() {
        let mut i = InputLine::new();
        type_str(&mut i, "   ");
        i.on_key(ctrl('w'));
        assert_eq!(i.buffer(), "");
    }

    #[test]
    fn ctrl_c_on_empty_quits() {
        let mut i = InputLine::new();
        assert_eq!(i.on_key(ctrl('c')), InputAction::Quit);
    }

    #[test]
    fn ctrl_c_on_non_empty_clears() {
        let mut i = InputLine::new();
        type_str(&mut i, "hello");
        assert_eq!(i.on_key(ctrl('c')), InputAction::None);
        assert_eq!(i.buffer(), "");
    }

    #[test]
    fn ctrl_d_on_empty_quits() {
        let mut i = InputLine::new();
        assert_eq!(i.on_key(ctrl('d')), InputAction::Quit);
    }

    #[test]
    fn ctrl_d_on_non_empty_deletes_at_cursor() {
        let mut i = InputLine::new();
        type_str(&mut i, "abc");
        i.on_key(ctrl('a'));
        i.on_key(ctrl('d'));
        assert_eq!(i.buffer(), "bc");
    }

    #[test]
    fn enter_submits_and_records_history() {
        let mut i = InputLine::new();
        type_str(&mut i, "hello");
        let action = i.on_key(key(KeyCode::Enter));
        assert_eq!(action, InputAction::Submit("hello".into()));
        assert_eq!(i.buffer(), "");
        assert_eq!(i.history.len(), 1);
    }

    #[test]
    fn enter_ignores_whitespace_only() {
        let mut i = InputLine::new();
        type_str(&mut i, "   ");
        let action = i.on_key(key(KeyCode::Enter));
        assert_eq!(action, InputAction::None);
        assert_eq!(i.history.len(), 0);
    }

    #[test]
    fn enter_dedupes_consecutive_duplicates() {
        let mut i = InputLine::new();
        type_str(&mut i, "hello");
        i.on_key(key(KeyCode::Enter));
        type_str(&mut i, "hello");
        i.on_key(key(KeyCode::Enter));
        assert_eq!(i.history.len(), 1);
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
    fn down_past_newest_with_no_draft_is_empty() {
        let mut i = InputLine::new();
        type_str(&mut i, "old");
        i.on_key(key(KeyCode::Enter));
        i.on_key(key(KeyCode::Up));
        i.on_key(key(KeyCode::Down));
        assert_eq!(i.buffer(), "");
    }

    #[test]
    fn history_cap_is_enforced() {
        let mut i = InputLine::new();
        i.history_cap = 3;
        for n in 0..5 {
            type_str(&mut i, &format!("m{n}"));
            i.on_key(key(KeyCode::Enter));
        }
        assert_eq!(i.history.len(), 3);
        assert_eq!(i.history.front().map(String::as_str), Some("m2"));
        assert_eq!(i.history.back().map(String::as_str), Some("m4"));
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
    fn esc_clears_buffer() {
        let mut i = InputLine::new();
        type_str(&mut i, "hello");
        i.on_key(key(KeyCode::Esc));
        assert_eq!(i.buffer(), "");
        assert_eq!(i.cursor_col(), 0);
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

    #[test]
    fn alt_modified_char_is_not_inserted() {
        let mut i = InputLine::new();
        let alt_b = KeyEvent::new(KeyCode::Char('b'), KeyModifiers::ALT);
        i.on_key(alt_b);
        assert_eq!(i.buffer(), "");
    }
}

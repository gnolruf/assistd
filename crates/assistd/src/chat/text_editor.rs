//! A text field with a cursor: the editing every chat input shares.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::text::{locate_cursor, wrap_input};

/// What a [`TextEditor`] key asks of its owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EditAction {
    Pending,
    Apply,
    Cancel,
}

/// How a [`TextEditor`] is laid out.
#[derive(Debug, Clone, Copy)]
pub(crate) enum EditorShape {
    /// One row, scrolled sideways to keep the cursor in view.
    Line,
    /// A wrapped box that takes line breaks.
    Box {
        /// Columns the box last wrapped at; 0 before it is first drawn.
        wrap_width: u16,
    },
}

/// A text field with a cursor.
#[derive(Debug)]
pub(crate) struct TextEditor {
    text: String,
    /// Byte offset, always on a char boundary.
    cursor: usize,
    shape: EditorShape,
}

impl TextEditor {
    /// Places the cursor at the end of `text`.
    pub(crate) fn new(text: String, shape: EditorShape) -> Self {
        let cursor = text.len();
        Self {
            text,
            cursor,
            shape,
        }
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// Replaces the text and places the cursor at its end.
    pub(crate) fn set_text(&mut self, text: String) {
        self.cursor = text.len();
        self.text = text;
    }

    /// Empties the field, returning what it held.
    pub(crate) fn take_text(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.text)
    }

    pub(crate) fn is_box(&self) -> bool {
        matches!(self.shape, EditorShape::Box { .. })
    }

    /// Records the columns a box wraps at, so ↑/↓ move by the rows on screen; a line
    /// ignores it.
    pub(crate) fn set_wrap_width(&mut self, width: u16) {
        if let EditorShape::Box { wrap_width } = &mut self.shape {
            *wrap_width = width;
        }
    }

    /// Characters before the cursor.
    pub(crate) fn cursor_chars(&self) -> usize {
        self.text[..self.cursor].chars().count()
    }

    pub(crate) fn on_key(&mut self, ev: KeyEvent) -> EditAction {
        let ctrl = ev.modifiers.contains(KeyModifiers::CONTROL);
        let alt = ev.modifiers.contains(KeyModifiers::ALT);
        let line_break = ev
            .modifiers
            .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT);
        match (ev.code, self.shape) {
            (KeyCode::Enter, EditorShape::Box { .. }) if line_break => self.insert('\n'),
            (KeyCode::Up, EditorShape::Box { .. }) => {
                self.move_row(false);
            }
            (KeyCode::Down, EditorShape::Box { .. }) => {
                self.move_row(true);
            }
            (KeyCode::Enter, _) => return EditAction::Apply,
            (KeyCode::Esc, _) => return EditAction::Cancel,
            (KeyCode::Char('u'), _) if ctrl => {
                self.take_text();
            }
            (KeyCode::Char(c), _) if !ctrl && !alt => self.insert(c),
            (KeyCode::Backspace, _) => self.backspace(),
            (KeyCode::Delete, _) => self.delete(),
            (KeyCode::Left, _) => self.move_left(),
            (KeyCode::Right, _) => self.move_right(),
            (KeyCode::Home, _) => self.move_to_start(),
            (KeyCode::End, _) => self.move_to_end(),
            _ => {}
        }
        EditAction::Pending
    }

    pub(crate) fn insert(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    pub(crate) fn insert_str(&mut self, text: &str) {
        self.text.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    pub(crate) fn backspace(&mut self) {
        if let Some(previous) = self.previous_boundary() {
            self.text.drain(previous..self.cursor);
            self.cursor = previous;
        }
    }

    /// Deletes the character under the cursor.
    pub(crate) fn delete(&mut self) {
        if let Some(next) = self.next_boundary() {
            self.text.drain(self.cursor..next);
        }
    }

    pub(crate) fn move_left(&mut self) {
        self.cursor = self.previous_boundary().unwrap_or(self.cursor);
    }

    pub(crate) fn move_right(&mut self) {
        self.cursor = self.next_boundary().unwrap_or(self.cursor);
    }

    pub(crate) fn move_to_start(&mut self) {
        self.cursor = 0;
    }

    pub(crate) fn move_to_end(&mut self) {
        self.cursor = self.text.len();
    }

    /// Deletes from the cursor to the end of the text.
    pub(crate) fn kill_to_end(&mut self) {
        self.text.truncate(self.cursor);
    }

    /// Deletes from the start of the text to the cursor.
    pub(crate) fn kill_to_start(&mut self) {
        self.text.drain(..self.cursor);
        self.cursor = 0;
    }

    /// Deletes the word before the cursor and the whitespace after it.
    pub(crate) fn kill_word_back(&mut self) {
        let before = &self.text[..self.cursor];
        let word_end = before.trim_end().len();
        let word_start = before[..word_end]
            .char_indices()
            .rev()
            .take_while(|(_, c)| !c.is_whitespace())
            .last()
            .map_or(word_end, |(index, _)| index);
        self.text.drain(word_start..self.cursor);
        self.cursor = word_start;
    }

    /// Moves to the same column of the wrapped row above or below, or to
    /// that row's end when it is shorter. Returns whether there was such a
    /// row; a line has none.
    pub(crate) fn move_row(&mut self, down: bool) -> bool {
        let EditorShape::Box { wrap_width } = self.shape else {
            return false;
        };
        let chars: Vec<char> = self.text.chars().collect();
        let rows = wrap_input(&chars, 0, wrap_width);
        let (row, column) = locate_cursor(&rows, self.cursor_chars());
        let target = if down {
            row.checked_add(1)
        } else {
            row.checked_sub(1)
        };
        let Some(&(start, end)) = target.and_then(|target| rows.get(target)) else {
            return false;
        };
        let char_offset = start + column.min(end - start);
        self.cursor = self
            .text
            .char_indices()
            .nth(char_offset)
            .map_or(self.text.len(), |(index, _)| index);
        true
    }

    fn previous_boundary(&self) -> Option<usize> {
        self.text[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(index, _)| index)
    }

    fn next_boundary(&self) -> Option<usize> {
        self.text[self.cursor..]
            .chars()
            .next()
            .map(|c| self.cursor + c.len_utf8())
    }
}

//! The numbered answers a dock offers under a question, the last of which takes typed text.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::selection::Selection;

/// What a key did to a [`ChoiceList`].
pub(crate) enum ChoiceKey<C> {
    Answered(C),
    /// It moved the selection or edited the typed text.
    Taken,
    /// The list has no use for it.
    Passed,
}

/// Numbered choices with one selected; text typed on the last row goes with its answer.
pub(crate) struct ChoiceList<C> {
    choices: Vec<C>,
    selected: Selection,
    typed: String,
    /// What Esc and Ctrl+C answer.
    dismissal: C,
}

impl<C: Copy> ChoiceList<C> {
    /// `choices`, which must not be empty, with the last row selected so a stray Enter
    /// answers with it.
    pub(crate) fn new(choices: Vec<C>, dismissal: C) -> Self {
        let selected = Selection::at(choices.len().saturating_sub(1));
        Self {
            choices,
            selected,
            typed: String::new(),
            dismissal,
        }
    }

    /// Each choice in display order, with whether it is the selected one.
    pub(crate) fn rows(&self) -> impl Iterator<Item = (C, bool)> + '_ {
        self.choices
            .iter()
            .enumerate()
            .map(|(index, choice)| (*choice, index == self.selected.index()))
    }

    /// The text on the last row, as typed.
    pub(crate) fn typed(&self) -> &str {
        &self.typed
    }

    /// The typed text trimmed; `None` when nothing but whitespace was typed.
    pub(crate) fn message(&self) -> Option<String> {
        Some(self.typed.trim())
            .filter(|message| !message.is_empty())
            .map(str::to_string)
    }

    /// Arrows move the selection and Enter answers with it. On the last row every character
    /// typed is text; on the others a row's number answers, the last row's selecting it.
    /// Esc or Ctrl+C dismisses.
    pub(crate) fn on_key(&mut self, ev: KeyEvent) -> ChoiceKey<C> {
        let last_row = self.choices.len().saturating_sub(1);
        let on_text_row = self.selected.index() == last_row;
        match ev.code {
            KeyCode::Esc => return ChoiceKey::Answered(self.dismissal),
            KeyCode::Char('c') if ev.modifiers.contains(KeyModifiers::CONTROL) => {
                return ChoiceKey::Answered(self.dismissal);
            }
            KeyCode::Enter => {
                return match self.choices.get(self.selected.index()) {
                    Some(choice) => ChoiceKey::Answered(*choice),
                    None => ChoiceKey::Taken,
                };
            }
            KeyCode::Up => self.selected.previous(),
            KeyCode::Down => self.selected.next(self.choices.len()),
            KeyCode::Backspace if on_text_row => {
                self.typed.pop();
            }
            KeyCode::Char(typed) if on_text_row => self.typed.push(typed),
            KeyCode::Char(digit) => match self.numbered_row(digit) {
                Some(row) if row == last_row => self.selected = Selection::at(row),
                Some(row) => return ChoiceKey::Answered(self.choices[row]),
                None => {}
            },
            _ => return ChoiceKey::Passed,
        }
        ChoiceKey::Taken
    }

    /// The row shown as `digit`, rows being numbered from 1.
    fn numbered_row(&self, digit: char) -> Option<usize> {
        let number = usize::try_from(digit.to_digit(10)?).ok()?;
        number
            .checked_sub(1)
            .filter(|row| *row < self.choices.len())
    }
}

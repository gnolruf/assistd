//! The box that loads a built-in or saved theme, saves the current colors as
//! a new one, or deletes a saved one; each change is handed back to the panel.

use std::collections::BTreeMap;

use assistd_config::ThemeConfig;
use crossterm::event::{KeyCode, KeyEvent};

use super::super::selection::Selection;
use super::super::text_editor::{EditAction, EditorShape, TextEditor};

/// What the panel should do after a key in the theme list.
pub(crate) enum ThemeListAction {
    Pending,
    /// The highlighted theme changed; the preview should follow it.
    Changed,
    Close,
    Load {
        name: String,
        theme: ThemeConfig,
    },
    /// Saves the current colors under the typed name.
    Save(String),
    Delete(String),
}

/// One theme the list offers.
pub(crate) struct ThemeEntry {
    pub name: String,
    pub theme: ThemeConfig,
    /// Shipped with assistd, so it cannot be deleted.
    pub built_in: bool,
}

/// What the list is doing with the selected row.
pub(crate) enum ThemeListState {
    Browse,
    /// Typing the name to save the current colors under.
    Naming(TextEditor),
    /// Waiting for y or n before deleting the selected saved theme.
    ConfirmDelete,
}

/// The built-in themes, then the saved ones, then a row that saves.
pub(crate) struct ThemeList {
    entries: Vec<ThemeEntry>,
    /// Index into `entries`; their count selects the save row.
    selected: Selection,
    state: ThemeListState,
}

impl ThemeList {
    /// Lists the built-in themes and `saved`, selecting `active` when it is one.
    pub(crate) fn new(saved: &BTreeMap<String, ThemeConfig>, active: Option<&str>) -> Self {
        let built_in = ThemeConfig::BUILT_IN
            .iter()
            .map(|&(name, theme)| ThemeEntry {
                name: name.to_string(),
                theme,
                built_in: true,
            });
        let saved = saved.iter().map(|(name, &theme)| ThemeEntry {
            name: name.clone(),
            theme,
            built_in: false,
        });
        let entries: Vec<ThemeEntry> = built_in.chain(saved).collect();
        let selected = active
            .and_then(|active| entries.iter().position(|entry| entry.name == active))
            .map_or_else(Selection::default, Selection::at);
        Self {
            entries,
            selected,
            state: ThemeListState::Browse,
        }
    }

    pub(crate) fn entries(&self) -> &[ThemeEntry] {
        &self.entries
    }

    pub(crate) fn selected(&self) -> usize {
        self.selected.index()
    }

    pub(crate) fn state(&self) -> &ThemeListState {
        &self.state
    }

    /// The selected theme; `None` on the save row.
    pub(crate) fn highlighted(&self) -> Option<&ThemeEntry> {
        self.entries.get(self.selected.index())
    }

    pub(crate) fn on_key(&mut self, ev: KeyEvent) -> ThemeListAction {
        match &mut self.state {
            ThemeListState::Browse => self.on_browse_key(ev),
            ThemeListState::Naming(editor) => match editor.on_key(ev) {
                EditAction::Pending => ThemeListAction::Pending,
                EditAction::Apply => ThemeListAction::Save(editor.text().trim().to_string()),
                EditAction::Cancel => {
                    self.state = ThemeListState::Browse;
                    ThemeListAction::Pending
                }
            },
            ThemeListState::ConfirmDelete => self.on_confirm_delete_key(ev),
        }
    }

    fn on_browse_key(&mut self, ev: KeyEvent) -> ThemeListAction {
        match (ev.code, self.highlighted()) {
            (KeyCode::Esc, _) => ThemeListAction::Close,
            (KeyCode::Up | KeyCode::Char('k'), _) => {
                self.selected.previous();
                ThemeListAction::Changed
            }
            (KeyCode::Down | KeyCode::Char('j'), _) => {
                self.selected.next(self.entries.len() + 1);
                ThemeListAction::Changed
            }
            (KeyCode::Enter, Some(entry)) => ThemeListAction::Load {
                name: entry.name.clone(),
                theme: entry.theme,
            },
            (KeyCode::Enter, None) => {
                self.state =
                    ThemeListState::Naming(TextEditor::new(String::new(), EditorShape::Line));
                ThemeListAction::Pending
            }
            (KeyCode::Char('d'), Some(entry)) if !entry.built_in => {
                self.state = ThemeListState::ConfirmDelete;
                ThemeListAction::Pending
            }
            _ => ThemeListAction::Pending,
        }
    }

    fn on_confirm_delete_key(&mut self, ev: KeyEvent) -> ThemeListAction {
        match ev.code {
            KeyCode::Char('y') => {
                self.state = ThemeListState::Browse;
                self.highlighted()
                    .map_or(ThemeListAction::Pending, |entry| {
                        ThemeListAction::Delete(entry.name.clone())
                    })
            }
            KeyCode::Char('n') | KeyCode::Esc => {
                self.state = ThemeListState::Browse;
                ThemeListAction::Pending
            }
            _ => ThemeListAction::Pending,
        }
    }
}

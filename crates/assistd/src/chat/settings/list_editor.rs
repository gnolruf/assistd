//! The box that edits a list setting entry by entry; each change is handed
//! back to the panel to apply.

use crossterm::event::{KeyCode, KeyEvent};

use super::catalog::ListItem;
use super::panel::{EditAction, EditorShape, TextEditor};

/// What the panel should do after a key in the list box.
pub(crate) enum ListAction {
    Pending,
    Close,
    /// Puts `value` at `index`, appending when `index` is the entry count.
    Store {
        index: usize,
        value: toml::Value,
    },
    Delete(usize),
    /// The open entry cannot be stored, for the reason given.
    Refused(String),
}

/// What the list box is doing with the selected entry.
pub(crate) enum ListState {
    Browse,
    /// Editing it, or adding one when the add row is selected.
    Edit(EntryEditor),
    /// Waiting for y or n before deleting it.
    ConfirmDelete,
}

/// Navigation over a list's entries, with at most one entry open for editing.
pub(crate) struct ListEditor {
    item: ListItem,
    /// Index of the selected entry; the entry count selects the add row.
    selected: usize,
    state: ListState,
}

impl ListEditor {
    pub(crate) fn new(item: ListItem) -> Self {
        Self {
            item,
            selected: 0,
            state: ListState::Browse,
        }
    }

    pub(crate) fn selected(&self) -> usize {
        self.selected
    }

    pub(crate) fn state(&self) -> &ListState {
        &self.state
    }

    /// Closes the open entry after it was stored.
    pub(crate) fn close_entry(&mut self) {
        self.state = ListState::Browse;
    }

    pub(crate) fn on_key(&mut self, ev: KeyEvent, entries: &[toml::Value]) -> ListAction {
        match &mut self.state {
            ListState::Edit(entry) => match entry.on_key(ev) {
                EditAction::Pending => ListAction::Pending,
                EditAction::Cancel => {
                    self.state = ListState::Browse;
                    ListAction::Pending
                }
                EditAction::Apply => match entry.value() {
                    Ok(value) => ListAction::Store {
                        index: self.selected,
                        value,
                    },
                    Err(reason) => ListAction::Refused(reason),
                },
            },
            ListState::ConfirmDelete => self.on_confirm_delete_key(ev),
            ListState::Browse => self.on_browse_key(ev, entries),
        }
    }

    fn on_confirm_delete_key(&mut self, ev: KeyEvent) -> ListAction {
        match ev.code {
            KeyCode::Char('y' | 'Y') => {
                self.state = ListState::Browse;
                ListAction::Delete(self.selected)
            }
            KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                self.state = ListState::Browse;
                ListAction::Pending
            }
            _ => ListAction::Pending,
        }
    }

    fn on_browse_key(&mut self, ev: KeyEvent, entries: &[toml::Value]) -> ListAction {
        let add_row = entries.len();
        match ev.code {
            KeyCode::Esc | KeyCode::Char('q') => return ListAction::Close,
            KeyCode::Up | KeyCode::Char('k') => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.selected = (self.selected + 1).min(add_row),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = add_row,
            KeyCode::Enter => {
                self.state =
                    ListState::Edit(EntryEditor::new(self.item, entries.get(self.selected)));
            }
            KeyCode::Char('a') | KeyCode::Insert => {
                self.selected = add_row;
                self.state = ListState::Edit(EntryEditor::new(self.item, None));
            }
            KeyCode::Char('d') | KeyCode::Delete | KeyCode::Backspace
                if self.selected < add_row =>
            {
                self.state = ListState::ConfirmDelete;
            }
            _ => {}
        }
        ListAction::Pending
    }
}

/// The fields of one entry being edited.
pub(crate) struct EntryEditor {
    /// The entry's text, or a search engine's name.
    value: TextEditor,
    /// A search engine's API key file; empty when it takes no key.
    key_file: Option<TextEditor>,
    key_file_focused: bool,
}

impl EntryEditor {
    /// Starts from `entry`, or blank when adding one.
    fn new(item: ListItem, entry: Option<&toml::Value>) -> Self {
        let field = |text: String| TextEditor::new(text, EditorShape::Line);
        let text_of = |value: Option<&toml::Value>| value.map(entry_text).unwrap_or_default();
        let (value, key_file) = match item {
            ListItem::Text => (field(text_of(entry)), None),
            ListItem::SearchEngine => match entry {
                Some(toml::Value::Table(table)) => (
                    field(text_of(table.get("name"))),
                    Some(field(text_of(table.get("api_key_file")))),
                ),
                _ => (field(text_of(entry)), Some(field(String::new()))),
            },
        };
        Self {
            value,
            key_file,
            key_file_focused: false,
        }
    }

    pub(crate) fn value_field(&self) -> &TextEditor {
        &self.value
    }

    pub(crate) fn key_file_field(&self) -> Option<&TextEditor> {
        self.key_file.as_ref()
    }

    pub(crate) fn key_file_focused(&self) -> bool {
        self.key_file_focused
    }

    fn on_key(&mut self, ev: KeyEvent) -> EditAction {
        match (ev.code, &mut self.key_file) {
            (KeyCode::Tab | KeyCode::BackTab | KeyCode::Up | KeyCode::Down, Some(_)) => {
                self.key_file_focused = !self.key_file_focused;
                EditAction::Pending
            }
            (_, Some(key_file)) if self.key_file_focused => key_file.on_key(ev),
            _ => self.value.on_key(ev),
        }
    }

    /// The entry the fields describe; a search engine with no key file is
    /// its bare name.
    fn value(&self) -> Result<toml::Value, String> {
        let value = self.value.text();
        if value.trim().is_empty() {
            return Err("an entry cannot be empty; Esc cancels".to_string());
        }
        let Some(key_file) = &self.key_file else {
            return Ok(toml::Value::from(value));
        };
        let name = toml::Value::from(value.trim());
        match key_file.text().trim() {
            "" => Ok(name),
            path => Ok(toml::Value::Table(toml::Table::from_iter([
                ("name".to_string(), name),
                ("api_key_file".to_string(), toml::Value::from(path)),
            ]))),
        }
    }
}

/// `entry` on one line: its text, or a keyed engine's name and key file.
pub(crate) fn entry_summary(entry: &toml::Value) -> String {
    let field = |name: &str| entry.get(name).and_then(toml::Value::as_str);
    match (field("name"), field("api_key_file")) {
        (Some(name), Some(key_file)) => format!("{name} (key: {key_file})"),
        _ => entry_text(entry),
    }
}

fn entry_text(entry: &toml::Value) -> String {
    match entry {
        toml::Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

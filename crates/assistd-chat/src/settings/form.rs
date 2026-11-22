//! A form of labelled fields, each text, a choice or a list: focus, the keys every field
//! kind takes, and the list box open over a list field.

use assistd_config::ListItem;
use crossterm::event::{KeyCode, KeyEvent};

use super::super::selection::cycled;
use super::super::text_editor::TextEditor;
use super::list_editor::{ListAction, ListEditor, entry_text};

/// What a field holds, as the form draws it.
pub(crate) enum FieldValue<'a> {
    Text(&'a TextEditor),
    /// Text drawn as a mask while its field is not focused.
    Secret(&'a TextEditor),
    /// One of a few options, switched with ←/→.
    Choice(&'a str),
    /// Entries edited in a list box, drawn in the form as `summary`.
    List {
        entries: &'a [String],
        summary: String,
    },
}

/// What a key in a form asks of whatever holds it.
pub(crate) enum FormAction<T> {
    Pending,
    Apply(T),
    Cancel,
    /// The form or its open list entry cannot be applied, for the reason given.
    Refused(String),
}

/// The fields a [`Form`] edits and what they make once applied.
pub(crate) trait FormFields {
    type Field: Copy + PartialEq;
    type Output;

    /// The fields shown now, in order; never empty.
    fn fields(&self) -> Vec<Self::Field>;

    fn label(field: Self::Field) -> &'static str;

    fn value(&self, field: Self::Field) -> FieldValue<'_>;

    /// The editor behind a text field; `None` for the other kinds.
    fn text_mut(&mut self, field: Self::Field) -> Option<&mut TextEditor>;

    /// Switches a choice field to its next or previous option.
    fn cycle(&mut self, field: Self::Field, forward: bool);

    /// The entries behind a list field; `None` for the other kinds.
    fn list_mut(&mut self, _field: Self::Field) -> Option<&mut Vec<String>> {
        None
    }

    /// `text` as it is stored at `index` of the list field's `entries`, or why it cannot be.
    fn list_entry(
        _field: Self::Field,
        text: String,
        _entries: &[String],
        _index: usize,
    ) -> Result<String, String> {
        Ok(text)
    }

    /// What the fields describe, or why they describe nothing.
    fn output(&self) -> Result<Self::Output, String>;
}

/// `F`'s fields with one focused, and at most one list field open in a list box.
pub(crate) struct Form<F: FormFields> {
    fields: F,
    focus: F::Field,
    open_list: Option<ListEditor>,
}

impl<F: FormFields> Form<F> {
    /// A form over `fields` with the first one focused.
    pub(crate) fn new(fields: F) -> Self {
        let focus = fields.fields()[0];
        Self {
            fields,
            focus,
            open_list: None,
        }
    }

    pub(crate) fn fields(&self) -> &F {
        &self.fields
    }

    pub(crate) fn focus(&self) -> F::Field {
        self.focus
    }

    /// The list box open over the focused field.
    pub(crate) fn open_list(&self) -> Option<&ListEditor> {
        self.open_list.as_ref()
    }

    /// The focused field's entries as a list box's; empty for a field that is no list.
    pub(crate) fn list_entries(&self) -> Vec<toml::Value> {
        match self.fields.value(self.focus) {
            FieldValue::List { entries, .. } => {
                entries.iter().cloned().map(toml::Value::from).collect()
            }
            FieldValue::Text(_) | FieldValue::Secret(_) | FieldValue::Choice(_) => Vec::new(),
        }
    }

    /// ↑/↓ and Tab move between fields, ←/→ switch a choice, typing edits a text field,
    /// Enter opens a list field and elsewhere applies the form, and Esc drops it.
    pub(crate) fn on_key(&mut self, ev: KeyEvent) -> FormAction<F::Output> {
        if self.open_list.is_some() {
            return self.on_list_key(ev);
        }
        match (ev.code, self.fields.value(self.focus)) {
            (KeyCode::Up | KeyCode::BackTab, _) => self.move_focus(false),
            (KeyCode::Down | KeyCode::Tab, _) => self.move_focus(true),
            (KeyCode::Left, FieldValue::Choice(_)) => self.fields.cycle(self.focus, false),
            (KeyCode::Right, FieldValue::Choice(_)) => self.fields.cycle(self.focus, true),
            (KeyCode::Enter, FieldValue::List { .. }) => {
                self.open_list = Some(ListEditor::new(ListItem::Text));
            }
            (KeyCode::Enter, _) => {
                return match self.fields.output() {
                    Ok(output) => FormAction::Apply(output),
                    Err(reason) => FormAction::Refused(reason),
                };
            }
            (KeyCode::Esc, _) => return FormAction::Cancel,
            _ => {
                if let Some(editor) = self.fields.text_mut(self.focus) {
                    editor.on_key(ev);
                }
            }
        }
        FormAction::Pending
    }

    fn move_focus(&mut self, forward: bool) {
        self.focus = cycled(&self.fields.fields(), self.focus, forward);
    }

    fn on_list_key(&mut self, ev: KeyEvent) -> FormAction<F::Output> {
        let entries = self.list_entries();
        let Some(list) = &mut self.open_list else {
            return FormAction::Pending;
        };
        match list.on_key(ev, &entries) {
            ListAction::Pending => {}
            ListAction::Close => self.open_list = None,
            ListAction::Store { index, value } => {
                if let Err(reason) = self.place_entry(index, entry_text(&value)) {
                    return FormAction::Refused(reason);
                }
                if let Some(list) = &mut self.open_list {
                    list.close_entry();
                }
            }
            ListAction::Delete(index) => {
                if let Some(entries) = self.fields.list_mut(self.focus) {
                    entries.remove(index);
                }
            }
            ListAction::Refused(reason) => return FormAction::Refused(reason),
        }
        FormAction::Pending
    }

    /// Puts `text` at `index` of the focused list, appending when `index` is its length.
    fn place_entry(&mut self, index: usize, text: String) -> Result<(), String> {
        let field = self.focus;
        let Some(entries) = self.fields.list_mut(field) else {
            return Ok(());
        };
        let text = F::list_entry(field, text, entries, index)?;
        match entries.get_mut(index) {
            Some(entry) => *entry = text,
            None => entries.push(text),
        }
        Ok(())
    }
}

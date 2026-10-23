//! Settings panel state and key handling. Changes stay in memory until
//! saved, and only a config that validates is written.

use std::ops::ControlFlow;

use assistd_config::{ConfigDocument, ConfigError};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::super::ui::{locate_cursor, wrap_input};
use super::catalog::{SECTIONS, Setting, SettingKind, Tab};
use super::list_editor::{ListAction, ListEditor};

/// Rows PgUp/PgDn move the selection.
const PAGE_ROWS: usize = 10;
const SAVED_NOTE: &str = "saved; restart the daemon to apply (systemctl --user restart assistd)";

/// One line of the settings list; sections are indexes into [`SECTIONS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Row {
    /// The section's title, and its [`enable`](super::catalog::Section::enable)
    /// checkbox when it has one.
    Heading(usize),
    Setting(usize, &'static Setting),
    /// Shows or hides the section's advanced settings.
    AdvancedToggle(usize),
}

impl Row {
    fn section(self) -> usize {
        match self {
            Self::Heading(section) | Self::Setting(section, _) | Self::AdvancedToggle(section) => {
                section
            }
        }
    }
}

pub(crate) enum PanelMode {
    Browse,
    Edit(TextEditor),
    /// Editing a list setting's entries.
    List(ListEditor),
    /// Closing with unsaved changes; waits for save, discard or cancel.
    ConfirmClose,
}

/// The outcome of the last action, shown under the list.
pub(crate) enum Feedback {
    Note(String),
    Error(String),
}

/// The settings list over an open config file.
pub(crate) struct SettingsPanel {
    document: ConfigDocument,
    tab: Tab,
    /// Per section, whether its advanced settings are listed.
    advanced_shown: [bool; SECTIONS.len()],
    /// Index into [`rows`](Self::rows); never a heading.
    selected: usize,
    mode: PanelMode,
    feedback: Option<Feedback>,
    /// Why the config would not validate, if it would not.
    issues: Vec<String>,
}

impl SettingsPanel {
    pub(crate) fn new(document: ConfigDocument) -> Self {
        let mut panel = Self {
            document,
            tab: Tab::ALL[0],
            advanced_shown: [false; SECTIONS.len()],
            selected: 0,
            mode: PanelMode::Browse,
            feedback: None,
            issues: Vec::new(),
        };
        panel.issues = validation_issues(&panel.document);
        panel.select_first();
        panel
    }

    pub(crate) fn document(&self) -> &ConfigDocument {
        &self.document
    }

    pub(crate) fn tab(&self) -> Tab {
        self.tab
    }

    pub(crate) fn selected(&self) -> usize {
        self.selected
    }

    pub(crate) fn mode(&self) -> &PanelMode {
        &self.mode
    }

    pub(crate) fn feedback(&self) -> Option<&Feedback> {
        self.feedback.as_ref()
    }

    pub(crate) fn issues(&self) -> &[String] {
        &self.issues
    }

    /// The current tab's rows.
    pub(crate) fn rows(&self) -> Vec<Row> {
        SECTIONS
            .iter()
            .enumerate()
            .filter(|(_, section)| section.tab == self.tab)
            .flat_map(|(index, section)| {
                let shown = self.advanced_shown[index];
                let toggle = (section.advanced_count() > 0).then_some(Row::AdvancedToggle(index));
                std::iter::once(Row::Heading(index))
                    .chain(
                        section
                            .settings
                            .iter()
                            .filter(move |setting| shown || !setting.advanced())
                            .map(move |setting| Row::Setting(index, setting)),
                    )
                    .chain(toggle)
            })
            .collect()
    }

    pub(crate) fn advanced_shown(&self, section: usize) -> bool {
        self.advanced_shown[section]
    }

    /// Whether `row` sits in a feature that is off, or that needs another
    /// feature that is off; locked rows are shown but skipped.
    pub(crate) fn is_locked(&self, row: Row) -> bool {
        let section = &SECTIONS[row.section()];
        let requirements_met = section.requires.iter().all(|key| self.is_on(key));
        match row {
            Row::Heading(_) => !requirements_met,
            Row::Setting(..) | Row::AdvancedToggle(_) => {
                !requirements_met
                    || section
                        .enable
                        .as_ref()
                        .is_some_and(|enable| !self.is_on(enable.key))
            }
        }
    }

    pub(crate) fn is_on(&self, key: &str) -> bool {
        self.current_value(key)
            .and_then(toml::Value::as_bool)
            .unwrap_or(false)
    }

    /// The entries in effect for list setting `key`.
    pub(crate) fn list_entries(&self, key: &str) -> &[toml::Value] {
        self.current_value(key)
            .and_then(toml::Value::as_array)
            .map_or(&[], Vec::as_slice)
    }

    /// The value in effect for `key`: the file's, else the default.
    pub(crate) fn current_value(&self, key: &str) -> Option<&toml::Value> {
        self.document
            .file_value(key)
            .or_else(|| self.document.default_value(key))
    }

    /// Breaks when the panel closes.
    pub(crate) fn on_key(&mut self, ev: KeyEvent) -> ControlFlow<()> {
        if !matches!(ev.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return ControlFlow::Continue(());
        }
        match self.mode {
            PanelMode::Browse => self.on_browse_key(ev),
            PanelMode::Edit(_) => {
                self.on_edit_key(ev);
                ControlFlow::Continue(())
            }
            PanelMode::List(_) => {
                self.on_list_key(ev);
                ControlFlow::Continue(())
            }
            PanelMode::ConfirmClose => self.on_confirm_close_key(ev),
        }
    }

    fn on_browse_key(&mut self, ev: KeyEvent) -> ControlFlow<()> {
        let ctrl = ev.modifiers.contains(KeyModifiers::CONTROL);
        match ev.code {
            KeyCode::Char('s') if ctrl => {
                self.save();
            }
            KeyCode::Char('c') if ctrl => return self.request_close(),
            KeyCode::Esc | KeyCode::Char('q') => return self.request_close(),
            KeyCode::Tab => self.switch_tab(true),
            KeyCode::BackTab => self.switch_tab(false),
            KeyCode::Up | KeyCode::Char('k') => self.step_selection(false, 1),
            KeyCode::Down | KeyCode::Char('j') => self.step_selection(true, 1),
            KeyCode::PageUp => self.step_selection(false, PAGE_ROWS),
            KeyCode::PageDown => self.step_selection(true, PAGE_ROWS),
            KeyCode::Home => self.step_selection(false, usize::MAX),
            KeyCode::End => self.step_selection(true, usize::MAX),
            KeyCode::Enter | KeyCode::Char(' ') => self.activate_selected(),
            KeyCode::Left => self.cycle_selected(false),
            KeyCode::Right => self.cycle_selected(true),
            KeyCode::Char('a') => {
                let toggle = Row::AdvancedToggle(self.selected_row().section());
                if !self.is_locked(toggle) {
                    self.toggle_advanced(toggle.section());
                }
            }
            KeyCode::Char('r') | KeyCode::Delete | KeyCode::Backspace => self.reset_selected(),
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn on_edit_key(&mut self, ev: KeyEvent) {
        let PanelMode::Edit(editor) = &mut self.mode else {
            return;
        };
        match editor.on_key(ev) {
            EditAction::Pending => {}
            EditAction::Cancel => {
                self.mode = PanelMode::Browse;
                self.feedback = None;
            }
            EditAction::Apply => {
                let text = editor.text.clone();
                self.apply_text(&text);
            }
        }
    }

    fn on_list_key(&mut self, ev: KeyEvent) {
        let Some(setting) = self.selected_setting() else {
            return;
        };
        let mut entries = self.list_entries(setting.key).to_vec();
        let PanelMode::List(editor) = &mut self.mode else {
            return;
        };
        match editor.on_key(ev, &entries) {
            ListAction::Pending => {}
            ListAction::Close => {
                self.mode = PanelMode::Browse;
                self.feedback = None;
            }
            ListAction::Store { index, value } => {
                match entries.get_mut(index) {
                    Some(entry) => *entry = value,
                    None => entries.push(value),
                }
                self.apply_entries(setting, entries);
            }
            ListAction::Delete(index) => {
                entries.remove(index);
                self.apply_entries(setting, entries);
            }
            ListAction::Refused(reason) => self.feedback = Some(Feedback::Error(reason)),
        }
    }

    /// Sets list `setting` to `entries`, closing the open entry, or reports
    /// why they were refused and leaves the entry open.
    fn apply_entries(&mut self, setting: &Setting, entries: Vec<toml::Value>) {
        let value = toml::Value::Array(entries);
        match self.change(|document| document.set(setting.key, &value)) {
            Ok(()) => {
                if let PanelMode::List(editor) = &mut self.mode {
                    editor.close_entry();
                }
                self.feedback = None;
            }
            Err(reason) => self.feedback = Some(Feedback::Error(reason)),
        }
    }

    fn on_confirm_close_key(&mut self, ev: KeyEvent) -> ControlFlow<()> {
        match ev.code {
            KeyCode::Char('s') => {
                if self.save() {
                    return ControlFlow::Break(());
                }
                self.mode = PanelMode::Browse;
            }
            KeyCode::Char('d') => return ControlFlow::Break(()),
            KeyCode::Esc => self.mode = PanelMode::Browse,
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn request_close(&mut self) -> ControlFlow<()> {
        if self.document.is_modified() {
            self.mode = PanelMode::ConfirmClose;
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    }

    fn selected_row(&self) -> Row {
        self.rows()[self.selected]
    }

    /// The setting on the selected row, the section checkbox included.
    pub(crate) fn selected_setting(&self) -> Option<&'static Setting> {
        match self.selected_row() {
            Row::Setting(_, setting) => Some(setting),
            Row::Heading(section) => SECTIONS[section].enable.as_ref(),
            Row::AdvancedToggle(_) => None,
        }
    }

    /// Shows the next or previous tab, wrapping at the ends.
    fn switch_tab(&mut self, forward: bool) {
        let count = Tab::ALL.len();
        let index = Tab::ALL
            .iter()
            .position(|tab| *tab == self.tab)
            .unwrap_or(0);
        let offset = if forward { 1 } else { count - 1 };
        self.tab = Tab::ALL[(index + offset) % count];
        self.select_first();
    }

    /// Selects the tab's first selectable row, or its first row when none is.
    fn select_first(&mut self) {
        self.selected = self
            .rows()
            .into_iter()
            .position(|row| self.is_selectable(row))
            .unwrap_or(0);
    }

    /// Moves up to `steps` selectable rows forward or back, stopping at the ends.
    fn step_selection(&mut self, forward: bool, steps: usize) {
        let selectable: Vec<bool> = self
            .rows()
            .into_iter()
            .map(|row| self.is_selectable(row))
            .collect();
        let is_selectable = |index: &usize| selectable[*index];
        for _ in 0..steps {
            let next = if forward {
                (self.selected + 1..selectable.len()).find(is_selectable)
            } else {
                (0..self.selected).rev().find(is_selectable)
            };
            match next {
                Some(index) => self.selected = index,
                None => break,
            }
        }
    }

    /// Unlocked settings, advanced toggles, and headings that are checkboxes.
    fn is_selectable(&self, row: Row) -> bool {
        let is_checkbox = match row {
            Row::Heading(section) => SECTIONS[section].enable.is_some(),
            Row::Setting(..) | Row::AdvancedToggle(_) => true,
        };
        is_checkbox && !self.is_locked(row)
    }

    fn toggle_advanced(&mut self, section: usize) {
        self.advanced_shown[section] = !self.advanced_shown[section];
        let toggle = Row::AdvancedToggle(section);
        if let Some(index) = self.rows().iter().position(|row| *row == toggle) {
            self.selected = index;
        }
    }

    fn activate_selected(&mut self) {
        let setting = match self.selected_row() {
            Row::AdvancedToggle(section) => return self.toggle_advanced(section),
            Row::Setting(_, setting) => setting,
            Row::Heading(section) => match SECTIONS[section].enable.as_ref() {
                Some(enable) => enable,
                None => return,
            },
        };
        match setting.kind {
            SettingKind::Toggle | SettingKind::Choice(_) => self.cycle_selected(true),
            SettingKind::External => {
                self.feedback = Some(Feedback::Note(format!(
                    "edit {} in {}",
                    setting.key,
                    self.document.path().display()
                )));
            }
            SettingKind::List(item) => {
                self.mode = PanelMode::List(ListEditor::new(item));
                self.feedback = None;
            }
            SettingKind::Integer
            | SettingKind::Decimal
            | SettingKind::Text
            | SettingKind::Prose => {
                let text = self
                    .current_value(setting.key)
                    .map(|value| editable_text(setting.kind, value))
                    .unwrap_or_default();
                let shape = if setting.kind == SettingKind::Prose {
                    EditorShape::Box { wrap_width: 0 }
                } else {
                    EditorShape::Line
                };
                self.mode = PanelMode::Edit(TextEditor::new(text, shape));
                self.feedback = None;
            }
        }
    }

    /// Flips a toggle, or moves a choice to its next or previous option.
    fn cycle_selected(&mut self, forward: bool) {
        let Some(setting) = self.selected_setting() else {
            return;
        };
        let current = self.current_value(setting.key);
        let next = match setting.kind {
            SettingKind::Toggle => {
                toml::Value::Boolean(!current.and_then(toml::Value::as_bool).unwrap_or(false))
            }
            SettingKind::Choice(options) => {
                let index = current
                    .and_then(toml::Value::as_str)
                    .and_then(|current| options.iter().position(|option| *option == current))
                    .unwrap_or(0);
                let offset = if forward { 1 } else { options.len() - 1 };
                toml::Value::from(options[(index + offset) % options.len()])
            }
            _ => return,
        };
        self.apply(setting, &next);
    }

    fn apply_text(&mut self, text: &str) {
        let Some(setting) = self.selected_setting() else {
            return;
        };
        match parse_input(setting.kind, text) {
            Ok(value) => self.apply(setting, &value),
            Err(reason) => self.feedback = Some(Feedback::Error(reason)),
        }
    }

    /// Sets `setting` and returns to browsing, or reports why the value
    /// was refused and stays in the current mode with the value unchanged.
    fn apply(&mut self, setting: &Setting, value: &toml::Value) {
        match self.change(|document| document.set(setting.key, value)) {
            Ok(()) => {
                self.mode = PanelMode::Browse;
                self.feedback = None;
            }
            Err(reason) => self.feedback = Some(Feedback::Error(reason)),
        }
    }

    fn reset_selected(&mut self) {
        let Some(setting) = self.selected_setting() else {
            return;
        };
        if self.document.file_value(setting.key).is_none() {
            return;
        }
        self.feedback = Some(match self.change(|document| document.reset(setting.key)) {
            Ok(()) => Feedback::Note(format!("{} reset to default", setting.key)),
            Err(reason) => Feedback::Error(reason),
        });
    }

    /// Makes `edit` and keeps it only when the config gains no validation
    /// problem; otherwise the document is left as it was and the error says why.
    fn change(
        &mut self,
        edit: impl FnOnce(&mut ConfigDocument) -> Result<(), ConfigError>,
    ) -> Result<(), String> {
        let mut candidate = self.document.clone();
        edit(&mut candidate).map_err(|e| e.to_string())?;
        let issues = validation_issues(&candidate);
        let introduced: Vec<&str> = issues
            .iter()
            .filter(|issue| !self.issues.contains(issue))
            .map(String::as_str)
            .collect();
        if !introduced.is_empty() {
            return Err(format!("not applied: {}", introduced.join("; ")));
        }
        self.document = candidate;
        self.issues = issues;
        Ok(())
    }

    /// Returns whether the file was written.
    fn save(&mut self) -> bool {
        let outcome = self.document.save();
        self.feedback = Some(match &outcome {
            Ok(()) => Feedback::Note(SAVED_NOTE.to_string()),
            Err(ConfigError::Validation(_)) => {
                Feedback::Error("not saved: fix the problems listed below first".to_string())
            }
            Err(e) => Feedback::Error(format!("not saved: {e}")),
        });
        outcome.is_ok()
    }

    /// Records the columns the open text box wraps at, so ↑/↓ move by
    /// the rows on screen.
    pub(crate) fn set_text_box_width(&mut self, width: u16) {
        if let PanelMode::Edit(TextEditor {
            shape: EditorShape::Box { wrap_width },
            ..
        }) = &mut self.mode
        {
            *wrap_width = width;
        }
    }
}

/// What a [`TextEditor`] key asks of its owner.
pub(crate) enum EditAction {
    Pending,
    Apply,
    Cancel,
}

/// How a [`TextEditor`] is laid out.
#[derive(Clone, Copy)]
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

    pub(crate) fn is_box(&self) -> bool {
        matches!(self.shape, EditorShape::Box { .. })
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
            (KeyCode::Up, EditorShape::Box { wrap_width }) => self.move_row(wrap_width, false),
            (KeyCode::Down, EditorShape::Box { wrap_width }) => self.move_row(wrap_width, true),
            (KeyCode::Enter, _) => return EditAction::Apply,
            (KeyCode::Esc, _) => return EditAction::Cancel,
            (KeyCode::Char('u'), _) if ctrl => {
                self.text.clear();
                self.cursor = 0;
            }
            (KeyCode::Char(c), _) if !ctrl && !alt => self.insert(c),
            (KeyCode::Backspace, _) => {
                if let Some(previous) = self.previous_boundary() {
                    self.text.drain(previous..self.cursor);
                    self.cursor = previous;
                }
            }
            (KeyCode::Delete, _) => {
                if let Some(next) = self.next_boundary() {
                    self.text.drain(self.cursor..next);
                }
            }
            (KeyCode::Left, _) => self.cursor = self.previous_boundary().unwrap_or(self.cursor),
            (KeyCode::Right, _) => self.cursor = self.next_boundary().unwrap_or(self.cursor),
            (KeyCode::Home, _) => self.cursor = 0,
            (KeyCode::End, _) => self.cursor = self.text.len(),
            _ => {}
        }
        EditAction::Pending
    }

    fn insert(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Moves to the same column of the wrapped row above or below, or to
    /// that row's end when it is shorter.
    fn move_row(&mut self, wrap_width: u16, down: bool) {
        let chars: Vec<char> = self.text.chars().collect();
        let rows = wrap_input(&chars, 0, wrap_width);
        let (row, column) = locate_cursor(&rows, self.cursor_chars());
        let target = if down {
            row.checked_add(1)
        } else {
            row.checked_sub(1)
        };
        if let Some(&(start, end)) = target.and_then(|target| rows.get(target)) {
            let char_offset = start + column.min(end - start);
            self.cursor = self
                .text
                .char_indices()
                .nth(char_offset)
                .map_or(self.text.len(), |(index, _)| index);
        }
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

/// Why `document`'s config would not validate; empty when it would.
fn validation_issues(document: &ConfigDocument) -> Vec<String> {
    match document.config().validate() {
        Err(ConfigError::Validation(issues)) => issues,
        Err(e) => vec![e.to_string()],
        Ok(()) => Vec::new(),
    }
}

/// `value` as the text an edit of it starts from.
pub(crate) fn editable_text(kind: SettingKind, value: &toml::Value) -> String {
    match (kind, value) {
        (_, toml::Value::String(text)) => text.clone(),
        (SettingKind::Decimal, toml::Value::Float(number)) => short_decimal(*number),
        (_, value) => value.to_string(),
    }
}

/// `number` to six decimal places without trailing zeros, hiding the noise
/// of an `f32` default widened to `f64`.
pub(crate) fn short_decimal(number: f64) -> String {
    format!("{number:.6}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn parse_input(kind: SettingKind, text: &str) -> Result<toml::Value, String> {
    match kind {
        SettingKind::Integer => text
            .trim()
            .parse()
            .map(toml::Value::Integer)
            .map_err(|_| "expected a whole number".to_string()),
        SettingKind::Decimal => text
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|number| number.is_finite())
            .map(toml::Value::Float)
            .ok_or_else(|| "expected a number".to_string()),
        SettingKind::Toggle
        | SettingKind::Choice(_)
        | SettingKind::Text
        | SettingKind::Prose
        | SettingKind::List(_)
        | SettingKind::External => Ok(toml::Value::String(text.to_string())),
    }
}

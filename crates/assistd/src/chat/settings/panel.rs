//! Settings panel state and key handling. Changes stay in memory until
//! saved, and only a config that validates is written.

use std::ops::ControlFlow;

use assistd_config::{ConfigDocument, ConfigError};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::catalog::{SECTIONS, Setting, SettingKind};

/// Rows PgUp/PgDn move the selection.
const PAGE_ROWS: usize = 10;
const SAVED_NOTE: &str = "saved; restart the daemon to apply (systemctl --user restart assistd)";

/// One line of the settings list; sections are indexes into [`SECTIONS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Row {
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
    Edit(LineEditor),
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
            advanced_shown: [false; SECTIONS.len()],
            selected: 0,
            mode: PanelMode::Browse,
            feedback: None,
            issues: Vec::new(),
        };
        panel.refresh_issues();
        panel.step_selection(true, 1);
        panel
    }

    pub(crate) fn document(&self) -> &ConfigDocument {
        &self.document
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

    pub(crate) fn rows(&self) -> Vec<Row> {
        SECTIONS
            .iter()
            .enumerate()
            .flat_map(|(index, section)| {
                let shown = self.advanced_shown[index];
                let toggle = (section.advanced_count() > 0).then_some(Row::AdvancedToggle(index));
                std::iter::once(Row::Heading(index))
                    .chain(
                        section
                            .settings
                            .iter()
                            .filter(move |setting| shown || !setting.advanced)
                            .map(move |setting| Row::Setting(index, setting)),
                    )
                    .chain(toggle)
            })
            .collect()
    }

    pub(crate) fn advanced_shown(&self, section: usize) -> bool {
        self.advanced_shown[section]
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
            KeyCode::Up | KeyCode::Char('k') => self.step_selection(false, 1),
            KeyCode::Down | KeyCode::Char('j') => self.step_selection(true, 1),
            KeyCode::PageUp => self.step_selection(false, PAGE_ROWS),
            KeyCode::PageDown => self.step_selection(true, PAGE_ROWS),
            KeyCode::Home => self.step_selection(false, usize::MAX),
            KeyCode::End => self.step_selection(true, usize::MAX),
            KeyCode::Enter | KeyCode::Char(' ') => self.activate_selected(),
            KeyCode::Left => self.cycle_selected(false),
            KeyCode::Right => self.cycle_selected(true),
            KeyCode::Char('a') => self.toggle_advanced(self.selected_row().section()),
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

    fn selected_setting(&self) -> Option<&'static Setting> {
        match self.selected_row() {
            Row::Setting(_, setting) => Some(setting),
            Row::Heading(_) | Row::AdvancedToggle(_) => None,
        }
    }

    /// Moves up to `steps` selectable rows forward or back, stopping at the ends.
    fn step_selection(&mut self, forward: bool, steps: usize) {
        let rows = self.rows();
        let selectable = |index: &usize| !matches!(rows[*index], Row::Heading(_));
        for _ in 0..steps {
            let next = if forward {
                (self.selected + 1..rows.len()).find(selectable)
            } else {
                (0..self.selected).rev().find(selectable)
            };
            match next {
                Some(index) => self.selected = index,
                None => break,
            }
        }
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
            Row::Heading(_) => return,
            Row::Setting(_, setting) => setting,
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
            SettingKind::Integer | SettingKind::Decimal | SettingKind::Text | SettingKind::List => {
                let text = self
                    .current_value(setting.key)
                    .map(|value| editable_text(setting.kind, value))
                    .unwrap_or_default();
                self.mode = PanelMode::Edit(LineEditor::new(text));
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
    /// was refused and stays in the current mode.
    fn apply(&mut self, setting: &Setting, value: &toml::Value) {
        match self.document.set(setting.key, value) {
            Ok(()) => {
                self.mode = PanelMode::Browse;
                self.feedback = None;
                self.refresh_issues();
            }
            Err(e) => self.feedback = Some(Feedback::Error(e.to_string())),
        }
    }

    fn reset_selected(&mut self) {
        let Some(setting) = self.selected_setting() else {
            return;
        };
        if self.document.file_value(setting.key).is_none() {
            return;
        }
        match self.document.reset(setting.key) {
            Ok(()) => {
                self.feedback = Some(Feedback::Note(format!("{} reset to default", setting.key)));
                self.refresh_issues();
            }
            Err(e) => self.feedback = Some(Feedback::Error(e.to_string())),
        }
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

    fn refresh_issues(&mut self) {
        self.issues = match self.document.config().validate() {
            Err(ConfigError::Validation(issues)) => issues,
            Err(e) => vec![e.to_string()],
            Ok(()) => Vec::new(),
        };
    }
}

enum EditAction {
    Pending,
    Apply,
    Cancel,
}

/// A single-line text field with a cursor.
pub(crate) struct LineEditor {
    text: String,
    /// Byte offset, always on a char boundary.
    cursor: usize,
}

impl LineEditor {
    /// Places the cursor at the end of `text`.
    fn new(text: String) -> Self {
        let cursor = text.len();
        Self { text, cursor }
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// Characters before the cursor.
    pub(crate) fn cursor_chars(&self) -> usize {
        self.text[..self.cursor].chars().count()
    }

    fn on_key(&mut self, ev: KeyEvent) -> EditAction {
        let ctrl = ev.modifiers.contains(KeyModifiers::CONTROL);
        match ev.code {
            KeyCode::Enter => return EditAction::Apply,
            KeyCode::Esc => return EditAction::Cancel,
            KeyCode::Char('u') if ctrl => {
                self.text.clear();
                self.cursor = 0;
            }
            KeyCode::Char(c) if !ctrl && !ev.modifiers.contains(KeyModifiers::ALT) => {
                self.text.insert(self.cursor, c);
                self.cursor += c.len_utf8();
            }
            KeyCode::Backspace => {
                if let Some(previous) = self.previous_boundary() {
                    self.text.drain(previous..self.cursor);
                    self.cursor = previous;
                }
            }
            KeyCode::Delete => {
                if let Some(next) = self.next_boundary() {
                    self.text.drain(self.cursor..next);
                }
            }
            KeyCode::Left => self.cursor = self.previous_boundary().unwrap_or(self.cursor),
            KeyCode::Right => self.cursor = self.next_boundary().unwrap_or(self.cursor),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.text.len(),
            _ => {}
        }
        EditAction::Pending
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
        SettingKind::List => match text.parse() {
            Ok(list @ toml::Value::Array(_)) => Ok(list),
            _ => Err(r#"expected a TOML array, e.g. ["a", "b"]"#.to_string()),
        },
        SettingKind::Toggle
        | SettingKind::Choice(_)
        | SettingKind::Text
        | SettingKind::External => Ok(toml::Value::String(text.to_string())),
    }
}

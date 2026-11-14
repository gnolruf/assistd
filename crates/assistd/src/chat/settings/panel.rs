//! Settings panel state and key handling. Changes stay in memory until
//! saved, and only a config that validates is written.

use std::collections::BTreeSet;
use std::ops::ControlFlow;

use assistd_config::{
    ConfigDocument, ConfigError, ConfigIssue, Consumer, SettingKind, ThemeColor, ThemeConfig,
    ThemeName,
};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::super::text_editor::{EditAction, EditorShape, TextEditor};
use super::super::theme::Theme;
use super::catalog::{SECTIONS, Setting, Tab};
use super::color_picker::{ColorPicker, PickerAction};
use super::list_editor::{ListAction, ListEditor};
use super::theme_list::{ThemeList, ThemeListAction};

const THEME_KEY: &str = "chat.theme";
pub(crate) const SAVED_THEMES_KEY: &str = "chat.themes";
/// Rows PgUp/PgDn move the selection.
const PAGE_ROWS: usize = 10;

/// Where the panel runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanelHost {
    /// In the chat, which takes the saved theme when the panel closes.
    Chat,
    /// On its own, as `assistd settings`.
    Standalone,
}

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
    /// Picking a theme color, with the theme as it would look once applied.
    Color {
        picker: ColorPicker,
        preview: Box<Theme>,
    },
    /// Choosing a theme to load, save or delete, with the highlighted one
    /// previewed.
    Themes {
        list: ThemeList,
        preview: Box<Theme>,
    },
    /// Closing with unsaved changes; waits for save, discard or cancel.
    ConfirmClose,
    /// Saving over a file that changed on disk; waits for reload, overwrite
    /// or cancel.
    ConfirmOverwrite,
}

/// The outcome of the last action, shown under the list.
pub(crate) enum Feedback {
    Note(String),
    Error(String),
}

/// The settings list over an open config file.
pub(crate) struct SettingsPanel {
    document: ConfigDocument,
    host: PanelHost,
    tab: Tab,
    /// Per section, whether its advanced settings are listed.
    advanced_shown: [bool; SECTIONS.len()],
    /// Index into [`rows`](Self::rows).
    selected: usize,
    mode: PanelMode,
    feedback: Option<Feedback>,
    /// Why the config would not load, if it would not.
    issues: Vec<ConfigIssue>,
}

impl SettingsPanel {
    pub(crate) fn new(document: ConfigDocument, host: PanelHost) -> Self {
        let mut panel = Self {
            document,
            host,
            tab: Tab::ALL[0],
            advanced_shown: [false; SECTIONS.len()],
            selected: 0,
            mode: PanelMode::Browse,
            feedback: None,
            issues: Vec::new(),
        };
        panel.issues = panel.document.issues();
        panel.select_first();
        panel
    }

    pub(crate) fn document(&self) -> &ConfigDocument {
        &self.document
    }

    /// The theme the document describes, unsaved changes included, or the
    /// color picker's preview while it is open.
    pub(crate) fn theme(&self) -> Theme {
        match &self.mode {
            PanelMode::Color { preview, .. } | PanelMode::Themes { preview, .. } => **preview,
            _ => Theme::new(&self.document.config().chat.theme),
        }
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

    pub(crate) fn issues(&self) -> &[ConfigIssue] {
        &self.issues
    }

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
            PanelMode::Color { .. } => {
                self.on_color_key(ev);
                ControlFlow::Continue(())
            }
            PanelMode::Themes { .. } => {
                self.on_themes_key(ev);
                ControlFlow::Continue(())
            }
            PanelMode::ConfirmClose => self.on_confirm_close_key(ev),
            PanelMode::ConfirmOverwrite => {
                self.on_confirm_overwrite_key(ev);
                ControlFlow::Continue(())
            }
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
            KeyCode::Char('x') => self.delete_unknown_keys(),
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
                let text = editor.text().to_string();
                self.apply_text(&text);
            }
        }
    }

    fn on_color_key(&mut self, ev: KeyEvent) {
        let Some(setting) = self.selected_setting() else {
            return;
        };
        let PanelMode::Color { picker, .. } = &mut self.mode else {
            return;
        };
        match picker.on_key(ev) {
            PickerAction::Pending => {}
            PickerAction::Changed => {
                let candidate = picker.candidate();
                let preview = self.preview_theme(setting, candidate);
                if let PanelMode::Color { preview: shown, .. } = &mut self.mode {
                    **shown = preview;
                }
            }
            PickerAction::Cancel => {
                self.mode = PanelMode::Browse;
                self.feedback = None;
            }
            PickerAction::Apply(color) => {
                self.apply(setting, &toml::Value::from(color.to_string()));
            }
        }
    }

    fn on_themes_key(&mut self, ev: KeyEvent) {
        let PanelMode::Themes { list, preview } = &mut self.mode else {
            return;
        };
        match list.on_key(ev) {
            ThemeListAction::Pending => {}
            ThemeListAction::Changed => {
                let highlighted = list.highlighted().map(|entry| entry.theme);
                **preview = Theme::new(&highlighted.unwrap_or(self.document.config().chat.theme));
            }
            ThemeListAction::Close => {
                self.mode = PanelMode::Browse;
                self.feedback = None;
            }
            ThemeListAction::Load { name, theme } => {
                self.feedback = Some(match self.change(|document| load_theme(document, &theme)) {
                    Ok(()) => {
                        self.mode = PanelMode::Browse;
                        Feedback::Note(format!("loaded the {name} theme"))
                    }
                    Err(reason) => Feedback::Error(reason),
                });
            }
            ThemeListAction::Save(name) => self.save_theme(&name),
            ThemeListAction::Delete(name) => {
                let key = format!("{SAVED_THEMES_KEY}.{name}");
                self.change_themes(|document| document.reset(&key), &name, "deleted");
            }
        }
    }

    /// Saves the current colors as theme `name`, refusing a name that is
    /// malformed or a built-in theme's.
    fn save_theme(&mut self, name: &str) {
        if let Err(refused) = name.parse::<ThemeName>() {
            self.feedback = Some(Feedback::Error(refused.to_string()));
            return;
        }
        let theme = self.document.config().chat.theme;
        self.change_themes(
            |document| store_theme(document, name, &theme),
            name,
            "saved",
        );
    }

    /// Makes `edit` to the saved themes and reopens the list on `name`.
    fn change_themes(
        &mut self,
        edit: impl FnOnce(&mut ConfigDocument) -> Result<(), ConfigError>,
        name: &str,
        verb: &str,
    ) {
        self.feedback = Some(match self.change(edit) {
            Ok(()) => {
                self.mode = self.theme_list_mode(Some(name));
                Feedback::Note(format!("{verb} the {name} theme; Ctrl+S writes it"))
            }
            Err(reason) => Feedback::Error(reason),
        });
    }

    fn theme_list_mode(&self, select: Option<&str>) -> PanelMode {
        let chat = &self.document.config().chat;
        let list = ThemeList::new(&chat.themes, select);
        let shown = list.highlighted().map_or(chat.theme, |entry| entry.theme);
        PanelMode::Themes {
            list,
            preview: Box::new(Theme::new(&shown)),
        }
    }

    /// The name of the built-in or saved theme the current colors match, or
    /// `custom` when none does.
    pub(crate) fn active_theme_name(&self) -> String {
        let chat = &self.document.config().chat;
        ThemeConfig::BUILT_IN
            .iter()
            .map(|(name, theme)| (*name, theme))
            .chain(
                chat.themes
                    .iter()
                    .map(|(name, theme)| (name.as_str(), theme)),
            )
            .find(|(_, theme)| **theme == chat.theme)
            .map_or_else(|| "custom".to_string(), |(name, _)| name.to_string())
    }

    /// The theme with `setting` set to `color`; the current one when the
    /// document refuses it.
    fn preview_theme(&self, setting: &Setting, color: ThemeColor) -> Theme {
        let mut candidate = self.document.clone();
        match candidate.set(setting.key, &toml::Value::from(color.to_string())) {
            Ok(()) => Theme::new(&candidate.config().chat.theme),
            Err(_) => Theme::new(&self.document.config().chat.theme),
        }
    }

    /// The color in effect for `setting`; `default` when it cannot be read.
    pub(crate) fn current_color(&self, setting: &Setting) -> ThemeColor {
        self.current_value(setting.key)
            .and_then(toml::Value::as_str)
            .and_then(|text| text.parse().ok())
            .unwrap_or(ThemeColor::Default)
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
                self.mode = PanelMode::Browse;
                if self.save() {
                    return ControlFlow::Break(());
                }
            }
            KeyCode::Char('d') => return ControlFlow::Break(()),
            KeyCode::Esc => self.mode = PanelMode::Browse,
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn on_confirm_overwrite_key(&mut self, ev: KeyEvent) {
        match ev.code {
            KeyCode::Char('r') => self.reload(),
            KeyCode::Char('o') => {
                self.mode = PanelMode::Browse;
                let outcome = self.document.overwrite();
                self.report_save(&outcome);
            }
            KeyCode::Esc => self.mode = PanelMode::Browse,
            _ => {}
        }
    }

    /// Reopens the file, dropping every change made here; a file that no
    /// longer loads leaves the panel as it was.
    fn reload(&mut self) {
        self.mode = PanelMode::Browse;
        match ConfigDocument::open(self.document.path()) {
            Ok(document) => {
                self.issues = document.issues();
                self.document = document;
                if !self.is_selectable(self.selected_row()) {
                    self.select_first();
                }
                self.feedback = Some(Feedback::Note("reloaded from disk".to_string()));
            }
            Err(e) => self.feedback = Some(Feedback::Error(format!("not reloaded: {e}"))),
        }
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
        match setting.kind() {
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
            SettingKind::Themes => {
                let active = self.active_theme_name();
                self.mode = self.theme_list_mode(Some(&active));
                self.feedback = None;
            }
            SettingKind::Color => {
                self.mode = PanelMode::Color {
                    picker: ColorPicker::new(self.current_color(setting)),
                    preview: Box::new(self.theme()),
                };
                self.feedback = None;
            }
            SettingKind::Integer
            | SettingKind::Decimal
            | SettingKind::Text
            | SettingKind::Prose => {
                let text = self
                    .current_value(setting.key)
                    .map(|value| editable_text(setting.kind(), value))
                    .unwrap_or_default();
                let shape = if setting.kind() == SettingKind::Prose {
                    EditorShape::Box { wrap_width: 0 }
                } else {
                    EditorShape::Line
                };
                self.mode = PanelMode::Edit(TextEditor::new(text, shape));
                self.feedback = None;
            }
        }
    }

    /// Flips a toggle, or moves a choice or color to its next or previous
    /// option; an RGB color moves to the first named one.
    fn cycle_selected(&mut self, forward: bool) {
        let Some(setting) = self.selected_setting() else {
            return;
        };
        let current = self.current_value(setting.key);
        let next = match setting.kind() {
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
            SettingKind::Color => toml::Value::from(
                next_named_color(self.current_color(setting), forward).to_string(),
            ),
            _ => return,
        };
        self.apply(setting, &next);
    }

    fn apply_text(&mut self, text: &str) {
        let Some(setting) = self.selected_setting() else {
            return;
        };
        match parse_input(setting.kind(), text) {
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

    /// Deletes the keys the schema does not know, and says so when some are
    /// left: one inside a list entry is out of the panel's reach.
    fn delete_unknown_keys(&mut self) {
        let unknown = self.document.unknown_keys().to_vec();
        if unknown.is_empty() {
            return;
        }
        let deleted =
            self.change(|document| unknown.iter().try_for_each(|key| document.reset(key)));
        self.feedback = Some(match deleted {
            Ok(()) if self.document.unknown_keys().is_empty() => {
                Feedback::Note("deleted the unknown keys; Ctrl+S writes it".to_string())
            }
            Ok(()) => Feedback::Error(format!(
                "delete the remaining unknown keys in {}",
                self.document.path().display()
            )),
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
        let issues = candidate.issues();
        let introduced: Vec<String> = issues
            .iter()
            .filter(|issue| !self.issues.contains(issue))
            .map(ConfigIssue::to_string)
            .collect();
        if !introduced.is_empty() {
            return Err(format!("not applied: {}", introduced.join("; ")));
        }
        self.document = candidate;
        self.issues = issues;
        Ok(())
    }

    /// Returns whether the file was written; one that changed on disk is
    /// left for [`PanelMode::ConfirmOverwrite`] to settle.
    fn save(&mut self) -> bool {
        let outcome = self.document.save();
        if matches!(outcome, Err(ConfigError::ChangedOnDisk { .. })) {
            self.mode = PanelMode::ConfirmOverwrite;
        }
        self.report_save(&outcome)
    }

    /// Shows how a save went and returns whether the file was written.
    fn report_save(&mut self, outcome: &Result<BTreeSet<Consumer>, ConfigError>) -> bool {
        self.feedback = Some(match outcome {
            Ok(changed) => Feedback::Note(saved_note(changed, self.host)),
            Err(ConfigError::Validation(_)) => {
                Feedback::Error("not saved: fix the problems listed below first".to_string())
            }
            Err(e) => Feedback::Error(format!("not saved: {e}")),
        });
        outcome.is_ok()
    }

    /// Records the open color picker's width, so ↑/↓ move by the rows on screen.
    pub(crate) fn set_color_picker_width(&mut self, width: u16) {
        if let PanelMode::Color { picker, .. } = &mut self.mode {
            picker.set_width(width);
        }
    }

    /// Records the columns the open text box wraps at, so ↑/↓ move by
    /// the rows on screen.
    pub(crate) fn set_text_box_width(&mut self, width: u16) {
        if let PanelMode::Edit(editor) = &mut self.mode {
            editor.set_wrap_width(width);
        }
    }
}

/// `saved`, then whatever must restart before the `changed` consumers' keys apply.
fn saved_note(changed: &BTreeSet<Consumer>, host: PanelHost) -> String {
    let restarts: Vec<&str> = changed
        .iter()
        .filter_map(|consumer| match consumer {
            Consumer::Daemon => Some("the daemon (systemctl --user restart assistd)"),
            Consumer::Tray => Some("the tray"),
            Consumer::Chat if host == PanelHost::Standalone => Some("the chat"),
            Consumer::Chat | Consumer::SettingsPanel => None,
        })
        .collect();
    if restarts.is_empty() {
        "saved".to_string()
    } else {
        format!("saved; restart {} to apply", restarts.join(" and "))
    }
}

/// Sets every `chat.theme` color to `theme`'s, leaving out those that match
/// the default so the file stays short.
fn load_theme(document: &mut ConfigDocument, theme: &ThemeConfig) -> Result<(), ConfigError> {
    for (field, value) in toml::Table::try_from(theme)? {
        let key = format!("{THEME_KEY}.{field}");
        if document.default_value(&key) == Some(&value) {
            document.reset(&key)?;
        } else {
            document.set(&key, &value)?;
        }
    }
    Ok(())
}

/// Writes `theme`'s colors as the saved theme `name`, one key each.
fn store_theme(
    document: &mut ConfigDocument,
    name: &str,
    theme: &ThemeConfig,
) -> Result<(), ConfigError> {
    for (field, value) in toml::Table::try_from(theme)? {
        document.set(&format!("{SAVED_THEMES_KEY}.{name}.{field}"), &value)?;
    }
    Ok(())
}

/// The named color after or before `color`, wrapping; the first for an RGB color.
fn next_named_color(color: ThemeColor, forward: bool) -> ThemeColor {
    let named: Vec<ThemeColor> = ThemeColor::named().collect();
    let next = named.iter().position(|c| *c == color).map_or(0, |index| {
        let offset = if forward { 1 } else { named.len() - 1 };
        (index + offset) % named.len()
    });
    named[next]
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
        | SettingKind::External
        | SettingKind::Color
        | SettingKind::Themes => Ok(toml::Value::String(text.to_string())),
    }
}

#[cfg(test)]
mod tests;

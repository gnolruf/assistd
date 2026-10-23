//! ratatui render for the settings panel.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};

use super::super::markdown::BLEY;
use super::super::ui::{locate_cursor, wrap_input};
use super::catalog::{SECTIONS, Setting, SettingKind, Tab};
use super::list_editor::{EntryEditor, ListState, entry_summary};
use super::panel::{Feedback, PanelMode, Row, SettingsPanel, TextEditor, short_decimal};
use crate::client::terminal_text::escape_controls_single_line;

/// Validation problems listed under the settings; more are counted.
const MAX_ISSUE_ROWS: usize = 3;
const POINTER: &str = "❯ ";
const NO_POINTER: &str = "  ";
/// Columns between the longest label and the values.
const LABEL_GAP: usize = 2;
const RULE: &str = "─";
const CHECKED: &str = "[✓]";
const UNCHECKED: &str = "[ ]";
/// Field labels of a search engine entry, padded to one width.
const ENGINE_LABEL: &str = "engine        ";
const KEY_FILE_LABEL: &str = "API key file  ";
/// Share of the list's rows the text or list box takes while open.
const TEXT_BOX_PERCENT: u16 = 60;
/// Lines kept below the selection before the list scrolls.
const SCROLL_MARGIN: usize = 5;

/// The title and file, the tab bar, the list (over the text or list box while one is open), then
/// the selected key's details, feedback, validation problems and key hints.
/// Mutates only the text box's wrap width.
pub(crate) fn render_settings(frame: &mut Frame<'_>, area: Rect, panel: &mut SettingsPanel) {
    let issue_rows = panel.issues().len().min(MAX_ISSUE_ROWS + 1);
    let [
        title_area,
        tab_area,
        rule_area,
        list_area,
        detail_area,
        feedback_area,
        issues_area,
        hint_area,
    ] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(u16::try_from(issue_rows).unwrap_or(u16::MAX)),
        Constraint::Length(1),
    ])
    .areas(area);
    render_title(frame, title_area, panel);
    render_tabs(frame, tab_area, panel.tab());
    let rule = RULE.repeat(usize::from(area.width));
    frame.render_widget(Paragraph::new(Span::styled(rule, dim())), rule_area);
    let list_area = if has_box(panel.mode()) {
        let [list_area, box_area] =
            Layout::vertical([Constraint::Min(3), Constraint::Percentage(TEXT_BOX_PERCENT)])
                .areas(list_area);
        match panel.mode() {
            PanelMode::List(_) => render_list_box(frame, box_area, panel),
            _ => render_text_box(frame, box_area, panel),
        }
        list_area
    } else {
        list_area
    };
    let panel = &*panel;
    render_list(frame, list_area, panel);
    render_detail(frame, detail_area, panel);
    render_feedback(frame, feedback_area, panel.feedback());
    render_issues(frame, issues_area, panel.issues());
    frame.render_widget(
        Paragraph::new(Span::styled(hint(panel.mode()), dim())),
        hint_area,
    );
}

fn render_title(frame: &mut Frame<'_>, area: Rect, panel: &SettingsPanel) {
    let mut spans = vec![
        Span::styled("Settings", accent().add_modifier(Modifier::BOLD)),
        Span::styled(format!("  {}", panel.document().path().display()), dim()),
    ];
    if panel.document().is_modified() {
        spans.push(Span::styled(
            "  · unsaved changes",
            Style::default().fg(Color::Yellow),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Every tab's title, the current one highlighted.
fn render_tabs(frame: &mut Frame<'_>, area: Rect, current: Tab) {
    let spans: Vec<Span<'static>> = Tab::ALL
        .iter()
        .flat_map(|&tab| {
            let style = if tab == current {
                accent().add_modifier(Modifier::BOLD | Modifier::REVERSED)
            } else {
                dim()
            };
            [
                Span::styled(format!(" {} ", tab.title()), style),
                Span::raw(" "),
            ]
        })
        .collect();
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The window of rows that keeps the selection clear of the list's edges,
/// with the cursor placed in the field being edited.
fn render_list(frame: &mut Frame<'_>, area: Rect, panel: &SettingsPanel) {
    let label_width = longest_label().min(usize::from(area.width) / 2);
    let value_width = usize::from(area.width).saturating_sub(NO_POINTER.len() + label_width);
    let (lines, selected_line) = list_lines(panel, label_width, value_width);
    let visible = usize::from(area.height).max(1);
    let start = scroll_start(selected_line, lines.len(), visible);
    let lines: Vec<Line<'static>> = lines.into_iter().skip(start).take(visible).collect();
    frame.render_widget(Paragraph::new(lines), area);
    if let PanelMode::Edit(editor) = panel.mode()
        && !editor.is_box()
    {
        let (_, cursor_column) = editor_window(editor, value_width);
        let row = selected_line.saturating_sub(start);
        let column = NO_POINTER.len() + label_width + cursor_column;
        frame.set_cursor_position(Position::new(
            area.x
                .saturating_add(u16::try_from(column).unwrap_or(u16::MAX))
                .min(area.right().saturating_sub(1)),
            area.y
                .saturating_add(u16::try_from(row).unwrap_or(u16::MAX)),
        ));
    }
}

/// Every row as listed with a blank line between sections, and the index
/// of the selected row's line.
fn list_lines(
    panel: &SettingsPanel,
    label_width: usize,
    value_width: usize,
) -> (Vec<Line<'static>>, usize) {
    let mut lines = Vec::new();
    let mut selected_line = 0;
    for (index, row) in panel.rows().into_iter().enumerate() {
        if matches!(row, Row::Heading(_)) && !lines.is_empty() {
            lines.push(Line::default());
        }
        let selected = index == panel.selected();
        if selected {
            selected_line = lines.len();
        }
        lines.push(row_line(panel, row, selected, label_width, value_width));
    }
    (lines, selected_line)
}

/// `row` as listed; dim throughout when it is locked.
fn row_line(
    panel: &SettingsPanel,
    row: Row,
    selected: bool,
    label_width: usize,
    value_width: usize,
) -> Line<'static> {
    let line = match row {
        Row::Heading(section) => return heading_line(panel, section, selected),
        Row::Setting(_, setting) => {
            let label = format!("{:label_width$}", setting.label());
            let value = match (selected, panel.mode()) {
                (true, PanelMode::Edit(editor)) if !editor.is_box() => {
                    Span::raw(editor_window(editor, value_width).0)
                }
                _ => value_span(panel, setting, value_width),
            };
            pointed_line(selected, vec![Span::raw(label), value])
        }
        Row::AdvancedToggle(section) => {
            let text = if panel.advanced_shown(section) {
                "▾ hide advanced".to_string()
            } else {
                format!("▸ show {} advanced", SECTIONS[section].advanced_count())
            };
            pointed_line(selected, vec![Span::styled(text, dim())])
        }
    };
    if panel.is_locked(row) {
        line.style(dim())
    } else {
        line
    }
}

/// The section's title followed by its checkbox when it has one; reversed
/// when selected and dim when locked.
fn heading_line(panel: &SettingsPanel, section_index: usize, selected: bool) -> Line<'static> {
    let section = &SECTIONS[section_index];
    let color = if panel.is_locked(Row::Heading(section_index)) {
        dim()
    } else {
        accent()
    };
    let mut style = color.add_modifier(Modifier::BOLD);
    if selected {
        style = style.add_modifier(Modifier::REVERSED);
    }
    let text = match &section.enable {
        Some(enable) => format!("{} {}", section.title, checkbox(panel.is_on(enable.key))),
        None => section.title.to_string(),
    };
    Line::styled(text, style)
}

fn pointed_line(selected: bool, mut spans: Vec<Span<'static>>) -> Line<'static> {
    if selected {
        spans.insert(
            0,
            Span::styled(POINTER, accent().add_modifier(Modifier::BOLD)),
        );
        Line::from(spans).style(Style::default().add_modifier(Modifier::BOLD))
    } else {
        spans.insert(0, Span::raw(NO_POINTER));
        Line::from(spans)
    }
}

/// The value in effect, cut to `width` columns, plain when the file sets
/// it and dim when it is the default.
fn value_span(panel: &SettingsPanel, setting: &Setting, width: usize) -> Span<'static> {
    let from_file = panel.document().file_value(setting.key).is_some();
    let text = panel.current_value(setting.key).map_or_else(
        || "unset".to_string(),
        |value| ellipsized(shown_value(setting.kind, value), width),
    );
    if from_file {
        Span::raw(text)
    } else {
        Span::styled(text, dim())
    }
}

/// The selected setting's full key, description and default.
fn render_detail(frame: &mut Frame<'_>, area: Rect, panel: &SettingsPanel) {
    let Some(setting) = panel.selected_setting() else {
        return;
    };
    let default = panel.document().default_value(setting.key).map_or_else(
        || "unset".to_string(),
        |value| shown_value(setting.kind, value),
    );
    let detail = format!(
        "{} : {} (default value: {default})",
        setting.key,
        setting.description()
    );
    let detail = ellipsized(detail, usize::from(area.width));
    frame.render_widget(Paragraph::new(Span::styled(detail, dim())), area);
}

fn render_feedback(frame: &mut Frame<'_>, area: Rect, feedback: Option<&Feedback>) {
    let span = match feedback {
        Some(Feedback::Note(text)) => Span::styled(text.clone(), Style::default().fg(Color::Green)),
        Some(Feedback::Error(text)) => Span::styled(
            escape_controls_single_line(text).into_owned(),
            Style::default().fg(Color::Red),
        ),
        None => return,
    };
    frame.render_widget(Paragraph::new(span), area);
}

fn render_issues(frame: &mut Frame<'_>, area: Rect, issues: &[String]) {
    let style = Style::default().fg(Color::Yellow);
    let mut lines: Vec<Line<'static>> = issues
        .iter()
        .take(MAX_ISSUE_ROWS)
        .map(|issue| Line::styled(format!("! {issue}"), style))
        .collect();
    if issues.len() > MAX_ISSUE_ROWS {
        let more = issues.len() - MAX_ISSUE_ROWS;
        lines.push(Line::styled(format!("! …and {more} more"), style));
    }
    frame.render_widget(Paragraph::new(lines), area);
}

fn hint(mode: &PanelMode) -> &'static str {
    match mode {
        PanelMode::Browse => {
            "Tab switch tab · ↑/↓ move · Enter edit · ←/→ change · r reset · a advanced · Ctrl+S save · Esc close"
        }
        PanelMode::Edit(editor) if editor.is_box() => {
            "Enter apply · Shift+Enter new line · Esc cancel · Ctrl+U clear"
        }
        PanelMode::Edit(_) => "Enter apply · Esc cancel · Ctrl+U clear",
        PanelMode::List(editor) => match editor.state() {
            ListState::Browse => "↑/↓ move · Enter edit · a add · d delete · Esc done",
            ListState::Edit(entry) if entry.key_file_field().is_some() => {
                "Enter apply · Tab switch field · Esc cancel · Ctrl+U clear"
            }
            ListState::Edit(_) => "Enter apply · Esc cancel · Ctrl+U clear",
            ListState::ConfirmDelete => "y delete · n keep",
        },
        PanelMode::ConfirmClose => {
            "unsaved changes: s save and close · d discard · Esc keep editing"
        }
    }
}

/// Whether the mode draws a box under the list.
fn has_box(mode: &PanelMode) -> bool {
    match mode {
        PanelMode::Edit(editor) => editor.is_box(),
        PanelMode::List(_) => true,
        PanelMode::Browse | PanelMode::ConfirmClose => false,
    }
}

/// The open editor when it is a text box.
fn open_text_box(panel: &SettingsPanel) -> Option<&TextEditor> {
    match panel.mode() {
        PanelMode::Edit(editor) if editor.is_box() => Some(editor),
        _ => None,
    }
}

/// A rounded, accent-bordered box titled with the selected setting's label.
fn box_block(panel: &SettingsPanel) -> Block<'static> {
    let label = panel
        .selected_setting()
        .map_or("", |setting| setting.label());
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(accent())
        .title(Span::styled(format!(" {label} "), accent()))
}

/// The edited text wrapped in a bordered box, scrolled to keep the cursor
/// in view.
fn render_text_box(frame: &mut Frame<'_>, area: Rect, panel: &mut SettingsPanel) {
    let block = box_block(panel);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    panel.set_text_box_width(inner.width);
    let Some(editor) = open_text_box(panel) else {
        return;
    };
    let chars: Vec<char> = editor.text().chars().collect();
    let rows = wrap_input(&chars, 0, inner.width);
    let (cursor_row, cursor_column) = locate_cursor(&rows, editor.cursor_chars());
    let visible = usize::from(inner.height).max(1);
    let start = (cursor_row + 1).saturating_sub(visible);
    let lines: Vec<Line<'static>> = rows
        .iter()
        .skip(start)
        .take(visible)
        .map(|&(row_start, row_end)| Line::raw(visible_chars(&chars[row_start..row_end])))
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
    frame.set_cursor_position(Position::new(
        inner
            .x
            .saturating_add(u16::try_from(cursor_column).unwrap_or(u16::MAX))
            .min(inner.right().saturating_sub(1)),
        inner
            .y
            .saturating_add(u16::try_from(cursor_row - start).unwrap_or(u16::MAX)),
    ));
}

/// The list setting's entries then an add row, scrolled to keep the
/// selection in view, with the open entry's fields in its place.
fn render_list_box(frame: &mut Frame<'_>, area: Rect, panel: &SettingsPanel) {
    let (PanelMode::List(editor), Some(setting)) = (panel.mode(), panel.selected_setting()) else {
        return;
    };
    let block = box_block(panel);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let width = usize::from(inner.width).saturating_sub(NO_POINTER.len());
    let entries = panel.list_entries(setting.key);
    let mut lines = Vec::new();
    let mut cursor = None;
    let mut selected_line = 0;
    for index in 0..=entries.len() {
        let selected = index == editor.selected();
        if selected {
            selected_line = lines.len();
        }
        match (selected, editor.state(), entries.get(index)) {
            (true, ListState::Edit(entry), _) => {
                cursor = Some(push_entry_lines(&mut lines, entry, width));
                selected_line = lines.len() - 1;
            }
            (true, ListState::ConfirmDelete, Some(entry)) => {
                lines.push(confirm_delete_line(entry, width));
            }
            (_, _, entry) => lines.push(list_box_line(entry, selected, width)),
        }
    }
    let visible = usize::from(inner.height).max(1);
    let start = scroll_start(selected_line, lines.len(), visible);
    let lines: Vec<Line<'static>> = lines.into_iter().skip(start).take(visible).collect();
    frame.render_widget(Paragraph::new(lines), inner);
    if let Some((row, column)) = cursor {
        frame.set_cursor_position(Position::new(
            inner
                .x
                .saturating_add(u16::try_from(column).unwrap_or(u16::MAX))
                .min(inner.right().saturating_sub(1)),
            inner
                .y
                .saturating_add(u16::try_from(row.saturating_sub(start)).unwrap_or(u16::MAX)),
        ));
    }
}

/// One entry, or the add row when `entry` is `None`.
fn list_box_line(entry: Option<&toml::Value>, selected: bool, width: usize) -> Line<'static> {
    let span = match entry {
        Some(entry) => Span::raw(ellipsized(
            escape_controls_single_line(&entry_summary(entry)).into_owned(),
            width,
        )),
        None => Span::styled("+ add entry", dim()),
    };
    pointed_line(selected, vec![span])
}

/// The entry awaiting deletion, asking y or n.
fn confirm_delete_line(entry: &toml::Value, width: usize) -> Line<'static> {
    let summary = escape_controls_single_line(&entry_summary(entry)).into_owned();
    let question = format!("delete {summary}? (y/n)");
    pointed_line(
        true,
        vec![Span::styled(
            ellipsized(question, width),
            Style::default().fg(Color::Yellow),
        )],
    )
}

/// Pushes the open entry's fields, one per line with the pointer on the
/// first, and returns the focused field's cursor as (line, column).
fn push_entry_lines(
    lines: &mut Vec<Line<'static>>,
    entry: &EntryEditor,
    width: usize,
) -> (usize, usize) {
    let Some(key_file) = entry.key_file_field() else {
        let (text, column) = editor_window(entry.value_field(), width);
        lines.push(pointed_line(true, vec![Span::raw(text)]));
        return (lines.len() - 1, NO_POINTER.len() + column);
    };
    let fields = [
        (ENGINE_LABEL, entry.value_field(), !entry.key_file_focused()),
        (KEY_FILE_LABEL, key_file, entry.key_file_focused()),
    ];
    let mut cursor = (0, 0);
    let field_width = width.saturating_sub(ENGINE_LABEL.len());
    for (index, (label, field, focused)) in fields.into_iter().enumerate() {
        let (text, column) = editor_window(field, field_width);
        let label_style = if focused { accent() } else { dim() };
        let spans = vec![Span::styled(label, label_style), Span::raw(text)];
        if focused {
            cursor = (lines.len(), NO_POINTER.len() + label.len() + column);
        }
        lines.push(pointed_line(index == 0, spans));
    }
    cursor
}

/// `chars` with control characters shown as `↵` so they cannot move the
/// terminal's cursor.
fn visible_chars(chars: &[char]) -> String {
    chars
        .iter()
        .map(|&c| if c.is_control() { '↵' } else { c })
        .collect()
}

/// `value` as listed: toggles as checkboxes, structured values as a count.
fn shown_value(kind: SettingKind, value: &toml::Value) -> String {
    match (kind, value) {
        (SettingKind::Toggle, toml::Value::Boolean(on)) => checkbox(*on).to_string(),
        (SettingKind::List(_), toml::Value::Array(entries)) if entries.is_empty() => {
            "(none)".to_string()
        }
        (SettingKind::List(_), toml::Value::Array(entries)) => entries
            .iter()
            .map(|entry| escape_controls_single_line(&entry_summary(entry)).into_owned())
            .collect::<Vec<_>>()
            .join(", "),
        (SettingKind::External, toml::Value::Array(entries)) => match entries.len() {
            1 => "1 entry".to_string(),
            count => format!("{count} entries"),
        },
        (_, toml::Value::String(text)) if text.is_empty() => "(empty)".to_string(),
        (_, toml::Value::String(text)) => escape_controls_single_line(text).into_owned(),
        (SettingKind::Decimal, toml::Value::Float(number)) => short_decimal(*number),
        (_, value) => value.to_string(),
    }
}

/// `text` cut to `width` characters, ending in `…` when anything was cut.
fn ellipsized(text: String, width: usize) -> String {
    if text.chars().count() <= width {
        return text;
    }
    let mut cut: String = text.chars().take(width.saturating_sub(1)).collect();
    cut.push('…');
    cut
}

fn checkbox(on: bool) -> &'static str {
    if on { CHECKED } else { UNCHECKED }
}

/// The slice of the editor's text that fits `width` columns with the cursor
/// in view, and the cursor's column within it.
fn editor_window(editor: &TextEditor, width: usize) -> (String, usize) {
    let cursor = editor.cursor_chars();
    let start = cursor.saturating_sub(width.saturating_sub(1));
    let chars: Vec<char> = editor.text().chars().skip(start).take(width).collect();
    (visible_chars(&chars), cursor - start)
}

fn longest_label() -> usize {
    SECTIONS
        .iter()
        .flat_map(|section| section.settings)
        .map(|setting| setting.label().chars().count())
        .max()
        .unwrap_or(0)
        + LABEL_GAP
}

/// First line of a `visible`-line window that keeps [`SCROLL_MARGIN`] lines
/// below `selected` in view, or as many as half the window allows.
fn scroll_start(selected: usize, total: usize, visible: usize) -> usize {
    if total <= visible {
        return 0;
    }
    let margin = SCROLL_MARGIN.min(visible.saturating_sub(1) / 2);
    (selected + 1 + margin)
        .saturating_sub(visible)
        .min(total - visible)
}

fn accent() -> Style {
    Style::default().fg(BLEY)
}

fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

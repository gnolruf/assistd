//! ratatui render for the settings panel.

use assistd_config::ThemeColor;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};

use super::super::theme::{Theme, color};
use super::super::ui::{locate_cursor, wrap_input};
use super::catalog::{SECTIONS, Setting, SettingKind, Tab};
use super::color_picker::{ColorPicker, PALETTE_CELL_WIDTH, PickerFocus};
use super::list_editor::{EntryEditor, ListEditor, ListState, entry_summary};
use super::panel::{Feedback, PanelMode, Row, SettingsPanel, TextEditor, short_decimal};
use super::preview::render_theme_preview;
use super::theme_list::{ThemeEntry, ThemeListState};
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
/// Share of the list's rows the Theme tab gives its preview.
const PREVIEW_PERCENT: u16 = 50;
/// Share of the list's rows the text or list box takes while open.
const TEXT_BOX_PERCENT: u16 = 60;
/// Two cells of solid color.
const SWATCH: &str = "██";
/// The custom color's swatch, wider so it reads beside the channels.
const WIDE_SWATCH: &str = "████████";
const CHANNEL_NAMES: [&str; 3] = ["red  ", "green", "blue "];
/// Pure red, green and blue, drawing each channel's gauge.
const CHANNEL_COLORS: [Color; 3] = [
    Color::Rgb(255, 0, 0),
    Color::Rgb(0, 255, 0),
    Color::Rgb(0, 0, 255),
];
const GAUGE_FULL: &str = "█";
const GAUGE_EMPTY: &str = "░";
/// Columns a theme's name is padded to in the theme list.
const THEME_NAME_WIDTH: usize = 16;
const SAVE_AS: &str = "save as: ";
/// Lines kept below the selection before the list scrolls.
const SCROLL_MARGIN: usize = 5;

/// The title, tab bar and list (over the text or list box while one is
/// open), then the selected key's details, feedback, problems and key hints.
/// Mutates only the text box's wrap width.
pub(crate) fn render_settings(frame: &mut Frame<'_>, area: Rect, panel: &mut SettingsPanel) {
    let theme = &panel.theme();
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
    render_title(frame, title_area, panel, theme);
    render_tabs(frame, tab_area, panel.tab(), theme);
    let rule = RULE.repeat(usize::from(area.width));
    frame.render_widget(Paragraph::new(Span::styled(rule, meta(theme))), rule_area);
    let list_area = if panel.tab() == Tab::Theme {
        let [list_area, preview_area] =
            Layout::vertical([Constraint::Min(3), Constraint::Percentage(PREVIEW_PERCENT)])
                .areas(list_area);
        if let Some(setting) = panel.selected_setting() {
            render_theme_preview(frame, preview_area, theme, setting.key);
        }
        list_area
    } else {
        list_area
    };
    let list_area = if has_box(panel.mode()) {
        let [list_area, box_area] =
            Layout::vertical([Constraint::Min(3), Constraint::Percentage(TEXT_BOX_PERCENT)])
                .areas(list_area);
        match panel.mode() {
            PanelMode::List(_) => render_list_box(frame, box_area, panel, theme),
            PanelMode::Color { .. } => render_color_box(frame, box_area, panel, theme),
            PanelMode::Themes { .. } => render_theme_box(frame, box_area, panel, theme),
            _ => render_text_box(frame, box_area, panel, theme),
        }
        list_area
    } else {
        list_area
    };
    let panel = &*panel;
    render_list(frame, list_area, panel, theme);
    render_detail(frame, detail_area, panel, theme);
    render_feedback(frame, feedback_area, panel.feedback(), theme);
    render_issues(frame, issues_area, panel.issues(), theme);
    frame.render_widget(
        Paragraph::new(Span::styled(hint(panel.mode()), meta(theme))),
        hint_area,
    );
}

fn render_title(frame: &mut Frame<'_>, area: Rect, panel: &SettingsPanel, theme: &Theme) {
    let mut spans = vec![
        Span::styled("Settings", accent(theme).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("  {}", panel.document().path().display()),
            meta(theme),
        ),
    ];
    if panel.document().is_modified() {
        spans.push(Span::styled(
            "  · unsaved changes",
            Style::default().fg(theme.warning),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_tabs(frame: &mut Frame<'_>, area: Rect, current: Tab, theme: &Theme) {
    let spans: Vec<Span<'static>> = Tab::ALL
        .iter()
        .flat_map(|&tab| {
            let style = if tab == current {
                accent(theme).add_modifier(Modifier::BOLD | Modifier::REVERSED)
            } else {
                meta(theme)
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
fn render_list(frame: &mut Frame<'_>, area: Rect, panel: &SettingsPanel, theme: &Theme) {
    let label_width = longest_label().min(usize::from(area.width) / 2);
    let value_width = usize::from(area.width).saturating_sub(NO_POINTER.len() + label_width);
    let (lines, selected_line) = list_lines(panel, label_width, value_width, theme);
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
        place_cursor(frame, area, row, column);
    }
}

/// Every row as listed with a blank line between sections, and the index
/// of the selected row's line.
fn list_lines(
    panel: &SettingsPanel,
    label_width: usize,
    value_width: usize,
    theme: &Theme,
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
        lines.push(row_line(
            panel,
            row,
            selected,
            label_width,
            value_width,
            theme,
        ));
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
    theme: &Theme,
) -> Line<'static> {
    let line = match row {
        Row::Heading(section) => return heading_line(panel, section, selected, theme),
        Row::Setting(_, setting) => {
            let label = format!("{:label_width$}", setting.label());
            let value = match (selected, panel.mode()) {
                (true, PanelMode::Edit(editor)) if !editor.is_box() => {
                    Span::raw(editor_window(editor, value_width).0)
                }
                _ => value_span(panel, setting, value_width, theme),
            };
            pointed_line(selected, theme, vec![Span::raw(label), value])
        }
        Row::AdvancedToggle(section) => {
            let text = if panel.advanced_shown(section) {
                "▾ hide advanced".to_string()
            } else {
                format!("▸ show {} advanced", SECTIONS[section].advanced_count())
            };
            pointed_line(selected, theme, vec![Span::styled(text, meta(theme))])
        }
    };
    if panel.is_locked(row) {
        line.style(meta(theme))
    } else {
        line
    }
}

/// The section's title followed by its checkbox when it has one; reversed
/// when selected and dim when locked.
fn heading_line(
    panel: &SettingsPanel,
    section_index: usize,
    selected: bool,
    theme: &Theme,
) -> Line<'static> {
    let section = &SECTIONS[section_index];
    let color = if panel.is_locked(Row::Heading(section_index)) {
        meta(theme)
    } else {
        accent(theme)
    };
    let mut style = color.add_modifier(Modifier::BOLD);
    if selected {
        style = style.patch(theme.selection());
    }
    let text = match &section.enable {
        Some(enable) => format!("{} {}", section.title, checkbox(panel.is_on(enable.key))),
        None => section.title.to_string(),
    };
    Line::styled(text, style)
}

fn pointed_line(selected: bool, theme: &Theme, mut spans: Vec<Span<'static>>) -> Line<'static> {
    if selected {
        spans.insert(
            0,
            Span::styled(POINTER, accent(theme).add_modifier(Modifier::BOLD)),
        );
        Line::from(spans).style(Style::default().add_modifier(Modifier::BOLD))
    } else {
        spans.insert(0, Span::raw(NO_POINTER));
        Line::from(spans)
    }
}

/// The value in effect, cut to `width` columns, plain when the file sets
/// it and dim when it is the default.
fn value_span(
    panel: &SettingsPanel,
    setting: &Setting,
    width: usize,
    theme: &Theme,
) -> Span<'static> {
    if setting.kind == SettingKind::Themes {
        return Span::styled(ellipsized(panel.active_theme_name(), width), accent(theme));
    }
    if setting.kind == SettingKind::Color {
        let shown = panel.current_color(setting);
        return Span::styled(
            ellipsized(format!("{SWATCH} {shown}"), width),
            Style::default().fg(color(shown)),
        );
    }
    let from_file = panel.document().file_value(setting.key).is_some();
    let text = panel.current_value(setting.key).map_or_else(
        || "unset".to_string(),
        |value| ellipsized(shown_value(setting.kind, value), width),
    );
    if from_file {
        Span::raw(text)
    } else {
        Span::styled(text, meta(theme))
    }
}

/// The selected setting's full key, description and default.
fn render_detail(frame: &mut Frame<'_>, area: Rect, panel: &SettingsPanel, theme: &Theme) {
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
    frame.render_widget(Paragraph::new(Span::styled(detail, meta(theme))), area);
}

fn render_feedback(frame: &mut Frame<'_>, area: Rect, feedback: Option<&Feedback>, theme: &Theme) {
    let span = match feedback {
        Some(Feedback::Note(text)) => {
            Span::styled(text.clone(), Style::default().fg(theme.success))
        }
        Some(Feedback::Error(text)) => Span::styled(
            escape_controls_single_line(text).into_owned(),
            Style::default().fg(theme.error),
        ),
        None => return,
    };
    frame.render_widget(Paragraph::new(span), area);
}

fn render_issues(frame: &mut Frame<'_>, area: Rect, issues: &[String], theme: &Theme) {
    let style = Style::default().fg(theme.warning);
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
        PanelMode::List(editor) => list_hint(editor.state()),
        PanelMode::Color { picker, .. } => match (picker.hex_input(), picker.focus()) {
            (Some(_), _) => "type 6 hex digits · Backspace erase · Esc stop typing",
            (None, PickerFocus::Palette(_)) => {
                "arrows pick · Tab channels · # type hex · Enter apply · Esc cancel"
            }
            (None, PickerFocus::Channel(_)) => {
                "↑/↓ channel · ←/→ adjust (Shift ×16) · Tab palette · # type hex · Enter apply · Esc cancel"
            }
        },
        PanelMode::Themes { list, .. } => match list.state() {
            ThemeListState::Browse => {
                "↑/↓ preview · Enter load or save · d delete saved · Esc close"
            }
            ThemeListState::Naming(_) => "type a name · Enter save · Esc cancel · Ctrl+U clear",
            ThemeListState::ConfirmDelete => "y delete · n keep",
        },
        PanelMode::ConfirmClose => {
            "unsaved changes: s save and close · d discard · Esc keep editing"
        }
    }
}

fn has_box(mode: &PanelMode) -> bool {
    match mode {
        PanelMode::Edit(editor) => editor.is_box(),
        PanelMode::List(_) | PanelMode::Color { .. } | PanelMode::Themes { .. } => true,
        PanelMode::Browse | PanelMode::ConfirmClose => false,
    }
}

/// A rounded, accent-bordered box titled with the selected setting's label.
fn box_block(panel: &SettingsPanel, theme: &Theme) -> Block<'static> {
    let label = panel
        .selected_setting()
        .map_or("", |setting| setting.label());
    titled_box(label, theme)
}

/// A rounded, accent-bordered box titled `title`.
pub(crate) fn titled_box(title: &str, theme: &Theme) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(accent(theme))
        .title(Span::styled(format!(" {title} "), accent(theme)))
}

/// The edited text wrapped in a bordered box, scrolled to keep the cursor
/// in view.
fn render_text_box(frame: &mut Frame<'_>, area: Rect, panel: &mut SettingsPanel, theme: &Theme) {
    let block = box_block(panel, theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    panel.set_text_box_width(inner.width);
    if let PanelMode::Edit(editor) = panel.mode() {
        render_wrapped_editor(frame, inner, editor, true);
    }
}

/// `editor`'s text wrapped into `area`, scrolled to keep the cursor in view, with the
/// terminal cursor placed when `show_cursor`.
pub(crate) fn render_wrapped_editor(
    frame: &mut Frame<'_>,
    area: Rect,
    editor: &TextEditor,
    show_cursor: bool,
) {
    let chars: Vec<char> = editor.text().chars().collect();
    let rows = wrap_input(&chars, 0, area.width);
    let (cursor_row, cursor_column) = locate_cursor(&rows, editor.cursor_chars());
    let visible = usize::from(area.height).max(1);
    let start = (cursor_row + 1).saturating_sub(visible);
    let lines: Vec<Line<'static>> = rows
        .iter()
        .skip(start)
        .take(visible)
        .map(|&(row_start, row_end)| Line::raw(visible_chars(&chars[row_start..row_end])))
        .collect();
    frame.render_widget(Paragraph::new(lines), area);
    if show_cursor {
        place_cursor(frame, area, cursor_row - start, cursor_column);
    }
}

/// The list setting's entries in the box titled with its label.
fn render_list_box(frame: &mut Frame<'_>, area: Rect, panel: &SettingsPanel, theme: &Theme) {
    let (PanelMode::List(editor), Some(setting)) = (panel.mode(), panel.selected_setting()) else {
        return;
    };
    let entries = panel.list_entries(setting.key);
    render_list_editor(frame, area, editor, entries, setting.label(), theme);
}

/// `entries` then an add row in a box titled `title`, scrolled to keep the selection in
/// view, with the open entry's fields in its place.
pub(crate) fn render_list_editor(
    frame: &mut Frame<'_>,
    area: Rect,
    editor: &ListEditor,
    entries: &[toml::Value],
    title: &str,
    theme: &Theme,
) {
    let block = titled_box(title, theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let width = usize::from(inner.width).saturating_sub(NO_POINTER.len());
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
                cursor = Some(push_entry_lines(&mut lines, entry, width, theme));
                selected_line = lines.len() - 1;
            }
            (true, ListState::ConfirmDelete, Some(entry)) => {
                lines.push(confirm_delete_line(entry, width, theme));
            }
            (_, _, entry) => lines.push(list_box_line(entry, selected, width, theme)),
        }
    }
    let visible = usize::from(inner.height).max(1);
    let start = scroll_start(selected_line, lines.len(), visible);
    let lines: Vec<Line<'static>> = lines.into_iter().skip(start).take(visible).collect();
    frame.render_widget(Paragraph::new(lines), inner);
    if let Some((row, column)) = cursor {
        place_cursor(frame, inner, row.saturating_sub(start), column);
    }
}

/// The keys the list box takes in `state`.
pub(crate) fn list_hint(state: &ListState) -> &'static str {
    match state {
        ListState::Browse => "↑/↓ move · Enter edit · a add · d delete · Esc done",
        ListState::Edit(entry) if entry.key_file_field().is_some() => {
            "Enter apply · Tab switch field · Esc cancel · Ctrl+U clear"
        }
        ListState::Edit(_) => "Enter apply · Esc cancel · Ctrl+U clear",
        ListState::ConfirmDelete => "y delete · n keep",
    }
}

/// Every theme with a strip of its colors, then the row that saves the
/// current colors, scrolled to keep the selection in view.
fn render_theme_box(frame: &mut Frame<'_>, area: Rect, panel: &SettingsPanel, theme: &Theme) {
    let PanelMode::Themes { list, .. } = panel.mode() else {
        return;
    };
    let block = box_block(panel, theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let width = usize::from(inner.width).saturating_sub(NO_POINTER.len());
    let deleting = matches!(list.state(), ThemeListState::ConfirmDelete);
    let mut lines: Vec<Line<'static>> = list
        .entries()
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let selected = index == list.selected();
            if selected && deleting {
                confirm_delete_line(&toml::Value::from(entry.name.clone()), width, theme)
            } else {
                pointed_line(selected, theme, theme_entry_spans(entry, theme))
            }
        })
        .collect();
    let mut cursor = None;
    let on_save_row = list.selected() == list.entries().len();
    match list.state() {
        ThemeListState::Naming(editor) => {
            let (text, column) = editor_window(editor, width.saturating_sub(SAVE_AS.len()));
            cursor = Some((lines.len(), NO_POINTER.len() + SAVE_AS.len() + column));
            let spans = vec![Span::styled(SAVE_AS, accent(theme)), Span::raw(text)];
            lines.push(pointed_line(true, theme, spans));
        }
        _ => lines.push(pointed_line(
            on_save_row,
            theme,
            vec![Span::styled("+ save current colors as…", meta(theme))],
        )),
    }
    let visible = usize::from(inner.height).max(1);
    let start = scroll_start(list.selected(), lines.len(), visible);
    let lines: Vec<Line<'static>> = lines.into_iter().skip(start).take(visible).collect();
    frame.render_widget(Paragraph::new(lines), inner);
    if let Some((row, column)) = cursor {
        place_cursor(frame, inner, row.saturating_sub(start), column);
    }
}

/// The theme's name, a swatch of each of its main colors, and whether it
/// ships with assistd.
fn theme_entry_spans(entry: &ThemeEntry, theme: &Theme) -> Vec<Span<'static>> {
    let colors = [
        entry.theme.accent,
        entry.theme.assistant_text,
        entry.theme.code,
        entry.theme.success,
        entry.theme.error,
        entry.theme.warning,
        entry.theme.link,
    ];
    let mut spans = vec![Span::raw(format!("{:THEME_NAME_WIDTH$}", entry.name))];
    spans.extend(
        colors
            .into_iter()
            .map(|named| Span::styled(SWATCH, Style::default().fg(color(named)))),
    );
    if entry.built_in {
        spans.push(Span::styled("  built-in", meta(theme)));
    }
    spans
}

/// The named colors as a grid of swatches, then the custom color and its
/// red, green and blue channels.
fn render_color_box(frame: &mut Frame<'_>, area: Rect, panel: &mut SettingsPanel, theme: &Theme) {
    let block = box_block(panel, theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    panel.set_color_picker_width(inner.width);
    let PanelMode::Color { picker, .. } = panel.mode() else {
        return;
    };
    let mut lines = vec![Line::styled("Named", meta(theme))];
    let columns = (usize::from(inner.width) / PALETTE_CELL_WIDTH).max(1);
    lines.extend(
        picker
            .palette()
            .chunks(columns)
            .enumerate()
            .map(|(row, colors)| palette_row(picker, row * columns, colors, theme)),
    );
    lines.push(Line::default());
    lines.push(custom_color_line(picker, theme));
    let gauge_width = usize::from(inner.width).saturating_sub(NO_POINTER.len() + 16);
    lines.extend((0..3).map(|channel| channel_line(picker, channel, gauge_width, theme)));
    frame.render_widget(Paragraph::new(lines), inner);
}

/// One row of palette cells, the first at index `first`.
fn palette_row(
    picker: &ColorPicker,
    first: usize,
    colors: &[ThemeColor],
    theme: &Theme,
) -> Line<'static> {
    let spans = colors.iter().enumerate().flat_map(|(offset, &named)| {
        let focused = picker.focus() == PickerFocus::Palette(first + offset);
        let (pointer, name_style) = if focused {
            (POINTER, Style::default().add_modifier(Modifier::BOLD))
        } else {
            (NO_POINTER, Style::default())
        };
        let name_width = PALETTE_CELL_WIDTH - POINTER.chars().count() - SWATCH.chars().count() - 1;
        [
            Span::styled(pointer, accent(theme).add_modifier(Modifier::BOLD)),
            Span::styled(SWATCH, Style::default().fg(color(named))),
            Span::styled(format!(" {:name_width$}", named.to_string()), name_style),
        ]
    });
    Line::from(spans.collect::<Vec<_>>())
}

/// The channels' color as a swatch and `#rrggbb`, or the hex being typed.
fn custom_color_line(picker: &ColorPicker, theme: &Theme) -> Line<'static> {
    let mixed = picker.mixed();
    let value = match picker.hex_input() {
        Some(typed) => Span::styled(format!("#{typed}▏"), accent(theme)),
        None => Span::raw(mixed.to_string()),
    };
    Line::from(vec![
        Span::styled("Custom  ", meta(theme)),
        Span::styled(WIDE_SWATCH, Style::default().fg(color(mixed))),
        Span::raw(" "),
        value,
    ])
}

/// A channel's name, value and a gauge of it `width` columns wide.
fn channel_line(
    picker: &ColorPicker,
    channel: usize,
    width: usize,
    theme: &Theme,
) -> Line<'static> {
    let value = picker.rgb()[channel];
    let filled = usize::from(value) * width / usize::from(u8::MAX);
    let spans = vec![
        Span::raw(format!("{} ◀ {value:>3} ▶ ", CHANNEL_NAMES[channel])),
        Span::styled(
            GAUGE_FULL.repeat(filled),
            Style::default().fg(CHANNEL_COLORS[channel]),
        ),
        Span::styled(GAUGE_EMPTY.repeat(width - filled), meta(theme)),
    ];
    pointed_line(
        picker.focus() == PickerFocus::Channel(channel),
        theme,
        spans,
    )
}

/// One entry, or the add row when `entry` is `None`.
fn list_box_line(
    entry: Option<&toml::Value>,
    selected: bool,
    width: usize,
    theme: &Theme,
) -> Line<'static> {
    let span = match entry {
        Some(entry) => Span::raw(ellipsized(
            escape_controls_single_line(&entry_summary(entry)).into_owned(),
            width,
        )),
        None => Span::styled("+ add entry", meta(theme)),
    };
    pointed_line(selected, theme, vec![span])
}

/// The entry awaiting deletion, asking y or n.
fn confirm_delete_line(entry: &toml::Value, width: usize, theme: &Theme) -> Line<'static> {
    let summary = escape_controls_single_line(&entry_summary(entry)).into_owned();
    let question = format!("delete {summary}? (y/n)");
    pointed_line(
        true,
        theme,
        vec![Span::styled(
            ellipsized(question, width),
            Style::default().fg(theme.warning),
        )],
    )
}

/// Pushes the open entry's fields, one per line with the pointer on the
/// first, and returns the focused field's cursor as (line, column).
fn push_entry_lines(
    lines: &mut Vec<Line<'static>>,
    entry: &EntryEditor,
    width: usize,
    theme: &Theme,
) -> (usize, usize) {
    let Some(key_file) = entry.key_file_field() else {
        let (text, column) = editor_window(entry.value_field(), width);
        lines.push(pointed_line(true, theme, vec![Span::raw(text)]));
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
        let label_style = if focused { accent(theme) } else { meta(theme) };
        let spans = vec![Span::styled(label, label_style), Span::raw(text)];
        if focused {
            cursor = (lines.len(), NO_POINTER.len() + label.len() + column);
        }
        lines.push(pointed_line(index == 0, theme, spans));
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
        (SettingKind::Themes, toml::Value::Table(saved)) if saved.is_empty() => {
            "none saved".to_string()
        }
        (SettingKind::Themes, toml::Value::Table(saved)) => {
            saved.keys().cloned().collect::<Vec<_>>().join(", ")
        }
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
pub(crate) fn editor_window(editor: &TextEditor, width: usize) -> (String, usize) {
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

/// Puts the terminal cursor `row` lines and `column` columns into `area`,
/// clamped to it.
pub(crate) fn place_cursor(frame: &mut Frame<'_>, area: Rect, row: usize, column: usize) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let offset = |value: usize| u16::try_from(value).unwrap_or(u16::MAX);
    frame.set_cursor_position(Position::new(
        area.x
            .saturating_add(offset(column))
            .min(area.right().saturating_sub(1)),
        area.y
            .saturating_add(offset(row))
            .min(area.bottom().saturating_sub(1)),
    ));
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

fn accent(theme: &Theme) -> Style {
    Style::default().fg(theme.accent)
}

fn meta(theme: &Theme) -> Style {
    Style::default().fg(theme.status_text)
}

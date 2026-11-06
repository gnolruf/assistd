//! The `/tasks` panel: the task list, and the editor with its trigger form and prompt box.

use assistd_ipc::{
    FinishOutcome, HostName, MissedFire, PermissionMode, TaskEvent, TaskId, TaskTrigger,
};
use chrono::{DateTime, Local, Utc};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Paragraph};

use super::super::settings::{TextEditor, list_hint, render_list_editor};
use super::super::tasks::{
    AT_FORMAT, EditorMode, EditorRow, ListedTask, Message, RefineNote, TaskEditor, TasksPanel,
    TriggerCheck, TriggerForm, TriggerRow, UpstreamChoice,
};
use super::super::theme::Theme;
use super::{locate_cursor, picker_scroll_start, render_rule, run_outcome, wrap_input};
use crate::client::terminal_text::escape_controls_single_line;

/// Rows each listed task takes: its name and its details.
const TASK_ENTRY_ROWS: usize = 2;
const DETAIL_INDENT: &str = "    ";
const POINTER: &str = "❯ ";
const NO_POINTER: &str = "  ";
/// Columns the editor's and form's labels are padded to.
const LABEL_WIDTH: usize = 14;
/// Rows the prompt box or trigger form takes while open.
const BOX_ROWS: u16 = 10;
const FIRE_FORMAT: &str = "%a %m-%d %H:%M";
const NEXT_FIRE_FORMAT: &str = "%Y-%m-%d %H:%M";
const LIST_HINT: &str = "↑/↓ select · Enter edit · n new · Space on/off · d delete · Esc close";
const DELETE_HINT: &str = "y delete · any other key keeps it";
const EDITOR_HINT: &str =
    "↑/↓ move · Enter open · ←/→ mode · d remove trigger · ctrl+s save · Esc back";
const FIELD_HINT: &str = "Enter apply · Esc cancel";
const PROMPT_HINT: &str = "Enter apply · shift+Enter new line · ctrl+r refine · Esc cancel";
const FORM_HINT: &str = "↑/↓ field · ←/→ choose · Enter apply · Esc cancel";
const DISCARD_HINT: &str = "y discard changes · any other key keeps editing";
/// The refining status at each count of dots, padded so the hint does not shift.
const REFINING_HINTS: [&str; 4] = [
    "refining    · Esc cancel",
    "refining.   · Esc cancel",
    "refining..  · Esc cancel",
    "refining... · Esc cancel",
];

/// A title, the view on top, then the panel's last message.
pub(super) fn render_tasks_panel(
    frame: &mut Frame<'_>,
    area: Rect,
    panel: &mut TasksPanel,
    theme: &Theme,
) {
    let [title_area, rule_area, body_area, message_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(area);
    frame.render_widget(Paragraph::new(title(panel)), title_area);
    render_rule(frame, rule_area, theme);
    render_message(frame, message_area, panel, theme);
    let upstreams = panel.upstream_choices();
    if let Some(editor) = panel.editor_mut() {
        render_editor(frame, body_area, editor, &upstreams, theme);
    } else {
        render_task_list(frame, body_area, panel, theme);
    }
}

/// The keys that work in the view on top, for the status line.
pub(super) fn tasks_panel_hint(panel: &TasksPanel) -> &'static str {
    match panel.editor() {
        Some(editor) => match editor.mode() {
            EditorMode::Browse => EDITOR_HINT,
            EditorMode::Name(_) => FIELD_HINT,
            EditorMode::Hosts(list) => list_hint(list.state()),
            EditorMode::Prompt(prompt) => prompt
                .refine_dots()
                .map_or(PROMPT_HINT, |dots| REFINING_HINTS[dots]),
            EditorMode::Trigger(_) => FORM_HINT,
            EditorMode::Discard => DISCARD_HINT,
        },
        None if panel.confirm_delete() => DELETE_HINT,
        None => LIST_HINT,
    }
}

/// `Tasks`, then the edited task's name and the view over it.
fn title(panel: &TasksPanel) -> Line<'static> {
    let bold = Style::default().add_modifier(Modifier::BOLD);
    let mut spans = vec![Span::styled("Tasks", bold)];
    if let Some(editor) = panel.editor() {
        let name = if editor.name().is_empty() {
            "new task".to_string()
        } else {
            escape_controls_single_line(editor.name()).into_owned()
        };
        spans.push(Span::raw(format!(" › {name}")));
        if editor.is_modified() {
            spans.push(Span::raw(" •"));
        }
    }
    Line::from(spans)
}

fn render_message(frame: &mut Frame<'_>, area: Rect, panel: &TasksPanel, theme: &Theme) {
    let line = if panel.confirm_delete() {
        let name = panel
            .tasks()
            .get(panel.selected())
            .map(|task| escape_controls_single_line(task.spec.name.as_str()).into_owned())
            .unwrap_or_default();
        Line::styled(
            format!("Delete {name} and its run history? y to confirm"),
            Style::default().fg(theme.warning),
        )
    } else {
        match panel.message() {
            Some(Message::Info(text)) => Line::styled(text.clone(), meta(theme)),
            Some(Message::Error(text)) => Line::styled(
                escape_controls_single_line(text).into_owned(),
                Style::default().fg(theme.error),
            ),
            None => Line::raw(""),
        }
    };
    frame.render_widget(Paragraph::new(line), area);
}

/// Every task with its triggers, next fire and last run, or a note when there are none.
fn render_task_list(frame: &mut Frame<'_>, area: Rect, panel: &TasksPanel, theme: &Theme) {
    let tasks = panel.tasks();
    if tasks.is_empty() {
        let note = if panel.loaded() {
            "No tasks yet. Press n to make one."
        } else {
            "Loading…"
        };
        frame.render_widget(Paragraph::new(Span::styled(note, meta(theme))), area);
        return;
    }
    let visible = (usize::from(area.height) / TASK_ENTRY_ROWS).max(1);
    let start = picker_scroll_start(panel.selected(), tasks.len(), visible);
    let lines: Vec<Line<'static>> = tasks
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .flat_map(|(index, task)| task_entry_rows(task, tasks, index == panel.selected(), theme))
        .collect();
    frame.render_widget(Paragraph::new(Text::from(lines)), area);
}

fn task_entry_rows(
    task: &ListedTask,
    tasks: &[ListedTask],
    selected: bool,
    theme: &Theme,
) -> [Line<'static>; 2] {
    let (pointer, name_style) = if selected {
        (POINTER, pointer_style(theme))
    } else {
        (NO_POINTER, Style::default())
    };
    let spec = &task.spec;
    let title = vec![
        Span::styled(
            format!(
                "{pointer}{}",
                escape_controls_single_line(spec.name.as_str())
            ),
            name_style,
        ),
        Span::styled(
            if spec.enabled {
                "  enabled"
            } else {
                "  disabled"
            },
            meta(theme),
        ),
    ];
    let mut details = vec![triggers_summary(&spec.triggers, tasks)];
    if let Some(at) = task.next_fire {
        details.push(format!("next {}", local(at, NEXT_FIRE_FORMAT)));
    }
    if let Some(state) = &task.last_run {
        details.push(format!("last {}", run_outcome(state)));
    }
    [
        Line::from(title),
        Line::styled(
            format!("{DETAIL_INDENT}{}", details.join(" · ")),
            meta(theme),
        ),
    ]
}

fn triggers_summary(triggers: &[TaskTrigger], tasks: &[ListedTask]) -> String {
    if triggers.is_empty() {
        return "no triggers".into();
    }
    let name_of = |id: TaskId| {
        tasks
            .iter()
            .find(|task| task.id == id)
            .map(|task| task.spec.name.as_str().to_string())
    };
    triggers
        .iter()
        .map(|trigger| trigger_summary(trigger, &name_of))
        .collect::<Vec<_>>()
        .join(", ")
}

/// What starts a run, in a few words; `name_of` names the tasks it may wait on.
fn trigger_summary(trigger: &TaskTrigger, name_of: &dyn Fn(TaskId) -> Option<String>) -> String {
    match trigger {
        TaskTrigger::Manual => "on request".into(),
        TaskTrigger::At { at } => format!("once at {}", at.format(AT_FORMAT)),
        TaskTrigger::Schedule { cron, missed } => {
            let skip = match missed {
                MissedFire::RunOnce => "",
                MissedFire::Skip => ", missed fires skipped",
            };
            format!("cron {}{skip}", escape_controls_single_line(cron))
        }
        TaskTrigger::Event {
            event,
            cooldown_secs,
        } => format!(
            "{} (cooldown {cooldown_secs}s)",
            event_summary(event, name_of)
        ),
    }
}

fn event_summary(event: &TaskEvent, name_of: &dyn Fn(TaskId) -> Option<String>) -> String {
    match event {
        TaskEvent::DaemonStarted => "when the daemon starts".into(),
        TaskEvent::SessionUnlocked => "when the session unlocks".into(),
        TaskEvent::PathChanged { path, .. } => format!(
            "when {} changes",
            escape_controls_single_line(&path.display().to_string())
        ),
        TaskEvent::TaskFinished { task, outcome } => {
            let name = name_of(*task).map_or_else(
                || format!("task {task}"),
                |name| escape_controls_single_line(&name).into_owned(),
            );
            let ending = match outcome {
                FinishOutcome::Succeeded => "succeeds",
                FinishOutcome::Failed => "fails",
                FinishOutcome::Either => "finishes",
            };
            format!("when {name} {ending}")
        }
    }
}

/// The rows, then the open prompt box or trigger form below them, then the note on the
/// last refine, then the editor's error.
fn render_editor(
    frame: &mut Frame<'_>,
    area: Rect,
    editor: &mut TaskEditor,
    upstreams: &[UpstreamChoice],
    theme: &Theme,
) {
    let box_rows = match editor.mode() {
        EditorMode::Prompt(_) | EditorMode::Trigger(_) | EditorMode::Hosts(_) => BOX_ROWS,
        EditorMode::Browse | EditorMode::Name(_) | EditorMode::Discard => 0,
    };
    let note_rows = u16::from(refine_note(editor).is_some());
    let error_rows = u16::from(editor.error().is_some());
    let [rows_area, box_area, note_area, error_area] = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(box_rows),
        Constraint::Length(note_rows),
        Constraint::Length(error_rows),
    ])
    .areas(area);
    render_editor_rows(frame, rows_area, editor, upstreams, theme);
    if let Some(error) = editor.error() {
        let line = Line::styled(
            escape_controls_single_line(error).into_owned(),
            Style::default().fg(theme.error),
        );
        frame.render_widget(Paragraph::new(line), error_area);
    }
    match editor.mode() {
        EditorMode::Trigger(form) => render_trigger_form(frame, box_area, form, upstreams, theme),
        EditorMode::Prompt(_) => {
            let inner = render_box(frame, box_area, "Prompt", theme);
            editor.set_prompt_width(inner.width);
            if let EditorMode::Prompt(prompt) = editor.mode() {
                render_text_box(frame, inner, prompt.text(), !prompt.refining());
            }
            if let Some(note) = refine_note(editor) {
                render_refine_note(frame, note_area, (note, editor.permission()), theme);
            }
        }
        EditorMode::Hosts(list) => {
            let entries = editor.host_entries();
            render_list_editor(frame, box_area, (list, &entries), "Allowed hosts", theme);
        }
        EditorMode::Browse | EditorMode::Name(_) | EditorMode::Discard => {}
    }
}

/// The open prompt box's note on its last refine.
fn refine_note(editor: &TaskEditor) -> Option<&RefineNote> {
    match editor.mode() {
        EditorMode::Prompt(prompt) => prompt.note(),
        _ => None,
    }
}

/// The recommended mode and why, in amber when it is not the task's `current` mode or runs
/// would still pause.
fn render_refine_note(
    frame: &mut Frame<'_>,
    area: Rect,
    (note, current): (&RefineNote, PermissionMode),
    theme: &Theme,
) {
    let color = if note.warns || note.mode != current {
        theme.warning
    } else {
        theme.status_text
    };
    let text = format!(
        "recommended mode: {} ({})",
        mode_name(note.mode),
        note.reason
    );
    let line = Line::styled(
        escape_controls_single_line(&text).into_owned(),
        Style::default().fg(color),
    );
    frame.render_widget(Paragraph::new(line), area);
}

/// Each row with the selected one picked out, scrolled to keep it in view.
fn render_editor_rows(
    frame: &mut Frame<'_>,
    area: Rect,
    editor: &TaskEditor,
    upstreams: &[UpstreamChoice],
    theme: &Theme,
) {
    let selected_row = editor.selected_row();
    let mut lines = Vec::new();
    let mut selected_line = 0;
    let mut cursor = None;
    for row in editor.rows() {
        if row == selected_row {
            selected_line = lines.len();
        }
        if let Some((label, text)) = inline_editor(editor, row) {
            let width = usize::from(area.width).saturating_sub(value_column());
            let (shown, column) = editor_window(text, width);
            cursor = Some((lines.len(), value_column() + column));
            lines.push(labelled(true, label, Span::raw(shown), theme));
            continue;
        }
        lines.extend(editor_row_lines(
            editor,
            row,
            row == selected_row,
            upstreams,
            theme,
        ));
    }
    let visible = usize::from(area.height).max(1);
    let start = picker_scroll_start(selected_line, lines.len(), visible);
    let shown: Vec<Line<'static>> = lines.into_iter().skip(start).take(visible).collect();
    frame.render_widget(Paragraph::new(Text::from(shown)), area);
    if let Some((line, column)) = cursor.filter(|(line, _)| *line >= start) {
        place_cursor(frame, area, line - start, column);
    }
}

/// The label and one-line editor drawn in place of `row` while it is being written.
fn inline_editor(editor: &TaskEditor, row: EditorRow) -> Option<(&'static str, &TextEditor)> {
    match (row, editor.mode()) {
        (EditorRow::Name, EditorMode::Name(text)) => Some(("Name", text)),
        _ => None,
    }
}

fn editor_row_lines(
    editor: &TaskEditor,
    row: EditorRow,
    selected: bool,
    upstreams: &[UpstreamChoice],
    theme: &Theme,
) -> Vec<Line<'static>> {
    match row {
        EditorRow::Name => vec![labelled(
            selected,
            "Name",
            placeholder(editor.name(), "(unnamed)", theme),
            theme,
        )],
        EditorRow::Prompt => vec![labelled(
            selected,
            "Prompt",
            prompt_preview(editor.prompt(), theme),
            theme,
        )],
        EditorRow::Mode => vec![labelled(
            selected,
            "Mode",
            mode_span(editor.permission(), selected, theme),
            theme,
        )],
        EditorRow::Hosts => vec![labelled(
            selected,
            "Allowed hosts",
            hosts_summary(editor, theme),
            theme,
        )],
        EditorRow::Enabled => vec![labelled(
            selected,
            "Enabled",
            Span::raw(if editor.enabled() { "[✓]" } else { "[ ]" }),
            theme,
        )],
        EditorRow::Trigger(index) => {
            trigger_row_lines(&editor.triggers()[index], index, selected, upstreams, theme)
        }
        EditorRow::AddTrigger => add_row_lines(
            editor.triggers().is_empty().then_some("Triggers"),
            "+ add trigger",
            selected,
            theme,
        ),
    }
}

/// An add row, after a `none` row labelled `empty_label` when the list is empty.
fn add_row_lines(
    empty_label: Option<&'static str>,
    add: &'static str,
    selected: bool,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let none =
        empty_label.map(|label| labelled(false, label, Span::styled("none", meta(theme)), theme));
    let add = labelled(
        selected,
        "",
        Span::styled(add, Style::default().fg(theme.accent)),
        theme,
    );
    none.into_iter().chain([add]).collect()
}

fn mode_name(mode: PermissionMode) -> &'static str {
    match mode {
        PermissionMode::Manual => "manual",
        PermissionMode::AcceptEdits => "accept edits",
        PermissionMode::Auto => "auto",
    }
}

/// The permission mode, with choice arrows while selected.
fn mode_span(mode: PermissionMode, selected: bool, theme: &Theme) -> Span<'static> {
    let name = mode_name(mode);
    if selected {
        Span::styled(format!("◂ {name} ▸"), Style::default().fg(theme.accent))
    } else {
        Span::raw(name)
    }
}

/// The allowed hosts on one line, or `none`.
fn hosts_summary(editor: &TaskEditor, theme: &Theme) -> Span<'static> {
    if editor.hosts().is_empty() {
        return Span::styled("none", meta(theme));
    }
    let hosts: Vec<&str> = editor.hosts().iter().map(HostName::as_str).collect();
    Span::raw(hosts.join(", "))
}

fn trigger_row_lines(
    row: &TriggerRow,
    index: usize,
    selected: bool,
    upstreams: &[UpstreamChoice],
    theme: &Theme,
) -> Vec<Line<'static>> {
    let label = if index == 0 { "Triggers" } else { "" };
    let name_of = |id: TaskId| {
        upstreams
            .iter()
            .find(|choice| choice.id == id)
            .map(|choice| choice.name.clone())
    };
    let summary = trigger_summary(&row.trigger, &name_of);
    let mut lines = vec![labelled(selected, label, Span::raw(summary), theme)];
    let indent = " ".repeat(value_column() + 2);
    let detail = match &row.check {
        TriggerCheck::Checking { .. } => Some(Span::styled("checking…", meta(theme))),
        TriggerCheck::Fires(fires) if fires.is_empty() => None,
        TriggerCheck::Fires(fires) => {
            let times: Vec<String> = fires.iter().map(|at| local(*at, FIRE_FORMAT)).collect();
            Some(Span::styled(
                format!("next {}", times.join(" · ")),
                meta(theme),
            ))
        }
        TriggerCheck::Refused(message) => Some(Span::styled(
            format!("⚠ {}", escape_controls_single_line(message)),
            Style::default().fg(theme.error),
        )),
    };
    lines.extend(detail.map(|span| Line::from(vec![Span::raw(indent), span])));
    lines
}

/// The prompt's first line, and how many more there are.
fn prompt_preview(prompt: &str, theme: &Theme) -> Span<'static> {
    let mut lines = prompt.lines();
    let Some(first) = lines.next().filter(|line| !line.trim().is_empty()) else {
        return Span::styled("(empty)", meta(theme));
    };
    let more = lines.count();
    let first = escape_controls_single_line(first);
    match more {
        0 => Span::raw(first.into_owned()),
        1 => Span::raw(format!("{first} (+1 line)")),
        more => Span::raw(format!("{first} (+{more} lines)")),
    }
}

fn placeholder(text: &str, empty: &'static str, theme: &Theme) -> Span<'static> {
    if text.is_empty() {
        Span::styled(empty, meta(theme))
    } else {
        Span::raw(escape_controls_single_line(text).into_owned())
    }
}

/// A row: the pointer when selected, `label` padded to [`LABEL_WIDTH`], then `value`.
fn labelled(selected: bool, label: &str, value: Span<'static>, theme: &Theme) -> Line<'static> {
    let (pointer, style) = if selected {
        (POINTER, pointer_style(theme))
    } else {
        (NO_POINTER, Style::default())
    };
    Line::from(vec![
        Span::styled(format!("{pointer}{label:<LABEL_WIDTH$}"), style),
        value,
    ])
}

/// The form's fields in a box, the focused one picked out, then why it would not apply.
fn render_trigger_form(
    frame: &mut Frame<'_>,
    area: Rect,
    form: &TriggerForm,
    upstreams: &[UpstreamChoice],
    theme: &Theme,
) {
    let title = if form.replaces.is_some() {
        "Edit trigger"
    } else {
        "Add trigger"
    };
    let inner = render_box(frame, area, title, theme);
    let mut lines = Vec::new();
    let mut cursor = None;
    for field in form.fields() {
        let focused = field == form.focus();
        let value = match form.text_field(field) {
            Some(text) => {
                let width = usize::from(inner.width).saturating_sub(value_column());
                let (shown, column) = editor_window(text, width);
                if focused {
                    cursor = Some((lines.len(), value_column() + column));
                }
                Span::raw(shown)
            }
            None => choice_span(&form.choice_text(field, upstreams), focused, theme),
        };
        lines.push(labelled(focused, field.label(), value, theme));
    }
    if let Some(error) = form.error() {
        lines.push(Line::styled(
            error.to_string(),
            Style::default().fg(theme.error),
        ));
    }
    frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    if let Some((line, column)) = cursor {
        place_cursor(frame, inner, line, column);
    }
}

fn choice_span(text: &str, focused: bool, theme: &Theme) -> Span<'static> {
    let text = escape_controls_single_line(text);
    if focused {
        Span::styled(format!("◂ {text} ▸"), Style::default().fg(theme.accent))
    } else {
        Span::raw(format!("  {text}"))
    }
}

/// A rounded box titled `title` over `area`, returning its inside.
fn render_box(frame: &mut Frame<'_>, area: Rect, title: &str, theme: &Theme) -> Rect {
    let accent = Style::default().fg(theme.accent);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(accent)
        .title(Span::styled(format!(" {title} "), accent));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    inner
}

/// `editor`'s text wrapped into `area`, scrolled to keep the cursor in view.
fn render_text_box(frame: &mut Frame<'_>, area: Rect, editor: &TextEditor, cursor: bool) {
    let chars: Vec<char> = editor.text().chars().collect();
    let rows = wrap_input(&chars, 0, area.width);
    let (cursor_row, cursor_column) = locate_cursor(&rows, editor.cursor_chars());
    let visible = usize::from(area.height).max(1);
    let start = (cursor_row + 1).saturating_sub(visible);
    let lines: Vec<Line<'static>> = rows
        .iter()
        .skip(start)
        .take(visible)
        .map(|&(row_start, row_end)| {
            let row: String = chars[row_start..row_end].iter().collect();
            Line::raw(escape_controls_single_line(&row).into_owned())
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), area);
    if cursor {
        place_cursor(frame, area, cursor_row - start, cursor_column);
    }
}

/// The slice of a one-line editor's text that fits `width` columns with the cursor in
/// view, and the cursor's column within it.
fn editor_window(editor: &TextEditor, width: usize) -> (String, usize) {
    let cursor = editor.cursor_chars();
    let start = cursor.saturating_sub(width.saturating_sub(1));
    let shown: String = editor.text().chars().skip(start).take(width).collect();
    (
        escape_controls_single_line(&shown).into_owned(),
        cursor - start,
    )
}

/// Puts the terminal cursor `row` lines and `column` columns into `area`, clamped to it.
fn place_cursor(frame: &mut Frame<'_>, area: Rect, row: usize, column: usize) {
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

/// The column a [`labelled`] row's value starts at.
fn value_column() -> usize {
    LABEL_WIDTH + POINTER.chars().count()
}

fn local(at: DateTime<Utc>, format: &str) -> String {
    at.with_timezone(&Local).format(format).to_string()
}

fn pointer_style(theme: &Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD)
}

fn meta(theme: &Theme) -> Style {
    Style::default().fg(theme.status_text)
}

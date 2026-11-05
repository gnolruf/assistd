//! The `/tasks` panel: the task list, the editor with its trigger form and prompt box,
//! and the plan and grants views over the editor.

use assistd_ipc::{FinishOutcome, MissedFire, TaskEvent, TaskId, TaskTrigger};
use chrono::{DateTime, Local, Utc};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Paragraph};

use super::super::plan_review::grant_label;
use super::super::settings::TextEditor;
use super::super::tasks::{
    AT_FORMAT, EditorMode, EditorRow, ListedTask, Message, Overlay, PlanSession, PlanState,
    TaskEditor, TasksPanel, TriggerCheck, TriggerForm, TriggerRow, UpstreamChoice,
};
use super::super::theme::Theme;
use super::plan_review::render_plan_review;
use super::{
    locate_cursor, picker_scroll_start, render_rule, render_transcript, run_outcome, wrap_input,
};
use crate::client::terminal_text::escape_controls_single_line;

/// Rows each listed task takes: its name and its details.
const TASK_ENTRY_ROWS: usize = 2;
const DETAIL_INDENT: &str = "    ";
const POINTER: &str = "❯ ";
const NO_POINTER: &str = "  ";
/// Columns the editor's and form's labels are padded to.
const LABEL_WIDTH: usize = 14;
/// Rows the prompt box, trigger form or feedback box takes while open.
const BOX_ROWS: u16 = 10;
const FIRE_FORMAT: &str = "%a %m-%d %H:%M";
const NEXT_FIRE_FORMAT: &str = "%Y-%m-%d %H:%M";
const LIST_HINT: &str = "↑/↓ select · Enter edit · n new · Space on/off · d delete · Esc close";
const DELETE_HINT: &str = "y delete · any other key keeps it";
const EDITOR_HINT: &str = "↑/↓ move · Enter open · d remove trigger · ctrl+s save · Esc back";
const FIELD_HINT: &str = "Enter apply · Esc cancel";
const PROMPT_HINT: &str = "Enter apply · shift+Enter new line · Esc cancel";
const FORM_HINT: &str = "↑/↓ field · ←/→ choose · Enter apply · Esc cancel";
const DISCARD_HINT: &str = "y discard changes · any other key keeps editing";
const PLANNING_HINT: &str = "planning… · Esc stop";
const REVIEW_HINT: &str = "↑/↓ select · Space tick · f feedback · a approve · Esc back";
const FEEDBACK_HINT: &str = "Enter re-plan · shift+Enter new line · Esc cancel";
const FAILED_HINT: &str = "Enter plan again · Esc back";
const GRANTS_HINT: &str = "↑/↓ select · d remove · Esc back";

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
    if panel.overlay().is_some() {
        render_overlay(frame, body_area, panel, theme);
    } else if let Some(editor) = panel.editor_mut() {
        render_editor(frame, body_area, editor, &upstreams, theme);
    } else {
        render_task_list(frame, body_area, panel, theme);
    }
}

/// The keys that work in the view on top, for the status line.
pub(super) fn tasks_panel_hint(panel: &TasksPanel) -> &'static str {
    match (panel.overlay(), panel.editor()) {
        (Some(Overlay::Grants { .. }), _) => GRANTS_HINT,
        (Some(Overlay::Plan(session)), _) => match &session.state {
            PlanState::Planning(_) => PLANNING_HINT,
            PlanState::Review {
                feedback: Some(_), ..
            } => FEEDBACK_HINT,
            PlanState::Review { .. } => REVIEW_HINT,
            PlanState::Failed(_) => FAILED_HINT,
        },
        (None, Some(editor)) => match editor.mode() {
            EditorMode::Browse => EDITOR_HINT,
            EditorMode::Name(_) => FIELD_HINT,
            EditorMode::Prompt(_) => PROMPT_HINT,
            EditorMode::Trigger(_) => FORM_HINT,
            EditorMode::Discard => DISCARD_HINT,
        },
        (None, None) if panel.confirm_delete() => DELETE_HINT,
        (None, None) => LIST_HINT,
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
    match panel.overlay() {
        Some(Overlay::Plan(_)) => spans.push(Span::raw(" › Plan")),
        Some(Overlay::Grants { .. }) => spans.push(Span::raw(" › Grants")),
        None => {}
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
    let mut title = vec![
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
    if spec.enabled && spec.plan.is_none() {
        title.push(Span::styled(
            "  ⚠ no approved plan",
            Style::default().fg(theme.warning),
        ));
    }
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

/// The rows, then the open prompt box or trigger form below them, then the editor's error.
fn render_editor(
    frame: &mut Frame<'_>,
    area: Rect,
    editor: &mut TaskEditor,
    upstreams: &[UpstreamChoice],
    theme: &Theme,
) {
    let box_rows = match editor.mode() {
        EditorMode::Prompt(_) | EditorMode::Trigger(_) => BOX_ROWS,
        EditorMode::Browse | EditorMode::Name(_) | EditorMode::Discard => 0,
    };
    let error_rows = u16::from(editor.error().is_some());
    let [rows_area, box_area, error_area] = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(box_rows),
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
            if let EditorMode::Prompt(text) = editor.mode() {
                render_text_box(frame, inner, text);
            }
        }
        EditorMode::Browse | EditorMode::Name(_) | EditorMode::Discard => {}
    }
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
        if let (EditorRow::Name, EditorMode::Name(text)) = (row, editor.mode()) {
            let width = usize::from(area.width).saturating_sub(value_column());
            let (shown, column) = editor_window(text, width);
            cursor = Some((lines.len(), value_column() + column));
            lines.push(labelled(true, "Name", Span::raw(shown), theme));
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
        EditorRow::Enabled => vec![labelled(
            selected,
            "Enabled",
            Span::raw(if editor.enabled() { "[✓]" } else { "[ ]" }),
            theme,
        )],
        EditorRow::Trigger(index) => {
            trigger_row_lines(&editor.triggers()[index], index, selected, upstreams, theme)
        }
        EditorRow::AddTrigger => {
            let mut lines = Vec::new();
            if editor.triggers().is_empty() {
                lines.push(labelled(
                    false,
                    "Triggers",
                    Span::styled("none", meta(theme)),
                    theme,
                ));
            }
            lines.push(labelled(
                selected,
                "",
                Span::styled("+ add trigger", Style::default().fg(theme.accent)),
                theme,
            ));
            lines
        }
        EditorRow::Plan => vec![labelled(
            selected,
            "Plan",
            plan_summary(editor, theme),
            theme,
        )],
        EditorRow::Grants => {
            let count = editor.stored().map_or(0, |spec| spec.grants.len());
            let text = match count {
                0 => "none".to_string(),
                1 => "1 grant".to_string(),
                count => format!("{count} grants"),
            };
            vec![labelled(
                selected,
                "Grants",
                Span::styled(text, meta(theme)),
                theme,
            )]
        }
    }
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

fn plan_summary(editor: &TaskEditor, theme: &Theme) -> Span<'static> {
    match editor.stored().and_then(|spec| spec.plan.as_ref()) {
        Some(plan) => Span::styled(
            format!(
                "approved · {} steps · Enter to plan again",
                plan.steps.len()
            ),
            Style::default().fg(theme.success),
        ),
        None if editor.enabled() => Span::styled(
            "⚠ none approved · Enter to plan",
            Style::default().fg(theme.warning),
        ),
        None => Span::styled("none · Enter to plan", meta(theme)),
    }
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

/// The plan view or the grants view.
fn render_overlay(frame: &mut Frame<'_>, area: Rect, panel: &mut TasksPanel, theme: &Theme) {
    if let Some(Overlay::Grants { selected }) = panel.overlay() {
        let selected = *selected;
        render_grants(frame, area, panel.editor(), selected, theme);
        return;
    }
    if let Some(pane) = panel.planning_pane_mut() {
        render_transcript(frame, area, pane, theme);
        return;
    }
    let feedback_open = matches!(
        panel.overlay(),
        Some(Overlay::Plan(PlanSession {
            state: PlanState::Review {
                feedback: Some(_),
                ..
            },
            ..
        }))
    );
    let [plan_area, feedback_area] = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(if feedback_open { BOX_ROWS } else { 0 }),
    ])
    .areas(area);
    let feedback_inner = feedback_open.then(|| render_box(frame, feedback_area, "Feedback", theme));
    if let Some(inner) = feedback_inner {
        panel.set_feedback_width(inner.width);
    }
    let Some(Overlay::Plan(session)) = panel.overlay() else {
        return;
    };
    match &session.state {
        PlanState::Review { review, feedback } => {
            render_plan_review(frame, plan_area, review, theme);
            if let (Some(text), Some(inner)) = (feedback, feedback_inner) {
                render_text_box(frame, inner, text);
            }
        }
        PlanState::Failed(message) => {
            let line = Line::styled(
                format!("Planning failed: {}", escape_controls_single_line(message)),
                Style::default().fg(theme.error),
            );
            frame.render_widget(Paragraph::new(line), plan_area);
        }
        PlanState::Planning(_) => {}
    }
}

/// The edited task's grants with the selected one picked out.
fn render_grants(
    frame: &mut Frame<'_>,
    area: Rect,
    editor: Option<&TaskEditor>,
    selected: usize,
    theme: &Theme,
) {
    let grants = editor
        .and_then(TaskEditor::stored)
        .map(|spec| spec.grants.as_slice())
        .unwrap_or_default();
    if grants.is_empty() {
        let note = Span::styled("No grants. Approving a plan adds them.", meta(theme));
        frame.render_widget(Paragraph::new(note), area);
        return;
    }
    let visible = usize::from(area.height).max(1);
    let start = picker_scroll_start(selected, grants.len(), visible);
    let lines: Vec<Line<'static>> = grants
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .map(|(index, grant)| {
            let label = escape_controls_single_line(&grant_label(grant)).into_owned();
            if index == selected {
                Line::styled(format!("{POINTER}{label}"), pointer_style(theme))
            } else {
                Line::raw(format!("{NO_POINTER}{label}"))
            }
        })
        .collect();
    frame.render_widget(Paragraph::new(Text::from(lines)), area);
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
fn render_text_box(frame: &mut Frame<'_>, area: Rect, editor: &TextEditor) {
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
    place_cursor(frame, area, cursor_row - start, cursor_column);
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

//! The `/tasks` panel: its tabs, the task list, and the editor with its trigger form and
//! prompt box.

use std::borrow::Cow;

use assistd_ipc::{
    FinishOutcome, HostName, MissedFire, PermissionMode, TaskEvent, TaskId, TaskTrigger,
};
use chrono::{DateTime, Local, Utc};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::Paragraph;

use super::super::selection::list_window;
use super::super::settings::{
    editor_window, list_hint, place_cursor, render_list_editor, render_wrapped_editor, titled_box,
};
use super::super::tasks::{
    AT_FORMAT, EditorMode, EditorRow, ListRow, ListedTask, Message, PanelTab, QueueView,
    RefineNote, TaskEditor, TasksPanel, TriggerCheck, TriggerForm, TriggerRow, UpstreamChoice,
};
use super::super::text_editor::TextEditor;
use super::super::theme::Theme;
use super::{render_rule, run_outcome};
use crate::client::terminal_text::escape_controls_single_line;

mod queue;
mod runs;

/// Rows each list entry takes: a task's name and its details, or the new-task row and a gap.
const TASK_ENTRY_ROWS: usize = 2;
const DETAIL_INDENT: &str = "    ";
const POINTER: &str = "❯ ";
const NO_POINTER: &str = "  ";
const KEPT_RUNS_LABEL: &str = "Keep runs";
/// Columns the editor's and form's labels are padded to.
const LABEL_WIDTH: usize = 14;
/// Rows the prompt box or trigger form takes while open.
const BOX_ROWS: u16 = 10;
const FIRE_FORMAT: &str = "%a %m-%d %H:%M";
const NEXT_FIRE_FORMAT: &str = "%Y-%m-%d %H:%M";
const LIST_HINT: &str =
    "↑/↓ select · Enter open · Space on/off · r run now · Tab queue · Esc close";
const NEW_TASK_HINT: &str = "↑/↓ select · Enter new task · Tab queue · Esc close";
const RUNS_HINT: &str = "↑/↓ select · ←/→ task · Enter resume session · Tab tasks · Esc close";
const FIELD_HINT: &str = "Enter apply · Esc cancel";
const KEPT_RUNS_HINT: &str = "Enter apply (empty for default) · Esc cancel";
const PROMPT_HINT: &str = "Enter apply · shift+Enter new line · ctrl+r refine · Esc cancel";
const FORM_HINT: &str = "↑/↓ field · ←/→ choose · Enter apply · Esc cancel";
const UNSAVED_HINT: &str = "Enter save · d discard · Esc keep editing";
const SAVING_HINT: &str = "saving… · Esc keep editing";
const DELETE_HINT: &str = "Enter delete · Esc keep";
const REMOVE_TRIGGER_HINT: &str = "Enter remove · Esc keep";
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
    frame.render_widget(Paragraph::new(title(panel, theme)), title_area);
    render_rule(frame, rule_area, theme);
    render_message(frame, message_area, panel, theme);
    let upstreams = panel.upstream_choices();
    if let Some(editor) = panel.editor_mut() {
        render_editor(frame, body_area, editor, &upstreams, theme);
        return;
    }
    match panel.tab() {
        PanelTab::Tasks => render_task_list(frame, body_area, panel, theme),
        PanelTab::Queue => queue::render_queue(frame, body_area, panel.queue(), theme),
        PanelTab::Runs => runs::render_runs(frame, body_area, panel, theme),
    }
}

/// The keys that work in the view on top, for the status line.
pub(super) fn tasks_panel_hint(panel: &TasksPanel) -> Cow<'static, str> {
    let hint = match panel.editor() {
        Some(editor) => match editor.mode() {
            EditorMode::Browse => return Cow::Owned(browse_hint(editor)),
            EditorMode::Name(_) => FIELD_HINT,
            EditorMode::KeptRuns(_) => KEPT_RUNS_HINT,
            EditorMode::Hosts(list) => list_hint(list.state()),
            EditorMode::Prompt(prompt) => prompt
                .refine_dots()
                .map_or(PROMPT_HINT, |dots| REFINING_HINTS[dots]),
            EditorMode::Trigger(_) => FORM_HINT,
            EditorMode::Unsaved => UNSAVED_HINT,
            EditorMode::SavingToClose { .. } => SAVING_HINT,
            EditorMode::ConfirmDelete => DELETE_HINT,
            EditorMode::ConfirmRemoveTrigger(_) => REMOVE_TRIGGER_HINT,
        },
        None => match panel.tab() {
            PanelTab::Tasks if panel.selected_row() == ListRow::NewTask => NEW_TASK_HINT,
            PanelTab::Tasks => LIST_HINT,
            PanelTab::Queue => return Cow::Owned(queue_hint(panel.queue())),
            PanelTab::Runs => RUNS_HINT,
        },
    };
    Cow::Borrowed(hint)
}

/// The Queue tab's keys, naming what `p` does to the queue now.
fn queue_hint(queue: &QueueView) -> String {
    let pause = if queue.snapshot().is_some_and(|snapshot| snapshot.paused) {
        "p resume"
    } else {
        "p pause"
    };
    format!("↑/↓ select · shift+↑/↓ move · x cancel · {pause} · Tab runs · Esc close")
}

/// The editor's keys for the selected row, with Ctrl+D once the task is saved.
fn browse_hint(editor: &TaskEditor) -> String {
    let row_keys = match editor.selected_row() {
        EditorRow::Mode => "←/→ mode",
        EditorRow::Trigger(_) => "Enter edit · Del remove",
        _ => "Enter open",
    };
    let delete = if editor.task().is_some() {
        " · ctrl+d delete task"
    } else {
        ""
    };
    format!("↑/↓ move · {row_keys} · ctrl+s save{delete} · Esc back")
}

/// The tabs, the one on screen picked out, or while editing `Tasks` and the edited task's
/// name.
fn title(panel: &TasksPanel, theme: &Theme) -> Line<'static> {
    let Some(editor) = panel.editor() else {
        return tab_bar(panel.tab(), theme);
    };
    let name = if editor.name().is_empty() {
        "new task".to_string()
    } else {
        escape_controls_single_line(editor.name()).into_owned()
    };
    let mut spans = vec![
        Span::styled("Tasks", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!(" › {name}")),
    ];
    if editor.is_modified() {
        spans.push(Span::raw(" •"));
    }
    Line::from(spans)
}

fn tab_bar(shown: PanelTab, theme: &Theme) -> Line<'static> {
    let spans = PanelTab::ALL.iter().flat_map(|tab| {
        let label = match tab {
            PanelTab::Tasks => " Tasks ",
            PanelTab::Queue => " Queue ",
            PanelTab::Runs => " Runs ",
        };
        let style = if *tab == shown {
            theme.title_style()
        } else {
            theme.meta_style()
        };
        [Span::styled(label, style), Span::raw(" ")]
    });
    Line::from(spans.collect::<Vec<_>>())
}

/// The editor's open question, else the panel's last message.
fn render_message(frame: &mut Frame<'_>, area: Rect, panel: &TasksPanel, theme: &Theme) {
    let upstreams = panel.upstream_choices();
    let question = panel
        .editor()
        .and_then(|editor| editor_question(editor, &upstreams));
    let line = match (question, panel.message()) {
        (Some(question), _) => Line::styled(question, theme.warning_style()),
        (None, Some(Message::Info(text))) => Line::styled(text.clone(), theme.meta_style()),
        (None, Some(Message::Error(text))) => Line::styled(
            escape_controls_single_line(text).into_owned(),
            theme.error_style(),
        ),
        (None, None) => Line::raw(""),
    };
    frame.render_widget(Paragraph::new(line), area);
}

/// What the editor is asking before it leaves or removes something, if anything;
/// `upstreams` name the tasks a trigger may wait on.
fn editor_question(editor: &TaskEditor, upstreams: &[UpstreamChoice]) -> Option<String> {
    match editor.mode() {
        EditorMode::ConfirmRemoveTrigger(index) => {
            let summary = trigger_summary(
                &editor.triggers()[*index].trigger,
                &upstream_name(upstreams),
            );
            Some(format!("Remove trigger \"{summary}\"?"))
        }
        EditorMode::Unsaved => Some("Save changes before leaving?".into()),
        EditorMode::ConfirmDelete if editor.name().is_empty() => {
            Some("Delete this task and its run history?".into())
        }
        EditorMode::ConfirmDelete => Some(format!(
            "Delete {} and its run history?",
            escape_controls_single_line(editor.name())
        )),
        _ => None,
    }
}

/// Every task with its triggers, next fire and last run, then the new-task row; a note
/// stands above that row while there are no tasks.
fn render_task_list(frame: &mut Frame<'_>, area: Rect, panel: &TasksPanel, theme: &Theme) {
    let tasks = panel.tasks();
    if !panel.loaded() {
        frame.render_widget(
            Paragraph::new(Span::styled("Loading…", theme.meta_style())),
            area,
        );
        return;
    }
    let selected = panel.selected_row();
    let note = tasks
        .is_empty()
        .then(|| Line::styled(format!("{NO_POINTER}No tasks yet."), theme.meta_style()));
    let selected_entry = match selected {
        ListRow::Task(index) => index,
        ListRow::NewTask => tasks.len(),
    };
    let visible = (usize::from(area.height) / TASK_ENTRY_ROWS).max(1);
    let entries = list_window(selected_entry, tasks.len() + 1, visible, 0).flat_map(|index| {
        let selected = index == selected_entry;
        match tasks.get(index) {
            Some(task) => task_entry_rows(task, tasks, selected, theme),
            None => new_task_rows(selected, theme),
        }
    });
    let lines: Vec<Line<'static>> = note.into_iter().chain(entries).collect();
    frame.render_widget(Paragraph::new(Text::from(lines)), area);
}

fn new_task_rows(selected: bool, theme: &Theme) -> [Line<'static>; TASK_ENTRY_ROWS] {
    let pointer = if selected { POINTER } else { NO_POINTER };
    let label = Span::styled("+ new task", theme.accent_style());
    [
        Line::from(vec![Span::styled(pointer, theme.title_style()), label]),
        Line::raw(""),
    ]
}

fn task_entry_rows(
    task: &ListedTask,
    tasks: &[ListedTask],
    selected: bool,
    theme: &Theme,
) -> [Line<'static>; TASK_ENTRY_ROWS] {
    let (pointer, name_style) = if selected {
        (POINTER, theme.title_style())
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
            theme.meta_style(),
        ),
    ];
    let mut details = vec![triggers_summary(&spec.triggers, tasks)];
    if let Some(at) = task.next_fire {
        details.push(format!("next {}", local(at, NEXT_FIRE_FORMAT)));
    }
    if let Some(state) = &task.last_run {
        details.push(format!("last {}", run_outcome(state)));
    }
    if let Some(failure) = &task.trigger_failure {
        details.push(escape_controls_single_line(failure).into_owned());
    }
    [
        Line::from(title),
        Line::styled(
            format!("{DETAIL_INDENT}{}", details.join(" · ")),
            theme.meta_style(),
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
        EditorMode::Browse
        | EditorMode::Name(_)
        | EditorMode::KeptRuns(_)
        | EditorMode::Unsaved
        | EditorMode::SavingToClose { .. }
        | EditorMode::ConfirmDelete
        | EditorMode::ConfirmRemoveTrigger(_) => 0,
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
            theme.error_style(),
        );
        frame.render_widget(Paragraph::new(line), error_area);
    }
    match editor.mode() {
        EditorMode::Trigger(form) => render_trigger_form(frame, box_area, form, upstreams, theme),
        EditorMode::Prompt(_) => {
            let inner = render_box(frame, box_area, "Prompt", theme);
            editor.set_prompt_width(inner.width);
            if let EditorMode::Prompt(prompt) = editor.mode() {
                render_wrapped_editor(frame, inner, prompt.text(), !prompt.refining());
            }
            if let Some(note) = refine_note(editor) {
                render_refine_note(frame, note_area, note, editor.permission(), theme);
            }
        }
        EditorMode::Hosts(list) => {
            let entries = editor.host_entries();
            render_list_editor(frame, box_area, list, &entries, "Allowed hosts", theme);
        }
        EditorMode::Browse
        | EditorMode::Name(_)
        | EditorMode::KeptRuns(_)
        | EditorMode::Unsaved
        | EditorMode::SavingToClose { .. }
        | EditorMode::ConfirmDelete
        | EditorMode::ConfirmRemoveTrigger(_) => {}
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
    note: &RefineNote,
    current: PermissionMode,
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
    let window = list_window(selected_line, lines.len(), visible, 0);
    let start = window.start;
    let shown: Vec<Line<'static>> = lines.drain(window).collect();
    frame.render_widget(Paragraph::new(Text::from(shown)), area);
    if let Some((line, column)) = cursor.filter(|(line, _)| *line >= start) {
        place_cursor(frame, area, line - start, column);
    }
}

/// The label and one-line editor drawn in place of `row` while it is being written.
fn inline_editor(editor: &TaskEditor, row: EditorRow) -> Option<(&'static str, &TextEditor)> {
    match (row, editor.mode()) {
        (EditorRow::Name, EditorMode::Name(text)) => Some(("Name", text)),
        (EditorRow::KeptRuns, EditorMode::KeptRuns(text)) => Some((KEPT_RUNS_LABEL, text)),
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
        EditorRow::Announce => vec![labelled(
            selected,
            "Announce reply",
            Span::raw(if editor.announce() { "[✓]" } else { "[ ]" }),
            theme,
        )],
        EditorRow::KeptRuns => vec![labelled(
            selected,
            KEPT_RUNS_LABEL,
            kept_runs_span(editor, theme),
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
    let none = empty_label.map(|label| {
        labelled(
            false,
            label,
            Span::styled("none", theme.meta_style()),
            theme,
        )
    });
    let add = labelled(selected, "", Span::styled(add, theme.accent_style()), theme);
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
        Span::styled(format!("◂ {name} ▸"), theme.accent_style())
    } else {
        Span::raw(name)
    }
}

/// The task's own kept-runs count, or `default`.
fn kept_runs_span(editor: &TaskEditor, theme: &Theme) -> Span<'static> {
    match editor.kept_runs() {
        Some(count) => Span::raw(count.to_string()),
        None => Span::styled("default", theme.meta_style()),
    }
}

/// The allowed hosts on one line, or `none`.
fn hosts_summary(editor: &TaskEditor, theme: &Theme) -> Span<'static> {
    if editor.hosts().is_empty() {
        return Span::styled("none", theme.meta_style());
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
    let summary = trigger_summary(&row.trigger, &upstream_name(upstreams));
    let mut lines = vec![labelled(selected, label, Span::raw(summary), theme)];
    let indent = " ".repeat(value_column() + 2);
    let detail = match &row.check {
        TriggerCheck::Checking { .. } => Some(Span::styled("checking…", theme.meta_style())),
        TriggerCheck::Fires(fires) if fires.is_empty() => None,
        TriggerCheck::Fires(fires) => {
            let times: Vec<String> = fires.iter().map(|at| local(*at, FIRE_FORMAT)).collect();
            Some(Span::styled(
                format!("next {}", times.join(" · ")),
                theme.meta_style(),
            ))
        }
        TriggerCheck::Refused(message) => Some(Span::styled(
            format!("⚠ {}", escape_controls_single_line(message)),
            theme.error_style(),
        )),
    };
    lines.extend(detail.map(|span| Line::from(vec![Span::raw(indent), span])));
    lines
}

/// Names a task by its id among `upstreams`.
fn upstream_name(upstreams: &[UpstreamChoice]) -> impl Fn(TaskId) -> Option<String> + '_ {
    |id| {
        upstreams
            .iter()
            .find(|choice| choice.id == id)
            .map(|choice| choice.name.clone())
    }
}

/// The prompt's first line, and how many more there are.
fn prompt_preview(prompt: &str, theme: &Theme) -> Span<'static> {
    let mut lines = prompt.lines();
    let Some(first) = lines.next().filter(|line| !line.trim().is_empty()) else {
        return Span::styled("(empty)", theme.meta_style());
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
        Span::styled(empty, theme.meta_style())
    } else {
        Span::raw(escape_controls_single_line(text).into_owned())
    }
}

/// A row: the pointer when selected, `label` padded to [`LABEL_WIDTH`], then `value`.
fn labelled(selected: bool, label: &str, value: Span<'static>, theme: &Theme) -> Line<'static> {
    let (pointer, style) = if selected {
        (POINTER, theme.title_style())
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
        lines.push(Line::styled(error.to_string(), theme.error_style()));
    }
    frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    if let Some((line, column)) = cursor {
        place_cursor(frame, inner, line, column);
    }
}

fn choice_span(text: &str, focused: bool, theme: &Theme) -> Span<'static> {
    let text = escape_controls_single_line(text);
    if focused {
        Span::styled(format!("◂ {text} ▸"), theme.accent_style())
    } else {
        Span::raw(format!("  {text}"))
    }
}

/// A box titled `title` over `area`, returning its inside.
fn render_box(frame: &mut Frame<'_>, area: Rect, title: &str, theme: &Theme) -> Rect {
    let block = titled_box(title, theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    inner
}

/// The column a [`labelled`] row's value starts at.
fn value_column() -> usize {
    LABEL_WIDTH + POINTER.chars().count()
}

fn local(at: DateTime<Utc>, format: &str) -> String {
    at.with_timezone(&Local).format(format).to_string()
}

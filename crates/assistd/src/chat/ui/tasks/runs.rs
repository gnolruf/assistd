//! The tasks panel's Runs tab: one task's runs, newest first, and the selected run's reply.

use assistd_ipc::{TaskId, TaskRunRecord, TaskRunState};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Wrap};

use super::super::super::tasks::TasksPanel;
use super::super::super::theme::Theme;
use super::super::{picker_scroll_start, run_outcome};
use super::{
    DETAIL_INDENT, FIRE_FORMAT, NO_POINTER, POINTER, local, meta, pointer_style, trigger_summary,
};
use crate::client::terminal_text::escape_controls_single_line;

/// Rows each run takes: its outcome and start, then its trigger and duration.
const RUN_ROWS: usize = 2;
/// Rows the selected run's reply takes below the list.
const REPLY_ROWS: u16 = 8;

/// The task's name, its runs, and the selected run's reply.
pub(super) fn render_runs(frame: &mut Frame<'_>, area: Rect, panel: &TasksPanel, theme: &Theme) {
    let Some(task) = panel.runs_task() else {
        let note = Line::styled(
            format!("{NO_POINTER}Select a task to see its runs."),
            meta(theme),
        );
        frame.render_widget(Paragraph::new(note), area);
        return;
    };
    let runs = panel.runs();
    let selected = runs.runs().get(runs.selected());
    let reply_rows = if selected.is_some_and(|run| run.reply.is_some()) {
        REPLY_ROWS
    } else {
        0
    };
    let [header_area, _, list_area, reply_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(reply_rows),
    ])
    .areas(area);
    let header = Line::from(vec![
        Span::raw("Runs of "),
        Span::styled(
            escape_controls_single_line(task.spec.name.as_str()).into_owned(),
            Style::default().add_modifier(Modifier::BOLD),
        ),
    ]);
    frame.render_widget(Paragraph::new(header), header_area);
    if let Some(reply) = selected.and_then(|run| run.reply.as_deref()) {
        render_reply(frame, reply_area, reply, theme);
    }
    if !runs.loaded() {
        frame.render_widget(
            Paragraph::new(Span::styled("Loading…", meta(theme))),
            list_area,
        );
        return;
    }
    if runs.runs().is_empty() {
        let empty = Line::styled(format!("{NO_POINTER}No runs yet."), meta(theme));
        frame.render_widget(Paragraph::new(empty), list_area);
        return;
    }
    let name_of = |id| {
        panel
            .tasks()
            .iter()
            .find(|listed| listed.id == id)
            .map(|listed| listed.spec.name.as_str().to_string())
    };
    let visible = (usize::from(list_area.height) / RUN_ROWS).max(1);
    let start = picker_scroll_start(runs.selected(), runs.runs().len(), visible);
    let lines: Vec<Line<'static>> = runs
        .runs()
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .flat_map(|(index, run)| run_rows(run, index == runs.selected(), &name_of, theme))
        .collect();
    frame.render_widget(Paragraph::new(Text::from(lines)), list_area);
}

fn run_rows(
    run: &TaskRunRecord,
    selected: bool,
    name_of: &dyn Fn(TaskId) -> Option<String>,
    theme: &Theme,
) -> [Line<'static>; RUN_ROWS] {
    let (pointer, style) = if selected {
        (POINTER, pointer_style(theme))
    } else {
        (NO_POINTER, Style::default())
    };
    let outcome_style = match run.state {
        TaskRunState::Failed { .. } => Style::default().fg(theme.error),
        _ => style,
    };
    let at = local(run.started_at.unwrap_or(run.queued_at), FIRE_FORMAT);
    let mut details = vec![trigger_summary(&run.trigger, name_of)];
    if let Some(took) = run_duration(run) {
        details.push(format!("took {took}"));
    }
    if run.session_id.is_some() {
        details.push("Enter to resume".into());
    }
    [
        Line::from(vec![
            Span::styled(format!("{pointer}run {} · ", run.run), style),
            Span::styled(run_outcome(&run.state), outcome_style),
            Span::styled(format!(" · {at}"), meta(theme)),
        ]),
        Line::styled(
            format!("{DETAIL_INDENT}{}", details.join(" · ")),
            meta(theme),
        ),
    ]
}

/// How long a finished run took from its start, as `42s`, `3m 4s` or `2h 5m`.
fn run_duration(run: &TaskRunRecord) -> Option<String> {
    let secs = (run.ended_at? - run.started_at?).num_seconds().max(0);
    Some(match secs {
        0..60 => format!("{secs}s"),
        60..3600 => format!("{}m {}s", secs / 60, secs % 60),
        _ => format!("{}h {}m", secs / 3600, secs % 3600 / 60),
    })
}

/// The run's final reply under a rule, wrapped, its control characters escaped.
fn render_reply(frame: &mut Frame<'_>, area: Rect, reply: &str, theme: &Theme) {
    let lines: Vec<Line<'static>> = std::iter::once(Line::styled("Reply", meta(theme)))
        .chain(
            reply
                .lines()
                .map(|line| Line::raw(escape_controls_single_line(line).into_owned())),
        )
        .collect();
    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        area,
    );
}

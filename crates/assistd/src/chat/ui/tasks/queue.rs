//! The tasks panel's Queue tab: whether the queue runs, what holds each lane, then every
//! unfinished run and waiting user turn.

use assistd_ipc::{LaneHolder, QueueItem, SuspendReason, TaskQueueSnapshot, TaskRunState};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::Paragraph;

use super::super::super::tasks::QueueView;
use super::super::super::theme::Theme;
use super::super::picker_scroll_start;
use super::{DETAIL_INDENT, NO_POINTER, POINTER, local, meta, pointer_style};
use crate::client::terminal_text::escape_controls_single_line;

/// Rows each queue item takes: what it is, then its state.
const ITEM_ROWS: usize = 2;
const RESUMES_FORMAT: &str = "%H:%M:%S";

/// The queue's state line, then its items.
pub(super) fn render_queue(frame: &mut Frame<'_>, area: Rect, queue: &QueueView, theme: &Theme) {
    let Some(snapshot) = queue.snapshot() else {
        let waiting = Span::styled("Waiting for the daemon's queue…", meta(theme));
        frame.render_widget(Paragraph::new(waiting), area);
        return;
    };
    let [status_area, _, items_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(area);
    frame.render_widget(Paragraph::new(status_line(snapshot, theme)), status_area);
    if snapshot.items.is_empty() {
        let empty = Line::styled(format!("{NO_POINTER}Nothing queued."), meta(theme));
        frame.render_widget(Paragraph::new(empty), items_area);
        return;
    }
    let visible = (usize::from(items_area.height) / ITEM_ROWS).max(1);
    let start = picker_scroll_start(queue.selected(), snapshot.items.len(), visible);
    let lines: Vec<Line<'static>> = snapshot
        .items
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .flat_map(|(index, item)| item_rows(item, index == queue.selected(), theme))
        .collect();
    frame.render_widget(Paragraph::new(Text::from(lines)), items_area);
}

/// Whether the queue starts runs, why not, and what holds each lane.
fn status_line(snapshot: &TaskQueueSnapshot, theme: &Theme) -> Line<'static> {
    let state = match (&snapshot.held, snapshot.paused) {
        (_, true) => Span::styled("paused", Style::default().fg(theme.warning)),
        (Some(hold), false) => {
            let until = hold.resumes_at.map_or_else(
                || " while you talk".to_string(),
                |at| format!(" until {}", local(at, RESUMES_FORMAT)),
            );
            Span::styled(
                format!("held for your conversation{until}"),
                Style::default().fg(theme.warning),
            )
        }
        (None, false) => Span::styled("running", Style::default().fg(theme.accent)),
    };
    let lanes = snapshot
        .lanes
        .iter()
        .map(|lane| match lane {
            LaneHolder::Main => "chat".to_string(),
            LaneHolder::Task { run } => format!("run {run}"),
            LaneHolder::Idle => "idle".to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    Line::from(vec![
        Span::raw("Queue "),
        state,
        Span::styled(format!(" · lanes: {lanes}"), meta(theme)),
    ])
}

fn item_rows(item: &QueueItem, selected: bool, theme: &Theme) -> [Line<'static>; ITEM_ROWS] {
    let (pointer, style) = if selected {
        (POINTER, pointer_style(theme))
    } else {
        (NO_POINTER, Style::default())
    };
    let (title, detail, detail_style) = match item {
        QueueItem::Run {
            run,
            task_name,
            state,
            prompt,
            ..
        } => {
            let title = format!(
                "run {run} · {}",
                escape_controls_single_line(task_name.as_str())
            );
            let detail = match prompt {
                Some(prompt) => format!(
                    "needs input: {} · Enter to answer",
                    escape_controls_single_line(&prompt.matched_pattern)
                ),
                None => queued_state(state),
            };
            let detail_style = if prompt.is_some() {
                Style::default().fg(theme.warning)
            } else {
                meta(theme)
            };
            (title, detail, detail_style)
        }
        QueueItem::UserTurn { preview, .. } => (
            format!("you: {}", escape_controls_single_line(preview)),
            "waiting its turn".to_string(),
            meta(theme),
        ),
    };
    [
        Line::styled(format!("{pointer}{title}"), style),
        Line::styled(format!("{DETAIL_INDENT}{detail}"), detail_style),
    ]
}

/// A queued run's state, with why it is suspended.
fn queued_state(state: &TaskRunState) -> String {
    match state {
        TaskRunState::Suspended {
            reason: SuspendReason::UserConversation,
        } => "suspended: your conversation holds the model".into(),
        TaskRunState::Suspended {
            reason: SuspendReason::GpuBusy,
        } => "suspended: another process needs the GPU".into(),
        TaskRunState::Running => "running".into(),
        TaskRunState::NeedsInput => "needs input".into(),
        TaskRunState::Queued => "queued".into(),
        TaskRunState::Succeeded | TaskRunState::Failed { .. } | TaskRunState::Cancelled => {
            "ending".into()
        }
    }
}

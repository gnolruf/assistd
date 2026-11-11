//! The tasks panel's Queue tab: whether the queue runs, what holds each lane, then every
//! unfinished run and waiting user turn.

use assistd_ipc::{LaneHolder, QueueItem, SuspendReason, TaskQueueSnapshot, TaskRunState};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::Paragraph;

use super::super::super::selection::list_window;
use super::super::super::tasks::QueueView;
use super::super::super::theme::Theme;
use super::{DETAIL_INDENT, NO_POINTER, POINTER, local};
use crate::client::terminal_text::escape_controls_single_line;

/// Rows each queue item takes: what it is, then its state.
const ITEM_ROWS: usize = 2;
const RESUMES_FORMAT: &str = "%H:%M:%S";

/// The queue's state line, then its items.
pub(super) fn render_queue(frame: &mut Frame<'_>, area: Rect, queue: &QueueView, theme: &Theme) {
    let Some(snapshot) = queue.snapshot() else {
        let waiting = Span::styled("Waiting for the daemon's queue…", theme.meta_style());
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
        let empty = Line::styled(format!("{NO_POINTER}Nothing queued."), theme.meta_style());
        frame.render_widget(Paragraph::new(empty), items_area);
        return;
    }
    let visible = (usize::from(items_area.height) / ITEM_ROWS).max(1);
    let selected = queue.selected();
    let lines: Vec<Line<'static>> = list_window(selected, snapshot.items.len(), visible, 0)
        .flat_map(|index| item_rows(&snapshot.items[index], index == selected, theme))
        .collect();
    frame.render_widget(Paragraph::new(Text::from(lines)), items_area);
}

/// Whether the queue starts runs, why not, and what holds each lane.
fn status_line(snapshot: &TaskQueueSnapshot, theme: &Theme) -> Line<'static> {
    let state = match (&snapshot.held, snapshot.paused) {
        (_, true) => Span::styled("paused", theme.warning_style()),
        (Some(hold), false) => {
            let until = hold.resumes_at.map_or_else(
                || " while you talk".to_string(),
                |at| format!(" until {}", local(at, RESUMES_FORMAT)),
            );
            Span::styled(
                format!("held for your conversation{until}"),
                theme.warning_style(),
            )
        }
        (None, false) => Span::styled("running", theme.accent_style()),
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
        Span::styled(format!(" · lanes: {lanes}"), theme.meta_style()),
    ])
}

fn item_rows(item: &QueueItem, selected: bool, theme: &Theme) -> [Line<'static>; ITEM_ROWS] {
    let (pointer, style) = if selected {
        (POINTER, theme.title_style())
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
                theme.warning_style()
            } else {
                theme.meta_style()
            };
            (title, detail, detail_style)
        }
        QueueItem::UserTurn { preview, .. } => (
            format!("you: {}", escape_controls_single_line(preview)),
            "waiting its turn".to_string(),
            theme.meta_style(),
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

//! The execute choice drawn under the transcript once a plan is made in plan mode.

use assistd_ipc::PermissionMode;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Paragraph};

use super::super::app::ExecuteChoice;
use super::super::theme::Theme;
use super::{CONFIRM_BORDER_ROWS, indented_rows, pointer_style, saturating_u16};

/// Draw the execute choice at the bottom of `area` in place of the input line, returning the
/// rows above it for the transcript.
pub(super) fn render_execute_dock(
    frame: &mut Frame<'_>,
    area: Rect,
    choices: &[ExecuteChoice],
    selected: usize,
    theme: &Theme,
) -> Rect {
    let width = usize::from(area.width.saturating_sub(2));
    let body = execute_rows(choices, selected, width, theme);
    let height = saturating_u16(body.len() + CONFIRM_BORDER_ROWS).min(area.height);
    let [output_area, dock_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(height)]).areas(area);
    let accent = Style::default().fg(theme.accent);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(accent)
        .title(Span::styled(
            " Execute this plan? ",
            accent.add_modifier(Modifier::BOLD),
        ));
    frame.render_widget(Paragraph::new(Text::from(body)).block(block), dock_area);
    output_area
}

/// The numbered choices, the selected one behind a pointer.
fn execute_rows(
    choices: &[ExecuteChoice],
    selected: usize,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    choices
        .iter()
        .enumerate()
        .flat_map(|(index, choice)| {
            let label = match choice {
                ExecuteChoice::Execute(PermissionMode::Manual) => "Yes, in manual mode",
                ExecuteChoice::Execute(PermissionMode::AcceptEdits) => "Yes, accepting edits",
                ExecuteChoice::Execute(PermissionMode::Auto) => "Yes, in auto mode",
                ExecuteChoice::KeepPlanning => "No, keep planning (Esc)",
            };
            let (pointer, style) = if index == selected {
                ("❯", pointer_style(theme))
            } else {
                (" ", Style::default())
            };
            let lead = format!("{pointer} {}. ", index + 1);
            indented_rows(&lead, label, style, width)
        })
        .collect()
}

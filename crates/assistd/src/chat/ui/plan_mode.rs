//! The plan review drawn under the transcript while the turn waits on its plan's verdict.

use std::borrow::Cow;

use assistd_ipc::PermissionMode;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Paragraph};

use super::super::app::{PlanChoice, PlanReview};
use super::super::theme::Theme;
use super::{CONFIRM_BORDER_ROWS, indented_rows, pointer_style, saturating_u16};

const FEEDBACK_PLACEHOLDER: &str = "No, keep planning (type to say what to change)";

/// Draw the plan review at the bottom of `area` in place of the input line, returning the rows
/// above it for the transcript, where the plan is.
pub(super) fn render_plan_review_dock(
    frame: &mut Frame<'_>,
    area: Rect,
    review: &PlanReview,
    theme: &Theme,
) -> Rect {
    let width = usize::from(area.width.saturating_sub(2));
    let mut body = vec![Line::from("Would you like to carry out this plan?")];
    body.extend(choice_rows(review, width, theme));
    body.push(Line::from(Span::styled(
        "PgUp/PgDn scrolls the plan · Esc keeps planning",
        Style::default().add_modifier(Modifier::DIM),
    )));
    let height = saturating_u16(body.len() + CONFIRM_BORDER_ROWS).min(area.height);
    let [output_area, dock_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(height)]).areas(area);
    let accent = Style::default().fg(theme.accent);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(accent)
        .title(Span::styled(
            " Plan ready ",
            accent.add_modifier(Modifier::BOLD),
        ));
    frame.render_widget(Paragraph::new(Text::from(body)).block(block), dock_area);
    output_area
}

/// The numbered choices, the selected one behind a pointer, and the typed feedback on the
/// keep-planning row.
fn choice_rows(review: &PlanReview, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    review
        .choices
        .iter()
        .enumerate()
        .flat_map(|(index, choice)| {
            let label: Cow<'_, str> = match choice {
                PlanChoice::Accept(PermissionMode::Auto) => "Yes, and use auto mode".into(),
                PlanChoice::Accept(PermissionMode::AcceptEdits) => {
                    "Yes, and auto-accept edits".into()
                }
                PlanChoice::Accept(PermissionMode::Manual) => {
                    "Yes, and manually approve edits".into()
                }
                PlanChoice::KeepPlanning if review.feedback.is_empty() => {
                    FEEDBACK_PLACEHOLDER.into()
                }
                PlanChoice::KeepPlanning => {
                    format!("No, keep planning: {}▏", review.feedback).into()
                }
            };
            let (pointer, style) = if index == review.selected {
                ("❯", pointer_style(theme))
            } else {
                (" ", Style::default())
            };
            let lead = format!("{pointer} {}. ", index + 1);
            indented_rows(&lead, &label, style, width)
        })
        .collect()
}

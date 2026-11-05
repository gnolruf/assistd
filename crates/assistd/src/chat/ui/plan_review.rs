//! A plan under review drawn as its summary, then each step with the actions it expects to
//! need approved, each marked with whether the work will ask about it.

use assistd_ipc::{ActionOutlook, AnticipatedAction};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::Paragraph;

use super::super::plan_review::{PlanReview, action_label, outlook_label};
use super::super::theme::Theme;
use crate::client::terminal_text::escape_controls_single_line;

const STEP_INDENT: &str = "   ";
const ACTION_INDENT: &str = "     · ";
const PROBLEM_INDENT: &str = "       ";

/// The review in `area` from its scroll offset, which is clamped to the lines it has.
pub(super) fn render_plan_review(
    frame: &mut Frame<'_>,
    area: Rect,
    review: &mut PlanReview,
    theme: &Theme,
) {
    let width = usize::from(area.width).max(1);
    let lines = plan_review_lines(review, width, theme);
    let visible = usize::from(area.height).max(1);
    review.clamp_scroll(lines.len().saturating_sub(visible));
    let shown: Vec<Line<'static>> = lines
        .into_iter()
        .skip(review.scroll())
        .take(visible)
        .collect();
    frame.render_widget(Paragraph::new(Text::from(shown)), area);
}

/// Every line of the review wrapped to `width`.
fn plan_review_lines(review: &PlanReview, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = wrapped(&review.plan().summary, "", width)
        .into_iter()
        .map(|row| Line::styled(row, Style::default().add_modifier(Modifier::BOLD)))
        .collect();
    for (number, step) in review.plan().steps.iter().enumerate() {
        lines.push(Line::raw(""));
        let heading = format!("{}. {}", number + 1, step.description);
        lines.extend(
            wrapped(&heading, STEP_INDENT, width)
                .into_iter()
                .map(Line::raw),
        );
        for anticipated in &step.actions {
            lines.extend(action_lines(anticipated, width, theme));
        }
    }
    lines
}

/// The action and its outlook, then any problem the daemon found with it.
fn action_lines(
    anticipated: &AnticipatedAction,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let outlook_style = match anticipated.outlook {
        ActionOutlook::Unasked => Style::default().fg(theme.success),
        ActionOutlook::JudgeDecides => Style::default().fg(theme.status_text),
        ActionOutlook::Asks => Style::default().fg(theme.warning),
    };
    let label = escape_controls_single_line(&action_label(&anticipated.action)).into_owned();
    let mut lines = vec![Line::from(vec![
        Span::raw(format!("{ACTION_INDENT}{label}")),
        Span::styled(
            format!("  {}", outlook_label(anticipated.outlook)),
            outlook_style,
        ),
    ])];
    if let Some(problem) = &anticipated.problem {
        let style = Style::default().fg(theme.error);
        lines.extend(
            wrapped(&format!("⚠ {problem}"), PROBLEM_INDENT, width)
                .into_iter()
                .map(|row| Line::styled(row, style)),
        );
    }
    lines
}

/// `text` wrapped to `width` with every row after `indent`, controls escaped.
fn wrapped(text: &str, indent: &str, width: usize) -> Vec<String> {
    let text = escape_controls_single_line(text);
    let options = textwrap::Options::new(width)
        .initial_indent(indent)
        .subsequent_indent(indent);
    textwrap::wrap(&text, options)
        .into_iter()
        .map(std::borrow::Cow::into_owned)
        .collect()
}

//! A plan under review drawn as its summary, then each step with its anticipated
//! actions as checkboxes, those that failed the daemon's checks flagged.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::Paragraph;

use super::super::plan_review::{PlanReview, grant_label};
use super::super::theme::Theme;
use super::picker_scroll_start;
use crate::client::terminal_text::escape_controls_single_line;

const STEP_INDENT: &str = "   ";
const ACTION_INDENT: &str = "     ";
const PROBLEM_INDENT: &str = "         ";
const CHECKED: &str = "[✓] ";
const UNCHECKED: &str = "[ ] ";

/// The review in `area`, scrolled to keep the selected action in view.
pub(super) fn render_plan_review(
    frame: &mut Frame<'_>,
    area: Rect,
    review: &PlanReview,
    theme: &Theme,
) {
    let width = usize::from(area.width).max(1);
    let (lines, selected_line) = plan_review_lines(review, width, theme);
    let visible = usize::from(area.height).max(1);
    let start = picker_scroll_start(selected_line, lines.len(), visible);
    let shown: Vec<Line<'static>> = lines.into_iter().skip(start).take(visible).collect();
    frame.render_widget(Paragraph::new(Text::from(shown)), area);
}

/// Every line of the review wrapped to `width`, and the line of the selected action.
fn plan_review_lines(
    review: &PlanReview,
    width: usize,
    theme: &Theme,
) -> (Vec<Line<'static>>, usize) {
    let mut lines: Vec<Line<'static>> = wrapped(&review.plan().summary, "", width)
        .into_iter()
        .map(|row| Line::styled(row, Style::default().add_modifier(Modifier::BOLD)))
        .collect();
    let mut action = 0;
    let mut selected_line = 0;
    for (number, step) in review.plan().steps.iter().enumerate() {
        lines.push(Line::raw(""));
        let heading = format!("{}. {}", number + 1, step.description);
        lines.extend(
            wrapped(&heading, STEP_INDENT, width)
                .into_iter()
                .map(Line::raw),
        );
        for anticipated in &step.actions {
            let selected = action == review.selected();
            if selected {
                selected_line = lines.len();
            }
            lines.push(action_line(
                &grant_label(&anticipated.grant),
                review.is_checked(action),
                selected,
                theme,
            ));
            if let Some(problem) = &anticipated.problem {
                let flagged = format!("⚠ {problem}");
                let style = Style::default().fg(theme.warning);
                lines.extend(
                    wrapped(&flagged, PROBLEM_INDENT, width)
                        .into_iter()
                        .map(|row| Line::styled(row, style)),
                );
            }
            action += 1;
        }
    }
    if review.action_count() == 0 {
        lines.push(Line::raw(""));
        lines.push(Line::styled(
            "The plan needs no approvals.",
            Style::default().fg(theme.status_text),
        ));
    }
    (lines, selected_line)
}

fn action_line(label: &str, checked: bool, selected: bool, theme: &Theme) -> Line<'static> {
    let checkbox = if checked { CHECKED } else { UNCHECKED };
    let style = if selected {
        theme.selection()
    } else {
        Style::default()
    };
    Line::from(vec![
        Span::raw(ACTION_INDENT),
        Span::styled(
            format!("{checkbox}{}", escape_controls_single_line(label)),
            style,
        ),
    ])
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

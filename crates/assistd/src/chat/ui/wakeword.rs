//! The `/wakeword` dock: enrollment progress, the saved phrase, and test results.

use assistd_ipc::SampleRejection;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Paragraph};

use super::super::app::{EnrollProgress, EnrollPrompt, SampleResult, TestProgress, WakeWordView};
use super::super::markdown::BLEY;
use super::{CONFIRM_BORDER_ROWS, TRANSCRIPT_MIN_ROWS, indented_rows, saturating_u16};

/// Draw the dock at the bottom of `area` in place of the input line,
/// returning the rows above it for the transcript.
pub(super) fn render_wake_word_dock(
    frame: &mut Frame<'_>,
    area: Rect,
    view: &WakeWordView,
) -> Rect {
    let width = usize::from(area.width.saturating_sub(2));
    let (title, body) = match view {
        WakeWordView::Enroll(progress) => (" Enroll wake phrase ", enroll_body(progress, width)),
        WakeWordView::Saved { phrase, variants } => {
            (" Wake phrase saved ", saved_body(phrase, variants, width))
        }
        WakeWordView::Test(progress) => (" Test wake phrase ", test_body(progress, width)),
    };
    let max_height = usize::from(area.height).saturating_sub(TRANSCRIPT_MIN_ROWS);
    let height = saturating_u16((body.len() + CONFIRM_BORDER_ROWS).min(max_height));
    let [output_area, dock_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(height)]).areas(area);
    let accent = Style::default().fg(BLEY);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(accent)
        .title(Span::styled(title, accent.add_modifier(Modifier::BOLD)));
    frame.render_widget(Paragraph::new(Text::from(body)).block(block), dock_area);
    output_area
}

/// What the status line suggests while the dock is open.
pub(super) fn wake_word_hint(view: &WakeWordView) -> &'static str {
    match view {
        WakeWordView::Enroll(progress) if progress.prompt.is_some_and(EnrollPrompt::can_save) => {
            "Enter save · Esc cancel"
        }
        WakeWordView::Enroll(_) => "Esc cancel",
        WakeWordView::Saved { .. } => "t test · Esc close",
        WakeWordView::Test(_) => "Esc stop",
    }
}

fn enroll_body(progress: &EnrollProgress, width: usize) -> Vec<Line<'static>> {
    let count = progress.prompt.map_or_else(String::new, |prompt| {
        let next = prompt.accepted.saturating_add(1);
        if prompt.can_save() {
            format!(" ({next})")
        } else {
            format!(" ({next}/{})", prompt.minimum)
        }
    });
    let mut body = indented_rows(
        "",
        &format!("Say \"{}\"{count}", progress.phrase),
        bold(),
        width,
    );
    if !progress.results.is_empty() {
        body.push(Line::default());
        body.extend(
            progress
                .results
                .iter()
                .flat_map(|result| sample_rows(result, width)),
        );
    }
    let hint = match progress.prompt {
        Some(prompt) if prompt.can_save() && prompt.consistent => {
            "Looks reliable. Keep going to cover more variations, or press Enter to save."
        }
        Some(prompt) if prompt.can_save() => {
            "Keep going to cover more variations, or press Enter to save."
        }
        _ => "Esc cancels.",
    };
    body.push(Line::default());
    body.extend(indented_rows("", hint, dim(), width));
    body
}

fn sample_rows(result: &SampleResult, width: usize) -> Vec<Line<'static>> {
    match result {
        SampleResult::Accepted { heard } => {
            indented_rows("✓ ", &format!("heard \"{heard}\""), good(), width)
        }
        SampleResult::Rejected { reason, heard } => {
            let heard = heard
                .as_ref()
                .map_or_else(String::new, |heard| format!(": heard \"{heard}\""));
            indented_rows(
                "✗ ",
                &format!("{}{heard}", rejection_text(*reason)),
                bad(),
                width,
            )
        }
    }
}

fn rejection_text(reason: SampleRejection) -> &'static str {
    match reason {
        SampleRejection::TooShort => "too short; say the whole phrase",
        SampleRejection::TooLong => "too long; say just the phrase",
        SampleRejection::Clipped => "clipped; turn the mic gain down",
        SampleRejection::DidNotMatch => "didn't sound like the phrase",
    }
}

fn saved_body(phrase: &str, variants: &[String], width: usize) -> Vec<Line<'static>> {
    let mut body = indented_rows(
        "✓ ",
        &format!("Saved \"{phrase}\""),
        good().add_modifier(Modifier::BOLD),
        width,
    );
    if !variants.is_empty() {
        body.extend(indented_rows(
            "  ",
            &format!("Also accepts: {}", variants.join(", ")),
            Style::default(),
            width,
        ));
    }
    body.push(Line::default());
    body.extend(indented_rows(
        "",
        "Press t to test it now, or Esc to close.",
        dim(),
        width,
    ));
    body
}

fn test_body(progress: &TestProgress, width: usize) -> Vec<Line<'static>> {
    let ask = progress.phrase.as_ref().map_or_else(
        || "Say the wake phrase to see what is heard.".to_string(),
        |phrase| format!("Say \"{phrase}\" to see what is heard."),
    );
    let mut body = indented_rows("", &ask, bold(), width);
    body.push(Line::default());
    if progress.results.is_empty() {
        body.extend(indented_rows("", "Listening…", dim(), width));
    }
    for result in &progress.results {
        let (lead, verdict, style) = if result.matched {
            ("✓ ", "matched", good())
        } else {
            ("✗ ", "no match", bad())
        };
        body.extend(indented_rows(
            lead,
            &format!("{verdict}: heard \"{}\"", result.heard),
            style,
            width,
        ));
    }
    body.push(Line::default());
    body.extend(indented_rows("", "Esc stops.", dim(), width));
    body
}

fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

fn dim() -> Style {
    Style::default().fg(Color::Gray).add_modifier(Modifier::DIM)
}

fn good() -> Style {
    Style::default().fg(Color::Green)
}

fn bad() -> Style {
    Style::default().fg(Color::Red)
}

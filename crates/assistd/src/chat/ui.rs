#![allow(elided_lifetimes_in_paths)] // ratatui Frame<'_> is mostly noise; project style is to elide it
//! Pure ratatui render for the chat TUI. No state mutation beyond the
//! output pane's wrap cache and the app's last-viewport-height sink, both
//! of which must be updated during layout.

use std::time::{Duration, Instant};

use assistd_core::{PresenceState, VoiceCaptureState};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui_image::StatefulImage;

use super::app::{App, ConfirmationModal};
use super::output::THUMBNAIL_ROWS;
use super::vram::{RamState, VramState};

/// Render the full chat TUI into `frame`, updating the app's cached layout state.
pub fn render(frame: &mut Frame, app: &mut App) {
    let frame_area = frame.area();
    let input_height =
        compute_input_height(frame_area.width, frame_area.height, app.input.buffer());
    let chunks = Layout::vertical([
        Constraint::Min(3),
        Constraint::Length(1),
        Constraint::Length(input_height),
    ])
    .split(frame_area);
    let output_area = chunks[0];
    let status_area = chunks[1];
    let input_area = chunks[2];

    app.set_output_height(output_area.height);

    render_output(frame, output_area, app);
    render_status(frame, status_area, app);
    render_input(frame, input_area, app);

    if let Some(modal) = &app.modal {
        render_confirmation_modal(frame, frame.area(), modal);
    }
}

fn render_confirmation_modal(frame: &mut Frame, area: Rect, modal: &ConfirmationModal) {
    let width = area.width.saturating_mul(3) / 5; // 60%
    let width = width.clamp(40, 100);
    let height = 10u16.min(area.height.saturating_sub(2));
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    let modal_area = Rect {
        x,
        y,
        width,
        height,
    };

    frame.render_widget(Clear, modal_area);
    let block = Block::default()
        .title(Span::styled(
            " Confirm destructive command ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(modal_area);
    frame.render_widget(block, modal_area);

    let text = Text::from(vec![
        Line::from(vec![
            Span::styled("pattern: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                modal.request.matched_pattern.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "command:",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(modal.request.script.clone()),
        Line::from(""),
        Line::from(Span::styled(
            "[Y]es / [N]o   (Enter = yes · Esc = no)",
            Style::default().fg(Color::Green),
        )),
    ]);
    let para = Paragraph::new(text).wrap(Wrap { trim: false });
    frame.render_widget(para, inner);
}

fn render_output(frame: &mut Frame, area: Rect, app: &mut App) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let slots = app.output.thumbnail_layout(area.width);
    let (lines, start) = app.output.render_view(area.width, area.height);
    let text = Text::from(lines.to_vec());
    let para = Paragraph::new(text).scroll((start, 0));
    frame.render_widget(para, area);

    let viewport_top = start as usize;
    let viewport_bottom = viewport_top.saturating_add(area.height as usize);
    const CAPTION_ROWS: u16 = 1;
    let image_height = THUMBNAIL_ROWS.saturating_sub(CAPTION_ROWS);
    if image_height == 0 {
        return;
    }
    for slot in slots {
        let image_start = slot.start_row + CAPTION_ROWS as usize;
        let image_end = slot.start_row + slot.height;
        if image_start < viewport_top || image_end > viewport_bottom {
            continue;
        }
        let local_row = (image_start - viewport_top) as u16;
        if local_row >= area.height {
            continue;
        }
        let avail = area.height - local_row;
        let h = image_height.min(avail);
        if h == 0 {
            continue;
        }
        let w = area.width.min(32);
        let rect = Rect {
            x: area.x,
            y: area.y + local_row,
            width: w,
            height: h,
        };
        let widget = StatefulImage::default();
        if let Some(state) = app.output.thumbnail_protocol_mut(slot.item_idx) {
            frame.render_stateful_widget(widget, rect, state);
        }
    }
}

fn render_status(frame: &mut Frame, area: Rect, app: &App) {
    if area.width == 0 {
        return;
    }

    let state = if app.generating {
        format!("{} generating", app.spinner_char())
    } else {
        "idle".to_string()
    };

    let snap = app.throughput.snapshot(Instant::now());
    let rate = snap.rate.map(|r| format!("{r:.0} tok/s"));

    let vram = match &app.resources.vram {
        VramState::Unknown => "…".to_string(),
        VramState::Disabled => "N/A".to_string(),
        VramState::Ok(info) => format!(
            "{:.1}/{:.1} GiB",
            info.used_mb as f64 / 1024.0,
            info.total_mb as f64 / 1024.0,
        ),
        VramState::Err(_) => "err".to_string(),
    };

    let ram = match &app.resources.ram {
        RamState::Unknown => "…".to_string(),
        RamState::Ok(info) => format!(
            "{:.1}/{:.1} GiB",
            info.used_mb as f64 / 1024.0,
            info.total_mb as f64 / 1024.0,
        ),
    };

    let reversed = Style::default().add_modifier(Modifier::REVERSED);
    let mut left_spans: Vec<Span> = Vec::new();
    left_spans.push(Span::styled(format!("model: {}", app.model_name), reversed));
    if let Some((dot, label)) = presence_dot(app.presence_state) {
        left_spans.push(Span::raw(" "));
        left_spans.push(Span::styled(
            "●",
            Style::default().fg(dot).add_modifier(Modifier::BOLD),
        ));
        left_spans.push(Span::styled(format!(" {label}"), reversed));
    }
    if let Some(remaining) = app.local_time_until_next_transition() {
        left_spans.push(Span::styled(
            format!(" ({})", format_countdown(remaining)),
            reversed,
        ));
    }
    if let Some((color, label)) = voice_indicator(app.listening) {
        left_spans.push(Span::raw(" "));
        left_spans.push(Span::styled(
            app.spinner_char().to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
        left_spans.push(Span::styled(format!(" {label}"), reversed));
    }

    let (vision_label, vision_color) = if app.vision_enabled {
        ("vision: on", Color::Green)
    } else {
        ("vision: off", Color::DarkGray)
    };
    left_spans.push(Span::styled(
        format!(" │ {vision_label}"),
        Style::default()
            .fg(vision_color)
            .add_modifier(Modifier::REVERSED),
    ));

    let pending_count = app.pending_attachments.len();
    if pending_count > 0 {
        left_spans.push(Span::raw(" "));
        let label = if pending_count == 1 {
            "📎×1".to_string()
        } else {
            format!("📎×{pending_count}")
        };
        left_spans.push(Span::styled(
            label,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::REVERSED),
        ));
    }
    left_spans.push(Span::styled(format!(" │ {state}"), reversed));
    if let Some(r) = rate {
        left_spans.push(Span::styled(format!(" │ {r}"), reversed));
    }
    left_spans.push(Span::styled(format!(" │ RAM: {ram}"), reversed));
    left_spans.push(Span::styled(format!(" │ VRAM: {vram}"), reversed));

    let left_len: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();

    let right = if let Some(notice) = app.notice() {
        notice.to_string()
    } else if app.output.scroll_offset() > 0 {
        format!(
            "↑{} · PgDn to follow · Ctrl+C quit",
            app.output.scroll_offset()
        )
    } else if app.presence_state.is_some() {
        "Ctrl+C quit · F2 cycle presence · PgUp/Dn scroll".to_string()
    } else {
        "Ctrl+C quit · PgUp/Dn scroll · ↑/↓ history".to_string()
    };

    let width = area.width as usize;
    let right_len = right.chars().count();

    let mut spans = left_spans;
    if left_len + right_len < width {
        let gap = width - left_len - right_len;
        spans.push(Span::styled(" ".repeat(gap), reversed));
        spans.push(Span::styled(
            right,
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::REVERSED),
        ));
    }

    let para = Paragraph::new(Line::from(spans));
    frame.render_widget(para, area);
}

fn presence_dot(s: Option<PresenceState>) -> Option<(Color, &'static str)> {
    match s? {
        PresenceState::Active => Some((Color::Green, "active")),
        PresenceState::Drowsy => Some((Color::Yellow, "drowsy")),
        PresenceState::Sleeping => Some((Color::Red, "sleeping")),
    }
}

fn voice_indicator(s: VoiceCaptureState) -> Option<(Color, &'static str)> {
    match s {
        VoiceCaptureState::Idle => None,
        VoiceCaptureState::Recording => Some((Color::Red, "Listening…")),
        VoiceCaptureState::Queued => Some((Color::Blue, "Processing…")),
        VoiceCaptureState::Transcribing => Some((Color::Yellow, "transcribing…")),
    }
}

fn format_countdown(d: Duration) -> String {
    let total = d.as_secs();
    if total >= 3600 {
        format!("{}h{:02}m", total / 3600, (total % 3600) / 60)
    } else if total >= 60 {
        format!("{}m", total / 60)
    } else {
        format!("{total}s")
    }
}

const INPUT_PROMPT: &str = "> ";

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let prompt_w = INPUT_PROMPT.chars().count() as u16;
    let width = area.width;
    let buf_chars: Vec<char> = app.input.buffer().chars().collect();

    let first_cap = width.saturating_sub(prompt_w) as usize;
    let mut lines: Vec<Line<'_>> = Vec::new();
    let mut idx = first_cap.min(buf_chars.len());
    let mut first = String::with_capacity(width as usize);
    first.push_str(INPUT_PROMPT);
    first.extend(&buf_chars[..idx]);
    lines.push(Line::from(first));
    while idx < buf_chars.len() {
        let take = (width as usize).min(buf_chars.len() - idx);
        let s: String = buf_chars[idx..idx + take].iter().collect();
        lines.push(Line::from(s));
        idx += take;
    }

    let para = Paragraph::new(Text::from(lines));
    frame.render_widget(para, area);

    let cursor_total = prompt_w + app.input.cursor_col();
    let cursor_y = cursor_total / width;
    let cursor_x = cursor_total % width;
    let cy = (area.y + cursor_y).min(area.y + area.height.saturating_sub(1));
    frame.set_cursor_position(Position::new(area.x + cursor_x, cy));
}

/// Number of rows the input area needs to display `buffer` plus the prompt,
/// wrapping at `frame_width` cells. Capped so the output pane keeps at least
/// its minimum height (3 rows) plus the 1-row status bar.
fn compute_input_height(frame_width: u16, frame_height: u16, buffer: &str) -> u16 {
    if frame_width == 0 {
        return 1;
    }
    let prompt_w = INPUT_PROMPT.chars().count() as u16;
    let total = prompt_w.saturating_add(buffer.chars().count() as u16);
    let needed = (total / frame_width).saturating_add(1);
    let cap = frame_height.saturating_sub(4).max(1);
    needed.clamp(1, cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_height_grows_when_buffer_overflows_width() {
        // width=10, prompt=2, so first row fits 8 chars of buffer.
        assert_eq!(compute_input_height(10, 24, ""), 1);
        assert_eq!(compute_input_height(10, 24, &"a".repeat(7)), 1);
        // total=10 → exactly fills row 0, cursor must wrap to row 1.
        assert_eq!(compute_input_height(10, 24, &"a".repeat(8)), 2);
        assert_eq!(compute_input_height(10, 24, &"a".repeat(18)), 3);
    }

    #[test]
    fn input_height_capped_to_leave_room_for_output_and_status() {
        // frame_height=6 → cap = 6-4 = 2.
        assert_eq!(compute_input_height(10, 6, &"a".repeat(100)), 2);
    }

    #[test]
    fn input_height_minimum_one_even_in_tiny_frame() {
        assert_eq!(compute_input_height(10, 0, ""), 1);
        assert_eq!(compute_input_height(0, 24, "anything"), 1);
    }
}

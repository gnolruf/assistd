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

pub fn render(frame: &mut Frame, app: &mut App) {
    let chunks = Layout::vertical([
        Constraint::Min(3),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(frame.area());
    let output_area = chunks[0];
    let status_area = chunks[1];
    let input_area = chunks[2];

    app.set_output_height(output_area.height);

    render_output(frame, output_area, app);
    render_status(frame, status_area, app);
    render_input(frame, input_area, app);

    // Modal overlay on top of the whole frame. Only drawn when a
    // destructive-command prompt is pending.
    if let Some(modal) = &app.modal {
        render_confirmation_modal(frame, frame.area(), modal);
    }
}

/// Centered Y/N prompt showing the pending destructive command. The agent
/// loop is blocked on the gate's oneshot until the user decides, so the
/// overlay is the only thing that matters.
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

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    const PROMPT: &str = "> ";
    let content = format!("{PROMPT}{}", app.input.buffer());
    let para = Paragraph::new(content);
    frame.render_widget(para, area);

    let cursor_x = area.x + PROMPT.chars().count() as u16 + app.input.cursor_col();
    let cursor_x = cursor_x.min(area.x + area.width.saturating_sub(1));
    frame.set_cursor_position(Position::new(cursor_x, area.y));
}

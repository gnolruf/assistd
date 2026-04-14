//! Pure ratatui render for the chat TUI. No state mutation beyond the
//! output pane's wrap cache and the app's last-viewport-height sink, both
//! of which must be updated during layout.

use std::time::{Duration, Instant};

use assistd_core::PresenceState;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::Paragraph;

use super::app::App;
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
}

fn render_output(frame: &mut Frame, area: Rect, app: &mut App) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let (lines, start) = app.output.render_view(area.width, area.height);
    let text = Text::from(lines.to_vec());
    let para = Paragraph::new(text).scroll((start, 0));
    frame.render_widget(para, area);
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

    // Reversed fg/bg for the whole bar (like before), applied per-span so we
    // can drop the modifier for the presence dot — a REVERSED colored fg
    // becomes a large colored background block, which reads as noise.
    let reversed = Style::default().add_modifier(Modifier::REVERSED);
    let mut left_spans: Vec<Span> = Vec::new();
    left_spans.push(Span::styled(format!("model: {}", app.model_name), reversed));
    let wake_started = app.presence.as_ref().and_then(|p| p.wake_in_progress());
    if let Some(started) = wake_started {
        // Override the presence segment during an active wake: the
        // regular dot + idle countdown are meaningless while the
        // daemon is mid-transition to Active.
        left_spans.push(Span::raw(" "));
        left_spans.push(Span::styled(
            format!(
                "{} waking up ({}s)",
                app.spinner_char(),
                started.elapsed().as_secs()
            ),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::REVERSED),
        ));
    } else {
        if let Some((dot, label)) = presence_dot(app.presence_state) {
            left_spans.push(Span::raw(" "));
            left_spans.push(Span::styled(
                "●",
                Style::default().fg(dot).add_modifier(Modifier::BOLD),
            ));
            left_spans.push(Span::styled(format!(" {label}"), reversed));
        }
        if let Some(remaining) = app
            .presence
            .as_ref()
            .and_then(|p| p.time_until_next_transition(&app.sleep_cfg))
        {
            left_spans.push(Span::styled(
                format!(" ({})", format_countdown(remaining)),
                reversed,
            ));
        }
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

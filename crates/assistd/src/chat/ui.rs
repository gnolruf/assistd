//! Pure ratatui render for the chat TUI. No state mutation beyond the
//! output pane's wrap cache and the app's last-viewport-height sink, both
//! of which must be updated during layout.

use std::time::Instant;

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

    let mut left_parts: Vec<String> = vec![format!("model: {}", app.model_name), state];
    if let Some(r) = rate {
        left_parts.push(r);
    }
    left_parts.push(format!("RAM: {ram}"));
    left_parts.push(format!("VRAM: {vram}"));
    let left = left_parts.join(" │ ");

    let right = if let Some(notice) = app.notice() {
        notice.to_string()
    } else if app.output.scroll_offset() > 0 {
        format!(
            "↑{} · PgDn to follow · Ctrl+C quit",
            app.output.scroll_offset()
        )
    } else {
        "Ctrl+C quit · PgUp/Dn scroll · ↑/↓ history".to_string()
    };

    let width = area.width as usize;
    let left_len = left.chars().count();
    let right_len = right.chars().count();

    let line = if left_len + right_len < width {
        let gap = width - left_len - right_len;
        Line::from(vec![
            Span::raw(left),
            Span::raw(" ".repeat(gap)),
            Span::styled(right, Style::default().fg(Color::DarkGray)),
        ])
    } else {
        Line::from(Span::raw(left))
    };

    let para = Paragraph::new(line).style(Style::default().add_modifier(Modifier::REVERSED));
    frame.render_widget(para, area);
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

//! ratatui render for the chat TUI. Mutates only the output pane's wrap
//! cache and the app's last viewport height, both of which are layout
//! outputs.

use std::time::{Duration, Instant};

use assistd_core::{PresenceState, VoiceCaptureState};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui_image::StatefulImage;

use super::app::{App, BranchPickerModal, ConfirmationModal};
use super::output::THUMBNAIL_ROWS;
use super::vram::{RamState, VramState};

pub fn render(frame: &mut Frame<'_>, app: &mut App) {
    let frame_area = frame.area();
    let input_height =
        compute_input_height(frame_area.width, frame_area.height, app.input.buffer());
    let suggestions = app.slash_suggestions();
    let popup_height = if suggestions.is_empty() {
        0
    } else {
        (suggestions.len() as u16).min(6)
    };
    let chunks = Layout::vertical([
        Constraint::Min(3),
        Constraint::Length(popup_height),
        Constraint::Length(1),
        Constraint::Length(input_height),
    ])
    .split(frame_area);
    let output_area = chunks[0];
    let popup_area = chunks[1];
    let status_area = chunks[2];
    let input_area = chunks[3];

    app.set_output_height(output_area.height);

    render_output(frame, output_area, app);
    if popup_height > 0 {
        render_slash_popup(frame, popup_area, &suggestions, app.slash_selected());
    }
    render_status(frame, status_area, app);
    render_input(frame, input_area, app);

    if let Some(picker) = &app.picker_modal {
        render_branch_picker_modal(frame, frame.area(), picker);
    }
    if let Some(modal) = &app.modal {
        render_confirmation_modal(frame, frame.area(), modal);
    }
}

fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

fn render_branch_picker_modal(frame: &mut Frame<'_>, area: Rect, picker: &BranchPickerModal) {
    let width = (area.width.saturating_mul(4) / 5).clamp(50, 120);
    let max_height = area.height.saturating_sub(2);
    let desired = picker.entries.len() as u16 + 4;
    let height = desired.min(max_height).max(6);
    let modal_area = centered_rect(area, width, height);

    frame.render_widget(Clear, modal_area);
    let block = Block::default()
        .title(Span::styled(
            " Resume conversation ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(modal_area);
    frame.render_widget(block, modal_area);

    let footer_h = 1u16;
    let list_h = inner.height.saturating_sub(footer_h);
    let list_area = Rect {
        x: inner.x,
        y: inner.y,
        width: inner.width,
        height: list_h,
    };
    let footer_area = Rect {
        x: inner.x,
        y: inner.y + list_h,
        width: inner.width,
        height: footer_h,
    };

    let visible = list_h as usize;
    let total = picker.entries.len();
    let mut start = 0usize;
    if total > visible {
        if picker.selected >= visible {
            start = picker.selected + 1 - visible;
        }
        if start + visible > total {
            start = total - visible;
        }
    }

    let lines: Vec<Line<'_>> = picker
        .entries
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .map(|(i, e)| {
            let marker = if e.is_active_session && e.is_current_in_session {
                "●"
            } else {
                " "
            };
            let parent = match (e.parent_branch_name.as_deref(), e.fork_point_seq) {
                (Some(p), Some(seq)) => format!("  (forked from {p}@{seq})"),
                _ => String::new(),
            };
            let session_label = e
                .session_title
                .as_deref()
                .filter(|t| !t.is_empty())
                .unwrap_or(e.session_short.as_str());
            let row = format!(
                " {marker} [{}] {}  · {} msgs{}",
                session_label, e.name, e.message_count, parent
            );
            let style = if i == picker.selected {
                Style::default()
                    .add_modifier(Modifier::REVERSED)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Line::from(Span::styled(row, style))
        })
        .collect();

    let para = Paragraph::new(Text::from(lines));
    frame.render_widget(para, list_area);

    let footer = Line::from(Span::styled(
        " ↑/↓ select · Enter resume · Esc cancel",
        Style::default().fg(Color::DarkGray),
    ));
    frame.render_widget(Paragraph::new(footer), footer_area);
}

fn render_slash_popup(
    frame: &mut Frame<'_>,
    area: Rect,
    suggestions: &[&'static (&'static str, &'static str)],
    selected: usize,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let lines: Vec<Line<'_>> = suggestions
        .iter()
        .enumerate()
        .take(area.height as usize)
        .map(|(i, (cmd, hint))| {
            let style = if i == selected {
                Style::default()
                    .add_modifier(Modifier::REVERSED)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Cyan)
            };
            let hint_style = if i == selected {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            if hint.is_empty() {
                Line::from(Span::styled(format!(" {cmd}"), style))
            } else {
                Line::from(vec![
                    Span::styled(format!(" {cmd} "), style),
                    Span::styled((*hint).to_string(), hint_style),
                ])
            }
        })
        .collect();
    let para = Paragraph::new(Text::from(lines));
    frame.render_widget(para, area);
}

/// Script lines shown in the confirmation modal before the rest is
/// summarised as a count.
const CONFIRM_SCRIPT_MAX_LINES: usize = 12;

/// One `Line` per script line, with control characters other than the
/// newline shown as their escape so a script cannot hide a command
/// behind a carriage return or terminal escape.
fn script_lines(script: &str) -> Vec<Line<'static>> {
    fn visible(raw: &str) -> String {
        let mut out = String::with_capacity(raw.len());
        for c in raw.chars() {
            if c.is_control() {
                out.extend(c.escape_default());
            } else {
                out.push(c);
            }
        }
        out
    }
    let lines: Vec<&str> = script.split('\n').collect();
    let shown = lines.len().min(CONFIRM_SCRIPT_MAX_LINES);
    let mut out: Vec<Line<'static>> = lines[..shown]
        .iter()
        .map(|raw| Line::from(visible(raw)))
        .collect();
    if lines.len() > shown {
        out.push(Line::from(Span::styled(
            format!("… {} more line(s)", lines.len() - shown),
            Style::default().fg(Color::DarkGray),
        )));
    }
    out
}

fn render_confirmation_modal(frame: &mut Frame<'_>, area: Rect, modal: &ConfirmationModal) {
    let script = script_lines(&modal.request.script);
    let line_count = modal.request.script.split('\n').count();
    let width = (area.width.saturating_mul(3) / 5).clamp(40, 100);
    let body_rows = 3 + script.len();
    let height = (body_rows + 3)
        .max(6)
        .min(area.height.saturating_sub(2) as usize) as u16;
    let modal_area = centered_rect(area, width, height);

    frame.render_widget(Clear, modal_area);
    let block = Block::default()
        .title(Span::styled(
            " Confirm command ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(modal_area);
    frame.render_widget(block, modal_area);
    let [body_area, footer_area] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);

    let command_label = if line_count == 1 {
        "command:".to_string()
    } else {
        format!("command ({line_count} lines):")
    };
    let mut body = vec![
        Line::from(vec![
            Span::styled("reason: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                modal.request.matched_pattern.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            command_label,
            Style::default().fg(Color::DarkGray),
        )),
    ];
    body.extend(script);
    let para = Paragraph::new(Text::from(body)).wrap(Wrap { trim: false });
    frame.render_widget(para, body_area);

    let footer = if modal.armed() {
        let always = &modal.request.always_allow;
        let text = if always.is_empty() {
            "[y] run it   [n] / Esc cancel".to_string()
        } else {
            format!(
                "[y] run once   [a] always allow {}   [n] / Esc cancel",
                always.join(", ")
            )
        };
        Span::styled(text, Style::default().fg(Color::Green))
    } else {
        Span::styled(
            "read the command…   [n] / Esc cancel",
            Style::default().fg(Color::DarkGray),
        )
    };
    frame.render_widget(Paragraph::new(Line::from(footer)), footer_area);
}

fn render_output(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let slots = app.output.thumbnail_layout(area.width);
    let (lines, viewport_top) = app.output.render_view(area.width, area.height);
    for (line, row) in lines.iter().zip(area.rows()) {
        frame.render_widget(line, row);
    }

    let viewport_bottom = viewport_top.saturating_add(area.height as usize);
    const CAPTION_ROWS: u16 = 1;
    let image_height = THUMBNAIL_ROWS.saturating_sub(CAPTION_ROWS);
    if image_height > 0 {
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

    if app.generating {
        let row = Rect {
            x: area.x,
            y: area.y + area.height.saturating_sub(1),
            width: area.width,
            height: 1,
        };
        let label = format!("{} Generating…", app.spinner_char());
        let para = Paragraph::new(Line::from(Span::styled(
            label,
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::DIM)
                .add_modifier(Modifier::ITALIC),
        )));
        frame.render_widget(Clear, row);
        frame.render_widget(para, row);
    }
}

fn render_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.width == 0 {
        return;
    }

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
    let mut left_spans: Vec<Span<'_>> = Vec::new();
    if let Some(title) = app.session_title.as_deref() {
        left_spans.push(Span::styled(
            truncate_title(title),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::REVERSED),
        ));
        left_spans.push(Span::styled(" │ ", reversed));
    }
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
    left_spans.push(Span::styled(" │ ", reversed));
    left_spans.push(Span::styled(
        vision_label,
        Style::default()
            .fg(vision_color)
            .add_modifier(Modifier::REVERSED),
    ));
    let (verbose_label, verbose_color) = if app.verbose {
        ("verbose: on", Color::Green)
    } else {
        ("verbose: off", Color::DarkGray)
    };
    left_spans.push(Span::styled(" │ ", reversed));
    left_spans.push(Span::styled(
        verbose_label,
        Style::default()
            .fg(verbose_color)
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

fn truncate_title(title: &str) -> String {
    const MAX_CHARS: usize = 32;
    if title.chars().count() <= MAX_CHARS {
        return title.to_string();
    }
    let head: String = title.chars().take(MAX_CHARS - 1).collect();
    format!("{}…", head.trim_end())
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

fn render_input(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let prompt_w = INPUT_PROMPT.chars().count() as u16;
    let width = area.width;
    let buf_chars: Vec<char> = app.input.buffer().chars().collect();
    let rows = wrap_input(&buf_chars, prompt_w, width);

    let mut lines: Vec<Line<'_>> = Vec::with_capacity(rows.len());
    for (i, &(s, e)) in rows.iter().enumerate() {
        let mut row = String::new();
        if i == 0 {
            row.push_str(INPUT_PROMPT);
        }
        row.extend(&buf_chars[s..e]);
        lines.push(Line::from(row));
    }
    let para = Paragraph::new(Text::from(lines));
    frame.render_widget(para, area);

    let cursor = app.input.cursor_col() as usize;
    let (row_idx, col) = locate_cursor(&rows, cursor);
    let mut cursor_x = col as u16;
    if row_idx == 0 {
        cursor_x = cursor_x.saturating_add(prompt_w);
    }
    let cursor_y = row_idx as u16;
    let cy = (area.y + cursor_y).min(area.y + area.height.saturating_sub(1));
    let cx = (area.x + cursor_x).min(area.x + area.width.saturating_sub(1));
    frame.set_cursor_position(Position::new(cx, cy));
}

fn compute_input_height(frame_width: u16, frame_height: u16, buffer: &str) -> u16 {
    if frame_width == 0 {
        return 1;
    }
    let prompt_w = INPUT_PROMPT.chars().count() as u16;
    let chars: Vec<char> = buffer.chars().collect();
    let needed = wrap_input(&chars, prompt_w, frame_width).len() as u16;
    let cap = frame_height.saturating_sub(4).max(1);
    needed.clamp(1, cap)
}

fn wrap_input(buf: &[char], prompt_w: u16, width: u16) -> Vec<(usize, usize)> {
    let n = buf.len();
    if width == 0 {
        return vec![(0, n)];
    }
    let mut rows: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let cap = if rows.is_empty() {
            (width as usize).saturating_sub(prompt_w as usize).max(1)
        } else {
            width as usize
        };
        if n - start <= cap {
            rows.push((start, n));
            break;
        }
        let end_max = start + cap;
        let mut break_at: Option<usize> = None;
        let mut i = end_max;
        while i > start {
            i -= 1;
            if buf[i].is_whitespace() {
                break_at = Some(i + 1);
                break;
            }
        }
        let mut row_end = break_at.unwrap_or(end_max);
        if row_end <= start {
            row_end = end_max;
        }
        rows.push((start, row_end));
        start = row_end;
    }
    if rows.is_empty() {
        rows.push((0, 0));
    }
    let (ls, le) = *rows.last().expect("rows non-empty");
    let last_visible = (le - ls)
        + if rows.len() == 1 {
            prompt_w as usize
        } else {
            0
        };
    if last_visible >= width as usize {
        rows.push((n, n));
    }
    rows
}

fn locate_cursor(rows: &[(usize, usize)], cursor: usize) -> (usize, usize) {
    for (idx, &(s, e)) in rows.iter().enumerate() {
        if cursor < e {
            return (idx, cursor - s);
        }
        if cursor == e {
            if idx + 1 < rows.len() {
                return (idx + 1, 0);
            }
            return (idx, cursor - s);
        }
    }
    let last = rows.len().saturating_sub(1);
    let (s, e) = rows.get(last).copied().unwrap_or((0, 0));
    (last, cursor.saturating_sub(s).min(e - s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_text(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    fn test_app() -> App {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        App::new(
            std::sync::Arc::new(assistd_ipc::IpcClient::with_path(
                "/tmp/assistd-test-nonexistent.sock",
            )),
            tx,
            "test-model".into(),
            assistd_core::Config::default().sleep,
            false,
            None,
        )
    }

    fn output_rows(app: &mut App, width: u16, height: u16) -> Vec<String> {
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height))
                .expect("test terminal");
        terminal
            .draw(|frame| render_output(frame, frame.area(), app))
            .expect("draw");
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                let row: String = (0..width).map(|x| buffer[(x, y)].symbol()).collect();
                row.trim_end().to_string()
            })
            .collect()
    }

    #[test]
    fn output_pane_draws_the_scrolled_window() {
        let mut app = test_app();
        for i in 0..30 {
            app.output.push_info(&format!("line {i}"));
        }
        assert_eq!(
            output_rows(&mut app, 20, 3),
            ["line 27", "line 28", "line 29"]
        );
        app.output.scroll_lines_up(10);
        assert_eq!(
            output_rows(&mut app, 20, 3),
            ["line 17", "line 18", "line 19"]
        );
        app.output.clear();
        app.output.push_info("only");
        assert_eq!(output_rows(&mut app, 20, 3), ["only", "", ""]);
    }

    #[test]
    fn centered_rect_fits_inside_a_narrow_area() {
        let area = Rect::new(3, 2, 20, 5);
        assert_eq!(centered_rect(area, 40, 8), area);
        assert_eq!(centered_rect(area, 10, 3), Rect::new(8, 3, 10, 3));
    }

    #[test]
    fn branch_picker_clamps_to_a_pane_narrower_than_its_floor() {
        let picker = BranchPickerModal {
            entries: Vec::new(),
            selected: 0,
        };
        let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(20, 4))
            .expect("test terminal");
        terminal
            .draw(|frame| render_branch_picker_modal(frame, frame.area(), &picker))
            .expect("draw");
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer[(0, 0)].symbol(), "┌");
        assert_eq!(buffer[(19, 3)].symbol(), "┘");
    }

    #[test]
    fn script_lines_splits_multiline_scripts() {
        let lines = script_lines("echo ok\nrm -rf ~");
        let texts: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(texts, vec!["echo ok", "rm -rf ~"]);
    }

    #[test]
    fn script_lines_make_control_characters_visible() {
        let lines = script_lines("echo ok\r\x1b[2Krm -rf ~\tnow");
        let texts: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(texts, vec!["echo ok\\r\\u{1b}[2Krm -rf ~\\tnow"]);
    }

    #[test]
    fn script_lines_summarise_the_overflow() {
        let script = (0..CONFIRM_SCRIPT_MAX_LINES + 5)
            .map(|i| format!("cmd{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let texts: Vec<String> = script_lines(&script).iter().map(line_text).collect();
        let mut expected: Vec<String> = (0..CONFIRM_SCRIPT_MAX_LINES)
            .map(|i| format!("cmd{i}"))
            .collect();
        expected.push("… 5 more line(s)".into());
        assert_eq!(texts, expected);
    }

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

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn wrap_input_breaks_at_word_boundary() {
        // width=10, prompt=2 → first row cap=8.
        // "hello world this is" should break after "hello ", not split a word.
        let rows = wrap_input(&chars("hello world this is"), 2, 10);
        let texts: Vec<String> = rows
            .iter()
            .map(|&(s, e)| chars("hello world this is")[s..e].iter().collect())
            .collect();
        assert_eq!(texts, vec!["hello ", "world ", "this is"]);
    }

    #[test]
    fn wrap_input_hard_breaks_when_word_is_longer_than_row() {
        let rows = wrap_input(&chars("hellotherefriend"), 2, 10);
        assert_eq!(rows, vec![(0, 8), (8, 16)]);
    }

    #[test]
    fn wrap_input_adds_phantom_row_when_last_row_full() {
        // The cursor after a row that fills the width needs a row of its own.
        let rows = wrap_input(&chars("aaaaaaaa"), 2, 10);
        assert_eq!(rows, vec![(0, 8), (8, 8)]);
    }

    #[test]
    fn wrap_input_empty_buffer_has_one_row() {
        let rows = wrap_input(&[], 2, 10);
        assert_eq!(rows, vec![(0, 0)]);
    }

    #[test]
    fn locate_cursor_jumps_to_next_row_on_boundary() {
        // "hello " then "world" with cursor right at the wrap → row 1 col 0.
        let buf = chars("hello world");
        let rows = wrap_input(&buf, 2, 10);
        // rows: [(0,6), (6,11)]
        assert_eq!(locate_cursor(&rows, 0), (0, 0));
        assert_eq!(locate_cursor(&rows, 5), (0, 5));
        assert_eq!(locate_cursor(&rows, 6), (1, 0));
        assert_eq!(locate_cursor(&rows, 11), (1, 5));
    }

    #[test]
    fn locate_cursor_lands_on_phantom_row_after_full_row() {
        let buf = chars("aaaaaaaa");
        let rows = wrap_input(&buf, 2, 10);
        // Cursor at end with full row → phantom row col 0.
        assert_eq!(locate_cursor(&rows, 8), (1, 0));
    }
}

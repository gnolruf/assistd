//! ratatui render for the chat TUI. Mutates only layout outputs: the
//! output pane's wrap cache and the app's last viewport height.

use std::time::{Duration, Instant};

use assistd_core::{PresenceState, VoiceCaptureState};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui_image::StatefulImage;

use super::app::{App, BranchListEntry, BranchPickerModal, ConfirmationModal};
use super::output::{OutputPane, THUMBNAIL_ROWS, ThumbnailSlot};
use super::vram::{RamState, VramState};

const INPUT_PROMPT: &str = "> ";
const SLASH_POPUP_MAX_ROWS: u16 = 6;
/// Script lines shown in the confirmation modal before the rest is
/// summarised as a count.
const CONFIRM_SCRIPT_MAX_LINES: usize = 12;
/// Rows of a thumbnail slot taken by its filename caption.
const THUMBNAIL_CAPTION_ROWS: u16 = 1;
const THUMBNAIL_MAX_COLS: u16 = 32;
const SESSION_TITLE_MAX_CHARS: usize = 32;

pub fn render(frame: &mut Frame<'_>, app: &mut App) {
    let frame_area = frame.area();
    let input_height =
        compute_input_height(frame_area.width, frame_area.height, app.input.buffer());
    let suggestions = app.slash_suggestions();
    let popup_height = (suggestions.len() as u16).min(SLASH_POPUP_MAX_ROWS);
    let [output_area, popup_area, status_area, input_area] = Layout::vertical([
        Constraint::Min(3),
        Constraint::Length(popup_height),
        Constraint::Length(1),
        Constraint::Length(input_height),
    ])
    .areas(frame_area);

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

/// Clear `area`, draw a bordered block titled `title` in `color`, and
/// return its inner area.
fn render_modal_frame(
    frame: &mut Frame<'_>,
    area: Rect,
    title: &'static str,
    color: Color,
) -> Rect {
    frame.render_widget(Clear, area);
    let block = Block::default()
        .title(Span::styled(
            title,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    inner
}

fn render_branch_picker_modal(frame: &mut Frame<'_>, area: Rect, picker: &BranchPickerModal) {
    let width = (area.width.saturating_mul(4) / 5).clamp(50, 120);
    let height = (picker.entries.len() as u16 + 4)
        .min(area.height.saturating_sub(2))
        .max(6);
    let inner = render_modal_frame(
        frame,
        centered_rect(area, width, height),
        " Resume conversation ",
        Color::Cyan,
    );

    let list_height = inner.height.saturating_sub(1);
    let list_area = Rect {
        height: list_height,
        ..inner
    };
    let footer_area = Rect {
        y: inner.y + list_height,
        height: 1,
        ..inner
    };

    let visible = usize::from(list_height);
    let start = picker_scroll_start(picker.selected, picker.entries.len(), visible);
    let lines: Vec<Line<'_>> = picker
        .entries
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .map(|(i, entry)| picker_row(entry, i == picker.selected))
        .collect();
    frame.render_widget(Paragraph::new(Text::from(lines)), list_area);

    let footer = Line::from(Span::styled(
        " ↑/↓ select · Enter resume · Esc cancel",
        Style::default().fg(Color::DarkGray),
    ));
    frame.render_widget(Paragraph::new(footer), footer_area);
}

/// First entry of a `visible`-row window that keeps `selected` in view.
fn picker_scroll_start(selected: usize, total: usize, visible: usize) -> usize {
    if total <= visible {
        return 0;
    }
    (selected + 1).saturating_sub(visible).min(total - visible)
}

fn picker_row(entry: &BranchListEntry, selected: bool) -> Line<'static> {
    let marker = if entry.is_active_session && entry.is_current_in_session {
        "●"
    } else {
        " "
    };
    let parent = match (entry.parent_branch_name.as_deref(), entry.fork_point_seq) {
        (Some(p), Some(seq)) => format!("  (forked from {p}@{seq})"),
        _ => String::new(),
    };
    let session_label = entry
        .session_title
        .as_deref()
        .filter(|t| !t.is_empty())
        .unwrap_or(entry.session_short.as_str());
    let row = format!(
        " {marker} [{session_label}] {}  · {} msgs{parent}",
        entry.name, entry.message_count
    );
    let style = if selected {
        selected_style()
    } else {
        Style::default()
    };
    Line::from(Span::styled(row, style))
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
            let (style, hint_style) = if i == selected {
                (selected_style(), reversed_style())
            } else {
                (
                    Style::default().fg(Color::Cyan),
                    Style::default().fg(Color::DarkGray),
                )
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
    frame.render_widget(Paragraph::new(Text::from(lines)), area);
}

/// One `Line` per script line, with control characters other than the
/// newline escaped so a script cannot hide a command behind a carriage
/// return or terminal escape.
fn script_lines(script: &str) -> Vec<Line<'static>> {
    let lines: Vec<&str> = script.split('\n').collect();
    let shown = lines.len().min(CONFIRM_SCRIPT_MAX_LINES);
    let mut out: Vec<Line<'static>> = lines[..shown]
        .iter()
        .map(|raw| Line::from(escape_controls(raw)))
        .collect();
    if lines.len() > shown {
        out.push(Line::from(Span::styled(
            format!("… {} more line(s)", lines.len() - shown),
            Style::default().fg(Color::DarkGray),
        )));
    }
    out
}

fn escape_controls(raw: &str) -> String {
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

fn render_confirmation_modal(frame: &mut Frame<'_>, area: Rect, modal: &ConfirmationModal) {
    let script = script_lines(&modal.request.script);
    let line_count = modal.request.script.split('\n').count();
    let width = (area.width.saturating_mul(3) / 5).clamp(40, 100);
    let body_rows = 3 + script.len();
    let height = (body_rows + 3)
        .max(6)
        .min(area.height.saturating_sub(2) as usize) as u16;
    let inner = render_modal_frame(
        frame,
        centered_rect(area, width, height),
        " Confirm command ",
        Color::Yellow,
    );
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

    let footer = Line::from(confirmation_footer(modal));
    frame.render_widget(Paragraph::new(footer), footer_area);
}

fn confirmation_footer(modal: &ConfirmationModal) -> Span<'static> {
    if !modal.armed() {
        return Span::styled(
            "read the command…   [n] / Esc cancel",
            Style::default().fg(Color::DarkGray),
        );
    }
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
    render_thumbnails(frame, area, &mut app.output, &slots, viewport_top);

    if app.generating {
        render_generating_row(frame, area, app.spinner_char());
    }
}

/// Draw each thumbnail whose image rows sit wholly inside the viewport
/// starting at wrapped row `viewport_top`.
fn render_thumbnails(
    frame: &mut Frame<'_>,
    area: Rect,
    output: &mut OutputPane,
    slots: &[ThumbnailSlot],
    viewport_top: usize,
) {
    let image_height = THUMBNAIL_ROWS.saturating_sub(THUMBNAIL_CAPTION_ROWS);
    let viewport_bottom = viewport_top.saturating_add(area.height as usize);
    for slot in slots {
        let image_start = slot.start_row + THUMBNAIL_CAPTION_ROWS as usize;
        let image_end = slot.start_row + slot.height;
        if image_start < viewport_top || image_end > viewport_bottom {
            continue;
        }
        let local_row = (image_start - viewport_top) as u16;
        if local_row >= area.height {
            continue;
        }
        let rect = Rect {
            x: area.x,
            y: area.y + local_row,
            width: area.width.min(THUMBNAIL_MAX_COLS),
            height: image_height.min(area.height - local_row),
        };
        if let Some(state) = output.thumbnail_protocol_mut(slot.item_idx) {
            frame.render_stateful_widget(StatefulImage::default(), rect, state);
        }
    }
}

fn render_generating_row(frame: &mut Frame<'_>, area: Rect, spinner: char) {
    let row = Rect {
        y: area.y + area.height.saturating_sub(1),
        height: 1,
        ..area
    };
    let para = Paragraph::new(Line::from(Span::styled(
        format!("{spinner} Generating…"),
        Style::default()
            .fg(Color::Gray)
            .add_modifier(Modifier::DIM)
            .add_modifier(Modifier::ITALIC),
    )));
    frame.render_widget(Clear, row);
    frame.render_widget(para, row);
}

/// Indicators on the left, and a notice or key hints right-aligned when
/// they fit.
fn render_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.width == 0 {
        return;
    }
    let mut spans = status_indicators(app);
    let left_len: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    let hint = status_hint(app);
    let hint_len = hint.chars().count();
    let width = area.width as usize;
    if left_len + hint_len < width {
        spans.push(Span::styled(
            " ".repeat(width - left_len - hint_len),
            reversed_style(),
        ));
        spans.push(Span::styled(
            hint,
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::REVERSED),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn status_indicators(app: &App) -> Vec<Span<'static>> {
    let reversed = reversed_style();
    let mut spans = Vec::new();
    if let Some(title) = app.session_title.as_deref() {
        spans.push(Span::styled(truncate_title(title), highlight_style()));
        spans.push(Span::styled(" │ ", reversed));
    }
    spans.push(Span::styled(format!("model: {}", app.model_name), reversed));
    if let Some((color, label)) = presence_dot(app.presence_state) {
        push_indicator(&mut spans, "●".to_string(), color, label);
    }
    if let Some(remaining) = app.local_time_until_next_transition() {
        spans.push(Span::styled(
            format!(" ({})", format_countdown(remaining)),
            reversed,
        ));
    }
    if let Some((color, label)) = voice_indicator(app.listening) {
        push_indicator(&mut spans, app.spinner_char().to_string(), color, label);
    }
    push_toggle(&mut spans, "vision", app.vision_enabled);
    push_toggle(&mut spans, "verbose", app.verbose);
    let pending_count = app.pending_attachments.len();
    if pending_count > 0 {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!("📎×{pending_count}"),
            highlight_style(),
        ));
    }
    if let Some(rate) = app.throughput.snapshot(Instant::now()).rate {
        spans.push(Span::styled(format!(" │ {rate:.0} tok/s"), reversed));
    }
    spans.push(Span::styled(
        format!(" │ RAM: {}", ram_label(&app.resources.ram)),
        reversed,
    ));
    spans.push(Span::styled(
        format!(" │ VRAM: {}", vram_label(&app.resources.vram)),
        reversed,
    ));
    spans
}

fn push_indicator(spans: &mut Vec<Span<'static>>, glyph: String, color: Color, label: &str) {
    spans.push(Span::raw(" "));
    spans.push(Span::styled(
        glyph,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(format!(" {label}"), reversed_style()));
}

fn push_toggle(spans: &mut Vec<Span<'static>>, name: &str, on: bool) {
    let (state, color) = if on {
        ("on", Color::Green)
    } else {
        ("off", Color::DarkGray)
    };
    spans.push(Span::styled(" │ ", reversed_style()));
    spans.push(Span::styled(
        format!("{name}: {state}"),
        Style::default().fg(color).add_modifier(Modifier::REVERSED),
    ));
}

fn status_hint(app: &App) -> String {
    if let Some(notice) = app.notice() {
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
    }
}

fn vram_label(state: &VramState) -> String {
    match state {
        VramState::Unknown => "…".to_string(),
        VramState::Disabled => "N/A".to_string(),
        VramState::Ok(info) => gib_label(info.used_mb, info.total_mb),
        VramState::Err(_) => "err".to_string(),
    }
}

fn ram_label(state: &RamState) -> String {
    match state {
        RamState::Unknown => "…".to_string(),
        RamState::Ok(info) => gib_label(info.used_mb, info.total_mb),
    }
}

fn gib_label(used_mb: u64, total_mb: u64) -> String {
    format!(
        "{:.1}/{:.1} GiB",
        used_mb as f64 / 1024.0,
        total_mb as f64 / 1024.0,
    )
}

fn reversed_style() -> Style {
    Style::default().add_modifier(Modifier::REVERSED)
}

fn selected_style() -> Style {
    Style::default()
        .add_modifier(Modifier::REVERSED)
        .add_modifier(Modifier::BOLD)
}

fn highlight_style() -> Style {
    Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD)
        .add_modifier(Modifier::REVERSED)
}

fn truncate_title(title: &str) -> String {
    if title.chars().count() <= SESSION_TITLE_MAX_CHARS {
        return title.to_string();
    }
    let head: String = title.chars().take(SESSION_TITLE_MAX_CHARS - 1).collect();
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

fn render_input(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let prompt_w = INPUT_PROMPT.chars().count() as u16;
    let buf_chars: Vec<char> = app.input.buffer().chars().collect();
    let rows = wrap_input(&buf_chars, prompt_w, area.width);

    let lines: Vec<Line<'_>> = rows
        .iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let mut row = String::new();
            if i == 0 {
                row.push_str(INPUT_PROMPT);
            }
            row.extend(&buf_chars[start..end]);
            Line::from(row)
        })
        .collect();
    frame.render_widget(Paragraph::new(Text::from(lines)), area);

    let (row_idx, col) = locate_cursor(&rows, app.input.cursor_col() as usize);
    let mut cursor_x = col as u16;
    if row_idx == 0 {
        cursor_x = cursor_x.saturating_add(prompt_w);
    }
    let cy = (area.y + row_idx as u16).min(area.y + area.height.saturating_sub(1));
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

/// `(start, end)` char ranges of each input row, breaking after the last
/// whitespace that fits. A row that fills the width is followed by an
/// empty one for the cursor.
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
        let row_end = buf[start..end_max]
            .iter()
            .rposition(|c| c.is_whitespace())
            .map_or(end_max, |i| start + i + 1);
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
    use std::sync::Arc;

    use assistd_core::Config;
    use assistd_ipc::IpcClient;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use tokio::sync::mpsc;

    use super::*;

    fn line_text(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    fn test_app() -> App {
        let (tx, _rx) = mpsc::channel(1);
        App::new(
            Arc::new(IpcClient::with_path("/tmp/assistd-test-nonexistent.sock")),
            tx,
            "test-model".into(),
            Config::default().sleep,
            false,
            None,
        )
    }

    fn output_rows(app: &mut App, width: u16, height: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("test terminal");
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
        let mut terminal = Terminal::new(TestBackend::new(20, 4)).expect("test terminal");
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
        assert_eq!(compute_input_height(10, 24, ""), 1);
        assert_eq!(compute_input_height(10, 24, &"a".repeat(7)), 1);
        assert_eq!(
            compute_input_height(10, 24, &"a".repeat(8)),
            2,
            "a full first row wraps the cursor onto a second"
        );
        assert_eq!(compute_input_height(10, 24, &"a".repeat(18)), 3);
    }

    #[test]
    fn input_height_capped_to_leave_room_for_output_and_status() {
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
        let buf = chars("hello world");
        let rows = wrap_input(&buf, 2, 10);
        assert_eq!(rows, [(0, 6), (6, 11)]);
        assert_eq!(locate_cursor(&rows, 0), (0, 0));
        assert_eq!(locate_cursor(&rows, 5), (0, 5));
        assert_eq!(locate_cursor(&rows, 6), (1, 0));
        assert_eq!(locate_cursor(&rows, 11), (1, 5));
    }

    #[test]
    fn locate_cursor_lands_on_phantom_row_after_full_row() {
        let buf = chars("aaaaaaaa");
        let rows = wrap_input(&buf, 2, 10);
        assert_eq!(locate_cursor(&rows, 8), (1, 0));
    }
}

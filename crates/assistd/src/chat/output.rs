//! Scrollable output pane: prose lines, tool blocks, thinking blocks and
//! thumbnails, wrapped to the viewport width on render.

use std::time::Instant;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui_image::protocol::StatefulProtocol;

/// Rows an inline thumbnail reserves.
pub const THUMBNAIL_ROWS: u16 = 8;

/// One tool invocation: command, output as the daemon delivered it
/// (already truncated, with its banner and footer), exit status, timing.
#[derive(Debug, Clone)]
pub struct ToolBlock {
    pub command: String,
    pub output: String,
    pub exit_code: i32,
    pub duration_ms: u64,
    pub expanded: bool,
}

pub struct ThumbnailItem {
    pub name: String,
    pub protocol: StatefulProtocol,
}

/// One reasoning phase, shown as an expandable block that auto-collapses
/// when it finishes.
#[derive(Debug, Clone)]
pub struct ThinkingBlock {
    pub text: String,
    pub started_at: Instant,
    /// `None` while still receiving deltas.
    pub ended_at: Option<Instant>,
    pub expanded: bool,
}

enum OutputItem {
    Text(Line<'static>),
    Tool(ToolBlock),
    Thumbnail(Box<ThumbnailItem>),
    Thinking(ThinkingBlock),
}

pub struct OutputPane {
    items: Vec<OutputItem>,
    open_assistant: Option<usize>,
    scroll_offset: u16,
    wrap_cache: Option<(u16, Vec<Line<'static>>)>,
    dirty: bool,
    /// Renders every thinking and tool block expanded, leaving their
    /// per-item flags untouched.
    verbose: bool,
}

impl Default for OutputPane {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputPane {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            open_assistant: None,
            scroll_offset: 0,
            wrap_cache: None,
            dirty: true,
            verbose: false,
        }
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        if self.verbose != verbose {
            self.verbose = verbose;
            self.dirty = true;
        }
    }

    /// Append a `> `-prefixed prompt line and a blank separator.
    pub fn push_user(&mut self, text: &str) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            format!("> {text}"),
            user_style(),
        )));
        self.items.push(OutputItem::Text(Line::from("")));
        self.dirty = true;
    }

    /// [`Self::push_user`] with a trailing 📎 tag naming the attachments.
    pub fn push_user_with_attachments(&mut self, text: &str, names: &[String]) {
        self.close_open_assistant();
        let tag = if names.len() == 1 {
            format!("  📎 {}", names[0])
        } else {
            format!("  📎 {} ({} files)", names.join(", "), names.len())
        };
        self.items.push(OutputItem::Text(single_span_line(
            format!("> {text}{tag}"),
            user_style(),
        )));
        self.items.push(OutputItem::Text(Line::from("")));
        self.dirty = true;
    }

    pub fn push_info(&mut self, text: &str) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            text.to_string(),
            info_style(),
        )));
        self.dirty = true;
    }

    /// Open a streaming assistant block for [`Self::append_assistant`].
    pub fn begin_assistant(&mut self) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            String::new(),
            assistant_style(),
        )));
        self.open_assistant = Some(self.items.len() - 1);
        self.dirty = true;
    }

    pub fn append_assistant(&mut self, delta: &str) {
        let mut idx = match self.open_assistant {
            Some(i) => i,
            None => {
                self.begin_assistant();
                self.items.len() - 1
            }
        };
        let mut fragments = delta.split('\n');
        if let Some(first) = fragments.next() {
            if let Some(line) = self.text_at_mut(idx) {
                append_to_line(line, first);
            }
        }
        for frag in fragments {
            self.items.push(OutputItem::Text(single_span_line(
                frag.to_string(),
                assistant_style(),
            )));
            idx = self.items.len() - 1;
        }
        self.open_assistant = Some(idx);
        self.dirty = true;
    }

    pub fn finish_assistant(&mut self) {
        if self.open_assistant.is_none() {
            return;
        }
        self.open_assistant = None;
        self.items.push(OutputItem::Text(Line::from("")));
        self.dirty = true;
    }

    pub fn push_error(&mut self, msg: &str) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            format!("!! {msg}"),
            error_style(),
        )));
        self.dirty = true;
    }

    /// Blocks whose body exceeds [`COLLAPSE_THRESHOLD`] lines start
    /// collapsed.
    pub fn push_tool_block(
        &mut self,
        command: String,
        output: String,
        exit_code: i32,
        duration_ms: u64,
    ) {
        self.close_open_assistant();
        let expanded = body_line_count(&output) <= COLLAPSE_THRESHOLD;
        self.items.push(OutputItem::Tool(ToolBlock {
            command,
            output,
            exit_code,
            duration_ms,
            expanded,
        }));
        self.dirty = true;
    }

    /// Toggle the most recent tool or thinking block.
    pub fn toggle_last_expandable(&mut self) -> bool {
        for item in self.items.iter_mut().rev() {
            match item {
                OutputItem::Tool(b) => {
                    b.expanded = !b.expanded;
                    self.dirty = true;
                    return true;
                }
                OutputItem::Thinking(t) => {
                    t.expanded = !t.expanded;
                    self.dirty = true;
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    /// Open a collapsed thinking block.
    pub fn begin_thinking(&mut self) {
        self.close_open_assistant();
        self.items.push(OutputItem::Thinking(ThinkingBlock {
            text: String::new(),
            started_at: Instant::now(),
            ended_at: None,
            expanded: false,
        }));
        self.dirty = true;
    }

    /// Append to the live thinking block, opening one if the trailing
    /// item is not a live block.
    pub fn append_thinking(&mut self, delta: &str) {
        let needs_new = !matches!(
            self.items.last(),
            Some(OutputItem::Thinking(t)) if t.ended_at.is_none()
        );
        if needs_new {
            self.begin_thinking();
        }
        if let Some(OutputItem::Thinking(t)) = self.items.last_mut() {
            t.text.push_str(delta);
        }
        self.dirty = true;
    }

    /// Stamp and collapse the live thinking block. No-op when none is
    /// live.
    pub fn finish_thinking(&mut self) {
        for item in self.items.iter_mut().rev() {
            if let OutputItem::Thinking(t) = item {
                if t.ended_at.is_none() {
                    t.ended_at = Some(Instant::now());
                    t.expanded = false;
                    self.dirty = true;
                }
                return;
            }
        }
    }

    /// Whole seconds the live thinking block has run, if any.
    pub fn live_thinking_seconds(&self) -> Option<u64> {
        for item in self.items.iter().rev() {
            if let OutputItem::Thinking(t) = item {
                if t.ended_at.is_none() {
                    return Some(t.started_at.elapsed().as_secs());
                }
            }
        }
        None
    }

    /// Force a rewrap on the next render.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.open_assistant = None;
        self.scroll_offset = 0;
        self.dirty = true;
    }

    /// Drop everything from the most recent user prompt onward. Returns
    /// the number of items removed.
    pub fn pop_last_user_exchange(&mut self) -> usize {
        let mut idx = self.items.len();
        while idx > 0 {
            idx -= 1;
            if let OutputItem::Text(line) = &self.items[idx]
                && line
                    .spans
                    .first()
                    .map(|s| s.content.starts_with("> "))
                    .unwrap_or(false)
            {
                let removed = self.items.len() - idx;
                self.items.truncate(idx);
                self.open_assistant = None;
                self.dirty = true;
                return removed;
            }
        }
        0
    }

    pub fn scroll_page_up(&mut self, viewport_height: u16) {
        let step = (viewport_height / 2).max(1);
        self.scroll_offset = self.scroll_offset.saturating_add(step);
    }

    pub fn scroll_page_down(&mut self, viewport_height: u16) {
        let step = (viewport_height / 2).max(1);
        self.scroll_offset = self.scroll_offset.saturating_sub(step);
    }

    pub fn scroll_lines_up(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(lines.max(1));
    }

    pub fn scroll_lines_down(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines.max(1));
    }

    pub fn reset_scroll(&mut self) {
        self.scroll_offset = 0;
    }

    /// Offset in wrapped lines; 0 is pinned to the bottom.
    pub fn scroll_offset(&self) -> u16 {
        self.scroll_offset
    }

    /// The wrapped lines and the index of the first one in the viewport.
    /// Clamps the scroll offset to the wrapped total.
    pub fn render_view(&mut self, width: u16, height: u16) -> (&[Line<'static>], u16) {
        let wrapped_len = self.wrapped(width).len();
        let max_offset = wrapped_len.saturating_sub(height as usize) as u16;
        if self.scroll_offset > max_offset {
            self.scroll_offset = max_offset;
        }
        let start = wrapped_len
            .saturating_sub(height as usize)
            .saturating_sub(self.scroll_offset as usize) as u16;
        let lines = &self
            .wrap_cache
            .as_ref()
            .expect("wrap_cache populated by self.wrapped() above")
            .1;
        (lines, start)
    }

    fn wrapped(&mut self, width: u16) -> &[Line<'static>] {
        let needs_rewrap = self.dirty
            || self
                .wrap_cache
                .as_ref()
                .map(|(w, _)| *w != width)
                .unwrap_or(true);
        if needs_rewrap {
            let wrapped = self.rewrap(width);
            self.wrap_cache = Some((width, wrapped));
            self.dirty = false;
        }
        &self
            .wrap_cache
            .as_ref()
            .expect("wrap_cache populated by branch above")
            .1
    }

    fn rewrap(&self, width: u16) -> Vec<Line<'static>> {
        if width == 0 {
            return self
                .items
                .iter()
                .map(|it| match it {
                    OutputItem::Text(l) => l.clone(),
                    OutputItem::Tool(b) => {
                        single_span_line(format!("$ {}", b.command), tool_call_style())
                    }
                    OutputItem::Thumbnail(t) => {
                        single_span_line(format!("📎 {}", t.name), info_style())
                    }
                    OutputItem::Thinking(t) => {
                        single_span_line(thinking_header_text(t), thinking_header_style())
                    }
                })
                .collect();
        }
        let mut out = Vec::with_capacity(self.items.len() * 2);
        for item in &self.items {
            match item {
                OutputItem::Text(line) => wrap_line_into(&mut out, line, width),
                OutputItem::Tool(b) => render_tool_block(&mut out, b, width, self.verbose),
                OutputItem::Thumbnail(t) => render_thumbnail_placeholder(&mut out, t),
                OutputItem::Thinking(t) => render_thinking_block(&mut out, t, width, self.verbose),
            }
        }
        out
    }

    fn close_open_assistant(&mut self) {
        if let Some(idx) = self.open_assistant.take() {
            let empty = matches!(
                self.items.get(idx),
                Some(OutputItem::Text(l)) if l.spans.iter().all(|s| s.content.is_empty())
            );
            if empty && idx + 1 == self.items.len() {
                self.items.pop();
            } else {
                self.items.push(OutputItem::Text(Line::from("")));
            }
            self.dirty = true;
        }
    }

    fn text_at_mut(&mut self, idx: usize) -> Option<&mut Line<'static>> {
        match self.items.get_mut(idx)? {
            OutputItem::Text(line) => Some(line),
            OutputItem::Tool(_) | OutputItem::Thumbnail(_) | OutputItem::Thinking(_) => None,
        }
    }

    /// Reserve [`THUMBNAIL_ROWS`] rows for the renderer to draw over.
    pub fn push_thumbnail(&mut self, name: String, protocol: StatefulProtocol) {
        self.close_open_assistant();
        self.items
            .push(OutputItem::Thumbnail(Box::new(ThumbnailItem {
                name,
                protocol,
            })));
        self.dirty = true;
    }

    /// Where each thumbnail sits in the wrapped output.
    pub fn thumbnail_layout(&mut self, width: u16) -> Vec<ThumbnailSlot> {
        let _ = self.wrapped(width);
        let mut slots = Vec::new();
        let mut row: usize = 0;
        for (idx, item) in self.items.iter().enumerate() {
            let height = match item {
                OutputItem::Text(line) => wrapped_text_rows(line, width),
                OutputItem::Tool(b) => wrapped_tool_rows(b, width, self.verbose),
                OutputItem::Thumbnail(_) => THUMBNAIL_ROWS as usize,
                OutputItem::Thinking(t) => wrapped_thinking_rows(t, width, self.verbose),
            };
            if matches!(item, OutputItem::Thumbnail(_)) {
                slots.push(ThumbnailSlot {
                    item_idx: idx,
                    start_row: row,
                    height,
                });
            }
            row += height;
        }
        slots
    }

    /// `None` when the item at `idx` is not a thumbnail.
    pub fn thumbnail_protocol_mut(&mut self, idx: usize) -> Option<&mut StatefulProtocol> {
        match self.items.get_mut(idx)? {
            OutputItem::Thumbnail(t) => Some(&mut t.protocol),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThumbnailSlot {
    pub item_idx: usize,
    /// First row of the reserved area in the wrapped output.
    pub start_row: usize,
    pub height: usize,
}

fn render_thumbnail_placeholder(out: &mut Vec<Line<'static>>, t: &ThumbnailItem) {
    out.push(single_span_line(format!("📎 {}", t.name), info_style()));
    for _ in 1..THUMBNAIL_ROWS {
        out.push(Line::from(""));
    }
}

fn wrapped_text_rows(line: &Line<'static>, width: u16) -> usize {
    if width == 0 {
        return 1;
    }
    let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    if content.is_empty() {
        return 1;
    }
    let n = textwrap::wrap(&content, width as usize).len();
    n.max(1)
}

fn wrapped_thinking_rows(t: &ThinkingBlock, width: u16, verbose: bool) -> usize {
    if width == 0 {
        return 1;
    }
    let inner_w = width.saturating_sub(2).max(1) as usize;
    let header = 1;
    let separator = 1;
    let show_body = (t.expanded || verbose) && !t.text.is_empty();
    let body: usize = if show_body {
        t.text
            .lines()
            .map(|line| textwrap::wrap(line, inner_w).len().max(1))
            .sum()
    } else {
        0
    };
    header + body + separator
}

fn wrapped_tool_rows(b: &ToolBlock, width: u16, verbose: bool) -> usize {
    if width == 0 {
        return 1;
    }
    let inner_w = width.saturating_sub(2).max(1) as usize;
    let mut rows: usize = 0;
    rows += textwrap::wrap(&format!("$ {}", b.command), inner_w)
        .len()
        .max(1);
    let body_lines = split_body(&b.output);
    let collapsed = !verbose && !b.expanded && body_lines.len() > COLLAPSE_THRESHOLD;
    let visible_idxs = visible_body_indices(&body_lines, collapsed);
    for i in &visible_idxs {
        let n = textwrap::wrap(&body_lines[*i], inner_w).len().max(1);
        rows += n;
    }
    if collapsed && body_lines.len() > visible_idxs.len() {
        rows += 1;
    }
    rows + 2
}

fn wrap_line_into(out: &mut Vec<Line<'static>>, line: &Line<'static>, width: u16) {
    let style = line.spans.first().map(|s| s.style).unwrap_or_default();
    let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    if content.is_empty() {
        out.push(Line::from(""));
        return;
    }
    let wrapped = textwrap::wrap(&content, width as usize);
    if wrapped.is_empty() {
        out.push(Line::from(""));
        return;
    }
    let pad_to_width = style.bg.is_some();
    for chunk in wrapped {
        let mut s = chunk.into_owned();
        if pad_to_width {
            let visible = s.chars().count();
            let cap = width as usize;
            if visible < cap {
                s.push_str(&" ".repeat(cap - visible));
            }
        }
        out.push(single_span_line(s, style));
    }
}

fn render_thinking_block(
    out: &mut Vec<Line<'static>>,
    t: &ThinkingBlock,
    width: u16,
    verbose: bool,
) {
    let bar = Span::styled("▎ ", thinking_bar_style());
    let inner_w = width.saturating_sub(2).max(1);
    push_barred(
        out,
        &bar,
        &thinking_header_text(t),
        thinking_header_style(),
        inner_w,
    );
    if (t.expanded || verbose) && !t.text.is_empty() {
        for line in t.text.lines() {
            if line.is_empty() {
                out.push(Line::from(vec![bar.clone()]));
            } else {
                push_barred(out, &bar, line, thinking_text_style(), inner_w);
            }
        }
    }
    out.push(Line::from(""));
}

fn thinking_header_text(t: &ThinkingBlock) -> String {
    let elapsed = match t.ended_at {
        Some(end) => end.saturating_duration_since(t.started_at),
        None => t.started_at.elapsed(),
    };
    let secs = elapsed.as_secs();
    if t.ended_at.is_some() {
        format!("✦ Thought for {secs}s")
    } else {
        format!("✻ Thinking… ({secs}s)")
    }
}

fn render_tool_block(out: &mut Vec<Line<'static>>, b: &ToolBlock, width: u16, verbose: bool) {
    let bar_color = if b.exit_code == 0 {
        Color::Green
    } else {
        Color::Red
    };
    let bar = Span::styled(
        "▎ ",
        Style::default().fg(bar_color).add_modifier(Modifier::BOLD),
    );
    let inner_w = width.saturating_sub(2).max(1);

    push_barred(
        out,
        &bar,
        &format!("$ {}", b.command),
        header_style(),
        inner_w,
    );

    let body_lines = split_body(&b.output);
    let collapsed = !verbose && !b.expanded && body_lines.len() > COLLAPSE_THRESHOLD;
    let visible_idxs = visible_body_indices(&body_lines, collapsed);

    for i in &visible_idxs {
        let line = &body_lines[*i];
        let style = if line.starts_with("[stderr] ") {
            stderr_style()
        } else {
            tool_result_style()
        };
        push_barred(out, &bar, line, style, inner_w);
    }
    if collapsed {
        let hidden = body_lines.len() - visible_idxs.len();
        if hidden > 0 {
            push_barred(
                out,
                &bar,
                &format!("… ({hidden} more lines, Tab to expand)"),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
                inner_w,
            );
        }
    }

    let footer = format!("[exit:{} | {}ms]", b.exit_code, b.duration_ms);
    let pad = (inner_w as usize).saturating_sub(footer.chars().count());
    out.push(Line::from(vec![
        bar.clone(),
        Span::raw(" ".repeat(pad)),
        Span::styled(
            footer,
            Style::default().fg(bar_color).add_modifier(Modifier::BOLD),
        ),
    ]));

    out.push(Line::from(""));
}

fn push_barred(
    out: &mut Vec<Line<'static>>,
    bar: &Span<'static>,
    content: &str,
    content_style: Style,
    inner_w: u16,
) {
    let wrapped = textwrap::wrap(content, inner_w as usize);
    if wrapped.is_empty() {
        out.push(Line::from(vec![bar.clone(), Span::raw("")]));
        return;
    }
    for chunk in wrapped {
        out.push(Line::from(vec![
            bar.clone(),
            Span::styled(chunk.into_owned(), content_style),
        ]));
    }
}

/// Body lines to draw: all of them, or the leading
/// [`COLLAPSED_HEAD_LINES`] plus every stderr line when collapsed.
fn visible_body_indices(body_lines: &[String], collapsed: bool) -> Vec<usize> {
    if !collapsed {
        return (0..body_lines.len()).collect();
    }
    let head = COLLAPSED_HEAD_LINES.min(body_lines.len());
    let mut idxs: Vec<usize> = (0..head).collect();
    idxs.extend(
        body_lines
            .iter()
            .enumerate()
            .skip(head)
            .filter(|(_, l)| l.starts_with("[stderr] "))
            .map(|(i, _)| i),
    );
    idxs
}

fn split_body(output: &str) -> Vec<String> {
    let mut lines: Vec<String> = output.lines().map(str::to_string).collect();
    if lines
        .last()
        .map(|l| l.starts_with("[exit:"))
        .unwrap_or(false)
    {
        lines.pop();
    }
    lines
}

fn body_line_count(output: &str) -> usize {
    let n = output.lines().count();
    if output.contains("[exit:") {
        n.saturating_sub(1)
    } else {
        n
    }
}

fn single_span_line(text: String, style: Style) -> Line<'static> {
    Line::from(Span::styled(text, style))
}

fn append_to_line(line: &mut Line<'static>, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(span) = line.spans.last_mut() {
        let mut owned = std::mem::take(&mut span.content).into_owned();
        owned.push_str(text);
        span.content = owned.into();
    } else {
        line.spans
            .push(Span::styled(text.to_string(), assistant_style()));
    }
}

fn user_style() -> Style {
    Style::default().bg(Color::DarkGray)
}

fn assistant_style() -> Style {
    Style::default()
}

fn error_style() -> Style {
    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
}

fn info_style() -> Style {
    Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM)
}

fn tool_call_style() -> Style {
    Style::default().fg(Color::Blue).add_modifier(Modifier::DIM)
}

fn tool_result_style() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn header_style() -> Style {
    Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD)
}

fn thinking_bar_style() -> Style {
    Style::default()
        .fg(Color::Gray)
        .add_modifier(Modifier::DIM)
        .add_modifier(Modifier::ITALIC)
}

fn thinking_header_style() -> Style {
    Style::default()
        .fg(Color::Gray)
        .add_modifier(Modifier::DIM)
        .add_modifier(Modifier::ITALIC)
}

fn thinking_text_style() -> Style {
    Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::ITALIC)
}

fn stderr_style() -> Style {
    Style::default().fg(Color::Red)
}

/// Body-line count above which a new tool block starts collapsed.
const COLLAPSE_THRESHOLD: usize = 20;
/// Leading body lines kept visible while collapsed; stderr lines stay
/// visible regardless.
const COLLAPSED_HEAD_LINES: usize = 10;

#[cfg(test)]
mod tests;

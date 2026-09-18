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
    let stderr_idxs: Vec<usize> = body_lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("[stderr] "))
        .map(|(i, _)| i)
        .collect();
    let visible_idxs: Vec<usize> = if collapsed {
        let mut idxs: Vec<usize> = (0..COLLAPSED_HEAD_LINES.min(body_lines.len())).collect();
        for &si in &stderr_idxs {
            if !idxs.contains(&si) {
                idxs.push(si);
            }
        }
        idxs.sort_unstable();
        idxs
    } else {
        (0..body_lines.len()).collect()
    };
    for i in &visible_idxs {
        let n = textwrap::wrap(&body_lines[*i], inner_w).len().max(1);
        rows += n;
    }
    if collapsed {
        let hidden = body_lines.len() - visible_idxs.len();
        if hidden > 0 {
            // The collapsed-tail line is short; treat it as one row.
            rows += 1;
        }
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
    let stderr_idxs: Vec<usize> = body_lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("[stderr] "))
        .map(|(i, _)| i)
        .collect();

    let collapsed = !verbose && !b.expanded && body_lines.len() > COLLAPSE_THRESHOLD;
    let visible_idxs: Vec<usize> = if collapsed {
        let mut idxs: Vec<usize> = (0..COLLAPSED_HEAD_LINES.min(body_lines.len())).collect();
        for &si in &stderr_idxs {
            if !idxs.contains(&si) {
                idxs.push(si);
            }
        }
        idxs.sort_unstable();
        idxs
    } else {
        (0..body_lines.len()).collect()
    };

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
mod tests {
    use super::*;

    fn line_text(line: &Line<'static>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    fn item_text(it: &OutputItem) -> String {
        match it {
            OutputItem::Text(l) => line_text(l),
            OutputItem::Tool(b) => format!("[tool:{}]", b.command),
            OutputItem::Thumbnail(t) => format!("[thumb:{}]", t.name),
            OutputItem::Thinking(t) => format!(
                "[thinking:{}:{}]",
                if t.ended_at.is_some() { "done" } else { "live" },
                t.text
            ),
        }
    }

    fn rendered_lines(p: &mut OutputPane, w: u16, h: u16) -> Vec<String> {
        let (lines, _) = p.render_view(w, h);
        lines.iter().map(line_text).collect()
    }

    #[test]
    fn push_user_prefixes_caret() {
        let mut p = OutputPane::new();
        p.push_user("hi");
        assert_eq!(item_text(&p.items[0]), "> hi");
    }

    #[test]
    fn begin_and_append_assistant_streams_into_one_line() {
        let mut p = OutputPane::new();
        p.begin_assistant();
        p.append_assistant("hello ");
        p.append_assistant("world");
        assert_eq!(item_text(&p.items[0]), "hello world");
        assert_eq!(p.open_assistant, Some(0));
    }

    #[test]
    fn append_without_begin_auto_starts() {
        let mut p = OutputPane::new();
        p.append_assistant("hey");
        assert_eq!(p.items.len(), 1);
        assert_eq!(item_text(&p.items[0]), "hey");
    }

    #[test]
    fn append_splits_on_embedded_newlines() {
        let mut p = OutputPane::new();
        p.append_assistant("line1\nline2\nline3");
        assert_eq!(p.items.len(), 3);
        assert_eq!(item_text(&p.items[0]), "line1");
        assert_eq!(item_text(&p.items[1]), "line2");
        assert_eq!(item_text(&p.items[2]), "line3");
        assert_eq!(p.open_assistant, Some(2));
    }

    #[test]
    fn append_trailing_newline_opens_blank_tail() {
        let mut p = OutputPane::new();
        p.append_assistant("hello\n");
        assert_eq!(p.items.len(), 2);
        assert_eq!(item_text(&p.items[0]), "hello");
        assert_eq!(item_text(&p.items[1]), "");
        assert_eq!(p.open_assistant, Some(1));
        p.append_assistant("more");
        assert_eq!(item_text(&p.items[1]), "more");
    }

    #[test]
    fn finish_assistant_closes_stream_and_adds_separator() {
        let mut p = OutputPane::new();
        p.begin_assistant();
        p.append_assistant("done");
        p.finish_assistant();
        assert_eq!(p.open_assistant, None);
        assert_eq!(p.items.len(), 2);
        assert_eq!(item_text(&p.items[0]), "done");
        assert_eq!(item_text(&p.items[1]), "");
    }

    #[test]
    fn push_user_mid_stream_closes_open_assistant() {
        let mut p = OutputPane::new();
        p.append_assistant("half");
        p.push_user("new question");
        assert_eq!(p.open_assistant, None);
        assert_eq!(p.items.len(), 4);
        assert_eq!(item_text(&p.items[2]), "> new question");
        assert_eq!(item_text(&p.items[3]), "");
    }

    #[test]
    fn push_error_adds_exclamation_prefix() {
        let mut p = OutputPane::new();
        p.push_error("boom");
        assert_eq!(item_text(&p.items[0]), "!! boom");
    }

    fn small_block(p: &mut OutputPane, cmd: &str, body: &str, exit: i32, ms: u64) {
        p.push_tool_block(cmd.into(), body.into(), exit, ms);
    }

    #[test]
    fn push_tool_block_creates_one_tool_item() {
        let mut p = OutputPane::new();
        small_block(&mut p, "ls /tmp", "a\nb\n[exit:0 | 5ms]", 0, 5);
        assert_eq!(p.items.len(), 1);
        assert!(matches!(p.items[0], OutputItem::Tool(_)));
    }

    #[test]
    fn tool_block_renders_header_body_footer_with_bar() {
        let mut p = OutputPane::new();
        small_block(&mut p, "ls /tmp", "a\nb\n[exit:0 | 5ms]", 0, 5);
        let rendered = rendered_lines(&mut p, 40, 10);
        assert!(rendered.iter().any(|l| l.contains("$ ls /tmp")));
        assert!(rendered.iter().any(|l| l.ends_with('a')));
        assert!(rendered.iter().any(|l| l.ends_with('b')));
        assert!(rendered.iter().any(|l| l.contains("[exit:0 | 5ms]")));
        for l in rendered.iter().filter(|l| !l.trim().is_empty()) {
            assert!(l.starts_with('▎'), "missing bar on: {l:?}");
        }
    }

    #[test]
    fn tool_block_collapsed_when_over_threshold() {
        let mut p = OutputPane::new();
        let body: String =
            (0..30).map(|i| format!("line {i}\n")).collect::<String>() + "[exit:0 | 1ms]";
        p.push_tool_block("seq 30".into(), body, 0, 1);
        let rendered = rendered_lines(&mut p, 60, 80);
        let head_visible = rendered.iter().filter(|l| l.contains("line ")).count();
        assert_eq!(head_visible, COLLAPSED_HEAD_LINES);
        assert!(
            rendered
                .iter()
                .any(|l| l.contains("more lines, Tab to expand"))
        );
    }

    #[test]
    fn tool_block_expanded_after_toggle() {
        let mut p = OutputPane::new();
        let body: String =
            (0..30).map(|i| format!("line {i}\n")).collect::<String>() + "[exit:0 | 1ms]";
        p.push_tool_block("seq 30".into(), body, 0, 1);
        assert!(p.toggle_last_expandable());
        let rendered = rendered_lines(&mut p, 60, 80);
        let head_visible = rendered.iter().filter(|l| l.contains("line ")).count();
        assert_eq!(head_visible, 30);
        assert!(!rendered.iter().any(|l| l.contains("more lines")));
    }

    #[test]
    fn nonzero_exit_pins_stderr_visible_when_collapsed() {
        let mut body = String::new();
        for i in 0..25 {
            body.push_str(&format!("stdout line {i}\n"));
        }
        body.push_str("[stderr] [grep]\tboom\n");
        body.push_str("[exit:1 | 3ms]");
        let mut p = OutputPane::new();
        p.push_tool_block("grep foo bar".into(), body, 1, 3);
        let rendered = rendered_lines(&mut p, 80, 80);
        assert!(rendered.iter().any(|l| l.contains("[stderr]")));
        assert!(rendered.iter().any(|l| l.contains("boom")));
    }

    #[test]
    fn nonzero_exit_uses_red_bar_color() {
        let mut p = OutputPane::new();
        small_block(&mut p, "false", "[exit:1 | 1ms]", 1, 1);
        let (lines, _) = p.render_view(40, 5);
        let header = lines
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.contains("$ false")))
            .expect("header line present");
        assert_eq!(header.spans[0].style.fg, Some(Color::Red));
    }

    #[test]
    fn zero_exit_uses_green_bar_color() {
        let mut p = OutputPane::new();
        small_block(&mut p, "true", "[exit:0 | 1ms]", 0, 1);
        let (lines, _) = p.render_view(40, 5);
        let header = lines
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.contains("$ true")))
            .expect("header");
        assert_eq!(header.spans[0].style.fg, Some(Color::Green));
    }

    #[test]
    fn very_long_single_line_output_wraps_under_bar() {
        let long = "x".repeat(500);
        let body = format!("{long}\n[exit:0 | 1ms]");
        let mut p = OutputPane::new();
        p.push_tool_block("yes | head".into(), body, 0, 1);
        let rendered = rendered_lines(&mut p, 40, 200);
        assert!(rendered.iter().filter(|l| l.contains("xxxx")).count() > 5);
        for l in rendered.iter().filter(|l| !l.trim().is_empty()) {
            assert!(l.starts_with('▎'), "missing bar: {l:?}");
        }
    }

    #[test]
    fn empty_output_still_shows_header_and_footer() {
        let mut p = OutputPane::new();
        small_block(&mut p, "true", "[exit:0 | 0ms]", 0, 0);
        let rendered = rendered_lines(&mut p, 40, 10);
        assert!(rendered.iter().any(|l| l.contains("$ true")));
        assert!(rendered.iter().any(|l| l.contains("[exit:0 | 0ms]")));
    }

    #[test]
    fn nonzero_exit_with_no_stderr_renders_red_bar_and_footer() {
        let mut p = OutputPane::new();
        small_block(&mut p, "exit 7", "[exit:7 | 1ms]", 7, 1);
        let rendered = rendered_lines(&mut p, 40, 10);
        assert!(rendered.iter().any(|l| l.contains("[exit:7 | 1ms]")));
        let (lines, _) = p.render_view(40, 10);
        let header = lines
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.contains("$ exit 7")))
            .unwrap();
        assert_eq!(header.spans[0].style.fg, Some(Color::Red));
    }

    #[test]
    fn truncation_banner_visible_in_block() {
        let mut body = String::new();
        body.push_str("a\nb\nc\n");
        body.push_str("--- output truncated (5000 lines, 50.0K) ---\n");
        body.push_str("Full output: /tmp/assistd-output/cmd-1.txt\n");
        body.push_str("[exit:0 | 9ms]");
        let mut p = OutputPane::new();
        p.push_tool_block("cat huge".into(), body, 0, 9);
        let rendered = rendered_lines(&mut p, 80, 80);
        assert!(rendered.iter().any(|l| l.contains("output truncated")));
        assert!(rendered.iter().any(|l| l.contains("Full output:")));
    }

    #[test]
    fn toggle_last_expandable_no_op_with_no_blocks() {
        let mut p = OutputPane::new();
        assert!(!p.toggle_last_expandable());
    }

    #[test]
    fn begin_thinking_creates_live_block_collapsed_by_default() {
        let mut p = OutputPane::new();
        p.begin_thinking();
        assert_eq!(p.items.len(), 1);
        match &p.items[0] {
            OutputItem::Thinking(t) => {
                assert!(t.ended_at.is_none());
                assert!(!t.expanded, "new live blocks start collapsed");
                assert!(t.text.is_empty());
            }
            _ => panic!("expected Thinking item"),
        }
        assert!(p.live_thinking_seconds().is_some());
    }

    #[test]
    fn append_thinking_streams_into_open_block() {
        let mut p = OutputPane::new();
        p.append_thinking("let me ");
        p.append_thinking("think");
        assert_eq!(p.items.len(), 1);
        match &p.items[0] {
            OutputItem::Thinking(t) => assert_eq!(t.text, "let me think"),
            _ => panic!("expected Thinking item"),
        }
    }

    #[test]
    fn append_thinking_opens_block_when_none_live() {
        let mut p = OutputPane::new();
        p.append_thinking("instant");
        match &p.items[0] {
            OutputItem::Thinking(t) => assert_eq!(t.text, "instant"),
            _ => panic!("expected Thinking item"),
        }
    }

    #[test]
    fn finish_thinking_stamps_ended_at_and_collapses() {
        let mut p = OutputPane::new();
        p.append_thinking("done thinking");
        p.finish_thinking();
        match &p.items[0] {
            OutputItem::Thinking(t) => {
                assert!(t.ended_at.is_some());
                assert!(!t.expanded);
            }
            _ => panic!("expected Thinking item"),
        }
        assert!(p.live_thinking_seconds().is_none());
    }

    #[test]
    fn finish_thinking_is_idempotent() {
        let mut p = OutputPane::new();
        p.append_thinking("x");
        p.finish_thinking();
        // Repeat calls are no-ops; no panic, no double-stamp shift.
        p.finish_thinking();
        p.finish_thinking();
        assert!(matches!(&p.items[0], OutputItem::Thinking(t) if t.ended_at.is_some()));
    }

    #[test]
    fn thinking_block_renders_collapsed_live_header_only() {
        let mut p = OutputPane::new();
        p.append_thinking("reasoning body line 1");
        let rendered = rendered_lines(&mut p, 60, 20);
        assert!(
            rendered.iter().any(|l| l.contains("Thinking…")),
            "missing live header: {rendered:?}"
        );
        // Body is hidden by default — Tab or verbose mode reveals it.
        assert!(
            !rendered.iter().any(|l| l.contains("reasoning body line 1")),
            "body should be hidden while collapsed: {rendered:?}"
        );
        for l in rendered.iter().filter(|l| !l.trim().is_empty()) {
            assert!(l.starts_with('▎'), "missing bar on: {l:?}");
        }
    }

    #[test]
    fn thinking_block_renders_body_after_tab_expand_while_live() {
        let mut p = OutputPane::new();
        p.append_thinking("reasoning body line 1");
        assert!(p.toggle_last_expandable());
        let rendered = rendered_lines(&mut p, 60, 20);
        assert!(rendered.iter().any(|l| l.contains("reasoning body line 1")));
    }

    #[test]
    fn verbose_mode_expands_live_thinking_without_per_item_toggle() {
        let mut p = OutputPane::new();
        p.append_thinking("verbose body");
        // Default: body hidden.
        let rendered = rendered_lines(&mut p, 60, 20);
        assert!(!rendered.iter().any(|l| l.contains("verbose body")));
        // Flip verbose on → body appears, per-item flag untouched.
        p.set_verbose(true);
        let rendered = rendered_lines(&mut p, 60, 20);
        assert!(rendered.iter().any(|l| l.contains("verbose body")));
        match &p.items[0] {
            OutputItem::Thinking(t) => assert!(!t.expanded, "per-item flag stays collapsed"),
            _ => panic!("expected Thinking item"),
        }
        // Flip verbose off → body hides again.
        p.set_verbose(false);
        let rendered = rendered_lines(&mut p, 60, 20);
        assert!(!rendered.iter().any(|l| l.contains("verbose body")));
    }

    #[test]
    fn verbose_mode_expands_collapsed_tool_block() {
        let mut p = OutputPane::new();
        let body: String =
            (0..30).map(|i| format!("line {i}\n")).collect::<String>() + "[exit:0 | 1ms]";
        p.push_tool_block("seq 30".into(), body, 0, 1);
        // Default: collapsed (only first COLLAPSED_HEAD_LINES visible).
        let rendered = rendered_lines(&mut p, 60, 80);
        let head_visible = rendered.iter().filter(|l| l.contains("line ")).count();
        assert_eq!(head_visible, COLLAPSED_HEAD_LINES);
        // Verbose ON: all 30 visible.
        p.set_verbose(true);
        let rendered = rendered_lines(&mut p, 60, 80);
        let head_visible = rendered.iter().filter(|l| l.contains("line ")).count();
        assert_eq!(head_visible, 30);
    }

    #[test]
    fn thinking_block_renders_past_tense_after_finish() {
        let mut p = OutputPane::new();
        p.append_thinking("body");
        p.finish_thinking();
        let rendered = rendered_lines(&mut p, 60, 20);
        assert!(
            rendered.iter().any(|l| l.contains("Thought for")),
            "missing past-tense header: {rendered:?}"
        );
        // Auto-collapsed: body is not rendered.
        assert!(
            !rendered.iter().any(|l| l.contains("body")),
            "body should be hidden after collapse: {rendered:?}"
        );
    }

    #[test]
    fn thinking_block_body_visible_after_toggle_when_collapsed() {
        let mut p = OutputPane::new();
        p.append_thinking("expand-me body");
        p.finish_thinking();
        assert!(p.toggle_last_expandable());
        let rendered = rendered_lines(&mut p, 60, 20);
        assert!(rendered.iter().any(|l| l.contains("expand-me body")));
    }

    #[test]
    fn begin_thinking_prunes_empty_open_assistant() {
        let mut p = OutputPane::new();
        // Mirrors the begin_submit flow which proactively opens an
        // empty assistant block before any Delta arrives.
        p.begin_assistant();
        assert_eq!(p.items.len(), 1);
        p.append_thinking("first thoughts");
        // The empty assistant line should have been pruned, not
        // left as a stray separator above the Thinking block.
        assert_eq!(p.items.len(), 1);
        assert!(matches!(&p.items[0], OutputItem::Thinking(_)));
        assert_eq!(p.open_assistant, None);
    }

    #[test]
    fn finished_thinking_then_new_reasoning_opens_fresh_block() {
        let mut p = OutputPane::new();
        p.append_thinking("phase one");
        p.finish_thinking();
        p.append_thinking("phase two");
        let thinking_count = p
            .items
            .iter()
            .filter(|i| matches!(i, OutputItem::Thinking(_)))
            .count();
        assert_eq!(thinking_count, 2, "expected two distinct phases");
    }

    #[test]
    fn toggle_last_expandable_prefers_most_recent_item() {
        // Most recent expandable is a Thinking block → Tab toggles it.
        let mut p = OutputPane::new();
        small_block(&mut p, "ls", "a\n[exit:0 | 1ms]", 0, 1);
        p.append_thinking("recent reasoning");
        p.finish_thinking();
        // After finish_thinking the block is collapsed; toggle expands it.
        assert!(p.toggle_last_expandable());
        match &p.items.last().unwrap() {
            OutputItem::Thinking(t) => assert!(t.expanded, "expected toggled-open"),
            _ => panic!("expected Thinking item last"),
        }
        // Tool block above stays untouched.
        match &p.items[0] {
            OutputItem::Tool(b) => assert!(b.expanded, "tool block must remain expanded"),
            _ => panic!("expected Tool item at index 0"),
        }

        // Most recent expandable is a Tool block → Tab toggles it.
        let mut q = OutputPane::new();
        q.append_thinking("earlier reasoning");
        q.finish_thinking();
        small_block(&mut q, "ls", "a\n[exit:0 | 1ms]", 0, 1);
        // The freshly-pushed tool block starts expanded (body ≤ 20 lines).
        match q.items.last().unwrap() {
            OutputItem::Tool(b) => assert!(b.expanded),
            _ => panic!("expected Tool item last"),
        }
        assert!(q.toggle_last_expandable());
        match q.items.last().unwrap() {
            OutputItem::Tool(b) => assert!(!b.expanded, "toggle should have collapsed"),
            _ => panic!("expected Tool item last"),
        }
    }

    #[test]
    fn live_thinking_seconds_returns_some_while_live_none_otherwise() {
        let mut p = OutputPane::new();
        assert_eq!(p.live_thinking_seconds(), None);
        p.append_thinking("x");
        assert!(p.live_thinking_seconds().is_some());
        p.finish_thinking();
        assert_eq!(p.live_thinking_seconds(), None);
    }

    #[test]
    fn pipe_chain_renders_as_single_block() {
        let mut p = OutputPane::new();
        small_block(
            &mut p,
            "cat foo | grep bar | wc -l",
            "2\n[exit:0 | 4ms]",
            0,
            4,
        );
        assert_eq!(p.items.len(), 1);
        let rendered = rendered_lines(&mut p, 80, 10);
        assert!(
            rendered
                .iter()
                .any(|l| l.contains("$ cat foo | grep bar | wc -l"))
        );
    }

    // --- scroll & wrap ---------------------------------------------------

    #[test]
    fn scroll_saturates_down_at_zero() {
        let mut p = OutputPane::new();
        p.scroll_page_down(10);
        assert_eq!(p.scroll_offset(), 0);
    }

    #[test]
    fn scroll_up_then_down_returns_to_zero() {
        let mut p = OutputPane::new();
        p.scroll_page_up(10);
        assert_eq!(p.scroll_offset(), 5);
        p.scroll_page_down(10);
        assert_eq!(p.scroll_offset(), 0);
    }

    #[test]
    fn reset_scroll_clears_offset() {
        let mut p = OutputPane::new();
        p.scroll_page_up(10);
        p.reset_scroll();
        assert_eq!(p.scroll_offset(), 0);
    }

    #[test]
    fn wrapping_splits_long_line() {
        let mut p = OutputPane::new();
        p.append_assistant("aaaa bbbb cccc dddd");
        let (wrapped, _) = p.render_view(10, 5);
        assert!(wrapped.len() > 1, "expected wrap, got {}", wrapped.len());
    }

    #[test]
    fn render_view_clamps_scroll_offset() {
        let mut p = OutputPane::new();
        for _ in 0..3 {
            p.append_assistant("line\n");
        }
        p.scroll_offset = 99;
        let (_wrapped, _start) = p.render_view(80, 10);
        assert_eq!(p.scroll_offset, 0);
    }

    #[test]
    fn render_view_zero_width_falls_back_to_raw_lines() {
        let mut p = OutputPane::new();
        p.push_user("hi");
        let (wrapped, _) = p.render_view(0, 5);
        assert_eq!(wrapped.len(), 2);
    }
}

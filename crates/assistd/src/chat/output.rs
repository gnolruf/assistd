//! Scrollable output pane: prose lines, tool blocks, thinking blocks and
//! thumbnails, wrapped to the viewport width on render.

use std::ops::Range;
use std::time::Instant;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui_image::protocol::StatefulProtocol;

/// Rows an inline thumbnail reserves.
pub const THUMBNAIL_ROWS: u16 = 8;
/// Body-line count above which a new tool block starts collapsed.
const COLLAPSE_THRESHOLD: usize = 20;
/// Leading body lines kept visible while collapsed; stderr lines stay
/// visible regardless.
const COLLAPSED_HEAD_LINES: usize = 10;

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
    scroll_offset: usize,
    wrap: WrapCache,
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
            wrap: WrapCache::default(),
            verbose: false,
        }
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Append a `> `-prefixed prompt line and a blank separator.
    pub fn push_user(&mut self, text: &str) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            format!("> {text}"),
            user_style(),
        )));
        self.items.push(OutputItem::Text(Line::from("")));
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
    }

    pub fn push_info(&mut self, text: &str) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            text.to_string(),
            info_style(),
        )));
    }

    /// Open a streaming assistant block for [`Self::append_assistant`].
    pub fn begin_assistant(&mut self) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            String::new(),
            assistant_style(),
        )));
        self.open_assistant = Some(self.items.len() - 1);
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
        if let Some(first) = fragments.next()
            && !first.is_empty()
            && let Some(line) = self.text_at_mut(idx)
        {
            append_to_line(line, first);
            self.wrap.invalidate(idx);
        }
        for frag in fragments {
            self.items.push(OutputItem::Text(single_span_line(
                frag.to_string(),
                assistant_style(),
            )));
            idx = self.items.len() - 1;
        }
        self.open_assistant = Some(idx);
    }

    pub fn finish_assistant(&mut self) {
        if self.open_assistant.is_none() {
            return;
        }
        self.open_assistant = None;
        self.items.push(OutputItem::Text(Line::from("")));
    }

    pub fn push_error(&mut self, msg: &str) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            format!("!! {msg}"),
            error_style(),
        )));
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
    }

    /// Toggle the most recent tool or thinking block.
    pub fn toggle_last_expandable(&mut self) -> bool {
        for (idx, item) in self.items.iter_mut().enumerate().rev() {
            let expanded = match item {
                OutputItem::Tool(b) => &mut b.expanded,
                OutputItem::Thinking(t) => &mut t.expanded,
                OutputItem::Text(_) | OutputItem::Thumbnail(_) => continue,
            };
            *expanded = !*expanded;
            self.wrap.invalidate(idx);
            return true;
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
            self.wrap.invalidate(self.items.len() - 1);
        }
    }

    /// Stamp and collapse the live thinking block. No-op when none is
    /// live.
    pub fn finish_thinking(&mut self) {
        for (idx, item) in self.items.iter_mut().enumerate().rev() {
            if let OutputItem::Thinking(t) = item {
                if t.ended_at.is_none() {
                    t.ended_at = Some(Instant::now());
                    t.expanded = false;
                    self.wrap.invalidate(idx);
                }
                return;
            }
        }
    }

    /// Whole seconds the live thinking block has run, if any.
    pub fn live_thinking_seconds(&self) -> Option<u64> {
        self.live_thinking()
            .map(|(_, t)| t.started_at.elapsed().as_secs())
    }

    /// Rewrap the live thinking block on the next render so its
    /// elapsed-time header advances. No-op when none is live.
    pub fn refresh_live_thinking(&mut self) {
        if let Some((idx, _)) = self.live_thinking() {
            self.wrap.invalidate(idx);
        }
    }

    fn live_thinking(&self) -> Option<(usize, &ThinkingBlock)> {
        self.items
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, item)| match item {
                OutputItem::Thinking(t) if t.ended_at.is_none() => Some((idx, t)),
                _ => None,
            })
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.wrap.truncate(0);
        self.open_assistant = None;
        self.scroll_offset = 0;
    }

    /// Drop everything from the most recent user prompt onward. Returns
    /// the number of items removed.
    pub fn pop_last_user_exchange(&mut self) -> usize {
        let Some(idx) = self.items.iter().rposition(|item| {
            matches!(
                item,
                OutputItem::Text(line)
                    if line.spans.first().is_some_and(|s| s.content.starts_with("> "))
            )
        }) else {
            return 0;
        };
        let removed = self.items.len() - idx;
        self.items.truncate(idx);
        self.wrap.truncate(idx);
        self.open_assistant = None;
        removed
    }

    pub fn scroll_page_up(&mut self, viewport_height: u16) {
        let step = usize::from((viewport_height / 2).max(1));
        self.scroll_offset = self.scroll_offset.saturating_add(step);
    }

    pub fn scroll_page_down(&mut self, viewport_height: u16) {
        let step = usize::from((viewport_height / 2).max(1));
        self.scroll_offset = self.scroll_offset.saturating_sub(step);
    }

    pub fn scroll_lines_up(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(usize::from(lines.max(1)));
    }

    pub fn scroll_lines_down(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(usize::from(lines.max(1)));
    }

    pub fn reset_scroll(&mut self) {
        self.scroll_offset = 0;
    }

    /// Offset in wrapped lines; 0 is pinned to the bottom.
    pub fn scroll_offset(&self) -> usize {
        self.scroll_offset
    }

    /// The wrapped lines inside a `height`-row viewport and the index of
    /// the first of them in the whole wrapped transcript. Clamps the
    /// scroll offset to the wrapped total.
    pub fn render_view(&mut self, width: u16, height: u16) -> (&[Line<'static>], usize) {
        self.sync_wrap(width);
        let lines = &self.wrap.lines;
        let height = usize::from(height);
        let max_offset = lines.len().saturating_sub(height);
        self.scroll_offset = self.scroll_offset.min(max_offset);
        let start = max_offset - self.scroll_offset;
        let end = lines.len().min(start + height);
        (&lines[start..end], start)
    }

    fn sync_wrap(&mut self, width: u16) {
        self.wrap.sync(&self.items, width, self.verbose);
    }

    fn close_open_assistant(&mut self) {
        if let Some(idx) = self.open_assistant.take() {
            let empty = matches!(
                self.items.get(idx),
                Some(OutputItem::Text(l)) if l.spans.iter().all(|s| s.content.is_empty())
            );
            if empty && idx + 1 == self.items.len() {
                self.items.pop();
                self.wrap.truncate(idx);
            } else {
                self.items.push(OutputItem::Text(Line::from("")));
            }
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
    }

    /// Where each thumbnail's reserved rows sit in the wrapped output that
    /// [`Self::render_view`] returns for the same `width`.
    pub fn thumbnail_layout(&mut self, width: u16) -> Vec<ThumbnailSlot> {
        self.sync_wrap(width);
        self.wrap
            .thumbnails
            .iter()
            .map(|&item_idx| {
                let rows = self.wrap.rows(item_idx);
                ThumbnailSlot {
                    item_idx,
                    start_row: rows.start,
                    height: rows.len(),
                }
            })
            .collect()
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

/// Wrapped rows of the leading `starts.len()` items, valid for one
/// `(width, verbose)` key. Every item mutated in place must be passed to
/// [`Self::invalidate`] and every removal to [`Self::truncate`]; items
/// appended past the cached prefix are picked up by [`Self::sync`].
#[derive(Default)]
struct WrapCache {
    key: Option<(u16, bool)>,
    lines: Vec<Line<'static>>,
    /// First row of each cached item; an item ends where the next starts.
    starts: Vec<usize>,
    /// Indices of cached thumbnail items, ascending.
    thumbnails: Vec<usize>,
    /// Cached items whose rows must be re-rendered.
    stale: Vec<usize>,
}

impl WrapCache {
    fn invalidate(&mut self, idx: usize) {
        if idx < self.starts.len() {
            self.stale.push(idx);
        }
    }

    fn truncate(&mut self, len: usize) {
        if let Some(&start) = self.starts.get(len) {
            self.lines.truncate(start);
            self.starts.truncate(len);
            self.thumbnails.retain(|&i| i < len);
            self.stale.retain(|&i| i < len);
        }
    }

    fn rows(&self, idx: usize) -> Range<usize> {
        let end = self
            .starts
            .get(idx + 1)
            .copied()
            .unwrap_or(self.lines.len());
        self.starts[idx]..end
    }

    fn sync(&mut self, items: &[OutputItem], width: u16, verbose: bool) {
        if self.key != Some((width, verbose)) {
            self.key = Some((width, verbose));
            self.truncate(0);
        }
        self.stale.sort_unstable();
        self.stale.dedup();
        let mut fresh = Vec::new();
        for idx in std::mem::take(&mut self.stale) {
            render_item(&mut fresh, &items[idx], width, verbose);
            let rows = self.rows(idx);
            let (old_len, new_len) = (rows.len(), fresh.len());
            self.lines.splice(rows, fresh.drain(..));
            if new_len != old_len {
                for start in &mut self.starts[idx + 1..] {
                    *start = *start - old_len + new_len;
                }
            }
        }
        for (idx, item) in items.iter().enumerate().skip(self.starts.len()) {
            self.starts.push(self.lines.len());
            if matches!(item, OutputItem::Thumbnail(_)) {
                self.thumbnails.push(idx);
            }
            render_item(&mut self.lines, item, width, verbose);
        }
    }
}

/// A zero width renders one unwrapped line per item.
fn render_item(out: &mut Vec<Line<'static>>, item: &OutputItem, width: u16, verbose: bool) {
    if width == 0 {
        out.push(match item {
            OutputItem::Text(l) => l.clone(),
            OutputItem::Tool(b) => single_span_line(format!("$ {}", b.command), tool_call_style()),
            OutputItem::Thumbnail(t) => single_span_line(format!("📎 {}", t.name), info_style()),
            OutputItem::Thinking(t) => {
                single_span_line(thinking_header_text(t), thinking_header_style())
            }
        });
        return;
    }
    match item {
        OutputItem::Text(line) => wrap_line_into(out, line, width),
        OutputItem::Tool(b) => render_tool_block(out, b, width, verbose),
        OutputItem::Thumbnail(t) => render_thumbnail_placeholder(out, t),
        OutputItem::Thinking(t) => render_thinking_block(out, t, width, verbose),
    }
}

fn render_thumbnail_placeholder(out: &mut Vec<Line<'static>>, t: &ThumbnailItem) {
    out.push(single_span_line(format!("📎 {}", t.name), info_style()));
    for _ in 1..THUMBNAIL_ROWS {
        out.push(Line::from(""));
    }
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
    if lines.last().is_some_and(|l| l.starts_with("[exit:")) {
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

#[cfg(test)]
mod tests;

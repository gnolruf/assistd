//! Scrollable output pane: prose lines, markdown replies, tool blocks,
//! thinking blocks and thumbnails, wrapped to the viewport width on render.

use std::ops::Range;
use std::time::Instant;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui_image::protocol::StatefulProtocol;

use super::markdown::render_markdown;

/// Rows an inline thumbnail reserves.
pub(super) const THUMBNAIL_ROWS: u16 = 8;
/// Body-line count above which a new tool block starts collapsed.
const COLLAPSE_THRESHOLD: usize = 20;
/// Leading body lines kept visible while collapsed; stderr lines stay
/// visible regardless.
const COLLAPSED_HEAD_LINES: usize = 10;
const NO_RESULT_OUTPUT: &str = "[stderr] no result received";
const NO_RESULT_EXIT_CODE: i32 = -1;
const UNANNOUNCED_COMMAND: &str = "<?>";

/// One tool invocation, shown from the moment it is called.
#[derive(Debug, Clone)]
pub(super) struct ToolBlock {
    pub command: String,
    pub state: ToolState,
    pub expanded: bool,
}

/// Whether a tool invocation has been answered yet.
#[derive(Debug, Clone)]
pub(super) enum ToolState {
    Running {
        started_at: Instant,
    },
    /// `output` is as the daemon delivered it: already truncated, with
    /// its banner and footer.
    Finished {
        output: String,
        exit_code: i32,
        duration_ms: u64,
    },
}

pub(super) struct ThumbnailItem {
    pub name: String,
    pub protocol: StatefulProtocol,
}

/// One reasoning phase, shown as an expandable block that auto-collapses
/// when it finishes.
#[derive(Debug, Clone)]
pub(super) struct ThinkingBlock {
    pub text: String,
    pub started_at: Instant,
    /// `None` while still receiving deltas.
    pub ended_at: Option<Instant>,
    pub expanded: bool,
}

enum OutputItem {
    Text(Line<'static>),
    /// One assistant reply as raw markdown, rendered on wrap.
    Assistant(String),
    Tool(ToolBlock),
    Thumbnail(Box<ThumbnailItem>),
    Thinking(ThinkingBlock),
}

pub(super) struct OutputPane {
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
    pub(super) fn new() -> Self {
        Self {
            items: Vec::new(),
            open_assistant: None,
            scroll_offset: 0,
            wrap: WrapCache::default(),
            verbose: false,
        }
    }

    pub(super) fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Append a `> `-prefixed prompt line and a blank separator.
    pub(super) fn push_user(&mut self, text: &str) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            format!("> {text}"),
            user_style(),
        )));
        self.items.push(OutputItem::Text(Line::from("")));
    }

    /// [`Self::push_user`] with a trailing 📎 tag naming the attachments.
    pub(super) fn push_user_with_attachments(&mut self, text: &str, names: &[String]) {
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

    pub(super) fn push_info(&mut self, text: &str) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            text.to_string(),
            info_style(),
        )));
    }

    /// Open a streaming assistant block for [`Self::append_assistant`].
    pub(super) fn begin_assistant(&mut self) {
        self.close_open_assistant();
        self.items.push(OutputItem::Assistant(String::new()));
        self.open_assistant = Some(self.items.len() - 1);
    }

    /// Extend the open reply's markdown; the block re-renders on the next
    /// wrap so formatting settles as the closing delimiters arrive.
    pub(super) fn append_assistant(&mut self, delta: &str) {
        let idx = match self.open_assistant {
            Some(idx) => idx,
            None => {
                self.begin_assistant();
                self.items.len() - 1
            }
        };
        if let Some(OutputItem::Assistant(text)) = self.items.get_mut(idx) {
            text.push_str(delta);
            self.wrap.invalidate(idx);
        }
    }

    pub(super) fn finish_assistant(&mut self) {
        if self.open_assistant.is_none() {
            return;
        }
        self.open_assistant = None;
        self.items.push(OutputItem::Text(Line::from("")));
    }

    pub(super) fn push_error(&mut self, msg: &str) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            format!("!! {msg}"),
            error_style(),
        )));
    }

    /// Append a finished tool block. Blocks whose body exceeds
    /// [`COLLAPSE_THRESHOLD`] lines start collapsed.
    pub(super) fn push_tool_block(
        &mut self,
        command: String,
        output: String,
        exit_code: i32,
        duration_ms: u64,
    ) {
        self.close_open_assistant();
        self.items.push(OutputItem::Tool(ToolBlock {
            command,
            expanded: starts_expanded(&output),
            state: ToolState::Finished {
                output,
                exit_code,
                duration_ms,
            },
        }));
    }

    pub(super) fn begin_tool_block(&mut self, command: String) {
        self.close_open_assistant();
        self.items.push(OutputItem::Tool(ToolBlock {
            command,
            state: ToolState::Running {
                started_at: Instant::now(),
            },
            expanded: true,
        }));
    }

    /// A result whose call was never announced gets a finished block of its
    /// own.
    pub(super) fn finish_tool_block(&mut self, output: String, exit_code: i32, duration_ms: u64) {
        let Some((idx, block)) = self.running_tool() else {
            self.push_tool_block(UNANNOUNCED_COMMAND.into(), output, exit_code, duration_ms);
            return;
        };
        block.expanded = starts_expanded(&output);
        block.state = ToolState::Finished {
            output,
            exit_code,
            duration_ms,
        };
        self.wrap.invalidate(idx);
    }

    /// Mark the running tool block failed. No-op when none is running.
    pub(super) fn abandon_running_tool(&mut self) {
        if self.running_tool().is_some() {
            self.finish_tool_block(NO_RESULT_OUTPUT.into(), NO_RESULT_EXIT_CODE, 0);
        }
    }

    fn running_tool(&mut self) -> Option<(usize, &mut ToolBlock)> {
        self.items
            .iter_mut()
            .enumerate()
            .rev()
            .find_map(|(idx, item)| match item {
                OutputItem::Tool(block) if matches!(block.state, ToolState::Running { .. }) => {
                    Some((idx, block))
                }
                _ => None,
            })
    }

    /// Toggle the most recent tool or thinking block.
    pub(super) fn toggle_last_expandable(&mut self) -> bool {
        for (idx, item) in self.items.iter_mut().enumerate().rev() {
            let expanded = match item {
                OutputItem::Tool(b) => &mut b.expanded,
                OutputItem::Thinking(t) => &mut t.expanded,
                OutputItem::Text(_) | OutputItem::Assistant(_) | OutputItem::Thumbnail(_) => {
                    continue;
                }
            };
            *expanded = !*expanded;
            self.wrap.invalidate(idx);
            return true;
        }
        false
    }

    /// Open a collapsed thinking block.
    pub(super) fn begin_thinking(&mut self) {
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
    pub(super) fn append_thinking(&mut self, delta: &str) {
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
    pub(super) fn finish_thinking(&mut self) {
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

    /// Whole seconds the live thinking or running tool block has run, if
    /// any.
    pub(super) fn live_block_seconds(&self) -> Option<u64> {
        self.live_block()
            .map(|(_, started_at)| started_at.elapsed().as_secs())
    }

    /// Rewrap the live block on the next render so its elapsed time
    /// advances. No-op when none is live.
    pub(super) fn refresh_live_block(&mut self) {
        if let Some((idx, _)) = self.live_block() {
            self.wrap.invalidate(idx);
        }
    }

    fn live_block(&self) -> Option<(usize, Instant)> {
        self.items
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, item)| match item {
                OutputItem::Thinking(t) if t.ended_at.is_none() => Some((idx, t.started_at)),
                OutputItem::Tool(ToolBlock {
                    state: ToolState::Running { started_at },
                    ..
                }) => Some((idx, *started_at)),
                _ => None,
            })
    }

    pub(super) fn clear(&mut self) {
        self.items.clear();
        self.wrap.truncate(0);
        self.open_assistant = None;
        self.scroll_offset = 0;
    }

    /// Drop everything from the most recent user prompt onward. Returns
    /// the number of items removed.
    pub(super) fn pop_last_user_exchange(&mut self) -> usize {
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

    pub(super) fn scroll_page_up(&mut self, viewport_height: u16) {
        let step = usize::from((viewport_height / 2).max(1));
        self.scroll_offset = self.scroll_offset.saturating_add(step);
    }

    pub(super) fn scroll_page_down(&mut self, viewport_height: u16) {
        let step = usize::from((viewport_height / 2).max(1));
        self.scroll_offset = self.scroll_offset.saturating_sub(step);
    }

    pub(super) fn scroll_lines_up(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(usize::from(lines.max(1)));
    }

    pub(super) fn scroll_lines_down(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(usize::from(lines.max(1)));
    }

    pub(super) fn reset_scroll(&mut self) {
        self.scroll_offset = 0;
    }

    /// Offset in wrapped lines; 0 is pinned to the bottom.
    pub(super) fn scroll_offset(&self) -> usize {
        self.scroll_offset
    }

    /// The wrapped lines inside a `height`-row viewport and the index of
    /// the first of them in the whole wrapped transcript. Clamps the
    /// scroll offset to the wrapped total.
    pub(super) fn render_view(&mut self, width: u16, height: u16) -> (&[Line<'static>], usize) {
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
                Some(OutputItem::Assistant(text)) if text.is_empty()
            );
            if empty && idx + 1 == self.items.len() {
                self.items.pop();
                self.wrap.truncate(idx);
            } else {
                self.items.push(OutputItem::Text(Line::from("")));
            }
        }
    }

    /// Reserve [`THUMBNAIL_ROWS`] rows for the renderer to draw over.
    pub(super) fn push_thumbnail(&mut self, name: String, protocol: StatefulProtocol) {
        self.close_open_assistant();
        self.items
            .push(OutputItem::Thumbnail(Box::new(ThumbnailItem {
                name,
                protocol,
            })));
    }

    /// Where each thumbnail's reserved rows sit in the wrapped output that
    /// [`Self::render_view`] returns for the same `width`.
    pub(super) fn thumbnail_layout(&mut self, width: u16) -> Vec<ThumbnailSlot> {
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
    pub(super) fn thumbnail_protocol_mut(&mut self, idx: usize) -> Option<&mut StatefulProtocol> {
        match self.items.get_mut(idx)? {
            OutputItem::Thumbnail(t) => Some(&mut t.protocol),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ThumbnailSlot {
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

/// A zero width renders every item unwrapped, one line per source line.
fn render_item(out: &mut Vec<Line<'static>>, item: &OutputItem, width: u16, verbose: bool) {
    if width == 0 {
        out.push(match item {
            OutputItem::Text(l) => l.clone(),
            OutputItem::Assistant(text) => {
                out.extend(text.lines().map(|line| Line::from(line.to_string())));
                return;
            }
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
        OutputItem::Assistant(text) => out.extend(render_markdown(text, usize::from(width))),
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
    let inner_w = width.saturating_sub(2).max(1);
    match &b.state {
        ToolState::Running { started_at } => {
            let bar = tool_bar(Color::Yellow);
            push_tool_header(out, &bar, &b.command, inner_w);
            let status = format!("running… ({}s)", started_at.elapsed().as_secs());
            push_barred(out, &bar, &status, tool_running_style(), inner_w);
        }
        ToolState::Finished {
            output,
            exit_code,
            duration_ms,
        } => {
            let bar_color = if *exit_code == 0 {
                Color::Green
            } else {
                Color::Red
            };
            let bar = tool_bar(bar_color);
            push_tool_header(out, &bar, &b.command, inner_w);
            let collapsible = !verbose && !b.expanded;
            push_tool_body(out, &bar, output, collapsible, inner_w);
            let footer = format!("[exit:{exit_code} | {duration_ms}ms]");
            push_tool_footer(out, &bar, footer, bar_color, inner_w);
        }
    }
    out.push(Line::from(""));
}

fn tool_bar(color: Color) -> Span<'static> {
    Span::styled(
        "▎ ",
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )
}

fn push_tool_header(
    out: &mut Vec<Line<'static>>,
    bar: &Span<'static>,
    command: &str,
    inner_w: u16,
) {
    push_barred(out, bar, &format!("$ {command}"), header_style(), inner_w);
}

fn push_tool_body(
    out: &mut Vec<Line<'static>>,
    bar: &Span<'static>,
    output: &str,
    collapsible: bool,
    inner_w: u16,
) {
    let body_lines = split_body(output);
    let collapsed = collapsible && body_lines.len() > COLLAPSE_THRESHOLD;
    let visible_idxs = visible_body_indices(&body_lines, collapsed);

    for i in &visible_idxs {
        let line = &body_lines[*i];
        let style = if line.starts_with("[stderr] ") {
            stderr_style()
        } else {
            tool_result_style()
        };
        push_barred(out, bar, line, style, inner_w);
    }
    if collapsed {
        let hidden = body_lines.len() - visible_idxs.len();
        if hidden > 0 {
            push_barred(
                out,
                bar,
                &format!("… ({hidden} more lines, Tab to expand)"),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
                inner_w,
            );
        }
    }
}

fn push_tool_footer(
    out: &mut Vec<Line<'static>>,
    bar: &Span<'static>,
    footer: String,
    color: Color,
    inner_w: u16,
) {
    let pad = (inner_w as usize).saturating_sub(footer.chars().count());
    out.push(Line::from(vec![
        bar.clone(),
        Span::raw(" ".repeat(pad)),
        Span::styled(
            footer,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
    ]));
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

fn starts_expanded(output: &str) -> bool {
    body_line_count(output) <= COLLAPSE_THRESHOLD
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

fn user_style() -> Style {
    Style::default().bg(Color::DarkGray)
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

fn tool_running_style() -> Style {
    Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::ITALIC)
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

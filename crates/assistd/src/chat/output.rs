//! Scrollable output pane with streaming-delta support and viewport-width
//! aware line wrapping.
//!
//! Items are heterogeneous: prose lines (user / assistant / error) live as
//! `OutputItem::Text(Line)`, while tool runs are first-class
//! `OutputItem::Tool(ToolBlock)` items expanded into bar-prefixed,
//! color-coded line groups at render time.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// A single tool invocation displayed as a cohesive block: command,
/// captured output (already condensed by Layer 2 to ≤200 lines / ≤50 KB),
/// exit status, and timing. Renders with a colored left-margin bar so the
/// block is visually distinct from prose.
///
/// The Layer-2 truncation indicator is conveyed by the truncation banner
/// already embedded in `output` — no separate flag is stored here.
#[derive(Debug, Clone)]
pub struct ToolBlock {
    pub command: String,
    /// Layer-2 `output` body verbatim: head + optional truncation banner +
    /// optional `[stderr] ...` line + `[exit:N | Xms]` footer.
    pub output: String,
    pub exit_code: i32,
    pub duration_ms: u64,
    pub expanded: bool,
}

#[derive(Debug)]
enum OutputItem {
    Text(Line<'static>),
    Tool(ToolBlock),
}

#[derive(Debug)]
pub struct OutputPane {
    items: Vec<OutputItem>,
    open_assistant: Option<usize>,
    scroll_offset: u16,
    wrap_cache: Option<(u16, Vec<Line<'static>>)>,
    dirty: bool,
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
        }
    }

    pub fn push_user(&mut self, text: &str) {
        self.close_open_assistant();
        self.items.push(OutputItem::Text(single_span_line(
            format!("> {text}"),
            user_style(),
        )));
        self.dirty = true;
    }

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
        // Resolve the open-assistant index through `begin_assistant` once
        // so the subsequent loop can assume a non-empty assistant line
        // without an Option unwrap surviving into the hot path.
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

    /// Append a tool invocation as a single block. Called once per
    /// `LlmEvent::ToolResult` after the corresponding `ToolCall` has been
    /// observed; the call/result pair is collapsed into one item so users
    /// see one cohesive entity instead of two loose line groups.
    ///
    /// New blocks start collapsed when their body exceeds
    /// [`COLLAPSE_THRESHOLD`] lines (Tab toggles the most recent).
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

    /// Toggle the expand/collapse state of the most-recently-appended tool
    /// block. Returns `true` if a toggle happened so the keybinding handler
    /// can decide whether to show feedback.
    pub fn toggle_last_tool_block(&mut self) -> bool {
        for item in self.items.iter_mut().rev() {
            if let OutputItem::Tool(b) = item {
                b.expanded = !b.expanded;
                self.dirty = true;
                return true;
            }
        }
        false
    }

    pub fn scroll_page_up(&mut self, viewport_height: u16) {
        let step = (viewport_height / 2).max(1);
        self.scroll_offset = self.scroll_offset.saturating_add(step);
    }

    pub fn scroll_page_down(&mut self, viewport_height: u16) {
        let step = (viewport_height / 2).max(1);
        self.scroll_offset = self.scroll_offset.saturating_sub(step);
    }

    pub fn reset_scroll(&mut self) {
        self.scroll_offset = 0;
    }

    pub fn scroll_offset(&self) -> u16 {
        self.scroll_offset
    }

    /// Returns the full wrapped line list and the index of the first line
    /// to render at the top of the viewport. The scroll offset is clamped
    /// in place so it never exceeds the wrapped total.
    pub fn render_view(&mut self, width: u16, height: u16) -> (&[Line<'static>], u16) {
        // Force cache population + grab the length without holding the
        // borrow, so the subsequent scroll-offset clamp is legal.
        let wrapped_len = self.wrapped(width).len();
        let max_offset = wrapped_len.saturating_sub(height as usize) as u16;
        if self.scroll_offset > max_offset {
            self.scroll_offset = max_offset;
        }
        let start = wrapped_len
            .saturating_sub(height as usize)
            .saturating_sub(self.scroll_offset as usize) as u16;
        // wrap_cache is guaranteed Some — self.wrapped() above
        // unconditionally populates it.
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
        // Invariant: wrap_cache is Some after the branch above; the only
        // way to reach here is either needs_rewrap was true (so we just
        // assigned Some) or needs_rewrap was false (which requires
        // wrap_cache.as_ref().map(...) to have returned Some, proving it's
        // Some here).
        &self
            .wrap_cache
            .as_ref()
            .expect("wrap_cache populated by branch above")
            .1
    }

    fn rewrap(&self, width: u16) -> Vec<Line<'static>> {
        if width == 0 {
            // Degraded path: render text items raw, tool blocks as a
            // single unwrapped header line. Keeps the zero-width test
            // green and avoids `textwrap` calls with width 0.
            return self
                .items
                .iter()
                .map(|it| match it {
                    OutputItem::Text(l) => l.clone(),
                    OutputItem::Tool(b) => {
                        single_span_line(format!("$ {}", b.command), tool_call_style())
                    }
                })
                .collect();
        }
        let mut out = Vec::with_capacity(self.items.len() * 2);
        for item in &self.items {
            match item {
                OutputItem::Text(line) => wrap_line_into(&mut out, line, width),
                OutputItem::Tool(b) => render_tool_block(&mut out, b, width),
            }
        }
        out
    }

    fn close_open_assistant(&mut self) {
        if self.open_assistant.take().is_some() {
            self.items.push(OutputItem::Text(Line::from("")));
            self.dirty = true;
        }
    }

    fn text_at_mut(&mut self, idx: usize) -> Option<&mut Line<'static>> {
        match self.items.get_mut(idx)? {
            OutputItem::Text(line) => Some(line),
            OutputItem::Tool(_) => None,
        }
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
    for chunk in wrapped {
        out.push(single_span_line(chunk.into_owned(), style));
    }
}

fn render_tool_block(out: &mut Vec<Line<'static>>, b: &ToolBlock, width: u16) {
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

    // Slice the Layer-2 body: drop the [exit:...] footer (we re-render it
    // ourselves so we can right-align timing and color the exit code) and
    // remember which lines start with [stderr] for pinning when collapsed.
    let body_lines = split_body(&b.output);
    let stderr_idxs: Vec<usize> = body_lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("[stderr] "))
        .map(|(i, _)| i)
        .collect();

    let collapsed = !b.expanded && body_lines.len() > COLLAPSE_THRESHOLD;
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

    // Footer: right-aligned [exit:N | Xms] in green/red bold.
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

    // Trailing blank so two adjacent blocks don't merge visually.
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

/// Strip the `[exit:N | Xms]` footer (always the last line of a Layer-2
/// `output`) and return the remaining body lines.
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

/// Count body lines (everything except the Layer-2 footer) to decide
/// whether a new block should start collapsed.
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
    Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD)
}

fn assistant_style() -> Style {
    Style::default()
}

fn error_style() -> Style {
    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
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

fn stderr_style() -> Style {
    Style::default().fg(Color::Red)
}

/// Body-line count above which a freshly-pushed tool block starts
/// collapsed. Tab toggles the most recent block.
const COLLAPSE_THRESHOLD: usize = 20;
/// Number of leading body lines to keep visible while collapsed. Stderr
/// lines past this index are still pinned visible — see `render_tool_block`.
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
        assert_eq!(p.items.len(), 3);
        assert_eq!(item_text(&p.items[2]), "> new question");
    }

    #[test]
    fn push_error_adds_exclamation_prefix() {
        let mut p = OutputPane::new();
        p.push_error("boom");
        assert_eq!(item_text(&p.items[0]), "!! boom");
    }

    // --- tool blocks -----------------------------------------------------

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
        assert!(p.toggle_last_tool_block());
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
    fn toggle_last_tool_block_no_op_with_no_blocks() {
        let mut p = OutputPane::new();
        assert!(!p.toggle_last_tool_block());
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
        assert_eq!(wrapped.len(), 1);
    }
}

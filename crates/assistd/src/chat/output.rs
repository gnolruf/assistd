//! Scrollable output pane with streaming-delta support and viewport-width
//! aware line wrapping.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

#[derive(Debug)]
pub struct OutputPane {
    lines: Vec<Line<'static>>,
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
            lines: Vec::new(),
            open_assistant: None,
            scroll_offset: 0,
            wrap_cache: None,
            dirty: true,
        }
    }

    pub fn push_user(&mut self, text: &str) {
        self.close_open_assistant();
        self.lines
            .push(single_span_line(format!("> {text}"), user_style()));
        self.dirty = true;
    }

    pub fn begin_assistant(&mut self) {
        self.close_open_assistant();
        self.lines
            .push(single_span_line(String::new(), assistant_style()));
        self.open_assistant = Some(self.lines.len() - 1);
        self.dirty = true;
    }

    pub fn append_assistant(&mut self, delta: &str) {
        if self.open_assistant.is_none() {
            self.begin_assistant();
        }
        let mut idx = self.open_assistant.expect("just set");
        let mut fragments = delta.split('\n');
        if let Some(first) = fragments.next() {
            append_to_line(&mut self.lines[idx], first);
        }
        for frag in fragments {
            self.lines
                .push(single_span_line(frag.to_string(), assistant_style()));
            idx = self.lines.len() - 1;
        }
        self.open_assistant = Some(idx);
        self.dirty = true;
    }

    pub fn finish_assistant(&mut self) {
        if self.open_assistant.is_none() {
            return;
        }
        self.open_assistant = None;
        self.lines.push(Line::from(""));
        self.dirty = true;
    }

    pub fn push_error(&mut self, msg: &str) {
        self.close_open_assistant();
        self.lines
            .push(single_span_line(format!("!! {msg}"), error_style()));
        self.dirty = true;
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
        let wrapped_len = self.wrapped(width).len();
        let max_offset = wrapped_len.saturating_sub(height as usize) as u16;
        if self.scroll_offset > max_offset {
            self.scroll_offset = max_offset;
        }
        let start = wrapped_len
            .saturating_sub(height as usize)
            .saturating_sub(self.scroll_offset as usize) as u16;
        (&self.wrap_cache.as_ref().unwrap().1, start)
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
        &self.wrap_cache.as_ref().unwrap().1
    }

    fn rewrap(&self, width: u16) -> Vec<Line<'static>> {
        if width == 0 {
            return self.lines.clone();
        }
        let mut out = Vec::with_capacity(self.lines.len());
        for line in &self.lines {
            let style = line.spans.first().map(|s| s.style).unwrap_or_default();
            let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
            if content.is_empty() {
                out.push(Line::from(""));
                continue;
            }
            let wrapped = textwrap::wrap(&content, width as usize);
            if wrapped.is_empty() {
                out.push(Line::from(""));
                continue;
            }
            for chunk in wrapped {
                out.push(single_span_line(chunk.into_owned(), style));
            }
        }
        out
    }

    fn close_open_assistant(&mut self) {
        if self.open_assistant.take().is_some() {
            self.lines.push(Line::from(""));
            self.dirty = true;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn line_text(line: &Line<'static>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn push_user_prefixes_caret() {
        let mut p = OutputPane::new();
        p.push_user("hi");
        assert_eq!(line_text(&p.lines[0]), "> hi");
    }

    #[test]
    fn begin_and_append_assistant_streams_into_one_line() {
        let mut p = OutputPane::new();
        p.begin_assistant();
        p.append_assistant("hello ");
        p.append_assistant("world");
        assert_eq!(line_text(&p.lines[0]), "hello world");
        assert_eq!(p.open_assistant, Some(0));
    }

    #[test]
    fn append_without_begin_auto_starts() {
        let mut p = OutputPane::new();
        p.append_assistant("hey");
        assert_eq!(p.lines.len(), 1);
        assert_eq!(line_text(&p.lines[0]), "hey");
    }

    #[test]
    fn append_splits_on_embedded_newlines() {
        let mut p = OutputPane::new();
        p.append_assistant("line1\nline2\nline3");
        assert_eq!(p.lines.len(), 3);
        assert_eq!(line_text(&p.lines[0]), "line1");
        assert_eq!(line_text(&p.lines[1]), "line2");
        assert_eq!(line_text(&p.lines[2]), "line3");
        assert_eq!(p.open_assistant, Some(2));
    }

    #[test]
    fn append_trailing_newline_opens_blank_tail() {
        let mut p = OutputPane::new();
        p.append_assistant("hello\n");
        // ["hello", ""]
        assert_eq!(p.lines.len(), 2);
        assert_eq!(line_text(&p.lines[0]), "hello");
        assert_eq!(line_text(&p.lines[1]), "");
        assert_eq!(p.open_assistant, Some(1));
        p.append_assistant("more");
        assert_eq!(line_text(&p.lines[1]), "more");
    }

    #[test]
    fn finish_assistant_closes_stream_and_adds_separator() {
        let mut p = OutputPane::new();
        p.begin_assistant();
        p.append_assistant("done");
        p.finish_assistant();
        assert_eq!(p.open_assistant, None);
        assert_eq!(p.lines.len(), 2);
        assert_eq!(line_text(&p.lines[0]), "done");
        assert_eq!(line_text(&p.lines[1]), "");
    }

    #[test]
    fn push_user_mid_stream_closes_open_assistant() {
        let mut p = OutputPane::new();
        p.append_assistant("half");
        p.push_user("new question");
        assert_eq!(p.open_assistant, None);
        // Expect: ["half", "", "> new question"]
        assert_eq!(p.lines.len(), 3);
        assert_eq!(line_text(&p.lines[2]), "> new question");
    }

    #[test]
    fn push_error_adds_exclamation_prefix() {
        let mut p = OutputPane::new();
        p.push_error("boom");
        assert_eq!(line_text(&p.lines[0]), "!! boom");
    }

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
        // total wrapped < viewport_height → max_offset == 0
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

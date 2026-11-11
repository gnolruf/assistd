//! Renders model markdown (emphasis, headings, lists, quotes, code,
//! tables) into styled terminal lines wrapped to a viewport width.

use pulldown_cmark::{Alignment, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use textwrap::core::display_width;

use super::theme::Theme;

const BULLET: &str = "• ";
const QUOTE_BAR: &str = "▎ ";
const COLUMN_SEPARATOR: &str = " │ ";
const COLUMN_RULE_JOINT: &str = "─┼─";
/// Narrowest a table column shrinks to before the table overflows.
const MIN_COLUMN_WIDTH: usize = 3;

/// Leading columns every physical line inside a block carries: a list
/// marker or quote bar on the first line, matching padding afterwards.
struct Prefix {
    first: Span<'static>,
    rest: Span<'static>,
    consumed: bool,
}

impl Prefix {
    fn marker(marker: String, style: Style) -> Self {
        let rest = Span::raw(" ".repeat(display_width(&marker)));
        Self {
            first: Span::styled(marker, style),
            rest,
            consumed: false,
        }
    }

    fn bar(bar: &'static str, style: Style) -> Self {
        Self {
            first: Span::styled(bar, style),
            rest: Span::styled(bar, style),
            consumed: false,
        }
    }

    fn current(&self) -> Span<'static> {
        if self.consumed {
            self.rest.clone()
        } else {
            self.first.clone()
        }
    }
}

/// A word: non-whitespace fragments that must stay on one line.
#[derive(Default)]
struct Word {
    fragments: Vec<Span<'static>>,
    width: usize,
}

struct TableBuilder {
    alignments: Vec<Alignment>,
    header: Vec<Vec<Span<'static>>>,
    rows: Vec<Vec<Vec<Span<'static>>>>,
    current_row: Vec<Vec<Span<'static>>>,
}

impl TableBuilder {
    fn new(alignments: Vec<Alignment>) -> Self {
        Self {
            alignments,
            header: Vec::new(),
            rows: Vec::new(),
            current_row: Vec::new(),
        }
    }

    /// The header, then each body row, with a rule under the header and
    /// between body rows.
    fn render(self, width: usize, theme: &Theme) -> Vec<Vec<Span<'static>>> {
        let columns = self.alignments.len().max(1);
        let natural = self.natural_widths(columns);
        let separators = display_width(COLUMN_SEPARATOR) * (columns - 1);
        let widths = fit_column_widths(&natural, width.saturating_sub(separators));
        let rule = vec![Span::styled(
            widths
                .iter()
                .map(|&w| "─".repeat(w))
                .collect::<Vec<_>>()
                .join(COLUMN_RULE_JOINT),
            rule_style(theme),
        )];
        let header = (!self.header.is_empty())
            .then_some((&self.header, Style::default().add_modifier(Modifier::BOLD)));
        let body = self.rows.iter().map(|row| (row, Style::default()));
        let mut out = Vec::new();
        for (index, (row, cell_style)) in header.into_iter().chain(body).enumerate() {
            if index > 0 {
                out.push(rule.clone());
            }
            out.extend(render_table_row(
                row,
                &widths,
                &self.alignments,
                cell_style,
                theme,
            ));
        }
        out
    }

    fn natural_widths(&self, columns: usize) -> Vec<usize> {
        let mut widths = vec![1; columns];
        for row in std::iter::once(&self.header).chain(&self.rows) {
            for (column, cell) in row.iter().enumerate().take(columns) {
                widths[column] = widths[column].max(spans_width(cell));
            }
        }
        widths
    }
}

struct Renderer<'a> {
    theme: &'a Theme,
    width: usize,
    lines: Vec<Line<'static>>,
    inline: Vec<Span<'static>>,
    styles: Vec<Style>,
    prefixes: Vec<Prefix>,
    list_counters: Vec<Option<u64>>,
    link_starts: Vec<(usize, String)>,
    code_block: Option<String>,
    table: Option<TableBuilder>,
    trailing_gap: bool,
    in_heading: bool,
}

impl<'a> Renderer<'a> {
    fn new(width: usize, theme: &'a Theme) -> Self {
        Self {
            theme,
            width,
            lines: Vec::new(),
            inline: Vec::new(),
            styles: Vec::new(),
            prefixes: Vec::new(),
            list_counters: Vec::new(),
            link_starts: Vec::new(),
            code_block: None,
            table: None,
            trailing_gap: false,
            in_heading: false,
        }
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush_inline();
        if self.trailing_gap {
            self.lines.pop();
        }
        self.lines
    }

    fn on_event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.on_start(tag),
            Event::End(tag) => self.on_end(tag),
            Event::Text(text) => self.on_text(&text),
            Event::Code(code) => self.push_inline(code.to_string(), self.inline_code_style()),
            Event::SoftBreak => self.push_inline(" ".to_string(), self.current_style()),
            Event::HardBreak => self.flush_inline(),
            Event::Rule => self.push_rule(),
            Event::TaskListMarker(checked) => {
                let marker = if checked { "[x] " } else { "[ ] " };
                self.push_inline(marker.to_string(), self.current_style());
            }
            Event::Html(html) | Event::InlineHtml(html) => {
                self.push_inline(html.to_string(), self.current_style());
            }
            Event::FootnoteReference(_) | Event::InlineMath(_) | Event::DisplayMath(_) => {}
        }
    }

    fn on_start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => self.flush_inline(),
            Tag::Heading { level, .. } => {
                self.flush_inline();
                self.push_style(heading_style(level, self.theme));
                self.in_heading = true;
            }
            Tag::BlockQuote(_) => {
                self.flush_inline();
                self.prefixes
                    .push(Prefix::bar(QUOTE_BAR, quote_bar_style(self.theme)));
                self.push_style(quote_text_style());
            }
            Tag::CodeBlock(_) => {
                self.flush_inline();
                self.code_block = Some(String::new());
            }
            Tag::List(start) => {
                self.flush_inline();
                self.list_counters.push(start);
            }
            Tag::Item => self.start_item(),
            Tag::Emphasis => self.push_modifier(Modifier::ITALIC),
            Tag::Strong => self.push_modifier(Modifier::BOLD),
            Tag::Strikethrough => self.push_modifier(Modifier::CROSSED_OUT),
            Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. } => {
                self.link_starts
                    .push((self.inline.len(), dest_url.to_string()));
                self.push_style(link_style(self.theme));
            }
            Tag::Table(alignments) => {
                self.flush_inline();
                self.table = Some(TableBuilder::new(alignments));
            }
            Tag::TableHead | Tag::TableRow => {
                if let Some(table) = &mut self.table {
                    table.current_row.clear();
                }
            }
            Tag::TableCell => self.inline.clear(),
            Tag::FootnoteDefinition(_)
            | Tag::DefinitionList
            | Tag::DefinitionListTitle
            | Tag::DefinitionListDefinition
            | Tag::HtmlBlock
            | Tag::MetadataBlock(_)
            | Tag::Superscript
            | Tag::Subscript => {}
        }
    }

    fn on_end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                self.flush_inline();
                self.push_gap();
            }
            TagEnd::Heading(_) => {
                self.flush_inline();
                self.pop_style();
                self.in_heading = false;
                self.push_gap();
            }
            TagEnd::BlockQuote(_) => {
                self.flush_inline();
                self.drop_trailing_gap();
                self.pop_style();
                self.prefixes.pop();
                self.push_gap();
            }
            TagEnd::CodeBlock => self.finish_code_block(),
            TagEnd::List(_) => {
                self.list_counters.pop();
                if self.list_counters.is_empty() {
                    self.push_gap();
                }
            }
            TagEnd::Item => {
                self.flush_inline();
                self.prefixes.pop();
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => self.pop_style(),
            TagEnd::Link | TagEnd::Image => self.finish_link(),
            TagEnd::Table => self.finish_table(),
            TagEnd::TableHead => {
                if let Some(table) = &mut self.table {
                    table.header = std::mem::take(&mut table.current_row);
                }
            }
            TagEnd::TableRow => {
                if let Some(table) = &mut self.table {
                    let row = std::mem::take(&mut table.current_row);
                    table.rows.push(row);
                }
            }
            TagEnd::TableCell => {
                if let Some(table) = &mut self.table {
                    table.current_row.push(std::mem::take(&mut self.inline));
                }
            }
            TagEnd::FootnoteDefinition
            | TagEnd::DefinitionList
            | TagEnd::DefinitionListTitle
            | TagEnd::DefinitionListDefinition
            | TagEnd::HtmlBlock
            | TagEnd::MetadataBlock(_)
            | TagEnd::Superscript
            | TagEnd::Subscript => {}
        }
    }

    fn on_text(&mut self, text: &str) {
        match &mut self.code_block {
            Some(code) => code.push_str(text),
            None => self.push_inline(text.to_string(), self.current_style()),
        }
    }

    fn start_item(&mut self) {
        self.flush_inline();
        let marker = match self.list_counters.last_mut() {
            Some(Some(counter)) => {
                let marker = format!("{counter}. ");
                *counter += 1;
                marker
            }
            Some(None) | None => BULLET.to_string(),
        };
        self.prefixes
            .push(Prefix::marker(marker, list_marker_style(self.theme)));
    }

    fn finish_link(&mut self) {
        self.pop_style();
        let Some((start, url)) = self.link_starts.pop() else {
            return;
        };
        let label: String = self
            .inline
            .get(start..)
            .unwrap_or_default()
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        if label != url {
            self.push_inline(format!(" ({url})"), link_url_style(self.theme));
        }
    }

    fn finish_code_block(&mut self) {
        let Some(code) = self.code_block.take() else {
            return;
        };
        let width = self.inner_width();
        for line in code.lines() {
            if line.is_empty() {
                self.push_line(Vec::new());
                continue;
            }
            let word = Word {
                width: display_width(line),
                fragments: vec![Span::styled(line.to_string(), code_block_style(self.theme))],
            };
            for piece in break_long_word(word, width) {
                self.push_line(piece.fragments);
            }
        }
        self.push_gap();
    }

    fn finish_table(&mut self) {
        let Some(table) = self.table.take() else {
            return;
        };
        for row in table.render(self.inner_width(), self.theme) {
            self.push_line(row);
        }
        self.push_gap();
    }

    fn push_rule(&mut self) {
        self.flush_inline();
        let rule = "─".repeat(self.inner_width());
        self.push_line(vec![Span::styled(rule, rule_style(self.theme))]);
        self.push_gap();
    }

    fn push_inline(&mut self, text: String, style: Style) {
        if !text.is_empty() {
            self.inline.push(Span::styled(text, style));
        }
    }

    fn flush_inline(&mut self) {
        let spans = std::mem::take(&mut self.inline);
        if spans.iter().all(|span| span.content.trim().is_empty()) {
            return;
        }
        for chunk in wrap_spans(&spans, self.inner_width()) {
            self.push_line(chunk);
        }
    }

    fn push_line(&mut self, content: Vec<Span<'static>>) {
        let mut spans: Vec<Span<'static>> = self.prefixes.iter().map(Prefix::current).collect();
        for prefix in &mut self.prefixes {
            prefix.consumed = true;
        }
        spans.extend(content);
        self.lines.push(Line::from(spans));
        self.trailing_gap = false;
    }

    /// One blank separator line after a block, never two in a row and
    /// never before the first line.
    fn push_gap(&mut self) {
        if self.lines.is_empty() || self.trailing_gap {
            return;
        }
        let bars: Vec<Span<'static>> = self
            .prefixes
            .iter()
            .map(|prefix| prefix.rest.clone())
            .filter(|span| !span.content.trim().is_empty())
            .collect();
        self.lines.push(Line::from(bars));
        self.trailing_gap = true;
    }

    fn drop_trailing_gap(&mut self) {
        if self.trailing_gap {
            self.lines.pop();
            self.trailing_gap = false;
        }
    }

    fn inner_width(&self) -> usize {
        let prefix_width: usize = self
            .prefixes
            .iter()
            .map(|prefix| display_width(&prefix.rest.content))
            .sum();
        self.width.saturating_sub(prefix_width).max(1)
    }

    /// Code inside a heading is drawn like the heading rather than fading.
    fn inline_code_style(&self) -> Style {
        if self.in_heading {
            self.current_style()
        } else {
            Style::default().fg(self.theme.code)
        }
    }

    fn current_style(&self) -> Style {
        self.styles
            .last()
            .copied()
            .unwrap_or_else(|| Style::default().fg(self.theme.assistant_text))
    }

    fn push_modifier(&mut self, modifier: Modifier) {
        self.push_style(self.current_style().add_modifier(modifier));
    }

    fn push_style(&mut self, style: Style) {
        self.styles.push(self.current_style().patch(style));
    }

    fn pop_style(&mut self) {
        self.styles.pop();
    }
}

/// Styled lines for `text`, each no wider than `width` columns except
/// where a table cannot shrink far enough to fit.
pub(super) fn render_markdown(text: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let mut renderer = Renderer::new(width.max(1), theme);
    for event in Parser::new_ext(text, parser_options()) {
        renderer.on_event(event);
    }
    renderer.finish()
}

/// Byte offset of the last complete line of `text` that starts a block which renders the
/// same without the text before it: one at column 0 after a blank line, outside a code
/// fence, that is not a list item. `None` when no line qualifies.
pub(super) fn last_block_boundary(text: &str) -> Option<usize> {
    let mut boundary = None;
    let mut fence = None;
    let mut after_blank = false;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let content = line.trim_end();
        match fence {
            Some(open) if closes_fence(content, open) => fence = None,
            Some(_) => {}
            None => {
                if after_blank && line.ends_with('\n') && starts_standalone_block(content) {
                    boundary = Some(offset);
                }
                fence = fence_opening(content);
            }
        }
        after_blank = fence.is_none() && content.is_empty();
        offset += line.len();
    }
    boundary
}

/// Whether `line` begins at column 0 with something other than a list item, whose
/// numbering and spacing depend on the items before it.
fn starts_standalone_block(line: &str) -> bool {
    let marker = line.trim_start_matches(|c: char| c.is_ascii_digit());
    let ordered = marker.len() < line.len() && marker.starts_with(['.', ')']);
    let bullet = line.starts_with(['-', '*', '+']);
    let item = (ordered || bullet) && marker[1..].chars().next().is_none_or(char::is_whitespace);
    !item && line.chars().next().is_some_and(|c| !c.is_whitespace())
}

/// The fence character and run length when `line` opens a fenced code block.
fn fence_opening(line: &str) -> Option<(char, usize)> {
    let line = line.trim_start();
    let fence = line.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let run = line.chars().take_while(|&c| c == fence).count();
    (run >= 3).then_some((fence, run))
}

/// Whether `line` is only a run of the `open` fence's character at least as long as it.
fn closes_fence(line: &str, (fence, run): (char, usize)) -> bool {
    let line = line.trim_start();
    line.len() >= run && line.chars().all(|c| c == fence)
}

fn parser_options() -> Options {
    Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS
}

/// Column widths that keep every column at its natural width when the
/// table fits, otherwise shrink the widest columns first, down to
/// [`MIN_COLUMN_WIDTH`].
fn fit_column_widths(natural: &[usize], available: usize) -> Vec<usize> {
    let natural: Vec<usize> = natural.iter().map(|&w| w.max(1)).collect();
    let capped_total = |cap: usize| natural.iter().map(|&w| w.min(cap)).sum::<usize>();
    let widest = natural.iter().copied().max().unwrap_or(0);
    if capped_total(widest) <= available {
        return natural;
    }
    let mut cap = MIN_COLUMN_WIDTH;
    let mut too_wide = widest.max(cap + 1);
    while too_wide - cap > 1 {
        let middle = cap + (too_wide - cap) / 2;
        if capped_total(middle) <= available {
            cap = middle;
        } else {
            too_wide = middle;
        }
    }
    let mut spare = available.saturating_sub(capped_total(cap));
    natural
        .iter()
        .map(|&w| {
            let widened = w > cap && spare > 0;
            spare -= usize::from(widened);
            w.min(cap) + usize::from(widened)
        })
        .collect()
}

/// The physical lines of one table row: cells wrapped to their column
/// width, padded per alignment and joined by a separator.
fn render_table_row(
    cells: &[Vec<Span<'static>>],
    widths: &[usize],
    alignments: &[Alignment],
    cell_style: Style,
    theme: &Theme,
) -> Vec<Vec<Span<'static>>> {
    let empty = Vec::new();
    let wrapped: Vec<Vec<Vec<Span<'static>>>> = widths
        .iter()
        .enumerate()
        .map(|(column, &width)| {
            let cell = cells.get(column).unwrap_or(&empty);
            let styled: Vec<Span<'static>> = cell
                .iter()
                .map(|span| Span::styled(span.content.clone(), span.style.patch(cell_style)))
                .collect();
            wrap_spans(&styled, width)
        })
        .collect();
    let height = wrapped.iter().map(Vec::len).max().unwrap_or(0).max(1);
    (0..height)
        .map(|row| {
            let mut line = Vec::new();
            for (column, &width) in widths.iter().enumerate() {
                if column > 0 {
                    line.push(Span::styled(COLUMN_SEPARATOR, rule_style(theme)));
                }
                let content = wrapped[column].get(row).cloned().unwrap_or_default();
                let alignment = alignments.get(column).copied().unwrap_or(Alignment::None);
                line.extend(pad_cell(content, width, alignment));
            }
            line
        })
        .collect()
}

fn pad_cell(
    mut content: Vec<Span<'static>>,
    width: usize,
    alignment: Alignment,
) -> Vec<Span<'static>> {
    let slack = width.saturating_sub(spans_width(&content));
    let (left, right) = match alignment {
        Alignment::None | Alignment::Left => (0, slack),
        Alignment::Right => (slack, 0),
        Alignment::Center => (slack / 2, slack - slack / 2),
    };
    if left > 0 {
        content.insert(0, Span::raw(" ".repeat(left)));
    }
    if right > 0 {
        content.push(Span::raw(" ".repeat(right)));
    }
    content
}

fn spans_width(spans: &[Span<'static>]) -> usize {
    spans.iter().map(|span| display_width(&span.content)).sum()
}

/// Greedy word wrap over styled spans. Words keep their per-fragment
/// styles; a word wider than `width` is broken at character boundaries.
fn wrap_spans(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut line: Vec<Span<'static>> = Vec::new();
    let mut line_width = 0;
    for word in split_words(spans)
        .into_iter()
        .flat_map(|word| break_long_word(word, width))
    {
        if line_width > 0 && line_width + 1 + word.width > width {
            lines.push(std::mem::take(&mut line));
            line_width = 0;
        }
        if line_width > 0 {
            line.push(Span::styled(" ", space_style(&line, &word)));
            line_width += 1;
        }
        line_width += word.width;
        line.extend(word.fragments);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

/// The style of a separating space: the shared style when both
/// neighbours agree, so underlines and strikes run through phrases.
fn space_style(line: &[Span<'static>], next: &Word) -> Style {
    let before = line.last().map(|span| span.style);
    let after = next.fragments.first().map(|span| span.style);
    match (before, after) {
        (Some(before), Some(after)) if before == after => before,
        _ => Style::default(),
    }
}

fn split_words(spans: &[Span<'static>]) -> Vec<Word> {
    let mut words = Vec::new();
    let mut current: Option<Word> = None;
    for span in spans {
        for chunk in span.content.split_inclusive(char::is_whitespace) {
            let text = chunk.trim_end_matches(char::is_whitespace);
            if !text.is_empty() {
                let word = current.get_or_insert_with(Word::default);
                word.width += display_width(text);
                word.fragments
                    .push(Span::styled(text.to_string(), span.style));
            }
            if text.len() != chunk.len()
                && let Some(word) = current.take()
            {
                words.push(word);
            }
        }
    }
    words.extend(current);
    words
}

fn break_long_word(word: Word, width: usize) -> Vec<Word> {
    if word.width <= width {
        return vec![word];
    }
    let mut pieces = Vec::new();
    let mut piece = Word::default();
    for fragment in word.fragments {
        let mut text = String::new();
        for ch in fragment.content.chars() {
            let ch_width = display_width(ch.encode_utf8(&mut [0; 4]));
            if piece.width > 0 && piece.width + ch_width > width {
                push_fragment(&mut piece, &mut text, fragment.style);
                pieces.push(std::mem::take(&mut piece));
            }
            text.push(ch);
            piece.width += ch_width;
        }
        push_fragment(&mut piece, &mut text, fragment.style);
    }
    if piece.width > 0 {
        pieces.push(piece);
    }
    pieces
}

fn push_fragment(piece: &mut Word, text: &mut String, style: Style) {
    if !text.is_empty() {
        piece
            .fragments
            .push(Span::styled(std::mem::take(text), style));
    }
}

fn heading_style(level: HeadingLevel, theme: &Theme) -> Style {
    let style = Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD);
    match level {
        HeadingLevel::H1 | HeadingLevel::H2 => style.add_modifier(Modifier::UNDERLINED),
        _ => style,
    }
}

fn code_block_style(theme: &Theme) -> Style {
    Style::default().fg(theme.code)
}

fn quote_bar_style(theme: &Theme) -> Style {
    Style::default().fg(theme.status_text)
}

fn quote_text_style() -> Style {
    Style::default().add_modifier(Modifier::ITALIC)
}

fn list_marker_style(theme: &Theme) -> Style {
    Style::default().fg(theme.accent)
}

fn link_style(theme: &Theme) -> Style {
    Style::default()
        .fg(theme.link)
        .add_modifier(Modifier::UNDERLINED)
}

fn link_url_style(theme: &Theme) -> Style {
    Style::default().fg(theme.status_text)
}

fn rule_style(theme: &Theme) -> Style {
    Style::default().fg(theme.border)
}

#[cfg(test)]
mod tests;

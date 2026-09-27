use super::*;

fn texts(lines: &[Line<'static>]) -> Vec<String> {
    lines
        .iter()
        .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
        .collect()
}

fn rendered(text: &str, width: usize) -> Vec<String> {
    texts(&render_markdown(text, width))
}

fn span_style(lines: &[Line<'static>], content: &str) -> Style {
    lines
        .iter()
        .flat_map(|line| line.spans.iter())
        .find(|span| span.content == content)
        .unwrap_or_else(|| panic!("no span {content:?} in {:?}", texts(lines)))
        .style
}

#[test]
fn plain_paragraph_wraps_at_word_boundaries() {
    assert_eq!(
        rendered("aaaa bbbb cccc dddd", 10),
        ["aaaa bbbb", "cccc dddd"]
    );
}

#[test]
fn inline_styles_map_to_modifiers() {
    let lines = render_markdown("**bold** *italic* ~~gone~~ `code`", 80);
    assert_eq!(texts(&lines), ["bold italic gone code"]);
    assert!(
        span_style(&lines, "bold")
            .add_modifier
            .contains(Modifier::BOLD)
    );
    assert!(
        span_style(&lines, "italic")
            .add_modifier
            .contains(Modifier::ITALIC)
    );
    assert!(
        span_style(&lines, "gone")
            .add_modifier
            .contains(Modifier::CROSSED_OUT)
    );
    assert_eq!(span_style(&lines, "bold").fg, Some(Color::White));
    assert_eq!(span_style(&lines, "code").fg, Some(Color::DarkGray));
    assert_eq!(span_style(&lines, "code").bg, None);
}

#[test]
fn nested_styles_accumulate() {
    let lines = render_markdown("***both***", 80);
    let style = span_style(&lines, "both");
    assert!(
        style
            .add_modifier
            .contains(Modifier::BOLD | Modifier::ITALIC)
    );
}

#[test]
fn styled_word_fragments_stay_on_one_line() {
    let lines = render_markdown("xxxx **bo**ld", 8);
    assert_eq!(texts(&lines), ["xxxx", "bold"]);
    assert!(
        span_style(&lines, "bo")
            .add_modifier
            .contains(Modifier::BOLD)
    );
    assert!(
        !span_style(&lines, "ld")
            .add_modifier
            .contains(Modifier::BOLD)
    );
}

#[test]
fn headings_are_bold_and_separated_from_body() {
    let lines = render_markdown("# Title\n\nBody text.\n\n### Sub\n\nMore.", 80);
    assert_eq!(
        texts(&lines),
        ["Title", "", "Body text.", "", "Sub", "", "More."]
    );
    let title = span_style(&lines, "Title");
    assert!(
        title
            .add_modifier
            .contains(Modifier::BOLD | Modifier::UNDERLINED)
    );
    assert_eq!(title.fg, Some(BLEY));
    let sub = span_style(&lines, "Sub");
    assert!(sub.add_modifier.contains(Modifier::BOLD));
    assert!(!sub.add_modifier.contains(Modifier::UNDERLINED));
    assert_eq!(sub.fg, Some(BLEY));
}

#[test]
fn code_inside_a_heading_is_white() {
    let lines = render_markdown("## The `run` tool\n\nUse `run` here.", 80);
    assert_eq!(texts(&lines), ["The run tool", "", "Use run here."]);
    let styles: Vec<Style> = lines
        .iter()
        .flat_map(|line| line.spans.iter())
        .filter(|span| span.content == "run")
        .map(|span| span.style)
        .collect();
    assert_eq!(styles.len(), 2);
    assert_eq!(styles[0].fg, Some(Color::White));
    assert_eq!(styles[1].fg, Some(Color::DarkGray));
}

#[test]
fn bullet_list_marks_items_and_indents_wrapped_continuations() {
    let lines = render_markdown("- short\n- a longer item that wraps", 14);
    assert_eq!(
        texts(&lines),
        ["• short", "• a longer", "  item that", "  wraps"]
    );
    assert_eq!(span_style(&lines, "• ").fg, Some(BLEY));
}

#[test]
fn ordered_list_counts_from_its_start_number() {
    assert_eq!(rendered("3. three\n4. four", 40), ["3. three", "4. four"]);
}

#[test]
fn nested_lists_indent_under_their_parent() {
    assert_eq!(
        rendered("- outer\n  - inner\n  - inner two\n- next", 40),
        ["• outer", "  • inner", "  • inner two", "• next"]
    );
}

#[test]
fn task_list_shows_checkbox_state() {
    assert_eq!(
        rendered("- [x] done\n- [ ] todo", 40),
        ["• [x] done", "• [ ] todo"]
    );
}

#[test]
fn loose_list_keeps_paragraph_gaps_between_items() {
    assert_eq!(rendered("- one\n\n- two\n", 40), ["• one", "", "• two"]);
}

#[test]
fn blockquote_bars_every_line_including_wraps() {
    let lines = render_markdown("> quoted words that wrap\n\nafter", 14);
    assert_eq!(
        texts(&lines),
        ["▎ quoted words", "▎ that wrap", "", "after"]
    );
    assert!(
        span_style(&lines, "quoted")
            .add_modifier
            .contains(Modifier::ITALIC)
    );
}

#[test]
fn fenced_code_keeps_indentation_and_breaks_long_lines_hard() {
    let text = "```rust\nfn main() {\n    let x = 1;\n\n}\n```\ntail";
    let lines = render_markdown(text, 12);
    assert_eq!(
        texts(&lines),
        ["fn main() {", "    let x = ", "1;", "", "}", "", "tail",]
    );
    for line in [&lines[0], &lines[4]] {
        assert!(
            line.spans
                .iter()
                .all(|s| s.style.fg == Some(Color::DarkGray))
        );
    }
}

#[test]
fn fenced_code_inside_a_list_sits_under_the_item() {
    assert_eq!(
        rendered(
            "- item
  ```
  x
  ```",
            8
        ),
        ["• item", "  x"]
    );
}

#[test]
fn unterminated_fence_renders_as_code_while_streaming() {
    assert_eq!(rendered("```\nls -la", 10), ["ls -la"]);
}

#[test]
fn partial_emphasis_renders_literally_until_closed() {
    assert_eq!(rendered("a **b", 40), ["a **b"]);
    let lines = render_markdown("a **bold** c", 40);
    assert_eq!(texts(&lines), ["a bold c"]);
}

#[test]
fn table_aligns_columns_and_bolds_header() {
    let text = "| Name | Qty |\n|:-----|----:|\n| apple | 3 |\n| kiwi | 12 |";
    let lines = render_markdown(text, 40);
    assert_eq!(
        texts(&lines),
        ["Name  │ Qty", "──────┼────", "apple │   3", "kiwi  │  12",]
    );
    assert!(
        span_style(&lines, "Name")
            .add_modifier
            .contains(Modifier::BOLD)
    );
    assert!(
        !span_style(&lines, "apple")
            .add_modifier
            .contains(Modifier::BOLD)
    );
}

#[test]
fn table_centers_cells_when_asked() {
    let text = "| Head |\n|:----:|\n| ab |";
    assert_eq!(rendered(text, 40), ["Head", "────", " ab "]);
}

#[test]
fn table_keeps_inline_styles_inside_cells() {
    let lines = render_markdown("| A |\n|---|\n| **x** |", 40);
    assert_eq!(texts(&lines), ["A", "─", "x"]);
    assert!(
        span_style(&lines, "x")
            .add_modifier
            .contains(Modifier::BOLD)
    );
}

#[test]
fn wide_table_shrinks_widest_column_and_wraps_its_cells() {
    let text = "| K | Description |\n|---|---|\n| a | one two three four |";
    assert_eq!(
        rendered(text, 16),
        [
            "K │ Description ",
            "──┼─────────────",
            "a │ one two     ",
            "  │ three four  ",
        ]
    );
}

#[test]
fn table_missing_cells_are_padded() {
    let text = "| A | B |\n|---|---|\n| only |";
    assert_eq!(rendered(text, 40), ["A    │ B", "─────┼──", "only │  "]);
}

#[test]
fn table_before_delimiter_row_streams_as_text() {
    assert_eq!(rendered("| A | B |", 40), ["| A | B |"]);
}

#[test]
fn link_shows_label_then_dimmed_url() {
    let lines = render_markdown("see [docs](https://x.y) now", 80);
    assert_eq!(texts(&lines), ["see docs (https://x.y) now"]);
    assert!(
        span_style(&lines, "docs")
            .add_modifier
            .contains(Modifier::UNDERLINED)
    );
    assert_eq!(
        span_style(&lines, "(https://x.y)").fg,
        Some(Color::DarkGray)
    );
}

#[test]
fn bare_url_is_not_repeated() {
    assert_eq!(rendered("<https://x.y>", 80), ["https://x.y"]);
}

#[test]
fn rule_spans_the_width() {
    assert_eq!(rendered("a\n\n---\n\nb", 5), ["a", "", "─────", "", "b"]);
}

#[test]
fn hard_break_starts_a_new_line_in_the_same_paragraph() {
    assert_eq!(rendered("one  \ntwo", 40), ["one", "two"]);
}

#[test]
fn soft_break_joins_as_a_space() {
    assert_eq!(rendered("one\ntwo", 40), ["one two"]);
}

#[test]
fn consecutive_blank_lines_collapse_and_trailing_gap_is_dropped() {
    assert_eq!(rendered("a\n\n\n\n\nb\n\n\n", 40), ["a", "", "b"]);
}

#[test]
fn overlong_word_is_broken_at_character_boundaries() {
    assert_eq!(rendered("abcdefghij klm", 4), ["abcd", "efgh", "ij", "klm"]);
}

#[test]
fn wide_characters_count_two_columns() {
    assert_eq!(rendered("日本語 テキスト", 8), ["日本語", "テキスト"]);
}

#[test]
fn width_one_never_panics() {
    let text = "# H\n\n- item\n\n> q\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\ncode\n```";
    assert!(!render_markdown(text, 1).is_empty());
    assert!(!render_markdown(text, 0).is_empty());
}

#[test]
fn empty_input_renders_nothing() {
    assert!(render_markdown("", 40).is_empty());
    assert!(render_markdown("\n\n", 40).is_empty());
}

#[test]
fn list_inside_blockquote_carries_both_prefixes() {
    assert_eq!(rendered("> - a\n> - b", 40), ["▎ • a", "▎ • b"]);
}

#[test]
fn item_with_nested_list_prints_its_text_first() {
    assert_eq!(
        rendered("1. top\n   - sub\n2. next", 40),
        ["1. top", "   • sub", "2. next"]
    );
}

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
fn bullet_list_marks_items_and_indents_wrapped_continuations() {
    let lines = render_markdown("- short\n- a longer item that wraps", 14);
    assert_eq!(
        texts(&lines),
        ["• short", "• a longer", "  item that", "  wraps"]
    );
    assert_eq!(span_style(&lines, "• ").fg, Some(BLEY));
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
fn unterminated_fence_renders_as_code_while_streaming() {
    assert_eq!(rendered("```\nls -la", 10), ["ls -la"]);
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
fn hard_break_inside_link_label_does_not_panic() {
    assert_eq!(
        rendered("*see* the [docs  \nhere](https://x.y)", 80),
        ["see the docs", "here (https://x.y)"]
    );
}

#[test]
fn consecutive_blank_lines_collapse_and_trailing_gap_is_dropped() {
    assert_eq!(rendered("a\n\n\n\n\nb\n\n\n", 40), ["a", "", "b"]);
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
fn item_with_nested_list_prints_its_text_first() {
    assert_eq!(
        rendered("1. top\n   - sub\n2. next", 40),
        ["1. top", "   • sub", "2. next"]
    );
}

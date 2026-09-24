use std::time::Duration;

use ratatui_image::picker::Picker;

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

fn item_texts(p: &OutputPane) -> Vec<String> {
    p.items.iter().map(item_text).collect()
}

fn rendered_lines(p: &mut OutputPane, w: u16, h: u16) -> Vec<String> {
    let (lines, _) = p.render_view(w, h);
    lines.iter().map(line_text).collect()
}

fn last_thinking(p: &mut OutputPane) -> &mut ThinkingBlock {
    match p.items.last_mut() {
        Some(OutputItem::Thinking(t)) => t,
        _ => panic!("expected a Thinking item last"),
    }
}

fn push_seq_30(p: &mut OutputPane) {
    let body: String =
        (0..30).map(|i| format!("line {i}\n")).collect::<String>() + "[exit:0 | 1ms]";
    p.push_tool_block("seq 30".into(), body, 0, 1);
}

fn body_rows(rendered: &[String]) -> Vec<&str> {
    rendered
        .iter()
        .map(String::as_str)
        .filter(|l| l.starts_with("▎ line "))
        .collect()
}

fn footer(inner_w: usize, text: &str) -> String {
    format!("▎ {}{text}", " ".repeat(inner_w - text.chars().count()))
}

#[test]
fn push_user_prefixes_caret_and_adds_separator() {
    let mut p = OutputPane::new();
    p.push_user("hi");
    assert_eq!(item_texts(&p), ["> hi", ""]);
}

#[test]
fn begin_and_append_assistant_streams_into_one_line() {
    let mut p = OutputPane::new();
    p.begin_assistant();
    p.append_assistant("hello ");
    p.append_assistant("world");
    assert_eq!(item_texts(&p), ["hello world"]);
    assert_eq!(p.open_assistant, Some(0));
}

#[test]
fn append_without_begin_auto_starts() {
    let mut p = OutputPane::new();
    p.append_assistant("hey");
    assert_eq!(item_texts(&p), ["hey"]);
}

#[test]
fn append_splits_on_embedded_newlines() {
    let mut p = OutputPane::new();
    p.append_assistant("line1\nline2\nline3");
    assert_eq!(item_texts(&p), ["line1", "line2", "line3"]);
    assert_eq!(p.open_assistant, Some(2));
}

#[test]
fn append_trailing_newline_opens_blank_tail() {
    let mut p = OutputPane::new();
    p.append_assistant("hello\n");
    assert_eq!(item_texts(&p), ["hello", ""]);
    assert_eq!(p.open_assistant, Some(1));
    p.append_assistant("more");
    assert_eq!(item_texts(&p), ["hello", "more"]);
}

#[test]
fn finish_assistant_closes_stream_and_adds_separator() {
    let mut p = OutputPane::new();
    p.begin_assistant();
    p.append_assistant("done");
    p.finish_assistant();
    assert_eq!(p.open_assistant, None);
    assert_eq!(item_texts(&p), ["done", ""]);
}

#[test]
fn push_user_mid_stream_closes_open_assistant() {
    let mut p = OutputPane::new();
    p.append_assistant("half");
    p.push_user("new question");
    assert_eq!(p.open_assistant, None);
    assert_eq!(item_texts(&p), ["half", "", "> new question", ""]);
}

#[test]
fn push_error_adds_exclamation_prefix() {
    let mut p = OutputPane::new();
    p.push_error("boom");
    assert_eq!(item_texts(&p), ["!! boom"]);
}

#[test]
fn tool_block_renders_header_body_and_one_right_aligned_footer() {
    let mut p = OutputPane::new();
    p.push_tool_block("ls /tmp".into(), "a\nb\n[exit:0 | 5ms]".into(), 0, 5);
    assert_eq!(
        rendered_lines(&mut p, 40, 10),
        [
            "▎ $ ls /tmp".to_string(),
            "▎ a".to_string(),
            "▎ b".to_string(),
            footer(38, "[exit:0 | 5ms]"),
            String::new(),
        ]
    );
}

#[test]
fn empty_output_still_shows_header_and_footer() {
    let mut p = OutputPane::new();
    p.push_tool_block("true".into(), "[exit:0 | 0ms]".into(), 0, 0);
    assert_eq!(
        rendered_lines(&mut p, 40, 10),
        [
            "▎ $ true".to_string(),
            footer(38, "[exit:0 | 0ms]"),
            String::new(),
        ]
    );
}

#[test]
fn tool_block_collapsed_when_over_threshold() {
    let mut p = OutputPane::new();
    push_seq_30(&mut p);
    let rendered = rendered_lines(&mut p, 60, 80);
    let expected: Vec<String> = (0..COLLAPSED_HEAD_LINES)
        .map(|i| format!("▎ line {i}"))
        .collect();
    assert_eq!(body_rows(&rendered), expected);
    assert!(
        rendered.contains(&"▎ … (20 more lines, Tab to expand)".to_string()),
        "{rendered:#?}"
    );
}

#[test]
fn tool_block_expanded_after_toggle() {
    let mut p = OutputPane::new();
    push_seq_30(&mut p);
    assert!(p.toggle_last_expandable());
    let rendered = rendered_lines(&mut p, 60, 80);
    assert_eq!(body_rows(&rendered).len(), 30);
    assert!(!rendered.iter().any(|l| l.contains("more lines")));
}

#[test]
fn verbose_mode_expands_collapsed_tool_block() {
    let mut p = OutputPane::new();
    push_seq_30(&mut p);
    assert_eq!(
        body_rows(&rendered_lines(&mut p, 60, 80)).len(),
        COLLAPSED_HEAD_LINES
    );
    p.set_verbose(true);
    assert_eq!(body_rows(&rendered_lines(&mut p, 60, 80)).len(), 30);
}

#[test]
fn collapsed_block_keeps_stderr_lines_visible() {
    let mut body = String::new();
    for i in 0..25 {
        body.push_str(&format!("stdout line {i}\n"));
    }
    body.push_str("[stderr] [grep]\tboom\n");
    body.push_str("[exit:1 | 3ms]");
    let mut p = OutputPane::new();
    p.push_tool_block("grep foo bar".into(), body, 1, 3);
    let rendered = rendered_lines(&mut p, 80, 80);
    assert!(
        rendered.contains(&"▎ [stderr] [grep]\tboom".to_string()),
        "{rendered:#?}"
    );
    assert!(
        rendered.contains(&"▎ … (15 more lines, Tab to expand)".to_string()),
        "{rendered:#?}"
    );
}

#[test]
fn bar_color_follows_exit_code() {
    for (exit_code, expected) in [(0, Color::Green), (1, Color::Red)] {
        let mut p = OutputPane::new();
        p.push_tool_block(
            "cmd".into(),
            format!("out\n[exit:{exit_code} | 1ms]"),
            exit_code,
            1,
        );
        let (lines, _) = p.render_view(40, 10);
        let barred: Vec<_> = lines.iter().filter(|l| !line_text(l).is_empty()).collect();
        assert_eq!(barred.len(), 3, "exit {exit_code}");
        for line in barred {
            assert_eq!(line.spans[0].style.fg, Some(expected), "exit {exit_code}");
        }
    }
}

#[test]
fn very_long_single_line_output_wraps_under_bar() {
    let body = format!("{}\n[exit:0 | 1ms]", "x".repeat(500));
    let mut p = OutputPane::new();
    p.push_tool_block("yes | head".into(), body, 0, 1);
    let rendered = rendered_lines(&mut p, 40, 200);
    assert_eq!(rendered.iter().filter(|l| l.starts_with("▎ x")).count(), 14);
    for l in rendered.iter().filter(|l| !l.trim().is_empty()) {
        assert!(l.starts_with('▎'), "missing bar: {l:?}");
    }
}

#[test]
fn toggle_last_expandable_is_false_without_blocks() {
    let mut p = OutputPane::new();
    p.push_user("hi");
    assert!(!p.toggle_last_expandable());
}

#[test]
fn begin_thinking_creates_live_block_collapsed_by_default() {
    let mut p = OutputPane::new();
    p.begin_thinking();
    assert_eq!(p.items.len(), 1);
    let t = last_thinking(&mut p);
    assert!(t.ended_at.is_none());
    assert!(!t.expanded);
    assert!(t.text.is_empty());
}

#[test]
fn append_thinking_streams_into_open_block() {
    let mut p = OutputPane::new();
    p.append_thinking("let me ");
    p.append_thinking("think");
    assert_eq!(item_texts(&p), ["[thinking:live:let me think]"]);
}

#[test]
fn finish_thinking_stamps_ended_at_and_collapses() {
    let mut p = OutputPane::new();
    p.append_thinking("done thinking");
    assert!(p.toggle_last_expandable());
    p.finish_thinking();
    let t = last_thinking(&mut p);
    assert!(t.ended_at.is_some());
    assert!(!t.expanded);
}

#[test]
fn finish_thinking_is_idempotent() {
    let mut p = OutputPane::new();
    p.append_thinking("x");
    p.finish_thinking();
    let first = last_thinking(&mut p).ended_at;
    p.finish_thinking();
    p.finish_thinking();
    assert_eq!(last_thinking(&mut p).ended_at, first);
}

#[test]
fn live_thinking_seconds_counts_only_the_live_block() {
    let mut p = OutputPane::new();
    assert_eq!(p.live_thinking_seconds(), None);
    p.append_thinking("x");
    last_thinking(&mut p).started_at -= Duration::from_secs(5);
    assert_eq!(p.live_thinking_seconds(), Some(5));
    p.finish_thinking();
    assert_eq!(p.live_thinking_seconds(), None);
}

#[test]
fn thinking_block_renders_collapsed_live_header_only() {
    let mut p = OutputPane::new();
    p.append_thinking("reasoning body line 1");
    last_thinking(&mut p).started_at -= Duration::from_secs(5);
    assert_eq!(rendered_lines(&mut p, 60, 20), ["▎ ✻ Thinking… (5s)", ""]);
}

#[test]
fn thinking_block_renders_body_after_tab_expand() {
    let mut p = OutputPane::new();
    p.append_thinking("reasoning body line 1");
    assert!(p.toggle_last_expandable());
    let rendered = rendered_lines(&mut p, 60, 20);
    assert!(rendered.contains(&"▎ reasoning body line 1".to_string()));
}

#[test]
fn verbose_mode_expands_live_thinking_without_per_item_toggle() {
    let mut p = OutputPane::new();
    p.append_thinking("verbose body");
    let shows_body = |p: &mut OutputPane| {
        rendered_lines(p, 60, 20)
            .iter()
            .any(|l| l.contains("verbose body"))
    };
    assert!(!shows_body(&mut p));
    p.set_verbose(true);
    assert!(shows_body(&mut p));
    assert!(!last_thinking(&mut p).expanded);
    p.set_verbose(false);
    assert!(!shows_body(&mut p));
}

#[test]
fn thinking_block_renders_past_tense_after_finish() {
    let mut p = OutputPane::new();
    p.append_thinking("body");
    p.finish_thinking();
    let t = last_thinking(&mut p);
    t.started_at = t.ended_at.expect("finished") - Duration::from_secs(3);
    assert_eq!(rendered_lines(&mut p, 60, 20), ["▎ ✦ Thought for 3s", ""]);
}

#[test]
fn begin_thinking_prunes_empty_open_assistant() {
    // A submit opens an empty assistant block before any delta; reasoning
    // arriving first must replace it rather than leave a stray blank line.
    let mut p = OutputPane::new();
    p.begin_assistant();
    p.append_thinking("first thoughts");
    assert_eq!(item_texts(&p), ["[thinking:live:first thoughts]"]);
    assert_eq!(p.open_assistant, None);
}

#[test]
fn finished_thinking_then_new_reasoning_opens_fresh_block() {
    let mut p = OutputPane::new();
    p.append_thinking("phase one");
    p.finish_thinking();
    p.append_thinking("phase two");
    assert_eq!(
        item_texts(&p),
        ["[thinking:done:phase one]", "[thinking:live:phase two]"]
    );
}

#[test]
fn toggle_last_expandable_prefers_most_recent_item() {
    let mut p = OutputPane::new();
    p.push_tool_block("ls".into(), "a\n[exit:0 | 1ms]".into(), 0, 1);
    p.append_thinking("recent reasoning");
    p.finish_thinking();
    assert!(p.toggle_last_expandable());
    assert!(last_thinking(&mut p).expanded);
    match &p.items[0] {
        OutputItem::Tool(b) => assert!(b.expanded, "tool block must remain expanded"),
        _ => panic!("expected Tool item at index 0"),
    }

    let mut q = OutputPane::new();
    q.append_thinking("earlier reasoning");
    q.finish_thinking();
    q.push_tool_block("ls".into(), "a\n[exit:0 | 1ms]".into(), 0, 1);
    assert!(q.toggle_last_expandable());
    match q.items.last() {
        Some(OutputItem::Tool(b)) => assert!(!b.expanded, "toggle should have collapsed"),
        _ => panic!("expected Tool item last"),
    }
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
fn wrapping_splits_long_line_at_word_boundaries() {
    let mut p = OutputPane::new();
    p.append_assistant("aaaa bbbb cccc dddd");
    assert_eq!(rendered_lines(&mut p, 10, 5), ["aaaa bbbb", "cccc dddd"]);
}

#[test]
fn render_view_clamps_scroll_offset() {
    let mut p = OutputPane::new();
    for _ in 0..15 {
        p.append_assistant("line\n");
    }
    p.scroll_offset = 99;
    let (lines, start) = p.render_view(80, 10);
    assert_eq!(lines.len(), 10);
    assert_eq!(start, 0);
    assert_eq!(p.scroll_offset, 6);
}

#[test]
fn render_view_zero_width_falls_back_to_raw_lines() {
    let mut p = OutputPane::new();
    p.push_user("hi");
    assert_eq!(rendered_lines(&mut p, 0, 5), ["> hi", ""]);
}

fn thumbnail_protocol() -> StatefulProtocol {
    Picker::halfblocks().new_resize_protocol(image::DynamicImage::new_rgb8(4, 4))
}

fn assert_matches_full_rewrap(p: &mut OutputPane, width: u16) {
    p.sync_wrap(width);
    let mut full = WrapCache::default();
    full.sync(&p.items, width, p.verbose);
    let incremental: Vec<String> = p.wrap.lines.iter().map(line_text).collect();
    let expected: Vec<String> = full.lines.iter().map(line_text).collect();
    assert_eq!(incremental, expected, "width {width}");
    assert_eq!(p.wrap.lines, full.lines, "width {width}");
    assert_eq!(p.wrap.starts, full.starts, "width {width}");
    assert_eq!(p.wrap.thumbnails, full.thumbnails, "width {width}");
}

/// A viewport width and the pane mutation applied before checking it.
type Step = (u16, fn(&mut OutputPane));

#[test]
fn incremental_rewrap_matches_full_rewrap() {
    let mut p = OutputPane::new();
    let steps: &[Step] = &[
        (40, |p| p.push_user("what is in /tmp and why")),
        (40, |p| p.begin_assistant()),
        (40, |p| p.append_thinking("let me look ")),
        (40, |p| {
            p.append_thinking("at the directory\n\nwith ls");
            last_thinking(p).started_at -= Duration::from_millis(5500);
        }),
        (40, |p| p.refresh_live_thinking()),
        (40, |p| assert!(p.toggle_last_expandable())),
        (40, |p| p.append_thinking(" and some more reasoning")),
        (40, |p| p.finish_thinking()),
        (40, push_seq_30),
        (40, |p| p.append_assistant("The directory holds ")),
        (40, |p| {
            p.append_assistant("thirty files that are all quite long")
        }),
        (13, |p| p.append_assistant("\nsecond line")),
        (13, |p| p.append_assistant("\n")),
        (13, |p| assert!(p.toggle_last_expandable())),
        (80, |p| p.append_assistant("third line")),
        (80, |p| {
            p.push_thumbnail("cat.png".into(), thumbnail_protocol())
        }),
        (80, |p| p.append_assistant("after the image")),
        (80, |p| p.set_verbose(true)),
        (9, |p| p.append_assistant(" still streaming")),
        (9, |p| p.set_verbose(false)),
        (9, |p| p.finish_assistant()),
        (0, |p| p.push_user("follow-up")),
        (25, |p| p.append_assistant("a reply to be undone")),
        (25, |p| assert_eq!(p.pop_last_user_exchange(), 3)),
        (25, |p| p.append_assistant("fresh reply")),
        (25, |p| p.clear()),
        (25, |p| p.push_info("empty again")),
    ];
    for &(width, step) in steps {
        step(&mut p);
        assert_matches_full_rewrap(&mut p, width);
    }
}

#[test]
fn streaming_delta_rewraps_only_the_open_line() {
    let mut p = OutputPane::new();
    push_seq_30(&mut p);
    p.append_assistant("hello");
    p.sync_wrap(40);
    let before_tail = p.wrap.starts[1];
    p.append_assistant(" world");
    assert_eq!(p.wrap.stale, [1]);
    p.sync_wrap(40);
    assert_eq!(p.wrap.starts, [0, before_tail]);
    assert_eq!(line_text(&p.wrap.lines[before_tail]), "hello world");
}

#[test]
fn scrolling_addresses_lines_past_u16_max() {
    let mut p = OutputPane::new();
    let total = 70_000;
    for i in 0..total {
        p.push_info(&format!("line {i}"));
    }
    let first_visible = |p: &mut OutputPane| {
        let (lines, start) = p.render_view(80, 10);
        (start, line_text(&lines[0]), lines.len())
    };
    assert_eq!(
        first_visible(&mut p),
        (total - 10, format!("line {}", total - 10), 10)
    );
    p.scroll_lines_up(2);
    assert_eq!(
        first_visible(&mut p),
        (total - 12, format!("line {}", total - 12), 10)
    );
    p.scroll_offset = total - 10 - 66_000;
    assert_eq!(first_visible(&mut p), (66_000, "line 66000".into(), 10));
    p.scroll_offset = usize::MAX;
    assert_eq!(first_visible(&mut p), (0, "line 0".into(), 10));
    assert_eq!(p.scroll_offset(), total - 10);
}

#[test]
fn thumbnail_slots_line_up_with_rendered_rows_at_narrow_widths() {
    let mut p = OutputPane::new();
    push_seq_30(&mut p);
    p.append_thinking("reasoning");
    p.finish_thinking();
    p.push_thumbnail("cat.png".into(), thumbnail_protocol());
    p.push_info("below");
    p.push_thumbnail("dog.png".into(), thumbnail_protocol());
    for width in [12, 20, 80] {
        let slots = p.thumbnail_layout(width);
        let (lines, top) = p.render_view(width, u16::MAX);
        assert_eq!(top, 0);
        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(slots.len(), 2, "width {width}");
        for (slot, name) in slots.iter().zip(["cat.png", "dog.png"]) {
            assert_eq!(slot.height, usize::from(THUMBNAIL_ROWS), "width {width}");
            assert_eq!(
                rendered[slot.start_row],
                format!("📎 {name}"),
                "width {width}"
            );
        }
        assert_eq!(rendered[slots[0].start_row + slots[0].height], "below");
    }
}

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

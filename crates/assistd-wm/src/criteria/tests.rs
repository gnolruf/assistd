use super::*;

fn id(raw: u64) -> WindowId {
    WindowId::new(raw).expect("test ids are non-zero")
}

#[test]
fn escape_for_criteria_handles_quotes_and_backslashes() {
    assert_eq!(escape_for_criteria("Firefox"), "Firefox");
    assert_eq!(escape_for_criteria(r#"a"b"#), r#"a\"b"#);
    assert_eq!(escape_for_criteria(r"a\b"), r"a\\b");
    assert_eq!(escape_for_criteria(r#"a"b\c"#), r#"a\"b\\c"#);
}

#[test]
fn format_workspace_target_by_number_or_quoted_name() {
    for (workspace, expected) in [
        (WorkspaceId::Num(3), "workspace number 3"),
        (WorkspaceId::Num(10), "workspace number 10"),
        (WorkspaceId::name("scratch"), r#"workspace "scratch""#),
        (
            WorkspaceId::name(r#"weird"name"#),
            r#"workspace "weird\"name""#,
        ),
    ] {
        assert_eq!(
            format_workspace_target(&workspace),
            expected,
            "{workspace:?}"
        );
    }
}

#[test]
fn workspace_id_parse_or_name() {
    assert_eq!("3".parse::<WorkspaceId>().unwrap(), WorkspaceId::Num(3));
    assert_eq!("03".parse::<WorkspaceId>().unwrap(), WorkspaceId::Num(3));
    assert_eq!(
        "scratch".parse::<WorkspaceId>().unwrap(),
        WorkspaceId::Name("scratch".into())
    );
    assert_eq!(
        "1:web".parse::<WorkspaceId>().unwrap(),
        WorkspaceId::Name("1:web".into())
    );
}

#[test]
fn focus_and_move_use_con_id_criteria() {
    assert_eq!(format_focus(&id(42)), r#"[con_id="42"] focus"#);
    assert_eq!(
        format_move_to_workspace(&id(42), &WorkspaceId::Num(3)),
        r#"[con_id="42"] move container to workspace number 3"#
    );
}

#[test]
fn resize_payload_uses_con_id_criteria() {
    assert_eq!(
        format_resize_width(&id(42), ResizeDir::Grow, 50),
        r#"[con_id="42"] resize grow width 50 px or 0 ppt"#
    );
    assert_eq!(
        format_resize_width(&id(1234567890), ResizeDir::Shrink, 5),
        r#"[con_id="1234567890"] resize shrink width 5 px or 0 ppt"#
    );
}

#[test]
fn layout_payload_emits_bare_form() {
    for (layout, expected) in [
        (Layout::Default, "layout default"),
        (Layout::Tabbed, "layout tabbed"),
        (Layout::Stacking, "layout stacking"),
        (Layout::SplitH, "layout splith"),
        (Layout::SplitV, "layout splitv"),
    ] {
        assert_eq!(format_layout(layout), expected);
    }
}

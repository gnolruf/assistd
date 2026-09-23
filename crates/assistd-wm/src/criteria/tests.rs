use super::*;
use crate::WindowId;

fn id(n: u64) -> WindowId {
    WindowId::new(n).expect("test ids are non-zero")
}

fn anchor(corner: AnchorCorner, ox: i32, oy: i32) -> PlacementAnchor {
    PlacementAnchor {
        corner,
        offset_x: ox,
        offset_y: oy,
        width: 360,
        height: 120,
    }
}

fn workspace_1920_1055() -> Rect {
    Rect {
        x: 0,
        y: 0,
        width: 1920,
        height: 1055,
    }
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
    for (ws, expected) in [
        (WorkspaceId::Num(3), "workspace number 3"),
        (WorkspaceId::Num(10), "workspace number 10"),
        (WorkspaceId::name("scratch"), r#"workspace "scratch""#),
        (
            WorkspaceId::name(r#"weird"name"#),
            r#"workspace "weird\"name""#,
        ),
    ] {
        assert_eq!(format_workspace_target(&ws), expected, "{ws:?}");
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
    for (l, expected) in [
        (Layout::Default, "layout default"),
        (Layout::Tabbed, "layout tabbed"),
        (Layout::Stacking, "layout stacking"),
        (Layout::SplitH, "layout splith"),
        (Layout::SplitV, "layout splitv"),
    ] {
        assert_eq!(format_layout(l), expected);
    }
}

#[test]
fn format_criteria_clause_per_variant_with_escaping() {
    for (criteria, expected) in [
        (
            PlacementCriteria::AppId("dev.assistd.popup".into()),
            r#"[app_id="dev.assistd.popup"]"#,
        ),
        (
            PlacementCriteria::Class("Firefox".into()),
            r#"[class="Firefox"]"#,
        ),
        (
            PlacementCriteria::Title("Inbox".into()),
            r#"[title="^Inbox$"]"#,
        ),
        (PlacementCriteria::ConId(id(42)), r#"[con_id="42"]"#),
        (
            PlacementCriteria::AppId(r#"a"b"#.into()),
            r#"[app_id="a\"b"]"#,
        ),
        (PlacementCriteria::Class(r"a\b".into()), r#"[class="a\\b"]"#),
    ] {
        assert_eq!(format_criteria_clause(&criteria), expected, "{criteria:?}");
    }
}

#[test]
fn place_floating_pixels_chains_commands_under_one_criteria() {
    let p = format_place_floating_pixels(
        &PlacementCriteria::Title("dev.assistd.popup".into()),
        anchor(AnchorCorner::BottomRight, -10, -30),
        workspace_1920_1055(),
    );
    assert_eq!(
        p,
        concat!(
            r#"[title="^dev.assistd.popup$"] floating enable, "#,
            r#"[title="^dev.assistd.popup$"] resize set 360 120, "#,
            r#"[title="^dev.assistd.popup$"] move position 1550 px 905 px, "#,
            r#"[title="^dev.assistd.popup$"] sticky enable"#,
        )
    );
}

#[test]
fn compute_target_position_per_corner() {
    // 360x120 window on a 1920x1055 workspace.
    for (corner, ox, oy, expected) in [
        (AnchorCorner::TopLeft, 10, 10, (10, 10)),
        (AnchorCorner::TopRight, -10, 10, (1550, 10)),
        (AnchorCorner::BottomLeft, 10, -10, (10, 925)),
        (AnchorCorner::BottomRight, -10, -30, (1550, 905)),
        (AnchorCorner::Center, 0, 0, (780, 467)),
    ] {
        assert_eq!(
            compute_target_position(anchor(corner, ox, oy), workspace_1920_1055()),
            expected,
            "{corner:?} ({ox}, {oy})"
        );
    }
}

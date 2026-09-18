use super::*;
use crate::WindowId;

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
fn format_workspace_target_numeric() {
    assert_eq!(
        format_workspace_target(&WorkspaceId::Num(3)),
        "workspace number 3"
    );
    assert_eq!(
        format_workspace_target(&WorkspaceId::Num(10)),
        "workspace number 10"
    );
}

#[test]
fn format_workspace_target_named() {
    assert_eq!(
        format_workspace_target(&WorkspaceId::name("scratch")),
        r#"workspace "scratch""#
    );
}

#[test]
fn format_workspace_target_named_with_quote_is_escaped() {
    assert_eq!(
        format_workspace_target(&WorkspaceId::name(r#"weird"name"#)),
        r#"workspace "weird\"name""#
    );
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
    let id = WindowId::new(42).unwrap();
    assert_eq!(format_focus(&id), r#"[con_id="42"] focus"#);
    assert_eq!(
        format_move_to_workspace(&id, &WorkspaceId::Num(3)),
        r#"[con_id="42"] move container to workspace number 3"#
    );
}

#[test]
fn resize_payload_uses_con_id_criteria() {
    let p = format_resize_width(&WindowId::new(42).unwrap(), ResizeDir::Grow, 50);
    assert_eq!(p, r#"[con_id="42"] resize grow width 50 px or 0 ppt"#);
    let p = format_resize_width(&WindowId::new(1234567890).unwrap(), ResizeDir::Shrink, 5);
    assert_eq!(
        p,
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
fn format_criteria_clause_each_variant() {
    assert_eq!(
        format_criteria_clause(&PlacementCriteria::AppId("dev.assistd.popup".into())),
        r#"[app_id="dev.assistd.popup"]"#
    );
    assert_eq!(
        format_criteria_clause(&PlacementCriteria::Class("Firefox".into())),
        r#"[class="Firefox"]"#
    );
    assert_eq!(
        format_criteria_clause(&PlacementCriteria::Title("Inbox".into())),
        r#"[title="^Inbox$"]"#
    );
    assert_eq!(
        format_criteria_clause(&PlacementCriteria::ConId(WindowId::new(42).unwrap())),
        r#"[con_id="42"]"#
    );
}

#[test]
fn format_criteria_clause_escapes_quotes_in_string_variants() {
    assert_eq!(
        format_criteria_clause(&PlacementCriteria::AppId(r#"a"b"#.into())),
        r#"[app_id="a\"b"]"#
    );
    assert_eq!(
        format_criteria_clause(&PlacementCriteria::Class(r"a\b".into())),
        r#"[class="a\\b"]"#
    );
}

#[test]
fn place_floating_pixels_bottom_right_default_offsets() {
    // 1920×1055 workspace, 360×120 popup, BottomRight (-10, -30):
    // TL = (1920 - 360 - 10, 1055 - 120 - 30) = (1550, 905).
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
fn place_floating_pixels_top_right_default_offsets() {
    // TL = (1920 - 360 - 10, 10) = (1550, 10).
    let p = format_place_floating_pixels(
        &PlacementCriteria::Title("dev.assistd.popup".into()),
        anchor(AnchorCorner::TopRight, -10, 10),
        workspace_1920_1055(),
    );
    assert_eq!(
        p,
        concat!(
            r#"[title="^dev.assistd.popup$"] floating enable, "#,
            r#"[title="^dev.assistd.popup$"] resize set 360 120, "#,
            r#"[title="^dev.assistd.popup$"] move position 1550 px 10 px, "#,
            r#"[title="^dev.assistd.popup$"] sticky enable"#,
        )
    );
}

#[test]
fn place_floating_pixels_top_left_default_offsets() {
    let p = format_place_floating_pixels(
        &PlacementCriteria::Title("dev.assistd.popup".into()),
        anchor(AnchorCorner::TopLeft, 10, 10),
        workspace_1920_1055(),
    );
    assert_eq!(
        p,
        concat!(
            r#"[title="^dev.assistd.popup$"] floating enable, "#,
            r#"[title="^dev.assistd.popup$"] resize set 360 120, "#,
            r#"[title="^dev.assistd.popup$"] move position 10 px 10 px, "#,
            r#"[title="^dev.assistd.popup$"] sticky enable"#,
        )
    );
}

#[test]
fn place_floating_pixels_bottom_left_negative_offset_y() {
    // TL = (10, 1055 - 120 - 10) = (10, 925).
    let p = format_place_floating_pixels(
        &PlacementCriteria::Title("dev.assistd.popup".into()),
        anchor(AnchorCorner::BottomLeft, 10, -10),
        workspace_1920_1055(),
    );
    assert_eq!(
        p,
        concat!(
            r#"[title="^dev.assistd.popup$"] floating enable, "#,
            r#"[title="^dev.assistd.popup$"] resize set 360 120, "#,
            r#"[title="^dev.assistd.popup$"] move position 10 px 925 px, "#,
            r#"[title="^dev.assistd.popup$"] sticky enable"#,
        )
    );
}

#[test]
fn place_floating_pixels_center_default_offsets() {
    // TL = ((1920 - 360) / 2, (1055 - 120) / 2) = (780, 467).
    let p = format_place_floating_pixels(
        &PlacementCriteria::Title("dev.assistd.popup".into()),
        anchor(AnchorCorner::Center, 0, 0),
        workspace_1920_1055(),
    );
    assert_eq!(
        p,
        concat!(
            r#"[title="^dev.assistd.popup$"] floating enable, "#,
            r#"[title="^dev.assistd.popup$"] resize set 360 120, "#,
            r#"[title="^dev.assistd.popup$"] move position 780 px 467 px, "#,
            r#"[title="^dev.assistd.popup$"] sticky enable"#,
        )
    );
}

#[test]
fn place_floating_pixels_for_con_id_uses_con_id_criteria() {
    let p = format_place_floating_pixels(
        &PlacementCriteria::ConId(WindowId::new(1234).unwrap()),
        anchor(AnchorCorner::TopRight, -10, 10),
        workspace_1920_1055(),
    );
    assert!(p.starts_with(r#"[con_id="1234"] floating enable"#));
}

#[test]
fn place_floating_pixels_escapes_quotes_in_app_id() {
    let p = format_place_floating_pixels(
        &PlacementCriteria::AppId(r#"a"b"#.into()),
        anchor(AnchorCorner::Center, 0, 0),
        workspace_1920_1055(),
    );
    assert!(
        p.starts_with(r#"[app_id="a\"b"] floating enable"#),
        "got: {p}"
    );
}

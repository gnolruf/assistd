//! Helpers for formatting i3/Sway criteria strings.
//!
//! Both compositors accept the same `[key="value"] action` syntax, so
//! this module is shared between [`crate::i3`] and [`crate::sway`].
//! Backslashes and quotes inside a criteria value need escaping; numeric
//! workspace targets get the more-robust `workspace number N` form.

use crate::{AnchorCorner, PlacementAnchor, PlacementCriteria, Rect, WorkspaceId};

/// Escape `\` and `"` inside a value that lands between `[class="..."]`
/// or `workspace "..."`. Backslashes are escaped first so we don't
/// double-escape the slashes inserted in front of quotes.
pub fn escape_for_criteria(s: &str) -> String {
    s.replace('\\', r"\\").replace('"', r#"\""#)
}

/// Numeric workspace IDs become `workspace number N` (robust to renames);
/// named workspaces become `workspace "<escaped name>"`. The enum
/// variant carries the choice; no parse round-trip on every call.
pub fn format_workspace_target(ws: &WorkspaceId) -> String {
    match ws {
        WorkspaceId::Num(n) => format!("workspace number {n}"),
        WorkspaceId::Name(s) => format!(r#"workspace "{}""#, escape_for_criteria(s)),
    }
}

/// Render a [`PlacementCriteria`] into the `[key="value"]` prefix both
/// compositors expect. `con_id` skips the escape pass — it's a
/// `NonZeroU64` and can't carry quotes. `Title` is anchored with
/// `^…$` so the regex i3/sway run against `_NET_WM_NAME` matches
/// only the full title, not any substring of it.
pub fn format_criteria_clause(c: &PlacementCriteria) -> String {
    match c {
        PlacementCriteria::AppId(s) => format!(r#"[app_id="{}"]"#, escape_for_criteria(s)),
        PlacementCriteria::Class(s) => format!(r#"[class="{}"]"#, escape_for_criteria(s)),
        PlacementCriteria::Title(s) => format!(r#"[title="^{}$"]"#, escape_for_criteria(s)),
        PlacementCriteria::ConId(id) => format!(r#"[con_id="{}"]"#, id.get()),
    }
}

/// Build the `floating enable, resize set W H, move position X Y` payload
/// for the configured anchor, given the focused workspace's pixel rect.
///
/// We use an explicit pixel position (`move position {X}px {Y}px`)
/// rather than `move position 100 ppt 100 ppt, move left Wpx` because
/// i3 silently clamps off-screen ppt-based positions to keep some
/// portion of the window visible. With clamping in play, the
/// `move position` becomes a near-no-op and the chained `move left`
/// then walks the window leftward by `W` pixels on each placement —
/// the popup ends up further off-screen with every wake event.
///
/// Pixel coordinates are output-relative (no `absolute` keyword) so
/// multi-monitor setups continue to anchor the popup to whichever
/// output the focused workspace currently lives on.
pub fn format_place_floating_pixels(
    c: &PlacementCriteria,
    anchor: PlacementAnchor,
    workspace: Rect,
) -> String {
    let prefix = format_criteria_clause(c);
    let (x, y) = compute_target_position(anchor, workspace);
    format!(
        "{prefix} floating enable, {prefix} resize set {} {}, \
         {prefix} move position {} px {} px",
        anchor.width, anchor.height, x, y,
    )
}

/// Top-left corner of the window, in output-relative pixels, for the
/// given anchor + offsets on a workspace of the given size. Negative
/// values are allowed and let users intentionally push the popup
/// off-screen if they really want to — we don't second-guess the
/// configured offsets.
pub fn compute_target_position(anchor: PlacementAnchor, workspace: Rect) -> (i32, i32) {
    let w = anchor.width as i32;
    let h = anchor.height as i32;
    let ww = workspace.width as i32;
    let wh = workspace.height as i32;
    match anchor.corner {
        AnchorCorner::TopLeft => (anchor.offset_x, anchor.offset_y),
        AnchorCorner::TopRight => (ww - w + anchor.offset_x, anchor.offset_y),
        AnchorCorner::BottomLeft => (anchor.offset_x, wh - h + anchor.offset_y),
        AnchorCorner::BottomRight => (ww - w + anchor.offset_x, wh - h + anchor.offset_y),
        AnchorCorner::Center => (
            (ww - w) / 2 + anchor.offset_x,
            (wh - h) / 2 + anchor.offset_y,
        ),
    }
}

#[cfg(test)]
mod tests {
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
        // "1:web" doesn't parse as u32 → Named (this matches the old
        // alias because i3 itself treats it as a name).
        assert_eq!(
            "1:web".parse::<WorkspaceId>().unwrap(),
            WorkspaceId::Name("1:web".into())
        );
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
                r#"[title="^dev.assistd.popup$"] move position 1550 px 905 px"#,
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
                r#"[title="^dev.assistd.popup$"] move position 1550 px 10 px"#,
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
                r#"[title="^dev.assistd.popup$"] move position 10 px 10 px"#,
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
                r#"[title="^dev.assistd.popup$"] move position 10 px 925 px"#,
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
                r#"[title="^dev.assistd.popup$"] move position 780 px 467 px"#,
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
}

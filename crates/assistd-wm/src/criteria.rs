//! Helpers for formatting i3/Sway criteria strings.
//!
//! Both compositors accept the same `[key="value"] action` syntax, so
//! this module is shared between [`crate::i3`] and [`crate::sway`].
//! Backslashes and quotes inside a criteria value need escaping; numeric
//! workspace targets get the more-robust `workspace number N` form.

use crate::{AnchorCorner, PlacementAnchor, PlacementCriteria, WorkspaceId};

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
/// `NonZeroU64` and can't carry quotes.
pub fn format_criteria_clause(c: &PlacementCriteria) -> String {
    match c {
        PlacementCriteria::AppId(s) => format!(r#"[app_id="{}"]"#, escape_for_criteria(s)),
        PlacementCriteria::Class(s) => format!(r#"[class="{}"]"#, escape_for_criteria(s)),
        PlacementCriteria::ConId(id) => format!(r#"[con_id="{}"]"#, id.get()),
    }
}

/// Build the chained `floating enable, resize set W H, move position …`
/// payload for the configured anchor. The output is one IPC string that
/// i3 / sway both accept; we use `<X> ppt <Y> ppt` to anchor at the
/// focused output's corner without querying its dimensions, then
/// `move left/right/up/down <N>px` to apply the offsets (and to inset
/// the window's far edges back inside the output for right/bottom
/// anchors).
pub fn format_place_floating_payload(c: &PlacementCriteria, anchor: PlacementAnchor) -> String {
    let prefix = format_criteria_clause(c);
    let geometry = format!(
        "{prefix} floating enable, {prefix} resize set {} {}",
        anchor.width, anchor.height
    );
    let position = match anchor.corner {
        AnchorCorner::TopLeft => format_corner_position(
            &prefix,
            "0 ppt 0 ppt",
            anchor.offset_x,
            anchor.offset_y,
            0,
            0,
        ),
        AnchorCorner::TopRight => format_corner_position(
            &prefix,
            "100 ppt 0 ppt",
            anchor.offset_x,
            anchor.offset_y,
            -(anchor.width as i64),
            0,
        ),
        AnchorCorner::BottomLeft => format_corner_position(
            &prefix,
            "0 ppt 100 ppt",
            anchor.offset_x,
            anchor.offset_y,
            0,
            -(anchor.height as i64),
        ),
        AnchorCorner::BottomRight => format_corner_position(
            &prefix,
            "100 ppt 100 ppt",
            anchor.offset_x,
            anchor.offset_y,
            -(anchor.width as i64),
            -(anchor.height as i64),
        ),
        AnchorCorner::Center => format_center_position(&prefix, anchor.offset_x, anchor.offset_y),
    };
    if position.is_empty() {
        geometry
    } else {
        format!("{geometry}, {position}")
    }
}

/// Build the position clause for a non-centre anchor. `base_position`
/// is the `<X> ppt <Y> ppt` form for the corner. `intrinsic_dx`/
/// `intrinsic_dy` shift back inward by the window's own dimensions for
/// right/bottom anchors (zero for left/top); `offset_x`/`offset_y` add
/// the user-configured offset on top of that.
fn format_corner_position(
    prefix: &str,
    base_position: &str,
    offset_x: i32,
    offset_y: i32,
    intrinsic_dx: i64,
    intrinsic_dy: i64,
) -> String {
    let dx = intrinsic_dx + offset_x as i64;
    let dy = intrinsic_dy + offset_y as i64;
    let mut parts = Vec::with_capacity(3);
    parts.push(format!("{prefix} move position {base_position}"));
    if let Some(clause) = horizontal_move_clause(prefix, dx) {
        parts.push(clause);
    }
    if let Some(clause) = vertical_move_clause(prefix, dy) {
        parts.push(clause);
    }
    parts.join(", ")
}

fn format_center_position(prefix: &str, offset_x: i32, offset_y: i32) -> String {
    let mut parts = Vec::with_capacity(3);
    parts.push(format!("{prefix} move position center"));
    if let Some(clause) = horizontal_move_clause(prefix, offset_x as i64) {
        parts.push(clause);
    }
    if let Some(clause) = vertical_move_clause(prefix, offset_y as i64) {
        parts.push(clause);
    }
    parts.join(", ")
}

/// `move left|right Npx` clause for a signed pixel delta. Returns
/// `None` for zero — i3/sway both treat `move right 0px` as a no-op,
/// but the suppressed clause keeps the rendered string minimal and the
/// snapshot tests readable.
fn horizontal_move_clause(prefix: &str, dx: i64) -> Option<String> {
    match dx.cmp(&0) {
        std::cmp::Ordering::Greater => Some(format!("{prefix} move right {dx}px")),
        std::cmp::Ordering::Less => Some(format!("{prefix} move left {}px", -dx)),
        std::cmp::Ordering::Equal => None,
    }
}

fn vertical_move_clause(prefix: &str, dy: i64) -> Option<String> {
    match dy.cmp(&0) {
        std::cmp::Ordering::Greater => Some(format!("{prefix} move down {dy}px")),
        std::cmp::Ordering::Less => Some(format!("{prefix} move up {}px", -dy)),
        std::cmp::Ordering::Equal => None,
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
    fn place_floating_payload_top_right_default_offsets() {
        let p = format_place_floating_payload(
            &PlacementCriteria::AppId("dev.assistd.popup".into()),
            anchor(AnchorCorner::TopRight, -10, 10),
        );
        // Default tray-popup geometry: 360×120 at top-right, inset 10px
        // from the right edge, 10px from the top. Window width pulls
        // top-left 370px to the left of the right edge; offset_y of 10
        // pushes it 10px down from the top.
        assert_eq!(
            p,
            concat!(
                r#"[app_id="dev.assistd.popup"] floating enable, "#,
                r#"[app_id="dev.assistd.popup"] resize set 360 120, "#,
                r#"[app_id="dev.assistd.popup"] move position 100 ppt 0 ppt, "#,
                r#"[app_id="dev.assistd.popup"] move left 370px, "#,
                r#"[app_id="dev.assistd.popup"] move down 10px"#,
            )
        );
    }

    #[test]
    fn place_floating_payload_top_left_default_offsets() {
        let p = format_place_floating_payload(
            &PlacementCriteria::AppId("dev.assistd.popup".into()),
            anchor(AnchorCorner::TopLeft, 10, 10),
        );
        assert_eq!(
            p,
            concat!(
                r#"[app_id="dev.assistd.popup"] floating enable, "#,
                r#"[app_id="dev.assistd.popup"] resize set 360 120, "#,
                r#"[app_id="dev.assistd.popup"] move position 0 ppt 0 ppt, "#,
                r#"[app_id="dev.assistd.popup"] move right 10px, "#,
                r#"[app_id="dev.assistd.popup"] move down 10px"#,
            )
        );
    }

    #[test]
    fn place_floating_payload_bottom_left_negative_offset_y() {
        let p = format_place_floating_payload(
            &PlacementCriteria::AppId("dev.assistd.popup".into()),
            anchor(AnchorCorner::BottomLeft, 10, -10),
        );
        // BottomLeft: intrinsic_dy = -height = -120; offset_y = -10
        // → dy = -130 → move up 130px.
        assert_eq!(
            p,
            concat!(
                r#"[app_id="dev.assistd.popup"] floating enable, "#,
                r#"[app_id="dev.assistd.popup"] resize set 360 120, "#,
                r#"[app_id="dev.assistd.popup"] move position 0 ppt 100 ppt, "#,
                r#"[app_id="dev.assistd.popup"] move right 10px, "#,
                r#"[app_id="dev.assistd.popup"] move up 130px"#,
            )
        );
    }

    #[test]
    fn place_floating_payload_bottom_right_default_offsets() {
        let p = format_place_floating_payload(
            &PlacementCriteria::AppId("dev.assistd.popup".into()),
            anchor(AnchorCorner::BottomRight, -10, -10),
        );
        // BottomRight: intrinsic_dx = -360, offset_x = -10 → -370 → move left 370px.
        // intrinsic_dy = -120, offset_y = -10 → -130 → move up 130px.
        assert_eq!(
            p,
            concat!(
                r#"[app_id="dev.assistd.popup"] floating enable, "#,
                r#"[app_id="dev.assistd.popup"] resize set 360 120, "#,
                r#"[app_id="dev.assistd.popup"] move position 100 ppt 100 ppt, "#,
                r#"[app_id="dev.assistd.popup"] move left 370px, "#,
                r#"[app_id="dev.assistd.popup"] move up 130px"#,
            )
        );
    }

    #[test]
    fn place_floating_payload_zero_offsets_omit_move_clauses() {
        // TopLeft with both offsets at 0 collapses to just the position
        // clause — no redundant "move right 0px".
        let p = format_place_floating_payload(
            &PlacementCriteria::AppId("x".into()),
            anchor(AnchorCorner::TopLeft, 0, 0),
        );
        assert_eq!(
            p,
            concat!(
                r#"[app_id="x"] floating enable, "#,
                r#"[app_id="x"] resize set 360 120, "#,
                r#"[app_id="x"] move position 0 ppt 0 ppt"#,
            )
        );
    }

    #[test]
    fn place_floating_payload_center_with_offsets() {
        let p = format_place_floating_payload(
            &PlacementCriteria::AppId("dev.assistd.popup".into()),
            anchor(AnchorCorner::Center, 0, 0),
        );
        assert_eq!(
            p,
            concat!(
                r#"[app_id="dev.assistd.popup"] floating enable, "#,
                r#"[app_id="dev.assistd.popup"] resize set 360 120, "#,
                r#"[app_id="dev.assistd.popup"] move position center"#,
            )
        );
        let p2 = format_place_floating_payload(
            &PlacementCriteria::AppId("dev.assistd.popup".into()),
            anchor(AnchorCorner::Center, 20, -20),
        );
        assert_eq!(
            p2,
            concat!(
                r#"[app_id="dev.assistd.popup"] floating enable, "#,
                r#"[app_id="dev.assistd.popup"] resize set 360 120, "#,
                r#"[app_id="dev.assistd.popup"] move position center, "#,
                r#"[app_id="dev.assistd.popup"] move right 20px, "#,
                r#"[app_id="dev.assistd.popup"] move up 20px"#,
            )
        );
    }

    #[test]
    fn place_floating_payload_for_con_id_uses_con_id_criteria() {
        let p = format_place_floating_payload(
            &PlacementCriteria::ConId(WindowId::new(1234).unwrap()),
            anchor(AnchorCorner::TopRight, -10, 10),
        );
        assert!(p.starts_with(r#"[con_id="1234"] floating enable"#));
    }

    #[test]
    fn place_floating_payload_escapes_quotes_in_app_id() {
        let p = format_place_floating_payload(
            &PlacementCriteria::AppId(r#"a"b"#.into()),
            anchor(AnchorCorner::Center, 0, 0),
        );
        assert!(
            p.starts_with(r#"[app_id="a\"b"] floating enable"#),
            "got: {p}"
        );
    }
}

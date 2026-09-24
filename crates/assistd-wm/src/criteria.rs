//! Formatting for the `[key="value"] action` command syntax that i3
//! and Sway share.

use crate::{
    AnchorCorner, Layout, PlacementAnchor, PlacementCriteria, Rect, ResizeDir, WindowId,
    WorkspaceId,
};

/// Escape `\` and `"` inside a quoted criteria value. Backslashes go
/// first so the ones inserted before quotes aren't doubled.
pub fn escape_for_criteria(s: &str) -> String {
    s.replace('\\', r"\\").replace('"', r#"\""#)
}

/// `workspace number N` for numeric ids, `workspace "<name>"` otherwise.
pub fn format_workspace_target(ws: &WorkspaceId) -> String {
    match ws {
        WorkspaceId::Num(n) => format!("workspace number {n}"),
        WorkspaceId::Name(s) => format!(r#"workspace "{}""#, escape_for_criteria(s)),
    }
}

/// Focuses `window`.
pub fn format_focus(window: &WindowId) -> String {
    format!(r#"[con_id="{}"] focus"#, window.get())
}

/// Moves `window`'s container to `workspace`.
pub fn format_move_to_workspace(window: &WindowId, workspace: &WorkspaceId) -> String {
    format!(
        r#"[con_id="{}"] move container to {}"#,
        window.get(),
        format_workspace_target(workspace)
    )
}

/// Grows or shrinks `window`'s width by `pixels`.
pub fn format_resize_width(window: &WindowId, direction: ResizeDir, pixels: u32) -> String {
    format!(
        r#"[con_id="{}"] resize {} width {} px or 0 ppt"#,
        window.get(),
        direction.as_str(),
        pixels,
    )
}

/// Acts on the focused container.
pub fn format_layout(layout: Layout) -> String {
    format!("layout {}", layout.as_str())
}

/// The `[key="value"]` prefix for a [`PlacementCriteria`]. `Title` is
/// anchored with `^…$` because the compositor treats it as a regex.
pub fn format_criteria_clause(c: &PlacementCriteria) -> String {
    match c {
        PlacementCriteria::AppId(s) => format!(r#"[app_id="{}"]"#, escape_for_criteria(s)),
        PlacementCriteria::Class(s) => format!(r#"[class="{}"]"#, escape_for_criteria(s)),
        PlacementCriteria::Title(s) => format!(r#"[title="^{}$"]"#, escape_for_criteria(s)),
        PlacementCriteria::ConId(id) => format!(r#"[con_id="{}"]"#, id.get()),
    }
}

/// The `floating enable, resize, move position, sticky enable` payload.
/// Positions are absolute pixels because i3's ppt-based positioning
/// silently clamps off-screen values.
pub fn format_place_floating_pixels(
    c: &PlacementCriteria,
    anchor: PlacementAnchor,
    workspace: Rect,
) -> String {
    let prefix = format_criteria_clause(c);
    let (x, y) = compute_target_position(anchor, workspace);
    format!(
        "{prefix} floating enable, {prefix} resize set {} {}, \
         {prefix} move position {} px {} px, {prefix} sticky enable",
        anchor.width, anchor.height, x, y,
    )
}

/// Top-left corner of the window in output-relative pixels. Negative
/// results are allowed.
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
mod tests;

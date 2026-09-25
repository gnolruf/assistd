//! Formatting for the `[key="value"] action` command syntax that i3
//! and Sway share.

use crate::{
    AnchorCorner, Layout, PlacementAnchor, PlacementCriteria, Rect, ResizeDir, WindowId,
    WorkspaceId,
};

/// Escape `\` and `"` inside a quoted criteria value. Backslashes go
/// first so the ones inserted before quotes aren't doubled.
pub fn escape_for_criteria(value: &str) -> String {
    value.replace('\\', r"\\").replace('"', r#"\""#)
}

/// `workspace number N` for numeric ids, `workspace "<name>"` otherwise.
pub fn format_workspace_target(workspace: &WorkspaceId) -> String {
    match workspace {
        WorkspaceId::Num(n) => format!("workspace number {n}"),
        WorkspaceId::Name(name) => format!(r#"workspace "{}""#, escape_for_criteria(name)),
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
pub fn format_criteria_clause(criteria: &PlacementCriteria) -> String {
    match criteria {
        PlacementCriteria::AppId(app_id) => {
            format!(r#"[app_id="{}"]"#, escape_for_criteria(app_id))
        }
        PlacementCriteria::Class(class) => format!(r#"[class="{}"]"#, escape_for_criteria(class)),
        PlacementCriteria::Title(title) => {
            format!(r#"[title="^{}$"]"#, escape_for_criteria(title))
        }
        PlacementCriteria::ConId(id) => format!(r#"[con_id="{}"]"#, id.get()),
    }
}

/// The `floating enable, resize, move position, sticky enable` payload.
/// Positions are absolute pixels because i3's ppt-based positioning
/// silently clamps off-screen values.
pub fn format_place_floating_pixels(
    criteria: &PlacementCriteria,
    anchor: PlacementAnchor,
    workspace: Rect,
) -> String {
    let prefix = format_criteria_clause(criteria);
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
    let free_width = workspace.width as i32 - anchor.width as i32;
    let free_height = workspace.height as i32 - anchor.height as i32;
    match anchor.corner {
        AnchorCorner::TopLeft => (anchor.offset_x, anchor.offset_y),
        AnchorCorner::TopRight => (free_width + anchor.offset_x, anchor.offset_y),
        AnchorCorner::BottomLeft => (anchor.offset_x, free_height + anchor.offset_y),
        AnchorCorner::BottomRight => (free_width + anchor.offset_x, free_height + anchor.offset_y),
        AnchorCorner::Center => (
            free_width / 2 + anchor.offset_x,
            free_height / 2 + anchor.offset_y,
        ),
    }
}

#[cfg(test)]
mod tests;

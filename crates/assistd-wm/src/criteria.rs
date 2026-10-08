//! Formatting for the `[key="value"] action` command syntax that i3
//! and Sway share.

use crate::{Layout, ResizeDir, WindowId, WorkspaceId};

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

#[cfg(test)]
mod tests;

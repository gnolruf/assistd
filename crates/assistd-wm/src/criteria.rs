//! Formatting for the `[key="value"] action` command syntax that i3
//! and Sway share.

use crate::{ExecCommand, Layout, ResizeDir, WindowId, WorkspaceId};

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

/// Runs `command` with the compositor's shell, after switching to
/// `workspace` when given.
pub fn format_exec(workspace: Option<&WorkspaceId>, command: &ExecCommand) -> String {
    let exec = format!(r#"exec "{}""#, command.as_str());
    match workspace {
        Some(workspace) => format!("{}; {exec}", format_workspace_target(workspace)),
        None => exec,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnquotableCommand;

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
    fn exec_payloads_keep_the_command_inside_one_quoted_string() {
        let command =
            ExecCommand::new("cd '/a;b' && exec nvim 'it'\\''s'".into()).expect("quotable");
        assert_eq!(
            format_exec(Some(&WorkspaceId::Num(2)), &command),
            r#"workspace number 2; exec "cd '/a;b' && exec nvim 'it'\''s'""#
        );
        for unquotable in ["nvim \"; kill", "nvim\n; kill", "nvim\r"] {
            assert_eq!(
                ExecCommand::new(unquotable.into()),
                Err(UnquotableCommand),
                "{unquotable:?}"
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
}

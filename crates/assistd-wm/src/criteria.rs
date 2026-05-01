//! Helpers for formatting i3/Sway criteria strings.
//!
//! Both compositors accept the same `[key="value"] action` syntax, so
//! this module is shared between [`crate::i3`] and [`crate::sway`].
//! Backslashes and quotes inside a criteria value need escaping; numeric
//! workspace targets get the more-robust `workspace number N` form.

use crate::WorkspaceId;

/// Escape `\` and `"` inside a value that lands between `[class="..."]`
/// or `workspace "..."`. Backslashes are escaped first so we don't
/// double-escape the slashes inserted in front of quotes.
pub fn escape_for_criteria(s: &str) -> String {
    s.replace('\\', r"\\").replace('"', r#"\""#)
}

/// Numeric workspace IDs become `workspace number N` (robust to renames);
/// non-numeric become `workspace "<escaped name>"`.
pub fn format_workspace_target(ws: &WorkspaceId) -> String {
    if let Ok(n) = ws.parse::<u32>() {
        format!("workspace number {n}")
    } else {
        format!(r#"workspace "{}""#, escape_for_criteria(ws))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_for_criteria_handles_quotes_and_backslashes() {
        assert_eq!(escape_for_criteria("Firefox"), "Firefox");
        assert_eq!(escape_for_criteria(r#"a"b"#), r#"a\"b"#);
        assert_eq!(escape_for_criteria(r"a\b"), r"a\\b");
        // Order matters: backslash inserted by quote-escape must NOT be
        // re-escaped. `a"b` → `a\"b` (single inserted slash, single quote).
        assert_eq!(escape_for_criteria(r#"a"b\c"#), r#"a\"b\\c"#);
    }

    #[test]
    fn format_workspace_target_numeric() {
        assert_eq!(format_workspace_target(&"3".into()), "workspace number 3");
        assert_eq!(format_workspace_target(&"10".into()), "workspace number 10");
    }

    #[test]
    fn format_workspace_target_named() {
        assert_eq!(
            format_workspace_target(&"scratch".into()),
            r#"workspace "scratch""#
        );
    }

    #[test]
    fn format_workspace_target_named_with_quote_is_escaped() {
        assert_eq!(
            format_workspace_target(&r#"weird"name"#.into()),
            r#"workspace "weird\"name""#
        );
    }

    #[test]
    fn format_workspace_target_u32_parseable_is_numeric() {
        // Any string that parses as `u32` becomes `workspace number N`,
        // including zero-padded variants — `"03"` → `workspace number 3`.
        // i3 normalizes the number, so the semantics match.
        assert_eq!(format_workspace_target(&"03".into()), "workspace number 3");
    }
}

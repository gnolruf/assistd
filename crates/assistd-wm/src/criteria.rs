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
/// named workspaces become `workspace "<escaped name>"`. The enum
/// variant carries the choice; no parse round-trip on every call.
pub fn format_workspace_target(ws: &WorkspaceId) -> String {
    match ws {
        WorkspaceId::Num(n) => format!("workspace number {n}"),
        WorkspaceId::Name(s) => format!(r#"workspace "{}""#, escape_for_criteria(s)),
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
}

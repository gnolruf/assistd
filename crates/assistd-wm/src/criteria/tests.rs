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
    let command = ExecCommand::new("cd '/a;b' && exec nvim 'it'\\''s'".into()).expect("quotable");
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

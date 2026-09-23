use super::*;

struct Stub(&'static str);

#[async_trait]
impl Command for Stub {
    fn name(&self) -> &str {
        self.0
    }
    fn summary(&self) -> &'static str {
        "stub command for tests"
    }
    fn help(&self) -> String {
        "stub help".to_string()
    }
    async fn run(&self, _input: CommandInput) -> CommandOutput {
        CommandOutput::ok(Vec::new())
    }
}

#[test]
fn unmatched_glob_reads_as_a_glob_not_a_missing_file() {
    let e = std::io::Error::from(std::io::ErrorKind::NotFound);
    assert_eq!(
        io_error_nav("ls", "/tmp/*.db-shm", &e),
        "[error] ls: no file matches /tmp/*.db-shm. Try: ls /tmp to see what is there\n"
    );
    assert_eq!(
        io_error_nav("ls", "/tmp/notes.txt", &e),
        "[error] ls: file not found: /tmp/notes.txt. Use: ls /tmp to see what is there\n"
    );
}

#[test]
fn path_through_a_file_points_at_the_offending_parent() {
    let e = std::io::Error::from(std::io::ErrorKind::NotADirectory);
    assert_eq!(
        io_error_nav("cat", "notes.txt/sub", &e),
        "[error] cat: notes.txt/sub: a parent component is not a directory. Check: ls notes.txt\n"
    );
}

#[test]
fn parent_dir_stops_at_the_first_metacharacter() {
    assert_eq!(parent_dir("/tmp/*.db-shm"), "/tmp");
    assert_eq!(parent_dir("docs/*.md"), "docs");
    assert_eq!(parent_dir("docs/*/"), "docs");
    assert_eq!(parent_dir("/a*/b*"), "/");
    assert_eq!(parent_dir("*.rs"), ".");
}

#[test]
fn parent_dir_of_a_plain_path_ignores_trailing_slashes() {
    assert_eq!(parent_dir("/tmp/notes.txt"), "/tmp");
    assert_eq!(parent_dir("/tmp/missing/"), "/tmp");
    assert_eq!(parent_dir("/notes.txt"), "/");
    assert_eq!(parent_dir("notes.txt"), ".");
}

#[test]
fn registry_resolves_by_name_and_lists_alphabetically() {
    let mut reg = CommandRegistry::new();
    reg.register(Stub("grep"));
    reg.register(Stub("cat"));
    reg.register(Stub("ls"));
    assert_eq!(reg.get("grep").map(|c| c.name()), Some("grep"));
    assert!(reg.get("nope").is_none());
    assert_eq!(reg.sorted_names(), ["cat", "grep", "ls"]);
    let summary = "stub command for tests";
    assert_eq!(
        reg.sorted_summaries(),
        [("cat", summary), ("grep", summary), ("ls", summary)]
    );
}

/// Acceptance for the error-message-as-navigation convention: every
/// registered command, when driven into a failure path, emits stderr
/// containing `[error] ` AND a hint word (`Use:` / `Try:` / `Check:` /
/// `Available:`). This is the gate that catches new commands added
/// without a convention-compliant error path.
///
/// Drives one case per command with an input that's guaranteed to
/// fail. `echo` has no failure mode and is skipped. Permission-denied
/// assertions live in platform-gated per-command tests because
/// creating an unreadable file portably is brittle.
#[test]
fn every_registered_command_emits_convention_compliant_error() {
    use crate::commands::{
        BashCommand, CatCommand, GrepCommand, HeadCommand, LsCommand, ScreenshotCommand,
        SeeCommand, SortCommand, TailCommand, UniqCommand, WcCommand, WebCommand, WmCommand,
        WriteCommand,
    };
    use assistd_wm::NoWindowManager;
    use std::sync::Arc as StdArc;

    fn contains_hint(s: &str) -> bool {
        s.contains("Use:") || s.contains("Try:") || s.contains("Check:") || s.contains("Available:")
    }

    async fn run_cmd<C: Command>(cmd: C, args: Vec<String>) -> CommandOutput {
        cmd.run(CommandInput { args, stdin: None }).await
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let cases: Vec<(&str, CommandOutput)> = vec![
        (
            "cat",
            rt.block_on(run_cmd(
                CatCommand,
                vec!["/nonexistent/assistd-convention-test".into()],
            )),
        ),
        (
            "ls",
            rt.block_on(run_cmd(
                LsCommand,
                vec!["/nonexistent/assistd-convention-test".into()],
            )),
        ),
        (
            "see",
            rt.block_on(run_cmd(
                SeeCommand::default(),
                vec!["/nonexistent/assistd-convention-test.png".into()],
            )),
        ),
        (
            "screenshot",
            rt.block_on(run_cmd(
                ScreenshotCommand::default(),
                vec!["--bogus-flag".into()],
            )),
        ),
        (
            "grep",
            rt.block_on(run_cmd(GrepCommand, vec!["-x".into(), "pat".into()])),
        ),
        (
            "write",
            rt.block_on(run_cmd(
                WriteCommand::permissive_for_tests(),
                vec!["/nonexistent/assistd-convention-test".into(), "x".into()],
            )),
        ),
        ("wc", rt.block_on(run_cmd(WcCommand, vec!["-q".into()]))),
        (
            "head",
            rt.block_on(run_cmd(HeadCommand, vec!["-n".into(), "lots".into()])),
        ),
        (
            "tail",
            rt.block_on(run_cmd(TailCommand, vec!["notes.md".into()])),
        ),
        ("sort", rt.block_on(run_cmd(SortCommand, vec!["-q".into()]))),
        (
            "uniq",
            rt.block_on(run_cmd(UniqCommand, vec!["notes.md".into()])),
        ),
        (
            "web",
            rt.block_on(run_cmd(
                WebCommand::new(),
                vec!["file:///etc/passwd".into()],
            )),
        ),
        // The timeout path deliberately emits a fixed, hint-free
        // line and is not covered here.
        (
            "bash",
            rt.block_on(run_cmd(
                {
                    use crate::policy::BashPolicyCfg;
                    use crate::policy::{AlwaysAllowGate, SandboxInfo};
                    use std::sync::Arc;
                    BashCommand::new(
                        Arc::new(BashPolicyCfg {
                            denylist: vec!["rm -rf /".into()],
                            ..Default::default()
                        }),
                        SandboxInfo::none(),
                        Arc::new(AlwaysAllowGate),
                    )
                },
                vec!["rm -rf /".into()],
            )),
        ),
        // wm against the disconnected NoWindowManager exercises the
        // mock-mode short-circuit; every subcommand returns the
        // same `[error] wm: compositor not connected. Check: …`
        // line, so any subcommand argv works as the failure driver.
        (
            "wm",
            rt.block_on(run_cmd(
                WmCommand::for_test(StdArc::new(NoWindowManager)),
                vec!["focus".into(), "Firefox".into()],
            )),
        ),
    ];

    for (name, out) in cases {
        assert_ne!(
            out.exit_code, 0,
            "{name}: failure input should exit non-zero"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(&format!("[error] {name}: ")),
            "{name}: stderr should say `[error] {name}: `, got {stderr:?}"
        );
        assert!(
            contains_hint(&stderr),
            "{name}: stderr missing recovery hint (Use:/Try:/Check:/Available:), got {stderr:?}"
        );
    }
}

/// Every registered production command has non-empty `help()` and
/// `summary()` text, with the summary inside the 80-char budget the
/// tool schema shows the model.
#[test]
fn every_registered_command_has_nonempty_help_and_summary() {
    let reg = crate::commands::test_registry();
    assert_eq!(reg.len(), 15);
    for (name, summary) in reg.sorted_summaries() {
        assert!(!summary.is_empty(), "{name} has empty summary");
        assert!(
            summary.len() <= 80,
            "{name} summary is {} chars (>80): {summary:?}",
            summary.len()
        );
        let help = reg.get(name).expect("command registered").help();
        assert!(
            help.contains("usage:"),
            "{name} help should contain `usage:` line, got {help:?}"
        );
    }
}

use std::sync::Arc;

use assistd_wm::NoWindowManager;

use super::*;
use crate::commands::{
    BashCommand, CatCommand, LsCommand, WebCommand, WmCommand, permissive_write, test_screenshot,
    test_see, test_wm,
};
use crate::policy::{
    AlwaysAllowGate, ApprovalGate, Approvals, BashPolicyCfg, DenyAllGate, SandboxInfo,
};

#[test]
fn unmatched_glob_reads_as_a_glob_not_a_missing_file() {
    let e = io::Error::from(ErrorKind::NotFound);
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
fn parent_dir_stops_at_the_first_metacharacter() {
    assert_eq!(parent_dir("/tmp/*.db-shm"), "/tmp");
    assert_eq!(parent_dir("docs/*.md"), "docs");
    assert_eq!(parent_dir("docs/*/"), "docs");
    assert_eq!(parent_dir("/a*/b*"), "/");
    assert_eq!(parent_dir("*.rs"), ".");
}

async fn run_cmd<C: Command>(cmd: C, args: Vec<String>) -> CommandOutput {
    cmd.run(CommandInput {
        args,
        stdin: None,
        scope: FileScope::Host,
        turn: TurnContext::default(),
    })
    .await
}

fn contains_hint(s: &str) -> bool {
    s.contains("Use:") || s.contains("Try:") || s.contains("Check:") || s.contains("Available:")
}

fn bash_declining_every_confirmation() -> BashCommand {
    BashCommand::new(
        Arc::new(BashPolicyCfg::default()),
        SandboxInfo::none(),
        Arc::new(DenyAllGate),
    )
}

fn wm_without_compositor() -> WmCommand {
    test_wm(Arc::new(NoWindowManager))
}

/// One failing invocation per command that writes its own error lines (`echo` cannot fail,
/// and the text programs print GNU's), paired with the command name its line must carry.
async fn failing_invocations() -> Vec<(&'static str, CommandOutput)> {
    let missing = || vec!["/nonexistent/assistd-convention-test".to_string()];
    vec![
        ("cat", run_cmd(CatCommand, missing()).await),
        ("ls", run_cmd(LsCommand, missing()).await),
        (
            "see",
            run_cmd(
                test_see(),
                vec!["/nonexistent/assistd-convention-test.png".into()],
            )
            .await,
        ),
        (
            "screenshot",
            run_cmd(test_screenshot(), vec!["--bogus-flag".into()]).await,
        ),
        (
            "write",
            run_cmd(
                permissive_write(),
                vec!["/nonexistent/assistd-convention-test".into(), "x".into()],
            )
            .await,
        ),
        (
            "web",
            run_cmd(
                WebCommand::new(
                    ApprovalGate::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved())),
                    None,
                ),
                vec!["fetch".into(), "file:///etc/passwd".into()],
            )
            .await,
        ),
        (
            "bash",
            run_cmd(
                bash_declining_every_confirmation(),
                vec!["touch /nonexistent/assistd-convention-test".into()],
            )
            .await,
        ),
        (
            "wm",
            run_cmd(
                wm_without_compositor(),
                vec!["focus".into(), "Firefox".into()],
            )
            .await,
        ),
    ]
}

#[test]
fn every_registered_command_emits_convention_compliant_error() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    for (name, out) in rt.block_on(failing_invocations()) {
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

#[test]
fn every_registered_command_has_nonempty_help_and_summary() {
    let reg = crate::commands::test_registry();
    assert_eq!(reg.len(), 15);
    for (name, synopsis, summary) in reg.sorted_listings() {
        assert!(!summary.is_empty(), "{name} has empty summary");
        assert!(
            summary.len() <= 80,
            "{name} summary is {} chars (>80): {summary:?}",
            summary.len()
        );
        let help = reg.get(name).expect("command registered").help();
        assert!(
            help.starts_with(&format!("usage: {name}")),
            "{name} help should open with a `usage: {name}` line, got {help:?}"
        );
        assert!(!synopsis.is_empty(), "{name} has empty synopsis");
    }
}

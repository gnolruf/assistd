//! Internal command abstraction. `Command`s are Rust handlers that the
//! chain executor dispatches to after the parser has converted a command
//! line into a [`crate::chain::Chain`] AST. They operate on raw bytes
//! (stdin → stdout) and surface a Unix-style `exit_code` so `&&`/`||`
//! composition works the way users expect.
//!
//! A `Command` is distinct from a [`crate::Tool`]: the LLM only ever sees
//! one `Tool` (`run`); the `CommandRegistry` is an internal lookup table
//! that `run` consults as it walks each chain stage.
//!
//! # Error-message-as-navigation convention
//!
//! Every stderr line a command emits must carry both *what went wrong* and
//! *what to do instead*, so the LLM recovers in one step instead of blind
//! retries. The format is:
//!
//! ```text
//! [error] <cmd>: <what-went-wrong>. <Hint>: <recovery>\n
//! ```
//!
//! - `<cmd>` — the command name (`cat`, `see`, `bash`, …) or a pseudo-tag
//!   for pre-dispatch failures (`parse`, `pipe`, `unknown command`).
//! - `<Hint>` — one of `Use:`, `Try:`, `Check:`, `Available:` — the first
//!   two phrase actionable alternatives; the latter two phrase diagnostics.
//! - `<recovery>` — either a concrete `run`-executable command the LLM can
//!   issue verbatim (e.g. `see photo.png`, `ls /dir`, `cat -b file.bin`)
//!   or a short check instruction (`ls -l <path>`).
//!
//! Use [`error_line`] to build a line, or [`io_error_nav`] to classify a
//! `std::io::Error` against the path that produced it. Never return a bare
//! non-zero `exit_code` without context — if a subprocess or downstream
//! library emitted stderr, forward it so the LLM can see *why*.

use anyhow::Result;
use async_trait::async_trait;

/// Format a single stderr line conforming to the error-navigation
/// convention. Emits exactly `[error] <cmd>: <what>. <hint>: <recovery>\n`.
///
/// `hint` is the label without the trailing colon (e.g. `"Use"`, `"Try"`,
/// `"Check"`, `"Available"`) — the colon and space are added for you.
pub fn error_line(
    cmd: &str,
    what: impl std::fmt::Display,
    hint: &str,
    recovery: impl std::fmt::Display,
) -> String {
    format!("[error] {cmd}: {what}. {hint}: {recovery}\n")
}

/// Classify a `std::io::Error` against the `path` that produced it and
/// emit a convention-compliant stderr line. Used by every file-touching
/// command so NotFound and PermissionDenied get uniform navigation hints.
pub fn io_error_nav(cmd: &str, path: &str, e: &std::io::Error) -> String {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::NotFound => error_line(
            cmd,
            format_args!("file not found: {path}"),
            "Use",
            "ls to check the path",
        ),
        ErrorKind::PermissionDenied => error_line(
            cmd,
            format_args!("permission denied: {path}"),
            "Check",
            format_args!("ls -l {path}"),
        ),
        _ => error_line(
            cmd,
            format_args!("{path}: {e}"),
            "Try",
            "a different path or check with ls",
        ),
    }
}

/// Input to a single chain stage.
pub struct CommandInput {
    /// Positional arguments **after** argv[0]. The command's own name
    /// is not included here — the registry has already resolved it.
    pub args: Vec<String>,
    /// Bytes piped in from the previous chain stage (or empty for the
    /// first command in a pipeline).
    pub stdin: Vec<u8>,
}

/// A side-channel payload a command can attach alongside its stdout. The
/// chain executor threads attachments through pipes untouched so `see X |
/// wc` still surfaces the image; `RunTool` base64-encodes them into the
/// JSON result so a caller (eventually the chat loop) can inject them
/// into the model's next turn as a vision input.
#[derive(Debug, Clone)]
pub enum Attachment {
    Image { mime: String, bytes: Vec<u8> },
}

/// Output of a single chain stage.
///
/// Returning `CommandOutput` inside `Result::Ok` (rather than surfacing
/// predictable failures as `Err`) is what lets `|| echo 'not found'`
/// catch a missing file: the `cat` handler reports `exit_code = 1` with
/// a friendly stderr, and the executor treats that as a triggerable
/// failure for `||`. `Err` is reserved for *catastrophic* failures
/// (spawn error, panic-in-trait, etc.) that should abort the whole
/// chain.
#[derive(Debug, Default, Clone)]
pub struct CommandOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: i32,
    pub attachments: Vec<Attachment>,
}

impl CommandOutput {
    pub fn ok(stdout: Vec<u8>) -> Self {
        Self {
            stdout,
            stderr: Vec::new(),
            exit_code: 0,
            attachments: Vec::new(),
        }
    }

    pub fn failed(exit_code: i32, stderr: impl Into<Vec<u8>>) -> Self {
        Self {
            stdout: Vec::new(),
            stderr: stderr.into(),
            exit_code,
            attachments: Vec::new(),
        }
    }
}

/// A single internal command (`cat`, `grep`, `bash`, …).
///
/// The `summary` / `help` split backs the progressive `--help` discovery
/// system: `summary` is a ≤80-char one-liner that [`CommandRegistry`]
/// aggregates for the `run` tool's Level-0 description (the list the LLM
/// sees in its tool schema); `help` is the full usage block a command
/// emits when invoked with insufficient arguments (Level-1). Commands
/// with subcommands can return subcommand-specific help from within their
/// own `run` body (Level-2).
#[async_trait]
pub trait Command: Send + Sync + 'static {
    fn name(&self) -> &str;
    /// One-line advertisement (≤80 chars, no trailing newline). Used to
    /// build the `run` tool's Level-0 description. Convention: terse verb
    /// phrase, e.g. `"filter lines matching a pattern (supports -i, -v, -c)"`.
    fn summary(&self) -> &'static str;
    /// Full usage block. Emitted verbatim on stdout when the command is
    /// called with insufficient arguments. Convention: first line begins
    /// with `usage: <name> …` so the LLM can visually disambiguate help
    /// output from a real `[<name>]\terror: …` failure.
    fn help(&self) -> String;
    async fn run(&self, input: CommandInput) -> Result<CommandOutput>;
}

/// Lookup table of registered commands.
#[derive(Default)]
pub struct CommandRegistry {
    commands: Vec<Box<dyn Command>>,
}

impl CommandRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<C: Command>(&mut self, cmd: C) {
        self.commands.push(Box::new(cmd));
    }

    pub fn get(&self, name: &str) -> Option<&dyn Command> {
        self.commands
            .iter()
            .find(|c| c.name() == name)
            .map(|c| c.as_ref())
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Command names, sorted alphabetically. Used to format the
    /// "Available: …" line in the unknown-command error so there's a
    /// single source of truth.
    pub fn sorted_names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.commands.iter().map(|c| c.name()).collect();
        v.sort_unstable();
        v
    }

    /// `(name, summary)` pairs, sorted alphabetically by name. Consumed
    /// by `RunTool::new` to build the dynamic Level-0 description the
    /// LLM sees in its tool schema.
    pub fn sorted_summaries(&self) -> Vec<(&str, &'static str)> {
        let mut v: Vec<(&str, &'static str)> = self
            .commands
            .iter()
            .map(|c| (c.name(), c.summary()))
            .collect();
        v.sort_unstable_by_key(|(n, _)| *n);
        v
    }
}

#[cfg(test)]
mod tests {
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
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput> {
            Ok(CommandOutput::ok(Vec::new()))
        }
    }

    #[test]
    fn sorted_names_is_alphabetical() {
        let mut reg = CommandRegistry::new();
        reg.register(Stub("grep"));
        reg.register(Stub("cat"));
        reg.register(Stub("ls"));
        assert_eq!(reg.sorted_names(), vec!["cat", "grep", "ls"]);
    }

    #[test]
    fn sorted_summaries_is_alphabetical_and_paired() {
        let mut reg = CommandRegistry::new();
        reg.register(Stub("grep"));
        reg.register(Stub("cat"));
        reg.register(Stub("ls"));
        let pairs = reg.sorted_summaries();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0, "cat");
        assert_eq!(pairs[1].0, "grep");
        assert_eq!(pairs[2].0, "ls");
        // Every pair carries a non-empty summary.
        for (name, summary) in &pairs {
            assert!(!summary.is_empty(), "{name} has empty summary");
        }
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
            BashCommand, CatCommand, GrepCommand, LsCommand, ScreenshotCommand, SeeCommand,
            WcCommand, WebCommand, WriteCommand,
        };

        fn contains_hint(s: &str) -> bool {
            s.contains("Use:")
                || s.contains("Try:")
                || s.contains("Check:")
                || s.contains("Available:")
        }

        async fn run_cmd<C: Command>(cmd: C, args: Vec<String>) -> CommandOutput {
            cmd.run(CommandInput {
                args,
                stdin: Vec::new(),
            })
            .await
            .expect("run returns Ok on handled failures")
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
                "web",
                rt.block_on(run_cmd(
                    WebCommand::new(),
                    vec!["file:///etc/passwd".into()],
                )),
            ),
            // bash denylist rejection exercises the policy error path,
            // which is a convention-compliant `[error] bash: … Try: …`
            // line. (The separate AC #2 timeout path deliberately emits a
            // fixed, hint-free format and is not covered by this test.)
            (
                "bash",
                rt.block_on(run_cmd(
                    {
                        use crate::commands::bash::BashPolicyCfg;
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
        ];

        for (name, out) in cases {
            assert_ne!(
                out.exit_code, 0,
                "{name}: failure input should exit non-zero"
            );
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(
                stderr.contains("[error] "),
                "{name}: stderr missing `[error] ` tag — got {stderr:?}"
            );
            assert!(
                stderr.contains(&format!("[error] {name}: ")),
                "{name}: stderr should say `[error] {name}: ` — got {stderr:?}"
            );
            assert!(
                contains_hint(&stderr),
                "{name}: stderr missing recovery hint (Use:/Try:/Check:/Available:) — got {stderr:?}"
            );
        }
    }

    /// Acceptance criterion #6: every registered production command has a
    /// non-empty `help()` return value. Also asserts `summary()` is
    /// non-empty and within the ≤80-char budget, since the Level-0
    /// description is the contract the LLM actually consumes.
    #[test]
    fn every_registered_command_has_nonempty_help_and_summary() {
        use crate::commands::{
            BashCommand, CatCommand, EchoCommand, GrepCommand, LsCommand, ScreenshotCommand,
            SeeCommand, WcCommand, WebCommand, WriteCommand,
        };
        let mut reg = CommandRegistry::new();
        reg.register(CatCommand);
        reg.register(LsCommand);
        reg.register(GrepCommand);
        reg.register(WcCommand);
        reg.register(EchoCommand);
        reg.register(WriteCommand::permissive_for_tests());
        reg.register(SeeCommand::default());
        reg.register(ScreenshotCommand::default());
        reg.register(WebCommand::new());
        reg.register(BashCommand::default());
        assert_eq!(reg.len(), 10);
        for (name, summary) in reg.sorted_summaries() {
            assert!(!summary.is_empty(), "{name} has empty summary");
            assert!(
                summary.len() <= 80,
                "{name} summary is {} chars (>80): {summary:?}",
                summary.len()
            );
            let help = reg.get(name).expect("command registered").help();
            assert!(!help.is_empty(), "{name} has empty help");
            assert!(
                help.contains("usage:"),
                "{name} help should contain `usage:` line — got {help:?}"
            );
        }
    }

    #[test]
    fn unknown_lookup_returns_none() {
        let reg = CommandRegistry::new();
        assert!(reg.get("nope").is_none());
    }

    #[tokio::test]
    async fn registered_command_is_runnable() {
        let mut reg = CommandRegistry::new();
        reg.register(Stub("ping"));
        let cmd = reg.get("ping").unwrap();
        let out = cmd
            .run(CommandInput {
                args: vec![],
                stdin: vec![],
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
    }
}

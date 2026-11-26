//! Internal commands: byte-oriented handlers with Unix exit codes that the
//! chain executor dispatches to from behind [`crate::RunTool`].

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, ErrorKind};
use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use assistd_ipc::{Access, Attachment};

use crate::TurnContext;
use crate::chain::Word;
use crate::policy::{ConfinedRoot, FileScope};

/// The recovery label on an error line: `Use`/`Try` offer an alternative,
/// `Check`/`Available` a diagnostic, `Install` a package, `Note` a condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hint {
    Use,
    Try,
    Check,
    Available,
    Install,
    Note,
}

impl Hint {
    /// The label as written on the wire, without the trailing colon.
    pub fn as_str(self) -> &'static str {
        match self {
            Hint::Use => "Use",
            Hint::Try => "Try",
            Hint::Check => "Check",
            Hint::Available => "Available",
            Hint::Install => "Install",
            Hint::Note => "Note",
        }
    }
}

impl fmt::Display for Hint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Format a stderr line as `[error] <cmd>: <what>. <hint>: <recovery>\n`.
/// `cmd` is the command name or a pre-dispatch tag (`parse`, `pipe`);
/// `recovery` is a command the model can run next or a short check.
pub fn error_line(
    cmd: &str,
    what: impl fmt::Display,
    hint: Hint,
    recovery: impl fmt::Display,
) -> String {
    format!("[error] {cmd}: {what}. {hint}: {recovery}\n")
}

/// A command's usage: the argument shape after its name, and the prose
/// printed below the usage line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub synopsis: String,
    pub body: String,
}

impl Usage {
    /// A usage of `synopsis` arguments explained by `body`.
    pub fn new(synopsis: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            synopsis: synopsis.into(),
            body: body.into(),
        }
    }

    /// The help text of the command called `name`: `usage: <name> <synopsis>`,
    /// a blank line, then the body.
    pub fn render(&self, name: &str) -> String {
        format!("usage: {name} {}\n\n{}", self.synopsis, self.body)
    }
}

/// Why a command failed: its exit code, what went wrong, and a recovery the
/// model can act on under a [`Hint`] label.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{what}. {hint}: {recovery}")]
pub struct CommandError {
    pub exit: i32,
    pub what: String,
    pub hint: Hint,
    pub recovery: String,
}

impl CommandError {
    /// A failure exiting `exit`.
    pub fn new(
        exit: i32,
        what: impl fmt::Display,
        hint: Hint,
        recovery: impl fmt::Display,
    ) -> Self {
        Self {
            exit,
            what: what.to_string(),
            hint,
            recovery: recovery.to_string(),
        }
    }

    /// A failure exiting 1.
    pub fn failed(what: impl fmt::Display, hint: Hint, recovery: impl fmt::Display) -> Self {
        Self::new(1, what, hint, recovery)
    }

    /// Arguments that could not be understood: exit 2 with a `Use:` hint.
    pub fn usage(what: impl fmt::Display, recovery: impl fmt::Display) -> Self {
        Self::new(2, what, Hint::Use, recovery)
    }

    /// An I/O error on `path`, exiting 1 with a hint suited to the error kind. A missing
    /// unquoted path that still holds glob metacharacters is an unmatched glob.
    pub fn io(path: &Word, e: &io::Error) -> Self {
        let dir = parent_dir(path.as_str());
        match e.kind() {
            ErrorKind::NotFound if !path.quoted && has_glob_meta(path.as_str()) => Self::failed(
                format_args!("no file matches {path}"),
                Hint::Try,
                format_args!("ls {dir} to see what is there"),
            ),
            ErrorKind::NotFound => Self::failed(
                format_args!("file not found: {path}"),
                Hint::Use,
                format_args!("ls {dir} to see what is there"),
            ),
            ErrorKind::NotADirectory => Self::failed(
                format_args!("{path}: a parent component is not a directory"),
                Hint::Check,
                format_args!("ls {dir}"),
            ),
            ErrorKind::PermissionDenied => Self::failed(
                format_args!("permission denied: {path}"),
                Hint::Check,
                format_args!("ls -l {path}"),
            ),
            ErrorKind::FileTooLarge => Self::failed(
                format_args!("{path}: {e}"),
                Hint::Use,
                format_args!("tail -n 200 {path} or grep PATTERN {path}"),
            ),
            ErrorKind::InvalidInput => Self::failed(
                format_args!("{path}: {e}"),
                Hint::Check,
                format_args!("ls -l {path}"),
            ),
            _ => Self::failed(
                format_args!("{path}: {e}"),
                Hint::Try,
                "a different path or check with ls",
            ),
        }
    }

    /// The stderr line reporting this failure of the command called `cmd`.
    pub fn line(&self, cmd: &str) -> String {
        error_line(cmd, &self.what, self.hint, &self.recovery)
    }
}

fn has_glob_meta(path: &str) -> bool {
    path.contains(['*', '?', '['])
}

/// Directory containing `path`, cut before any glob metacharacter so the
/// hint names a directory that can actually be listed.
fn parent_dir(path: &str) -> &str {
    let head = path
        .find(['*', '?', '['])
        .map_or_else(|| path.trim_end_matches('/'), |i| &path[..i]);
    match head.rfind('/') {
        Some(0) => "/",
        Some(cut) => &head[..cut],
        None => ".",
    }
}

/// Input to a single chain stage.
#[derive(Debug)]
pub struct CommandInput {
    /// Arguments after `argv[0]`, each with whether it was quoted.
    pub args: Vec<Word>,
    /// Bytes piped from the previous stage; `None` when nothing was piped,
    /// as distinct from an upstream stage that printed nothing.
    pub stdin: Option<Vec<u8>>,
    /// Where the paths in `args` resolve; a command that opens files must
    /// open them through it.
    pub scope: FileScope,
    /// The turn the chain runs in, whose router answers the command's prompts.
    pub turn: TurnContext,
}

/// Output of a single chain stage, whose non-zero `exit_code` `&&` and `||`
/// react to.
#[derive(Debug, Default, Clone)]
pub struct CommandOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: i32,
    pub attachments: Vec<Attachment>,
    /// What the command touched; it may miss things, and a script's reads are never known.
    pub accesses: Vec<Access>,
}

impl CommandOutput {
    /// Construct a successful output with the given stdout bytes.
    pub fn ok(stdout: Vec<u8>) -> Self {
        Self {
            stdout,
            ..Self::default()
        }
    }

    /// Construct a failed output with the given exit code and stderr bytes.
    pub fn failed(exit_code: i32, stderr: impl Into<Vec<u8>>) -> Self {
        Self {
            stderr: stderr.into(),
            exit_code,
            ..Self::default()
        }
    }

    /// Usage text on stdout with exit 2, the reply to insufficient arguments.
    pub fn usage(help: String) -> Self {
        Self {
            stdout: help.into_bytes(),
            exit_code: 2,
            ..Self::default()
        }
    }

    /// This output, also reporting that its command touched `accesses`.
    pub fn touching(mut self, accesses: impl IntoIterator<Item = Access>) -> Self {
        self.accesses.extend(accesses);
        self
    }

    /// Append `next`'s streams, attachments and accesses to this output and adopt
    /// its exit code: the shape `&&`, `||` and `;` produce.
    pub fn then(mut self, next: Self) -> Self {
        self.stdout.extend(next.stdout);
        self.stderr.extend(next.stderr);
        self.attachments.extend(next.attachments);
        self.accesses.extend(next.accesses);
        self.exit_code = next.exit_code;
        self
    }
}

/// A single internal command (`cat`, `grep`, `bash`, …). Its
/// [`Command::summary`] and synopsis are listed in the `run` tool description
/// and its [`Command::help`] is returned for insufficient arguments.
#[async_trait]
pub trait Command: fmt::Debug + Send + Sync + 'static {
    /// Name the command is dispatched by (e.g. `"cat"`).
    fn name(&self) -> &str;
    /// A terse verb phrase of at most 80 chars with no trailing newline.
    fn summary(&self) -> &'static str;
    /// The argument shape listed beside the summary, and the prose of the help text.
    fn usage(&self) -> Usage;
    /// The [`Command::usage`] as help text, written verbatim to stdout.
    fn help(&self) -> String {
        self.usage().render(self.name())
    }
    /// Whether `~` and globs in the arguments expand before [`Command::run`];
    /// `false` passes every word as written.
    fn expands_args(&self) -> bool {
        true
    }
    /// Whether the command's output can hold data from this machine; `false` only when it
    /// holds nothing but what the web returned.
    fn reads_host(&self) -> bool {
        true
    }
    /// Execute the command. `Err` is a failure with nothing else to report;
    /// one that still has output sets the exit code of an `Ok`.
    async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError>;
}

/// Lookup table of registered commands, keyed by name, and the
/// [`FileScope`] their arguments resolve in.
#[derive(Debug, Default)]
pub struct CommandRegistry {
    commands: BTreeMap<String, Box<dyn Command>>,
    scope: FileScope,
}

impl CommandRegistry {
    /// Create an empty registry whose commands see the host's files.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty registry whose commands, globs and `~` resolve only beneath `root`.
    pub(crate) fn confined(root: Arc<ConfinedRoot>) -> Self {
        Self {
            commands: BTreeMap::new(),
            scope: FileScope::Confined(root),
        }
    }

    /// Where the registered commands' arguments resolve.
    pub(crate) fn scope(&self) -> &FileScope {
        &self.scope
    }

    /// Register a command by value.
    pub fn register<C: Command>(&mut self, cmd: C) {
        self.commands.insert(cmd.name().to_string(), Box::new(cmd));
    }

    /// Drop every command whose name `keep` refuses.
    pub fn retain(&mut self, keep: impl Fn(&str) -> bool) {
        self.commands.retain(|name, _| keep(name));
    }

    /// Look up a registered command by its `name()`.
    pub fn get(&self, name: &str) -> Option<&dyn Command> {
        self.commands.get(name).map(AsRef::as_ref)
    }

    /// Number of registered commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Returns `true` when no commands have been registered.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Command names, sorted alphabetically.
    pub fn sorted_names(&self) -> Vec<&str> {
        self.commands.keys().map(String::as_str).collect()
    }

    /// `(name, synopsis, summary)` triples, sorted alphabetically by name.
    pub fn sorted_listings(&self) -> Vec<(&str, String, &'static str)> {
        self.commands
            .values()
            .map(|c| (c.name(), c.usage().synopsis, c.summary()))
            .collect()
    }
}

/// Run `cmd` on `input`, turning a [`CommandError`] into its exit code and
/// stderr line.
pub async fn run_rendered<C: Command + ?Sized>(cmd: &C, input: CommandInput) -> CommandOutput {
    cmd.run(input)
        .await
        .unwrap_or_else(|error| CommandOutput::failed(error.exit, error.line(cmd.name())))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use assistd_wm::NoWindowManager;

    use super::*;
    use crate::commands::{
        BashCommand, LsCommand, WebCommand, WmCommand, permissive_write, test_cat, test_sandbox,
        test_screenshot, test_see, test_wm,
    };
    use crate::policy::{ApprovalGate, Approvals};
    use crate::test_support::{AlwaysAllowGate, DenyAllGate, bash_policy};

    #[test]
    fn unmatched_glob_reads_as_a_glob_not_a_missing_file() {
        let e = io::Error::from(ErrorKind::NotFound);
        let line = |path: &Word| CommandError::io(path, &e).line("ls");
        assert_eq!(
            line(&Word::bare("/tmp/*.db-shm")),
            "[error] ls: no file matches /tmp/*.db-shm. Try: ls /tmp to see what is there\n"
        );
        for literal in [
            Word::bare("/tmp/notes.txt"),
            Word::quoted("/tmp/[notes].txt"),
        ] {
            assert_eq!(
                line(&literal),
                format!(
                    "[error] ls: file not found: {literal}. Use: ls /tmp to see what is there\n"
                )
            );
        }
    }

    #[test]
    fn parent_dir_stops_at_the_first_metacharacter() {
        assert_eq!(parent_dir("/tmp/*.db-shm"), "/tmp");
        assert_eq!(parent_dir("docs/*.md"), "docs");
        assert_eq!(parent_dir("docs/*/"), "docs");
        assert_eq!(parent_dir("/a*/b*"), "/");
        assert_eq!(parent_dir("*.rs"), ".");
    }

    async fn run_cmd<C: Command>(cmd: C, args: &[&str]) -> CommandOutput {
        run_rendered(
            &cmd,
            CommandInput {
                args: args.iter().copied().map(Word::bare).collect(),
                stdin: None,
                scope: FileScope::Host,
                turn: TurnContext::default(),
            },
        )
        .await
    }

    fn contains_hint(s: &str) -> bool {
        s.contains("Use:") || s.contains("Try:") || s.contains("Check:") || s.contains("Available:")
    }

    fn bash_declining_every_confirmation() -> BashCommand {
        BashCommand::new(
            Arc::new(bash_policy()),
            test_sandbox(),
            Arc::new(DenyAllGate),
        )
    }

    fn wm_without_compositor() -> WmCommand {
        test_wm(Arc::new(NoWindowManager))
    }

    /// One failing invocation per command that writes its own error lines (`echo` cannot fail,
    /// and the text programs print GNU's), paired with the command name its line must carry.
    async fn failing_invocations() -> Vec<(&'static str, CommandOutput)> {
        let missing = &["/nonexistent/assistd-convention-test"];
        vec![
            ("cat", run_cmd(test_cat(), missing).await),
            ("ls", run_cmd(LsCommand, missing).await),
            (
                "see",
                run_cmd(test_see(), &["/nonexistent/assistd-convention-test.png"]).await,
            ),
            (
                "screenshot",
                run_cmd(test_screenshot(), &["--bogus-flag"]).await,
            ),
            (
                "write",
                run_cmd(
                    permissive_write(),
                    &["/nonexistent/assistd-convention-test", "x"],
                )
                .await,
            ),
            (
                "web",
                run_cmd(
                    WebCommand::new(
                        ApprovalGate::new(
                            Arc::new(AlwaysAllowGate),
                            Arc::new(Approvals::unsaved()),
                        ),
                        None,
                    ),
                    &["fetch", "file:///etc/passwd"],
                )
                .await,
            ),
            (
                "bash",
                run_cmd(
                    bash_declining_every_confirmation(),
                    &["touch /nonexistent/assistd-convention-test"],
                )
                .await,
            ),
            (
                "wm",
                run_cmd(wm_without_compositor(), &["focus", "Firefox"]).await,
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
}

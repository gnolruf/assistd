//! Internal command abstraction: Rust handlers the chain executor
//! dispatches to, operating on raw bytes with a Unix-style exit code so
//! `&&`/`||` composition works. The model never sees a [`Command`]
//! directly, only the [`crate::RunTool`] that walks the chain.

use std::collections::BTreeMap;
use std::fmt;

use async_trait::async_trait;

/// The recovery label on an error line. `Use` and `Try` introduce an
/// alternative to run; `Check` and `Available` introduce a diagnostic;
/// `Install` names a package; `Note` explains a condition.
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

/// Format a stderr line in the error-navigation convention every
/// command follows: `[error] <cmd>: <what>. <hint>: <recovery>\n`.
///
/// Each line says both what went wrong and what to do instead, so the
/// model recovers in one step rather than retrying blindly. `cmd` is the
/// command name, or a pseudo-tag (`parse`, `pipe`, `unknown command`) for
/// failures before dispatch. `recovery` is either a command the model can
/// run verbatim (`see photo.png`, `ls /dir`) or a short check instruction
/// (`ls -l <path>`). A command should not return a bare non-zero exit
/// code: it emits such a line, or forwards the subprocess stderr that
/// explains the failure.
pub fn error_line(
    cmd: &str,
    what: impl fmt::Display,
    hint: Hint,
    recovery: impl fmt::Display,
) -> String {
    format!("[error] {cmd}: {what}. {hint}: {recovery}\n")
}

/// Classify a `std::io::Error` against the `path` that produced it and
/// build an [`error_line`] whose hint suits the error kind.
pub fn io_error_nav(cmd: &str, path: &str, e: &std::io::Error) -> String {
    use std::io::ErrorKind;
    match e.kind() {
        // A path still carrying glob metacharacters got here because the
        // expander found nothing to match and passed the pattern through
        // (POSIX behaviour). Saying "file not found" sends the model
        // looking for a file it never asked for.
        ErrorKind::NotFound if has_glob_meta(path) => error_line(
            cmd,
            format_args!("no file matches {path}"),
            Hint::Try,
            format_args!("ls {} to see what is there", parent_dir(path)),
        ),
        ErrorKind::NotFound => error_line(
            cmd,
            format_args!("file not found: {path}"),
            Hint::Use,
            format_args!("ls {} to see what is there", parent_dir(path)),
        ),
        ErrorKind::NotADirectory => error_line(
            cmd,
            format_args!("{path}: a parent component is not a directory"),
            Hint::Check,
            format_args!("ls {}", parent_dir(path)),
        ),
        ErrorKind::PermissionDenied => error_line(
            cmd,
            format_args!("permission denied: {path}"),
            Hint::Check,
            format_args!("ls -l {path}"),
        ),
        ErrorKind::FileTooLarge => error_line(
            cmd,
            format_args!("{path}: {e}"),
            Hint::Use,
            format_args!("bash \"tail -n 200 {path}\" or bash \"grep PATTERN {path}\""),
        ),
        ErrorKind::InvalidInput => error_line(
            cmd,
            format_args!("{path}: {e}"),
            Hint::Check,
            format_args!("ls -l {path}"),
        ),
        _ => error_line(
            cmd,
            format_args!("{path}: {e}"),
            Hint::Try,
            "a different path or check with ls",
        ),
    }
}

fn has_glob_meta(path: &str) -> bool {
    path.contains(['*', '?', '['])
}

/// Directory containing `path`, cut before the first glob metacharacter
/// when there is one, so the hint points at a directory that can
/// actually be listed.
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
pub struct CommandInput {
    /// Positional arguments after `argv[0]`; the command's own name is
    /// not included.
    pub args: Vec<String>,
    /// Bytes piped in from the previous chain stage. `None` when the
    /// command is not on the right of a pipe, so a filter can tell
    /// "nothing was piped" (reply with usage) from "the upstream stage
    /// produced nothing" (an empty result).
    pub stdin: Option<Vec<u8>>,
}

/// A side-channel payload a command attaches alongside its stdout. The
/// chain executor threads attachments through pipes untouched, so
/// `see X | wc` still surfaces the image.
#[derive(Debug, Clone)]
pub enum Attachment {
    Image { mime: String, bytes: Vec<u8> },
}

/// Output of a single chain stage.
///
/// Every failure is reported here, as a non-zero `exit_code` with a
/// stderr line, so `&&` and `||` can react to it: `cat missing || echo
/// 'not found'` runs the `echo`.
#[derive(Debug, Default, Clone)]
pub struct CommandOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: i32,
    pub attachments: Vec<Attachment>,
}

impl CommandOutput {
    /// Construct a successful output with the given stdout bytes.
    pub fn ok(stdout: Vec<u8>) -> Self {
        Self {
            stdout,
            stderr: Vec::new(),
            exit_code: 0,
            attachments: Vec::new(),
        }
    }

    /// Construct a failed output with the given exit code and stderr bytes.
    pub fn failed(exit_code: i32, stderr: impl Into<Vec<u8>>) -> Self {
        Self {
            stdout: Vec::new(),
            stderr: stderr.into(),
            exit_code,
            attachments: Vec::new(),
        }
    }

    /// Construct the reply to a call with insufficient arguments: the
    /// usage text on stdout with exit 2, so it reads as help rather
    /// than a failure.
    pub fn usage(help: String) -> Self {
        Self {
            stdout: help.into_bytes(),
            exit_code: 2,
            ..Self::default()
        }
    }

    /// The reply to a call whose arguments could not be understood:
    /// `[error] <cmd>: <what>. Use: <recovery>` with exit 2.
    pub fn usage_error(cmd: &str, what: impl fmt::Display, recovery: impl fmt::Display) -> Self {
        Self::failed(2, error_line(cmd, what, Hint::Use, recovery).into_bytes())
    }

    /// Append `next`'s streams and attachments to this output and adopt
    /// its exit code: the shape `&&`, `||` and `;` produce.
    pub fn then(mut self, next: Self) -> Self {
        self.stdout.extend(next.stdout);
        self.stderr.extend(next.stderr);
        self.attachments.extend(next.attachments);
        self.exit_code = next.exit_code;
        self
    }
}

/// A single internal command (`cat`, `grep`, `bash`, …).
///
/// Help is progressive: [`Command::summary`] is the one-liner listed in
/// the `run` tool's description, [`Command::help`] is the full usage
/// block returned when the command is called with insufficient
/// arguments, and commands with subcommands return subcommand help from
/// their own [`Command::run`].
#[async_trait]
pub trait Command: Send + Sync + 'static {
    /// Name the command is dispatched by (e.g. `"cat"`).
    fn name(&self) -> &str;
    /// One line of at most 80 chars with no trailing newline, phrased as
    /// a terse verb phrase such as
    /// `"filter lines matching a pattern (supports -i, -v, -c)"`.
    fn summary(&self) -> &'static str;
    /// Full usage block, written verbatim to stdout. The first line
    /// begins with `usage: <name> …` so help reads distinctly from an
    /// `[error] <name>: …` failure.
    fn help(&self) -> String;
    /// Execute the command with the given input and return its output.
    /// Failures are reported through the output's exit code and stderr.
    async fn run(&self, input: CommandInput) -> CommandOutput;
}

/// Lookup table of registered commands, keyed by name.
#[derive(Default)]
pub struct CommandRegistry {
    commands: BTreeMap<String, Box<dyn Command>>,
}

impl CommandRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a command by value.
    pub fn register<C: Command>(&mut self, cmd: C) {
        self.commands.insert(cmd.name().to_string(), Box::new(cmd));
    }

    /// Look up a registered command by its `name()`.
    pub fn get(&self, name: &str) -> Option<&dyn Command> {
        self.commands.get(name).map(|c| c.as_ref())
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

    /// `(name, summary)` pairs, sorted alphabetically by name.
    pub fn sorted_summaries(&self) -> Vec<(&str, &'static str)> {
        self.commands
            .values()
            .map(|c| (c.name(), c.summary()))
            .collect()
    }
}

#[cfg(test)]
mod tests;

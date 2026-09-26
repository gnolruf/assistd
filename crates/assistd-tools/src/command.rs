//! Internal commands: byte-oriented handlers with Unix exit codes that the
//! chain executor dispatches to from behind [`crate::RunTool`].

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, ErrorKind};

use async_trait::async_trait;

/// The recovery label on an [`error_line`]: `Use`/`Try` offer an alternative,
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

/// Build an [`error_line`] for an I/O error on `path`, with a hint suited to
/// the error kind. A missing path that still holds glob metacharacters is
/// reported as an unmatched glob rather than a missing file.
pub fn io_error_nav(cmd: &str, path: &str, e: &io::Error) -> String {
    match e.kind() {
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
pub struct CommandInput {
    /// Arguments after `argv[0]`.
    pub args: Vec<String>,
    /// Bytes piped from the previous stage; `None` when nothing was piped,
    /// as distinct from an upstream stage that printed nothing.
    pub stdin: Option<Vec<u8>>,
}

/// A side-channel payload alongside stdout, carried through pipes untouched
/// so `see X | wc` still surfaces the image.
#[derive(Debug, Clone)]
pub enum Attachment {
    Image { mime: String, bytes: Vec<u8> },
}

/// Output of a single chain stage. Failures are a non-zero `exit_code` plus
/// a stderr line, so `&&` and `||` can react to them.
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

    /// Usage text on stdout with exit 2, the reply to insufficient arguments.
    pub fn usage(help: String) -> Self {
        Self {
            stdout: help.into_bytes(),
            exit_code: 2,
            ..Self::default()
        }
    }

    /// `[error] <cmd>: <what>. Use: <recovery>` with exit 2, the reply to
    /// arguments that could not be understood.
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

/// A single internal command (`cat`, `grep`, `bash`, …). Its
/// [`Command::summary`] is listed in the `run` tool description and its
/// [`Command::help`] is returned for insufficient arguments.
#[async_trait]
pub trait Command: Send + Sync + 'static {
    /// Name the command is dispatched by (e.g. `"cat"`).
    fn name(&self) -> &str;
    /// A terse verb phrase of at most 80 chars with no trailing newline.
    fn summary(&self) -> &'static str;
    /// Full usage block, written verbatim to stdout; its first line begins
    /// `usage: <name>`.
    fn help(&self) -> String;
    /// Whether `~` and globs in the arguments expand before [`Command::run`];
    /// `false` passes every word as written.
    fn expands_args(&self) -> bool {
        true
    }
    /// Execute the command; failures are reported through exit code and stderr.
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

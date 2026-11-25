//! `wm open`: the command line that launches an installed application as
//! the user, through the compositor, to open a file when none is named.

use std::borrow::Cow;
use std::path::Path;

use assistd_utils::path::expand_tilde_from_env;
use assistd_wm::{ExecCommand, UnquotableCommand, WorkspaceId};

use super::default_app::{DefaultAppError, default_app};
use super::desktop_entry::{DesktopEntry, EntryError, Executable, Targets};
use crate::chain::Word;
use crate::command::Hint;

/// Terminal emulators a `Terminal=true` application runs in, by preference,
/// with the arguments that come before its command line.
const TERMINALS: &[(&str, &[&str])] = &[
    ("xdg-terminal-exec", &[]),
    ("kitty", &[]),
    ("foot", &[]),
    ("alacritty", &["-e"]),
    ("wezterm", &["start", "--"]),
    ("ghostty", &["-e"]),
    ("gnome-terminal", &["--"]),
    ("konsole", &["-e"]),
    ("xfce4-terminal", &["-x"]),
    ("xterm", &["-e"]),
];

/// What a launch names first: an application by desktop id, or a file for
/// its default application.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Subject {
    App(String),
    File(String),
}

/// `wm open [--workspace <ws>] [--cwd <dir>] <app|file> [target...]`.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct OpenArgs {
    pub(super) workspace: Option<WorkspaceId>,
    cwd: Option<String>,
    pub(super) subject: Subject,
    /// Every file or URL the application is handed, a [`Subject::File`] first.
    targets: Vec<String>,
}

impl OpenArgs {
    /// # Errors
    /// What could not be understood, for a usage error.
    pub(super) fn parse(args: &[Word]) -> Result<Self, String> {
        let mut workspace = None;
        let mut cwd = None;
        let mut rest = args;
        loop {
            match rest {
                [flag, value, tail @ ..] if flag == "--workspace" => {
                    workspace = Some(WorkspaceId::from(value.as_str()));
                    rest = tail;
                }
                [flag, value, tail @ ..] if flag == "--cwd" => {
                    cwd = Some(value.text.clone());
                    rest = tail;
                }
                [flag, ..] if flag.text.starts_with('-') => {
                    return Err(format!(
                        "'{flag}' is not an option, or is missing its value"
                    ));
                }
                [file, ..] if names_path(&file.text) => {
                    return Ok(Self {
                        workspace,
                        cwd,
                        subject: Subject::File(file.text.clone()),
                        targets: rest.iter().map(|target| target.text.clone()).collect(),
                    });
                }
                [app, targets @ ..] => {
                    return Ok(Self {
                        workspace,
                        cwd,
                        subject: Subject::App(app.text.clone()),
                        targets: targets.iter().map(|target| target.text.clone()).collect(),
                    });
                }
                [] => return Err("no application or file named".to_string()),
            }
        }
    }

    /// Whether the launch names a file, URL, or working directory, any of
    /// which the sandbox may have planted.
    pub(super) fn has_targets_or_cwd(&self) -> bool {
        !self.targets.is_empty() || self.cwd.is_some()
    }
}

/// Why a launch could not be prepared.
#[derive(Debug, thiserror::Error)]
pub(super) enum OpenError {
    #[error(transparent)]
    Entry(#[from] EntryError),
    #[error(transparent)]
    DefaultApp(#[from] DefaultAppError),
    #[error("{0} takes no files or URLs")]
    NoTargets(String),
    #[error("{0} is a terminal emulator, which wm open starts without a command")]
    TerminalCommand(String),
    #[error("{0} takes only one file or URL")]
    OneTarget(String),
    #[error("'{0}' is not an absolute path")]
    NotAPath(String),
    #[error("'{0}' is neither an absolute path nor a URL")]
    NotAPathOrUrl(String),
    #[error("--cwd '{0}' is not an existing directory")]
    NotADirectory(String),
    #[error("no terminal emulator was found to run {0} in")]
    NoTerminal(String),
    #[error(transparent)]
    Unquotable(#[from] UnquotableCommand),
}

impl OpenError {
    /// The hint label and recovery that follow this error.
    pub(super) fn recovery(&self) -> (Hint, Cow<'static, str>) {
        match self {
            Self::Entry(EntryError::NotFound(_)) => (
                Hint::Check,
                "ls /usr/share/applications ~/.local/share/applications; a desktop id is a file \
                 name there without .desktop"
                    .into(),
            ),
            Self::Entry(EntryError::NotLaunchable { .. }) => {
                (Hint::Try, "a different desktop id".into())
            }
            Self::Entry(EntryError::Untrusted(_)) => (
                Hint::Note,
                "only entries out of the sandbox's reach are launched; the user can install it \
                 under ~/.local/share/applications"
                    .into(),
            ),
            Self::DefaultApp(DefaultAppError::NotAFile(_)) => (
                Hint::Use,
                "the absolute or ~/ path of an existing file".into(),
            ),
            Self::DefaultApp(_) => (
                Hint::Use,
                "wm open <app> <file>, naming the desktop id of an application that opens it"
                    .into(),
            ),
            Self::NoTargets(_) | Self::OneTarget(_) => (Hint::Use, "fewer files, or none".into()),
            Self::TerminalCommand(_) => (
                Hint::Use,
                "the program's own desktop id, which opens in a terminal by itself \
                 (wm open --cwd ~/project nvim), or the terminal alone for a shell there \
                 (wm open --cwd ~/project kitty)"
                    .into(),
            ),
            Self::NotAPath(_) => (Hint::Use, "an absolute or ~/ path".into()),
            Self::NotAPathOrUrl(_) => (Hint::Use, "an absolute or ~/ path, or a URL".into()),
            Self::NotADirectory(_) => (Hint::Use, "an existing absolute or ~/ directory".into()),
            Self::NoTerminal(_) => {
                let names: Vec<&str> = TERMINALS.iter().map(|(name, _)| *name).collect();
                (Hint::Install, names.join(", ").into())
            }
            Self::Unquotable(_) => (Hint::Try, "a path without double quotes".into()),
        }
    }
}

/// A launch ready for the user's confirmation and the compositor.
#[derive(Debug)]
pub(super) struct PreparedLaunch {
    pub(super) entry: DesktopEntry,
    pub(super) command: ExecCommand,
    /// What the command line runs, a terminal emulator included.
    pub(super) executable: Executable,
    /// Whether an entry that takes URLs is handed a target, which it may fetch or, when a
    /// local file, load and follow to any host.
    pub(super) opens_url: bool,
}

impl PreparedLaunch {
    /// Find the entry `args` names, or the default one for its file, and
    /// prepare it; reads the filesystem, so it blocks.
    pub(super) fn find(args: &OpenArgs) -> Result<Self, OpenError> {
        let entry = match &args.subject {
            Subject::App(id) => DesktopEntry::find(id)?,
            Subject::File(file) => default_app(&expand_tilde_from_env(file))?,
        };
        let path_env = std::env::var_os("PATH").unwrap_or_default();
        let executable = entry.executable(&path_env);
        Self::new(entry, executable, args)
    }

    /// The shell command line that runs `entry`, whose `Exec` runs `executable`, as `args`
    /// asks.
    pub(super) fn new(
        entry: DesktopEntry,
        mut executable: Executable,
        args: &OpenArgs,
    ) -> Result<Self, OpenError> {
        let targets = resolve_targets(&entry, &args.targets)?;
        let opens_url = !targets.is_empty() && matches!(entry.targets(), Targets::Urls { .. });
        let mut argv = entry.command_line(&targets);
        if entry.terminal {
            let terminal = terminal_prefix(&entry.name)?;
            if let Some(program) = terminal.first() {
                executable = executable.running_also(Path::new(program));
            }
            argv.splice(0..0, terminal);
        }
        let exec = format!("exec {}", shell_words(&argv));
        let line = match &args.cwd {
            Some(cwd) => format!("cd {} && {exec}", quote_word(&resolve_cwd(cwd)?)),
            None => exec,
        };
        let command = ExecCommand::new(line)?;
        Ok(Self {
            entry,
            command,
            executable,
            opens_url,
        })
    }
}

/// Whether `word` is a file rather than a desktop id, which holds no `/`.
fn names_path(word: &str) -> bool {
    word.starts_with('/') || word.starts_with("~/")
}

fn resolve_targets(entry: &DesktopEntry, raw: &[String]) -> Result<Vec<String>, OpenError> {
    let (urls, many) = match entry.targets() {
        Targets::None if raw.is_empty() => return Ok(Vec::new()),
        Targets::None if entry.is_terminal_emulator => {
            return Err(OpenError::TerminalCommand(entry.name.clone()));
        }
        Targets::None => return Err(OpenError::NoTargets(entry.name.clone())),
        Targets::Files { many } => (false, many),
        Targets::Urls { many } => (true, many),
    };
    if !many && raw.len() > 1 {
        return Err(OpenError::OneTarget(entry.name.clone()));
    }
    raw.iter()
        .map(|target| resolve_target(target, urls))
        .collect()
}

/// `raw` as a URL when `urls` allows one, else as an absolute path with
/// `~` expanded; neither can be taken for an option.
fn resolve_target(raw: &str, urls: bool) -> Result<String, OpenError> {
    if urls && is_url(raw) {
        return Ok(raw.to_string());
    }
    let path = expand_tilde_from_env(raw);
    match path.into_os_string().into_string() {
        Ok(path) if path.starts_with('/') => Ok(path),
        _ if urls => Err(OpenError::NotAPathOrUrl(raw.to_string())),
        _ => Err(OpenError::NotAPath(raw.to_string())),
    }
}

/// `scheme:rest`, the scheme starting with a letter as RFC 3986 requires.
fn is_url(raw: &str) -> bool {
    raw.split_once(':').is_some_and(|(scheme, rest)| {
        !rest.is_empty()
            && scheme.starts_with(|c: char| c.is_ascii_alphabetic())
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "+.-".contains(c))
    })
}

fn resolve_cwd(raw: &str) -> Result<String, OpenError> {
    let path = expand_tilde_from_env(raw);
    if path.is_absolute() && path.is_dir() {
        path.into_os_string()
            .into_string()
            .map_err(|_| OpenError::NotADirectory(raw.to_string()))
    } else {
        Err(OpenError::NotADirectory(raw.to_string()))
    }
}

/// The first of [`TERMINALS`] on `PATH`, by absolute path, with its
/// arguments.
fn terminal_prefix(app: &str) -> Result<Vec<String>, OpenError> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    TERMINALS
        .iter()
        .find_map(|(program, args)| {
            let found = std::env::split_paths(&path)
                .map(|dir| dir.join(program))
                .find(|candidate| candidate.is_file())?
                .into_os_string()
                .into_string()
                .ok()?;
            Some(
                std::iter::once(found)
                    .chain(args.iter().map(|arg| (*arg).to_string()))
                    .collect(),
            )
        })
        .ok_or_else(|| OpenError::NoTerminal(app.to_string()))
}

fn shell_words(argv: &[String]) -> String {
    argv.iter()
        .map(|word| quote_word(word))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `word` as `sh` needs it typed to stay one word.
fn quote_word(word: &str) -> Cow<'_, str> {
    let plain = !word.is_empty()
        && word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./=:,+@%".contains(c));
    if plain {
        Cow::Borrowed(word)
    } else {
        Cow::Owned(format!("'{}'", word.replace('\'', r"'\''")))
    }
}

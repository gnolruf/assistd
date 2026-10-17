//! `wm open`: the command line that launches an installed application as
//! the user, through the compositor.

use std::borrow::Cow;

use assistd_utils::path::expand_tilde_from_env;
use assistd_wm::{ExecCommand, UnquotableCommand, WorkspaceId};

use super::desktop_entry::{DesktopEntry, EntryError, Targets};
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

/// `wm open [--workspace <ws>] [--cwd <dir>] <app> [target...]`.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct OpenArgs {
    pub(super) workspace: Option<WorkspaceId>,
    cwd: Option<String>,
    app: String,
    targets: Vec<String>,
}

impl OpenArgs {
    /// # Errors
    /// What could not be understood, for a usage error.
    pub(super) fn parse(args: &[String]) -> Result<Self, String> {
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
                    cwd = Some(value.clone());
                    rest = tail;
                }
                [flag, ..] if flag.starts_with('-') => {
                    return Err(format!(
                        "'{flag}' is not an option, or is missing its value"
                    ));
                }
                [app, targets @ ..] => {
                    return Ok(Self {
                        workspace,
                        cwd,
                        app: app.clone(),
                        targets: targets.to_vec(),
                    });
                }
                [] => return Err("no application named".to_string()),
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
}

impl PreparedLaunch {
    /// Find the entry `args` names and prepare it; reads the filesystem,
    /// so it blocks.
    pub(super) fn find(args: &OpenArgs) -> Result<Self, OpenError> {
        Self::new(DesktopEntry::find(&args.app)?, args)
    }

    /// The shell command line that runs `entry` as `args` asks.
    pub(super) fn new(entry: DesktopEntry, args: &OpenArgs) -> Result<Self, OpenError> {
        let targets = resolve_targets(&entry, &args.targets)?;
        let mut argv = entry.command_line(&targets);
        if entry.terminal {
            argv.splice(0..0, terminal_prefix(&entry.name)?);
        }
        let exec = format!("exec {}", shell_words(&argv));
        let line = match &args.cwd {
            Some(cwd) => format!("cd {} && {exec}", quote_word(&resolve_cwd(cwd)?)),
            None => exec,
        };
        let command = ExecCommand::new(line)?;
        Ok(Self { entry, command })
    }
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

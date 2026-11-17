//! Command review: the classifier, the shell-word helpers it shares, and why a script needs the
//! user's confirmation.

use std::fmt;

use self::variables::is_assignment;
use super::allowlist::ProgramApproval;
use super::confirm::Risk;
use super::shell::{COMMAND_PREFIX_WORDS, Word};

mod classify;
mod evaluated;
mod options;
mod paths;
mod placeholder;
mod programs;
mod variables;
mod wrappers;

pub use classify::{Action, Analysis, Class, Effect, analyze, classify};

/// How many scripts deep (`eval`, `trap`, …) the classifier looks before
/// calling a script unvetted.
const MAX_NESTED_SCRIPTS: usize = 16;

/// Reserved words, which run nothing themselves when unquoted.
const KEYWORDS: &[&str] = &[
    "!", "[[", "]]", "{", "}", "case", "do", "done", "elif", "else", "esac", "fi", "for",
    "function", "if", "in", "select", "then", "until", "while",
];

/// Builtins that run no code and reach nothing outside the shell.
const SAFE_BUILTINS: &[&str] = &[
    ":", "[", "bg", "break", "caller", "cd", "compopt", "continue", "declare", "dirs", "disown",
    "echo", "exit", "export", "false", "fg", "getopts", "hash", "help", "jobs", "let", "local",
    "logout", "popd", "printf", "pushd", "pwd", "read", "readonly", "return", "set", "shift",
    "shopt", "suspend", "test", "times", "true", "type", "typeset", "ulimit", "umask", "unalias",
    "unset", "wait",
];

/// The wrapper that appends arguments read from stdin to its command.
const FEEDS_ARGUMENTS: &str = "xargs";

/// Where `find -exec` and [`FEEDS_ARGUMENTS`] put an input item inside
/// their command.
const INPUT_PLACEHOLDER: &str = "{}";

/// Why a command needs the user's confirmation before it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Confirmation {
    /// Neither the read-only nor the edit list vets something it does; holds why, for the
    /// prompt, and what an "always allow" answer would approve, possibly nothing.
    Unvetted {
        why: String,
        always_allow: Vec<ProgramApproval>,
    },
    /// It touches a sensitive path, so it asks every time; holds what and where.
    Sensitive(String),
    /// It edits these paths, as written.
    Edit(Vec<String>),
}

impl Confirmation {
    /// What an "always allow" answer would approve; possibly nothing.
    pub fn always_allow(&self) -> &[ProgramApproval] {
        match self {
            Self::Unvetted { always_allow, .. } => always_allow,
            Self::Sensitive(_) | Self::Edit(_) => &[],
        }
    }

    /// Routine for an edit, elevated for anything else.
    pub fn risk(&self) -> Risk {
        match self {
            Self::Edit(_) => Risk::Routine,
            Self::Unvetted { .. } | Self::Sensitive(_) => Risk::Elevated,
        }
    }
}

impl fmt::Display for Confirmation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unvetted { why, .. } | Self::Sensitive(why) => f.write_str(why),
            Self::Edit(paths) if paths.is_empty() => f.write_str("edits files"),
            Self::Edit(paths) => write!(f, "edits `{}`", paths.join("`, `")),
        }
    }
}

/// A word that reads as a subcommand (`build`, `run-tests`) rather than a
/// path, pattern, or toolchain (`+nightly`).
fn is_subcommand(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_alphanumeric())
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn short_options(arg: &str) -> Option<&str> {
    arg.strip_prefix('-')
        .filter(|opts| !opts.is_empty() && !opts.starts_with('-'))
}

fn command_start(words: &[Word]) -> Option<usize> {
    let mut at = 0;
    while let Some(word) = words.get(at) {
        let bare = !word.quoted && !word.dynamic;
        if is_assignment(&word.text) || (bare && COMMAND_PREFIX_WORDS.contains(&word.text.as_str()))
        {
            at += 1;
        } else if bare && word.text == "function" {
            at += 2;
        } else {
            return Some(at);
        }
    }
    None
}

pub(super) fn basename(word: &str) -> &str {
    word.rsplit_once('/').map_or(word, |(_, name)| name)
}

/// A command word's last path component, lowercased, without a trailing
/// version (`python3.12` is `python`).
pub(super) fn program(word: &str) -> String {
    let name = basename(word).to_ascii_lowercase();
    match name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.' || c == '-') {
        "" => name,
        trimmed => trimmed.to_string(),
    }
}

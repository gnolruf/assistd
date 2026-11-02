//! Command review: the classifier, the shell-word helpers it shares, and why a script needs the
//! user's confirmation.

use std::{fmt, mem};

use super::allowlist::ProgramApproval;
use super::confirm::Risk;
use super::shell::{self, Script, Word};

mod classify;
mod evaluated;
mod placeholder;

pub use classify::{Action, Analysis, Class, Delete, Effect, analyze, classify};

/// How many scripts deep (`eval`, `trap`, …) the classifier looks before
/// calling a script unvetted.
const MAX_NESTED_SCRIPTS: usize = 16;

/// Reserved words that leave the word after them in command position.
const COMMAND_PREFIX_WORDS: &[&str] = &[
    "!", "{", "if", "then", "else", "elif", "do", "while", "until",
];

/// Builtins and programs that run a command given as their arguments, with
/// the number of operands before that command.
const WRAPPERS: &[(&str, usize)] = &[
    ("builtin", 0),
    ("command", 0),
    ("coproc", 0),
    ("env", 0),
    ("exec", 0),
    ("nice", 0),
    ("nohup", 0),
    ("stdbuf", 0),
    ("time", 0),
    ("timeout", 1),
    ("xargs", 0),
];

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

/// Characters that make one word a command line rather than a program
/// name (`env -S 'rm -rf ~'`).
const SHELL_SYNTAX: &str = ";&|()<>`$";

/// Wrapper options that take the next word as their value. Only documented
/// value options belong here: a flag listed by mistake hides the command.
const VALUE_OPTIONS: &[(&str, &[&str])] = &[
    ("env", &["-u", "--unset", "-C", "--chdir"]),
    ("nice", &["-n", "--adjustment"]),
    (
        "stdbuf",
        &["-i", "-o", "-e", "--input", "--output", "--error"],
    ),
    ("timeout", &["-s", "--signal", "-k", "--kill-after"]),
    (
        "xargs",
        &[
            "-a",
            "--arg-file",
            "-d",
            "--delimiter",
            "-E",
            "-I",
            "-L",
            "-n",
            "--max-args",
            "-P",
            "--max-procs",
            "-s",
            "--max-chars",
            "--process-slot-var",
        ],
    ),
];

/// The wrapper that appends arguments read from stdin to its command.
const FEEDS_ARGUMENTS: &str = "xargs";

/// Where `find -exec` and [`FEEDS_ARGUMENTS`] put an input item inside
/// their command.
const INPUT_PLACEHOLDER: &str = "{}";

/// `find` flags after which it runs a command.
const FIND_EXEC_FLAGS: &[&str] = &["-exec", "-execdir", "-ok", "-okdir"];

/// Files a redirection may write without confirmation, besides `/dev/fd/N`
/// and files under `/tmp` or the scratch directory.
const HARMLESS_TARGETS: &[&str] = &["/dev/null", "/dev/stdout", "/dev/stderr"];

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

struct Candidate {
    at: usize,
    may_be_command: bool,
}

/// Where [`candidates`] is within a command word's arguments.
#[derive(Default)]
struct Scan {
    /// The command word runs a command after this many more operands.
    pending: Option<usize>,
    /// Flags after which it runs a command (`find -exec`).
    exec_flags: Option<&'static [&'static str]>,
    /// No exec flag has been seen yet, so its arguments are its own.
    before_exec: bool,
    value_options: &'static [&'static str],
    /// The previous word was one of `value_options`.
    value_next: bool,
    /// The previous word was an option that may take this one as its value.
    after_option: bool,
}

impl Scan {
    fn enter(&mut self, word: &Word, next: Option<&Word>) {
        self.pending = wrapper_operands(word, next);
        self.value_options = if word.dynamic_name {
            &[]
        } else {
            value_options(&program(&word.text))
        };
        if let Some(flags) = exec_flags(word) {
            self.exec_flags = Some(flags);
            self.before_exec = true;
        }
    }

    /// Classify the argument `word` at `at`; `None` when it can be neither
    /// a command nor one of its arguments.
    fn classify(&mut self, at: usize, word: &Word, next: Option<&Word>) -> Option<Candidate> {
        if self
            .exec_flags
            .is_some_and(|flags| flags.contains(&word.text.as_str()))
        {
            self.before_exec = false;
            self.pending = Some(0);
            self.after_option = false;
            return None;
        }
        if self.before_exec || mem::take(&mut self.value_next) {
            return None;
        }
        if !word.quoted && word.text == "--" {
            self.pending = self.pending.map(|_| 0);
            self.after_option = false;
            return None;
        }
        let option = !word.dynamic && word.text.starts_with('-');
        let operand = !option && !self.after_option && !is_assignment(&word.text);
        let may_be_command = match self.pending {
            Some(operands) if operand && operands > 0 => {
                self.pending = Some(operands - 1);
                false
            }
            Some(_) if operand => {
                self.enter(word, next);
                true
            }
            _ => {
                self.pending.is_some_and(|operands| {
                    operands == 0 || (self.after_option && is_command_line(word))
                }) && !option
                    && !is_assignment(&word.text)
            }
        };
        self.value_next = option && self.value_options.contains(&word.text.as_str());
        self.after_option = option
            && !self.value_next
            && !word.text.contains('=')
            && (word.text.len() == 2 || word.text.starts_with("--"));
        Some(Candidate { at, may_be_command })
    }
}

/// The words of a simple command that may name a command it runs:
/// `start`, and the words after a wrapper or `find -exec`. Later words are
/// that command's arguments, still checked against patterns.
fn candidates(words: &[Word], start: usize) -> Vec<Candidate> {
    let mut out = vec![Candidate {
        at: start,
        may_be_command: true,
    }];
    let mut scan = Scan::default();
    scan.enter(&words[start], words.get(start + 1));
    if scan.pending.is_none() && scan.exec_flags.is_none() {
        return out;
    }
    for (at, word) in words.iter().enumerate().skip(start + 1) {
        out.extend(scan.classify(at, word, words.get(at + 1)));
    }
    out
}

/// A word that reads as a subcommand (`build`, `run-tests`) rather than a
/// path, pattern, or toolchain (`+nightly`).
fn is_subcommand(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_alphanumeric())
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// A literal redirection target that is a standard stream, `/dev/null`, or
/// a path under `/tmp` or `scratch` with no `..` a symlink could lead out
/// through.
fn is_harmless_target(target: &Word, scratch: Option<&str>) -> bool {
    let path = target.text.as_str();
    !target.dynamic
        && (HARMLESS_TARGETS.contains(&path)
            || path
                .strip_prefix("/dev/fd/")
                .is_some_and(|fd| !fd.is_empty() && fd.bytes().all(|b| b.is_ascii_digit()))
            || stays_under(path, "/tmp")
            || scratch.is_some_and(|scratch| stays_under(path, scratch)))
}

fn stays_under(path: &str, dir: &str) -> bool {
    if !path.starts_with('/') {
        return false;
    }
    let path: Vec<&str> = path_components(path).collect();
    let dir: Vec<&str> = path_components(dir).collect();
    path.len() > dir.len() && path.starts_with(&dir) && !path.contains(&"..")
}

fn path_components(path: &str) -> impl Iterator<Item = &str> {
    path.split('/')
        .filter(|component| !matches!(*component, "" | "."))
}

/// The first variable the script may change that the classifier cannot see
/// past: the search path, the dynamic loader's, bash's startup file, or a
/// file of options a program reads (`rg --pre`).
fn changed_sensitive_variable(script: &Script) -> Option<&str> {
    script
        .commands
        .iter()
        .flat_map(|cmd| &cmd.words)
        .flat_map(|word| changed_variables(&word.text))
        .find(|name| {
            matches!(
                *name,
                "PATH" | "BASH_ENV" | "GCONV_PATH" | "RIPGREP_CONFIG_PATH"
            ) || name.starts_with("LD_")
        })
}

/// Variables a word may assign, wherever it sits: `NAME=…`, a bare
/// `NAME` (`read NAME`, `for NAME in`, `export NAME`), a nameref target
/// (`declare -n ref=NAME`), or a `${NAME:=…}` default.
fn changed_variables(text: &str) -> impl Iterator<Item = &str> {
    let whole = shell::is_name(text).then_some(text);
    let (assigned, value) = match text.split_once('=') {
        Some((name, value)) => {
            let name = name.strip_suffix('+').unwrap_or(name);
            let name = name.split_once('[').map_or(name, |(base, _)| base);
            (
                shell::is_name(name).then_some(name),
                shell::is_name(value).then_some(value),
            )
        }
        None => (None, None),
    };
    let defaults = text.match_indices("${").filter_map(|(at, _)| {
        let rest = &text[at + 2..];
        let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))?;
        let (name, tail) = rest.split_at(end);
        (tail.starts_with(":=") || tail.starts_with('=')).then_some(name)
    });
    whole
        .into_iter()
        .chain(assigned)
        .chain(value)
        .chain(defaults)
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

/// How many operands `word` reads before the command it runs, if it is a
/// wrapper.
fn wrapper_operands(word: &Word, next: Option<&Word>) -> Option<usize> {
    if word.dynamic_name {
        return None;
    }
    let program = program(&word.text);
    let program = program.as_str();
    if program == "command" && next.is_some_and(|n| matches!(n.text.as_str(), "-v" | "-V")) {
        return None;
    }
    WRAPPERS
        .iter()
        .find(|&&(w, _)| w == program)
        .map(|&(_, operands)| operands)
}

/// A literal word holding a whole command line (`env -S 'rm -rf ~'`),
/// not a program name.
fn is_command_line(word: &Word) -> bool {
    !word.dynamic
        && word
            .text
            .contains(|c: char| c.is_whitespace() || SHELL_SYNTAX.contains(c))
}

fn exec_flags(word: &Word) -> Option<&'static [&'static str]> {
    (!word.dynamic_name && program(&word.text) == "find").then_some(FIND_EXEC_FLAGS)
}

fn value_options(program: &str) -> &'static [&'static str] {
    VALUE_OPTIONS
        .iter()
        .find(|&&(p, _)| p == program)
        .map_or(&[], |&(_, options)| options)
}

/// The first of `env`'s options with a value attached (`-Sx`, `-iS`,
/// `--split-string=x`), any of which may hide an `-S` command line.
fn env_attached_option(args: &[Word]) -> Option<&str> {
    let value_options = value_options("env");
    let mut value_next = false;
    for arg in args {
        let text = arg.text.as_str();
        if mem::take(&mut value_next) {
            continue;
        }
        if !text.starts_with('-') || text == "--" {
            return None;
        }
        if text.contains('=') || (!text.starts_with("--") && text.len() > 2) {
            return Some(text);
        }
        value_next = value_options.contains(&text);
    }
    None
}

fn is_assignment(word: &str) -> bool {
    word.split_once('=').is_some_and(|(name, _)| {
        let name = name.strip_suffix('+').unwrap_or(name);
        let name = name.split_once('[').map_or(name, |(base, _)| base);
        name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

fn is_stdin_path(path: &str) -> bool {
    matches!(path, "-" | "/dev/stdin")
        || path.starts_with("/dev/fd/")
        || path.starts_with("/proc/self/fd/")
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

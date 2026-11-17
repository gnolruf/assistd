//! Command review: the classifier, the shell-word helpers it shares, and why a script needs the
//! user's confirmation.

use std::fmt;
use std::ops::Range;
use std::path::Path;

use super::allowlist::ProgramApproval;
use super::confirm::Risk;
use super::path_class::{admits_edit, canonical_existing};
use super::shell::{self, COMMAND_PREFIX_WORDS, Script, Word};

mod classify;
mod evaluated;
mod options;
mod placeholder;
mod programs;

pub use classify::{Action, Analysis, Class, Effect, analyze, classify};

/// How many scripts deep (`eval`, `trap`, …) the classifier looks before
/// calling a script unvetted.
const MAX_NESTED_SCRIPTS: usize = 16;

/// How many wrappers deep a command may sit before what it runs is called unvetted.
const MAX_WRAPPERS: usize = 16;

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

/// `command`'s options that make it describe a command instead of running it.
const COMMAND_DESCRIBES: &[options::Flag] = &[options::Flag::Short('v'), options::Flag::Short('V')];

/// `env`'s options whose value is a command line it runs.
const COMMAND_LINE_OPTIONS: &[options::Flag] = &[
    options::Flag::Short('S'),
    options::Flag::Long("split-string"),
];

/// The wrapper that appends arguments read from stdin to its command.
const FEEDS_ARGUMENTS: &str = "xargs";

/// Where `find -exec` and [`FEEDS_ARGUMENTS`] put an input item inside
/// their command.
const INPUT_PLACEHOLDER: &str = "{}";

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

/// A word of a simple command that names a command it runs.
struct Candidate {
    at: usize,
    /// One past the last of its arguments.
    end: usize,
    /// It is a wrapper given no command, so it runs one only when something appends it.
    awaits_command: bool,
}

/// The commands a simple command runs: its own, and those its wrappers and `find -exec` run.
#[derive(Default)]
struct Commands {
    candidates: Vec<Candidate>,
    /// Why the first wrapper whose options hide what it runs does so.
    hidden: Option<String>,
}

impl Commands {
    /// Note the command that `span` of `words` is, and the commands it runs in turn.
    fn follow(&mut self, words: &[Word], span: Range<usize>, depth: usize) {
        let word = &words[span.start];
        let args = span.start + 1..span.end;
        let candidate = self.candidates.len();
        self.candidates.push(Candidate {
            at: span.start,
            end: span.end,
            awaits_command: false,
        });
        if depth > MAX_WRAPPERS {
            self.hidden
                .get_or_insert_with(|| format!("`{}` is wrapped too deeply to vet", word.text));
        } else if let Some(wrapper) = wrapper_named(word) {
            let awaits_command = self.follow_wrapper(wrapper, words, args, depth);
            self.candidates[candidate].awaits_command = awaits_command;
        } else if let Some(finder) = per_item_runner(word) {
            let walked = options::walk(finder, &words[args.clone()]);
            for command in walked.commands.iter().filter(|command| !command.is_empty()) {
                let span = args.start + command.start..args.start + command.end;
                self.follow(words, span, depth + 1);
            }
        }
    }

    /// Follow `wrapper`, given the `args` of `words`, to the command after its operands and to
    /// each command line an option gives it; whether it has neither.
    fn follow_wrapper(
        &mut self,
        wrapper: &programs::Wrapper,
        words: &[Word],
        args: Range<usize>,
        depth: usize,
    ) -> bool {
        let own = &words[args.clone()];
        let walked = options::walk(&wrapper.options, own);
        if let Some(option) = walked.unlisted {
            self.hidden.get_or_insert_with(|| {
                format!(
                    "`{} {}` is not a vetted option, so what it runs cannot be told",
                    wrapper.name, option.text
                )
            });
            return false;
        }
        if wrapper.name == "command" && walked.gives(COMMAND_DESCRIBES) {
            return false;
        }
        let mut runs = false;
        for given in &walked.options {
            if !COMMAND_LINE_OPTIONS.contains(&given.flag) {
                continue;
            }
            runs = true;
            match given.value_at {
                Some(line) => {
                    let line = args.start + line;
                    self.follow(words, line..line + 1, depth + 1);
                }
                None => {
                    self.hidden.get_or_insert_with(|| {
                        format!(
                            "`{} {}` may run a command the lists cannot see",
                            wrapper.name, given.flag
                        )
                    });
                }
            }
        }
        let command = walked
            .operands
            .iter()
            .filter(|&&at| !is_assignment(&own[at].text))
            .nth(wrapper.operands);
        if let Some(&command) = command {
            self.follow(words, args.start + command..args.end, depth + 1);
        }
        !runs && command.is_none()
    }
}

/// The commands the simple command `words` runs, its own starting at `start`.
fn commands(words: &[Word], start: usize) -> Commands {
    let mut found = Commands::default();
    found.follow(words, start..words.len(), 0);
    found
}

/// A word that reads as a subcommand (`build`, `run-tests`) rather than a
/// path, pattern, or toolchain (`+nightly`).
fn is_subcommand(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_alphanumeric())
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// A literal redirection target that is a standard stream, `/dev/null`, a path under `/tmp`
/// behind no hidden entry or `..`, or one that also stays so under the canonical `scratch`
/// once the symlinks already there are followed. Blocks on the filesystem.
fn is_harmless_target(target: &Word, scratch: Option<&str>) -> bool {
    let path = target.text.as_str();
    !target.dynamic
        && (HARMLESS_TARGETS.contains(&path)
            || path
                .strip_prefix("/dev/fd/")
                .is_some_and(|fd| !fd.is_empty() && fd.bytes().all(|b| b.is_ascii_digit()))
            || stays_under(path, "/tmp")
            || scratch.is_some_and(|scratch| {
                stays_under(path, scratch) && resolves_under(path, Path::new(scratch))
            }))
}

/// Whether `path` still lands where an edit of `dir` may, with the symlinks on disk followed.
fn resolves_under(path: &str, dir: &Path) -> bool {
    canonical_existing(Path::new(path)).is_some_and(|real| admits_edit(dir, &real))
}

/// Whether `path` is written strictly below `dir`, naming no hidden entry or `..` there.
fn stays_under(path: &str, dir: &str) -> bool {
    if !path.starts_with('/') {
        return false;
    }
    let path: Vec<&str> = path_components(path).collect();
    let dir: Vec<&str> = path_components(dir).collect();
    path.len() > dir.len()
        && path.starts_with(&dir)
        && !path[dir.len()..].iter().any(|name| name.starts_with('.'))
}

fn path_components(path: &str) -> impl Iterator<Item = &str> {
    path.split('/')
        .filter(|component| !matches!(*component, "" | "."))
}

/// The first variable the script may change that the classifier cannot see
/// past: the search path, the dynamic loader's, bash's startup file, or a
/// file of options a program reads (`rg --pre`).
fn changed_sensitive_variable(script: &Script) -> Option<&str> {
    changed_in(script).find(|name| {
        matches!(
            *name,
            "PATH" | "BASH_ENV" | "GCONV_PATH" | "RIPGREP_CONFIG_PATH"
        ) || name.starts_with("LD_")
    })
}

/// Every variable `script` may change: those its words may assign, and those a `{NAME}>`
/// redirection gives a file descriptor.
fn changed_in(script: &Script) -> impl Iterator<Item = &str> {
    script.commands.iter().flat_map(|cmd| {
        let by_words = cmd
            .words
            .iter()
            .flat_map(|word| changed_variables(&word.text));
        let by_redirections = cmd
            .fd_variables
            .iter()
            .filter_map(|variable| target_name(variable));
        by_words.chain(by_redirections)
    })
}

/// Variables a word may assign, wherever it sits: `NAME=…`, a bare `NAME` or `NAME[…]`
/// (`read NAME`, `for NAME in`, `export NAME`), one attached to a short option
/// (`printf -vNAME`), a nameref target (`declare -n ref=NAME`), or a `${NAME:=…}` default.
fn changed_variables(text: &str) -> impl Iterator<Item = &str> {
    let whole = target_name(text);
    let assigned = assigned_name(text);
    let attached = short_options(text).into_iter().flat_map(|cluster| {
        cluster
            .char_indices()
            .skip(1)
            .filter_map(|(at, _)| target_name(&cluster[at..]))
    });
    let value = text
        .split_once('=')
        .map(|(_, value)| value)
        .filter(|value| shell::is_name(value));
    let defaults = text.match_indices("${").filter_map(|(at, _)| {
        let rest = &text[at + 2..];
        let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))?;
        let (name, tail) = rest.split_at(end);
        (tail.starts_with(":=") || tail.starts_with('=')).then_some(name)
    });
    whole
        .into_iter()
        .chain(assigned)
        .chain(attached)
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

/// The wrapper `word` names when it is in command position.
fn wrapper_named(word: &Word) -> Option<&'static programs::Wrapper> {
    (!word.dynamic_name)
        .then(|| programs::wrapper(&program(&word.text)))
        .flatten()
}

/// The options of the program `word` names, when it runs a command once for each item it
/// finds, as `find -exec` does.
fn per_item_runner(word: &Word) -> Option<&'static options::OptionSpec> {
    (!word.dynamic_name)
        .then(|| programs::read_only(&program(&word.text)))
        .flatten()
        .map(|program| &program.options)
        .filter(|options| options.runs_commands())
}

/// A literal word holding a whole command line (`env -S 'rm -rf ~'`),
/// not a program name.
fn is_command_line(word: &Word) -> bool {
    !word.dynamic
        && word
            .text
            .contains(|c: char| c.is_whitespace() || SHELL_SYNTAX.contains(c))
}

/// The variable `word` assigns, as `NAME=…`, `NAME+=…` or `NAME[…]=…`.
fn assigned_name(word: &str) -> Option<&str> {
    let (name, _) = word.split_once('=')?;
    target_name(name.strip_suffix('+').unwrap_or(name))
}

/// The variable `target` names, as `NAME` or `NAME[…]`.
fn target_name(target: &str) -> Option<&str> {
    let name = target.split_once('[').map_or(target, |(base, _)| base);
    shell::is_name(name).then_some(name)
}

fn is_assignment(word: &str) -> bool {
    assigned_name(word).is_some()
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

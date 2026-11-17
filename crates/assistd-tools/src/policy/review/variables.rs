//! The variables a script may change or export, which decide what the programs it runs do.

use super::{SAFE_BUILTINS, command_start, short_options};
use crate::policy::shell::{self, Script, Word};

/// Variables a script may export without leaving it unvetted, besides `LC_*`: the locale,
/// time zone, terminal and colour settings, which run nothing.
const HARMLESS_EXPORTS: &[&str] = &["LANG", "LANGUAGE", "NO_COLOR", "TERM", "TZ"];

/// Builtins that export the names they declare when given `-x`.
const EXPORTING_DECLARATIONS: &[&str] = &["declare", "local", "readonly", "typeset"];

/// The first variable the script may change that the classifier cannot see
/// past: the search path, the dynamic loader's, bash's startup file, or a
/// file of options a program reads (`rg --pre`).
pub(super) fn changed_sensitive_variable(script: &Script) -> Option<&str> {
    changed_in(script).find(|name| {
        matches!(
            *name,
            "PATH" | "BASH_ENV" | "GCONV_PATH" | "RIPGREP_CONFIG_PATH"
        ) || name.starts_with("LD_")
    })
}

/// Every variable `script` may change: those its words may assign, and those a `{NAME}>`
/// redirection gives a file descriptor.
pub(super) fn changed_in(script: &Script) -> impl Iterator<Item = &str> {
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

/// The first `GIT_*` variable the script may change, which can make `git` run any program.
pub(super) fn changed_git_variable(script: &Script) -> Option<&str> {
    changed_in(script).find(|name| name.starts_with("GIT_"))
}

/// The first variable the script passes to the programs it runs, other than the
/// [`HARMLESS_EXPORTS`] and `LC_*`: a program may read one as a command to run (`EDITOR`,
/// `PAGER`, `CARGO_TARGET_*_RUNNER`), which no list or approval vets.
pub(super) fn exported_variable(script: &Script) -> Option<&str> {
    script
        .commands
        .iter()
        .flat_map(|cmd| exported_names(&cmd.words))
        .find(|name| !(HARMLESS_EXPORTS.contains(name) || name.starts_with("LC_")))
}

/// The names one command exports: assignments before a program or after `env`, the operands
/// of `export` or a `-x` declaration, and every variable after `set -a`.
fn exported_names(words: &[Word]) -> Vec<&str> {
    let Some(start) = command_start(words) else {
        return Vec::new();
    };
    let operands = &words[start + 1..];
    let has_option = |letter: char| {
        operands
            .iter()
            .any(|word| short_options(&word.text).is_some_and(|cluster| cluster.contains(letter)))
    };
    let runs_program = !SAFE_BUILTINS.contains(&words[start].text.as_str());
    let mut names: Vec<&str> = words[..start]
        .iter()
        .filter(|word| runs_program && is_assignment(&word.text))
        .map(exported_name)
        .collect();
    let declared = match words[start].text.as_str() {
        "export" => true,
        command => EXPORTING_DECLARATIONS.contains(&command) && has_option('x'),
    };
    if declared {
        names.extend(
            operands
                .iter()
                .filter(|word| !word.text.starts_with('-'))
                .map(exported_name),
        );
    }
    if words[start].text == "set"
        && (has_option('a') || operands.iter().any(|word| word.text == "allexport"))
    {
        names.push("all its variables");
    }
    if let Some(env) = words.iter().position(|word| word.text == "env") {
        names.extend(
            words[env + 1..]
                .iter()
                .filter(|word| is_assignment(&word.text))
                .map(exported_name),
        );
    }
    names
}

/// The variable name `word` assigns or declares.
fn exported_name(word: &Word) -> &str {
    let name = word
        .text
        .split_once('=')
        .map_or(word.text.as_str(), |(name, _)| name);
    name.strip_suffix('+').unwrap_or(name)
}

/// Whether the script may change variable `name`.
pub(super) fn changes(script: &Script, name: &str) -> bool {
    changed_in(script).any(|changed| changed == name)
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

pub(super) fn is_assignment(word: &str) -> bool {
    assigned_name(word).is_some()
}

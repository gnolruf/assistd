//! Text a wrapper replaces with run-time input before running its command:
//! `{}` after `find -exec`, and `xargs`'s replace string.

use super::{FEEDS_ARGUMENTS, INPUT_PLACEHOLDER, exec_flags, program};
use crate::policy::shell::Word;

/// GNU `xargs` short options and what each takes.
const XARGS_SHORT_OPTIONS: &[(char, Arity)] = &[
    ('0', Arity::Flag),
    ('a', Arity::Required),
    ('d', Arity::Required),
    ('E', Arity::Required),
    ('e', Arity::Optional),
    ('I', Arity::Required),
    ('i', Arity::Optional),
    ('L', Arity::Required),
    ('l', Arity::Optional),
    ('n', Arity::Required),
    ('o', Arity::Flag),
    ('P', Arity::Required),
    ('p', Arity::Flag),
    ('r', Arity::Flag),
    ('s', Arity::Required),
    ('t', Arity::Flag),
    ('x', Arity::Flag),
];

/// GNU `xargs` long options and what each takes.
const XARGS_LONG_OPTIONS: &[(&str, Arity)] = &[
    ("arg-file", Arity::Required),
    ("delimiter", Arity::Required),
    ("eof", Arity::Optional),
    ("exit", Arity::Flag),
    ("help", Arity::Flag),
    ("interactive", Arity::Flag),
    ("max-args", Arity::Required),
    ("max-chars", Arity::Required),
    ("max-lines", Arity::Optional),
    ("max-procs", Arity::Required),
    ("no-run-if-empty", Arity::Flag),
    ("null", Arity::Flag),
    ("open-tty", Arity::Flag),
    ("process-slot-var", Arity::Required),
    ("replace", Arity::Optional),
    ("show-limits", Arity::Flag),
    ("verbose", Arity::Flag),
    ("version", Arity::Flag),
];

/// The `xargs` options that set the replace string.
const XARGS_REPLACE_OPTIONS: &[&str] = &["I", "i", "replace"];

/// Text a wrapper replaces with run-time input, so a command word holding
/// it is only known at run time.
pub(super) enum Placeholder<'w> {
    Text(&'w str),
    /// The replace string is itself only known at run time.
    Unknown,
}

impl Placeholder<'_> {
    pub(super) fn fills(&self, word: &Word) -> bool {
        match self {
            Self::Text(text) => word.text.contains(text),
            Self::Unknown => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arity {
    Flag,
    /// A value attached or in the next word.
    Required,
    /// A value only when attached.
    Optional,
}

/// One option word, as getopt reads it.
struct ParsedOption<'w> {
    name: &'w str,
    arity: Arity,
    attached: Option<&'w str>,
}

/// What `word`, in command position with `args`, fills in at run time in
/// the command it runs.
pub(super) fn placeholders<'w>(word: &Word, args: &'w [Word]) -> Vec<Placeholder<'w>> {
    if exec_flags(word).is_some() {
        return vec![Placeholder::Text(INPUT_PLACEHOLDER)];
    }
    if word.dynamic_name || program(&word.text) != FEEDS_ARGUMENTS {
        return Vec::new();
    }
    let mut out = vec![Placeholder::Text(INPUT_PLACEHOLDER)];
    out.extend(xargs_replace_strings(args));
    out
}

/// The replace strings `xargs` options `args` set, read as GNU getopt
/// does: up to the first operand, `--`, or an option it rejects.
fn xargs_replace_strings(args: &[Word]) -> Vec<Placeholder<'_>> {
    let mut out = Vec::new();
    let mut words = args.iter();
    while let Some(word) = words.next() {
        if word.dynamic && word.text.starts_with('-') {
            out.push(Placeholder::Unknown);
            break;
        }
        let Some(option) = parse_xargs_option(&word.text) else {
            break;
        };
        let value = match (option.attached, option.arity) {
            (Some(attached), _) => Some((word, attached)),
            (None, Arity::Required) => words.next().map(|next| (next, next.text.as_str())),
            (None, _) => None,
        };
        if XARGS_REPLACE_OPTIONS.contains(&option.name) {
            out.push(match value {
                Some((source, _)) if source.dynamic => Placeholder::Unknown,
                Some((_, text)) => Placeholder::Text(text),
                None => Placeholder::Text(INPUT_PLACEHOLDER),
            });
        }
    }
    out
}

/// `text` as an `xargs` option word; `None` for an operand, `--`, or an
/// option `xargs` rejects.
fn parse_xargs_option(text: &str) -> Option<ParsedOption<'_>> {
    let option = text.strip_prefix('-').filter(|o| !o.is_empty())?;
    match option.strip_prefix('-') {
        Some(long) => parse_long_option(long),
        None => parse_short_cluster(option),
    }
}

/// A long option, which may be abbreviated to any prefix naming only it.
fn parse_long_option(long: &str) -> Option<ParsedOption<'_>> {
    let (given, attached) = long
        .split_once('=')
        .map_or((long, None), |(name, value)| (name, Some(value)));
    if given.is_empty() {
        return None;
    }
    let exact = XARGS_LONG_OPTIONS.iter().find(|&&(name, _)| name == given);
    let mut prefixed = XARGS_LONG_OPTIONS
        .iter()
        .filter(|&&(name, _)| name.starts_with(given));
    let &(name, arity) = exact.or_else(|| prefixed.next().filter(|_| prefixed.next().is_none()))?;
    (attached.is_none() || arity != Arity::Flag).then_some(ParsedOption {
        name,
        arity,
        attached,
    })
}

/// A cluster of short flags, ending at the first option that takes a
/// value, which takes the rest of the cluster.
fn parse_short_cluster(cluster: &str) -> Option<ParsedOption<'_>> {
    for (at, flag) in cluster.char_indices() {
        let end = at + flag.len_utf8();
        let &(_, arity) = XARGS_SHORT_OPTIONS.iter().find(|&&(c, _)| c == flag)?;
        if arity != Arity::Flag || end == cluster.len() {
            let rest = &cluster[end..];
            return Some(ParsedOption {
                name: &cluster[at..end],
                arity,
                attached: (!rest.is_empty()).then_some(rest),
            });
        }
    }
    None
}

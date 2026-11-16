//! One reading of a program's arguments against the options listed for it: which words are
//! options, values, operands or a command it runs, and which option the list lacks.

use std::fmt;
use std::ops::Range;

use super::INPUT_PLACEHOLDER;
use crate::policy::shell::Word;

/// The word that ends the command a [`Takes::Command`] primary runs.
const COMMAND_END: &str = ";";

/// The word that also ends it when it follows [`INPUT_PLACEHOLDER`].
const BATCH_END: &str = "+";

/// An option, as the program tables name one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Flag {
    /// A short option, alone or in a cluster.
    Short(char),
    /// A GNU long option, spelled in full.
    Long(&'static str),
    /// A word taken whole, as a `find` primary is.
    Primary(&'static str),
}

impl fmt::Display for Flag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Short(option) => write!(f, "-{option}"),
            Self::Long(name) => write!(f, "--{name}"),
            Self::Primary(word) => f.write_str(word),
        }
    }
}

/// What follows a primary.
#[derive(Debug, Clone, Copy)]
pub(super) enum Takes {
    Nothing,
    /// The next word, as its value.
    Value,
    /// A command and its arguments, up to the word that ends them.
    Command,
}

/// Where a program stops taking words as options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OptionsEnd {
    /// At `--`, so an option may follow an operand.
    Terminator,
    /// At `--` or its first operand.
    FirstOperand,
    /// Nowhere, and `--` is not an option.
    Never,
}

/// The options a program takes. A long option must be spelled in full: an abbreviation counts
/// as an option that is not listed.
#[derive(Debug)]
pub(super) struct OptionSpec {
    /// Short options that take no value.
    pub flags: &'static str,
    /// Short options whose value is the rest of their word, or else the next word.
    pub values: &'static str,
    /// Short options whose value is the rest of their word, when there is one.
    pub optional: &'static str,
    /// Long options that take no value.
    pub long_flags: &'static [&'static str],
    /// Long options whose value follows `=`, or else is the next word.
    pub long_values: &'static [&'static str],
    /// Long options whose value follows `=`, when there is one.
    pub long_optional: &'static [&'static str],
    /// Words taken whole, each with what follows it.
    pub primaries: &'static [(&'static str, Takes)],
    pub end: OptionsEnd,
}

impl OptionSpec {
    /// No options, read up to `--`.
    pub(super) const NONE: Self = Self {
        flags: "",
        values: "",
        optional: "",
        long_flags: &[],
        long_values: &[],
        long_optional: &[],
        primaries: &[],
        end: OptionsEnd::Terminator,
    };

    /// No options, read up to the first operand.
    pub(super) const LEADING: Self = Self {
        end: OptionsEnd::FirstOperand,
        ..Self::NONE
    };

    /// Whether one of its primaries runs a command.
    pub(super) fn runs_commands(&self) -> bool {
        self.primaries
            .iter()
            .any(|(_, takes)| matches!(takes, Takes::Command))
    }
}

/// A listed option found among a program's arguments.
#[derive(Debug)]
pub(super) struct Given {
    pub flag: Flag,
    /// Its value, when it was given one: the rest of its own word, or the word after it.
    pub value: Option<Word>,
    /// Where in the arguments its value is, when that is the word after it.
    pub value_at: Option<usize>,
}

/// A program's arguments as its [`OptionSpec`] reads them.
#[derive(Debug, Default)]
pub(super) struct Walked<'w> {
    pub options: Vec<Given>,
    /// Where in the arguments its operands are.
    pub operands: Vec<usize>,
    /// Where in the arguments the commands its primaries run are.
    pub commands: Vec<Range<usize>>,
    /// The first option its spec does not list; the words after it are read as though it took
    /// no value.
    pub unlisted: Option<&'w Word>,
    /// The arguments before this one were read where an option may be given.
    pub options_end: usize,
}

impl Walked<'_> {
    /// Whether any of `flags` was given.
    pub(super) fn gives(&self, flags: &[Flag]) -> bool {
        self.options.iter().any(|given| flags.contains(&given.flag))
    }

    /// The values given to any of `flags`.
    pub(super) fn values_of<'a>(&'a self, flags: &'a [Flag]) -> impl Iterator<Item = &'a Word> {
        self.options
            .iter()
            .filter(|given| flags.contains(&given.flag))
            .filter_map(|given| given.value.as_ref())
    }
}

struct Walk<'s, 'w> {
    spec: &'s OptionSpec,
    args: &'w [Word],
    /// The next argument to read.
    next: usize,
    found: Walked<'w>,
}

impl<'w> Walk<'_, 'w> {
    /// Read the argument `word` at `at`, with the words it takes after it.
    fn read(&mut self, at: usize, word: &'w Word) {
        let text = word.text.as_str();
        let primary = self.spec.primaries.iter().find(|(name, _)| *name == text);
        if let Some(&(name, takes)) = primary {
            self.primary(name, takes);
        } else if text == "--" && self.spec.end != OptionsEnd::Never {
            self.end_options(at, at + 1);
        } else if let Some(long) = text.strip_prefix("--") {
            self.long(word, long);
        } else if let Some(cluster) = text.strip_prefix('-').filter(|rest| !rest.is_empty()) {
            self.cluster(word, cluster);
        } else {
            self.found.operands.push(at);
            if self.spec.end == OptionsEnd::FirstOperand {
                self.end_options(at + 1, at + 1);
            }
        }
    }

    /// Stop taking options at `end`, reading every argument from `operands` on as an operand.
    fn end_options(&mut self, end: usize, operands: usize) {
        self.found.options_end = end;
        self.found.operands.extend(operands..self.args.len());
        self.next = self.args.len();
    }

    fn primary(&mut self, name: &'static str, takes: Takes) {
        match takes {
            Takes::Nothing => self.give(Flag::Primary(name), None, false),
            Takes::Value => self.give(Flag::Primary(name), None, true),
            Takes::Command => {
                let end = self.command_end();
                self.found.commands.push(self.next..end);
                self.next = end + 1;
            }
        }
    }

    /// Where the command starting at the next argument ends.
    fn command_end(&self) -> usize {
        let ends = |at: usize| {
            let text = self.args[at].text.as_str();
            text == COMMAND_END
                || (text == BATCH_END
                    && at > self.next
                    && self.args[at - 1].text == INPUT_PLACEHOLDER)
        };
        (self.next..self.args.len())
            .find(|&at| ends(at))
            .unwrap_or(self.args.len())
    }

    /// Read `long`, the text after the `--` of `word`.
    fn long(&mut self, word: &'w Word, long: &str) {
        let (name, attached) = match long.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (long, None),
        };
        let spec = self.spec;
        let listed =
            |names: &'static [&'static str]| names.iter().copied().find(|listed| *listed == name);
        let value = attached.map(|value| word.part(value));
        if let Some(name) = listed(spec.long_flags).filter(|_| attached.is_none()) {
            self.give(Flag::Long(name), None, false);
        } else if let Some(name) = listed(spec.long_values) {
            self.give(Flag::Long(name), value, true);
        } else if let Some(name) = listed(spec.long_optional) {
            self.give(Flag::Long(name), value, false);
        } else {
            self.found.unlisted.get_or_insert(word);
        }
    }

    /// Read `cluster`, the short options after the `-` of `word`, up to the first that takes
    /// the rest as its value.
    fn cluster(&mut self, word: &'w Word, cluster: &str) {
        let spec = self.spec;
        for (at, option) in cluster.char_indices() {
            if spec.flags.contains(option) {
                self.give(Flag::Short(option), None, false);
                continue;
            }
            let rest = &cluster[at + option.len_utf8()..];
            let takes_next = spec.values.contains(option);
            if takes_next || spec.optional.contains(option) {
                let attached = (!rest.is_empty()).then(|| word.part(rest));
                self.give(Flag::Short(option), attached, takes_next);
            } else {
                self.found.unlisted.get_or_insert(word);
            }
            return;
        }
    }

    /// Note `flag` as given with its `attached` value, or with the next word when it has none
    /// and `takes_next`.
    fn give(&mut self, flag: Flag, attached: Option<Word>, takes_next: bool) {
        let value_at =
            (attached.is_none() && takes_next && self.next < self.args.len()).then_some(self.next);
        if value_at.is_some() {
            self.next += 1;
        }
        let value = attached.or_else(|| value_at.map(|at| self.args[at].clone()));
        self.found.options.push(Given {
            flag,
            value,
            value_at,
        });
    }
}

/// Read `args` as a program that takes the options in `spec` does.
pub(super) fn walk<'w>(spec: &OptionSpec, args: &'w [Word]) -> Walked<'w> {
    let mut walk = Walk {
        spec,
        args,
        next: 0,
        found: Walked {
            options_end: args.len(),
            ..Walked::default()
        },
    };
    while let Some(word) = args.get(walk.next) {
        let at = walk.next;
        walk.next += 1;
        walk.read(at, word);
    }
    walk.found
}

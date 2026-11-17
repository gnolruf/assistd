//! Wrappers, which run the command their arguments name: every command one simple command
//! runs, found by following them and `find -exec`.

use std::ops::Range;

use super::variables::is_assignment;
use super::{options, program, programs};
use crate::policy::shell::Word;

/// How many wrappers deep a command may sit before what it runs is called unvetted.
const MAX_WRAPPERS: usize = 16;

/// Wrappers that are reserved words, so quoting one runs a program of that name instead.
const WRAPPER_KEYWORDS: &[&str] = &["coproc", "time"];

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

/// A word of a simple command that names a command it runs.
pub(super) struct Candidate {
    pub(super) at: usize,
    /// One past the last of its arguments.
    pub(super) end: usize,
    /// It is a wrapper given no command, so it runs one only when something appends it.
    pub(super) awaits_command: bool,
}

/// The commands a simple command runs: its own, and those its wrappers and `find -exec` run.
#[derive(Default)]
pub(super) struct Commands {
    pub(super) candidates: Vec<Candidate>,
    /// Why the first wrapper whose options hide what it runs does so.
    pub(super) hidden: Option<String>,
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
pub(super) fn commands(words: &[Word], start: usize) -> Commands {
    let mut found = Commands::default();
    found.follow(words, start..words.len(), 0);
    found
}

/// The wrapper `word` names when it is in command position.
pub(super) fn wrapper_named(word: &Word) -> Option<&'static programs::Wrapper> {
    (!word.dynamic_name)
        .then(|| programs::wrapper(&program(&word.text)))
        .flatten()
}

/// The options of the program `word` names, when it runs a command once for each item it
/// finds, as `find -exec` does.
pub(super) fn per_item_runner(word: &Word) -> Option<&'static options::OptionSpec> {
    (!word.dynamic_name)
        .then(|| programs::read_only(&program(&word.text)))
        .flatten()
        .map(|program| &program.options)
        .filter(|options| options.runs_commands())
}

/// The wrapper `word` runs as: a reserved-word wrapper only when unquoted.
pub(super) fn run_as_wrapper(word: &Word) -> Option<&'static programs::Wrapper> {
    let name = word.text.as_str();
    programs::wrapper(name).filter(|_| !(word.quoted && WRAPPER_KEYWORDS.contains(&name)))
}

/// A literal word holding a whole command line (`env -S 'rm -rf ~'`),
/// not a program name.
pub(super) fn is_command_line(word: &Word) -> bool {
    !word.dynamic
        && word
            .text
            .contains(|c: char| c.is_whitespace() || SHELL_SYNTAX.contains(c))
}

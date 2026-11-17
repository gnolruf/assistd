//! What a wrapper takes from run-time input: the text it replaces before running its command
//! (`{}` after `find -exec`, `xargs`'s replace string), and the file `xargs` reads items from.

use super::options::{self, Flag};
use super::{FEEDS_ARGUMENTS, INPUT_PLACEHOLDER, per_item_runner, wrapper_named};
use crate::policy::shell::Word;

/// The `xargs` options that set the replace string.
const XARGS_REPLACE_OPTIONS: &[Flag] = &[Flag::Short('I'), Flag::Short('i'), Flag::Long("replace")];

/// The `xargs` options naming a file it reads its items from, in place of stdin.
pub(super) const XARGS_ARG_FILE_OPTIONS: &[Flag] = &[Flag::Short('a'), Flag::Long("arg-file")];

/// Text a wrapper replaces with run-time input, so a command word holding
/// it is only known at run time.
pub(super) enum Placeholder {
    Text(String),
    /// The replace string is itself only known at run time.
    Unknown,
}

impl Placeholder {
    pub(super) fn fills(&self, word: &Word) -> bool {
        match self {
            Self::Text(text) => word.text.contains(text.as_str()),
            Self::Unknown => true,
        }
    }
}

/// What `word`, in command position with `args`, fills in at run time in
/// the command it runs.
pub(super) fn placeholders(word: &Word, args: &[Word]) -> Vec<Placeholder> {
    let input = Placeholder::Text(INPUT_PLACEHOLDER.to_string());
    match wrapper_named(word).filter(|wrapper| wrapper.name == FEEDS_ARGUMENTS) {
        Some(xargs) => {
            let mut out = vec![input];
            out.extend(replace_strings(&options::walk(&xargs.options, args)));
            out
        }
        None if per_item_runner(word).is_some() => vec![input],
        None => Vec::new(),
    }
}

/// The replace strings the options `xargs` was given set.
fn replace_strings(walked: &options::Walked<'_>) -> Vec<Placeholder> {
    let unlisted = walked.unlisted.map(|_| Placeholder::Unknown);
    walked
        .options
        .iter()
        .filter(|given| XARGS_REPLACE_OPTIONS.contains(&given.flag))
        .map(|given| match &given.value {
            Some(value) if value.dynamic => Placeholder::Unknown,
            Some(value) => Placeholder::Text(value.text.clone()),
            None => Placeholder::Text(INPUT_PLACEHOLDER.to_string()),
        })
        .chain(unlisted)
        .collect()
}

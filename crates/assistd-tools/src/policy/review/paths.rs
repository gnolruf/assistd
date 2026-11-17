//! Paths as a script writes them: which are known before it runs, how each is spelled, and
//! which a redirection may write without confirmation.

use std::path::Path;

use crate::policy::path_class::{admits_edit, canonical_existing};
use crate::policy::shell::Word;

/// Characters after which the text of a word is only known at run time.
const DYNAMIC_CHARS: &[char] = &['$', '`', '*', '?', '[', '{', '('];

/// The spellings of `$HOME` a path may start with.
const HOME_VARIABLES: &[&str] = &["${HOME}", "$HOME"];

/// Characters that start an expansion whose value may name any path, `..` and absolute paths
/// included; a glob only matches existing entries below its directory.
pub(super) const ANY_PATH_CHARS: &[char] = &['$', '`', '{'];

/// Files a redirection may write without confirmation, besides `/dev/fd/N`
/// and files under `/tmp` or the scratch directory.
const HARMLESS_TARGETS: &[&str] = &["/dev/null", "/dev/stdout", "/dev/stderr"];

/// A literal redirection target that is a standard stream, `/dev/null`, a path under `/tmp`
/// behind no hidden entry or `..`, or one that also stays so under the canonical `scratch`
/// once the symlinks already there are followed. Blocks on the filesystem.
pub(super) fn is_harmless_target(target: &Word, scratch: Option<&str>) -> bool {
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

pub(super) fn is_stdin_path(path: &str) -> bool {
    matches!(path, "-" | "/dev/stdin")
        || path.starts_with("/dev/fd/")
        || path.starts_with("/proc/self/fd/")
}

/// Whether `path` resolves against a working directory.
pub(super) fn is_relative(path: &str) -> bool {
    !path.starts_with(['/', '~'])
}

/// `word` as a path known before it runs, spelled as [`spelled`] does; `None` when it is only
/// known at run time.
pub(super) fn static_path(word: &Word, home_changed: bool) -> Option<String> {
    if !word.dynamic {
        return Some(spelled(word, home_changed));
    }
    home_as_tilde(&word.text, home_changed).filter(|path| !path.contains(DYNAMIC_CHARS))
}

/// `word`'s text with a leading `$HOME` spelled `~` when it may be, and a leading `~` that
/// bash leaves as written spelled `./~`.
pub(super) fn spelled(word: &Word, home_changed: bool) -> String {
    if word.literal_tilde {
        return format!("./{}", word.text);
    }
    word.dynamic
        .then(|| home_as_tilde(&word.text, home_changed))
        .flatten()
        .unwrap_or_else(|| word.text.clone())
}

fn home_as_tilde(text: &str, home_changed: bool) -> Option<String> {
    if home_changed {
        return None;
    }
    HOME_VARIABLES
        .iter()
        .find_map(|variable| text.strip_prefix(variable))
        .filter(|rest| rest.is_empty() || rest.starts_with('/'))
        .map(|rest| format!("~{rest}"))
}

/// The directory a path only known at run time lies in, from its text before the first part
/// only known at run time.
pub(super) fn static_prefix(text: &str) -> String {
    let known = text.find(DYNAMIC_CHARS).map_or(text, |end| &text[..end]);
    match known.rfind('/') {
        Some(0) => "/".to_string(),
        Some(end) => known[..end].to_string(),
        None => ".".to_string(),
    }
}

/// The directory `text` lists every entry of, as `dir/*` or `dir/{*,.*}` do.
pub(super) fn every_entry_of(text: &str) -> Option<String> {
    let (dir, name) = text.rsplit_once('/').unwrap_or((".", text));
    let selects_all = |pattern: &str| !pattern.is_empty() && pattern.chars().all(|c| c == '*');
    let all = selects_all(name)
        || name
            .strip_prefix('{')
            .and_then(|list| list.strip_suffix('}'))
            .is_some_and(|list| list.split(',').any(selects_all));
    let dir = if dir.is_empty() { "/" } else { dir };
    (all && !dir.contains(DYNAMIC_CHARS)).then(|| dir.to_string())
}

/// `name` inside `dir`, with one `/` between them.
pub(super) fn joined_path(dir: &str, name: &str) -> String {
    format!("{}/{name}", dir.trim_end_matches('/'))
}

/// The relative `path` below `dir`, or as written below the directory the script started in.
pub(super) fn below(dir: Option<&str>, path: &str) -> String {
    dir.map_or_else(|| path.to_string(), |dir| joined_path(dir, path))
}

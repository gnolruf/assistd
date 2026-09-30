//! Word expansion between parsing and dispatch: `~` to `$HOME` and globs
//! to the files they name.

use std::fs;
use std::path::{Path, PathBuf};

use assistd_utils::path::expand_tilde_from_env;
use glob::{MatchOptions, Pattern};
use thiserror::Error;
use tokio::task::JoinError;

use super::Word;

/// Most paths a single glob may expand to.
pub const MAX_GLOB_MATCHES: usize = 1_000;

/// Most directory entries a single glob may read while expanding.
pub const MAX_GLOB_ENTRIES: usize = 100_000;

/// Bash's glob defaults: a leading dot must be matched literally, so `cat *`
/// never pulls hidden files into the model's context.
const MATCH_OPTIONS: MatchOptions = MatchOptions {
    case_sensitive: true,
    require_literal_separator: false,
    require_literal_leading_dot: true,
};

/// A glob that outgrew its budget, or an expansion that never finished.
#[derive(Debug, Error)]
pub enum ExpandError {
    #[error("{pattern}: glob matches more than {MAX_GLOB_MATCHES} paths")]
    TooManyMatches { pattern: String },
    #[error("{pattern}: glob reads more than {MAX_GLOB_ENTRIES} directory entries")]
    TooManyEntries { pattern: String },
    #[error("glob expansion stopped before finishing: {0}")]
    Interrupted(#[from] JoinError),
}

/// One `/`-separated piece of a glob pattern.
enum Component {
    Literal(String),
    Glob(Pattern),
}

/// A glob expanded one component at a time, counting every directory entry
/// it reads against [`MAX_GLOB_ENTRIES`].
struct GlobWalk<'a> {
    pattern: &'a str,
    entries_read: usize,
}

impl GlobWalk<'_> {
    fn descend(
        &mut self,
        parents: &[PathBuf],
        component: &Component,
    ) -> Result<Vec<PathBuf>, ExpandError> {
        let mut children = Vec::new();
        for parent in parents {
            match component {
                Component::Literal(name) => {
                    let child = parent.join(name);
                    if fs::symlink_metadata(&child).is_ok() {
                        children.push(child);
                    }
                }
                Component::Glob(pattern) => {
                    children.extend(self.matching_children(parent, pattern)?);
                }
            }
        }
        Ok(children)
    }

    fn matching_children(
        &mut self,
        parent: &Path,
        pattern: &Pattern,
    ) -> Result<Vec<PathBuf>, ExpandError> {
        let dir = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        let Ok(entries) = fs::read_dir(dir) else {
            return Ok(Vec::new());
        };
        let mut names = Vec::new();
        for entry in entries.flatten() {
            self.entries_read += 1;
            if self.entries_read > MAX_GLOB_ENTRIES {
                return Err(ExpandError::TooManyEntries {
                    pattern: self.pattern.to_owned(),
                });
            }
            if let Ok(name) = entry.file_name().into_string()
                && pattern.matches_with(&name, MATCH_OPTIONS)
            {
                names.push(name);
            }
        }
        names.sort_unstable();
        Ok(names.iter().map(|name| parent.join(name)).collect())
    }
}

/// Expand words into the argument list a command receives, reading
/// directories on the blocking pool. Only unquoted words expand, and only
/// `~` and globs; an unmatched glob stays as written.
///
/// # Errors
/// A glob past [`MAX_GLOB_MATCHES`] or [`MAX_GLOB_ENTRIES`] fails the whole
/// expansion.
pub async fn expand_args(words: &[Word]) -> Result<Vec<String>, ExpandError> {
    let words = words.to_vec();
    tokio::task::spawn_blocking(move || expand_words(&words)).await?
}

fn expand_words(words: &[Word]) -> Result<Vec<String>, ExpandError> {
    let mut args = Vec::with_capacity(words.len());
    for word in words {
        args.extend(expand_word(word)?);
    }
    Ok(args)
}

fn expand_word(word: &Word) -> Result<Vec<String>, ExpandError> {
    if word.quoted {
        return Ok(vec![word.text.clone()]);
    }
    let tilde = expand_tilde_from_env(&word.text)
        .to_string_lossy()
        .into_owned();
    Ok(expand_glob(&tilde)?.unwrap_or_else(|| vec![tilde]))
}

fn expand_glob(pattern: &str) -> Result<Option<Vec<String>>, ExpandError> {
    if !pattern.contains(['*', '?', '[']) {
        return Ok(None);
    }
    let Some(components) = parse_components(pattern) else {
        return Ok(None);
    };
    let mut walk = GlobWalk {
        pattern,
        entries_read: 0,
    };
    let root = if pattern.starts_with('/') {
        PathBuf::from("/")
    } else {
        PathBuf::new()
    };
    let mut paths = vec![root];
    for component in &components {
        paths = walk.descend(&paths, component)?;
    }
    if pattern.ends_with('/') {
        paths.retain(|path| path.is_dir());
    }
    if paths.len() > MAX_GLOB_MATCHES {
        return Err(ExpandError::TooManyMatches {
            pattern: pattern.to_owned(),
        });
    }
    let matches: Vec<String> = paths
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    Ok((!matches.is_empty()).then_some(matches))
}

fn parse_components(pattern: &str) -> Option<Vec<Component>> {
    pattern
        .split('/')
        .filter(|piece| !piece.is_empty())
        .map(|piece| {
            if piece.contains(['*', '?', '[']) {
                Pattern::new(&collapse_stars(piece))
                    .ok()
                    .map(Component::Glob)
            } else {
                Some(Component::Literal(piece.to_owned()))
            }
        })
        .collect()
}

fn collapse_stars(piece: &str) -> String {
    let mut collapsed = String::with_capacity(piece.len());
    for ch in piece.chars() {
        if !(ch == '*' && collapsed.ends_with('*')) {
            collapsed.push(ch);
        }
    }
    collapsed
}

#[cfg(test)]
mod tests;

//! Home-directory expansion, lexical cleaning and symlink resolution for user-supplied paths.

use std::fs;
use std::path::{Component, Path, PathBuf};

/// What follows the tilde when `raw` is exactly `~` or starts with `~/`;
/// `None` for any other spelling, including `~user`.
pub fn tilde_remainder(raw: &str) -> Option<&str> {
    match raw.strip_prefix('~')? {
        "" => Some(""),
        rest => rest.strip_prefix('/'),
    }
}

/// `~` and `~/rest` resolved against `home`; any other `raw` is returned as
/// written.
pub fn expand_tilde(raw: &str, home: &Path) -> PathBuf {
    match tilde_remainder(raw) {
        Some("") => home.to_path_buf(),
        Some(rest) => home.join(rest),
        None => PathBuf::from(raw),
    }
}

/// [`expand_tilde`] against `$HOME`; `raw` is returned as written when it
/// is unset.
pub fn expand_tilde_from_env(raw: &str) -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) => expand_tilde(raw, Path::new(&home)),
        None => PathBuf::from(raw),
    }
}

/// Collapse `.` and `..` components without touching disk. A leading
/// `/..` is discarded, as the kernel does.
pub fn lexical_clean(path: &Path) -> PathBuf {
    let mut out: Vec<Component<'_>> = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match out.last() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir) => {}
                _ => out.push(component),
            },
            _ => out.push(component),
        }
    }
    let mut result = PathBuf::new();
    for component in out {
        result.push(component.as_os_str());
    }
    result
}

/// The deepest existing ancestor of `path`, or `path` itself, and the rest of `path` below it.
pub fn split_at_existing(path: &Path) -> (PathBuf, PathBuf) {
    let mut anchor = path.to_path_buf();
    let mut tail = PathBuf::new();
    loop {
        if anchor.symlink_metadata().is_ok() {
            return (anchor, tail);
        }
        let Some(parent) = anchor.parent() else {
            return (path.to_path_buf(), PathBuf::new());
        };
        let Some(file_name) = anchor.file_name() else {
            return (anchor, tail);
        };
        let mut new_tail = PathBuf::from(file_name);
        if !tail.as_os_str().is_empty() {
            new_tail.push(&tail);
        }
        tail = new_tail;
        anchor = parent.to_path_buf();
    }
}

/// `path` with its deepest existing ancestor canonicalized and the rest cleaned
/// lexically; `None` when that ancestor cannot be resolved.
pub fn canonical_existing(path: &Path) -> Option<PathBuf> {
    let (anchor, tail) = split_at_existing(path);
    let anchor = fs::canonicalize(anchor).ok()?;
    Some(lexical_clean(&anchor.join(tail)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tilde_remainder_accepts_only_bare_and_slash_forms() {
        assert_eq!(tilde_remainder("~"), Some(""));
        assert_eq!(tilde_remainder("~/docs/a.txt"), Some("docs/a.txt"));
        assert_eq!(tilde_remainder("~alice/x"), None);
        assert_eq!(tilde_remainder("/tmp/~"), None);
        assert_eq!(tilde_remainder("plain"), None);
    }
}

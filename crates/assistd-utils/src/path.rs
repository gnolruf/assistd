//! Home-directory expansion for user-supplied paths.

use std::path::{Path, PathBuf};

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

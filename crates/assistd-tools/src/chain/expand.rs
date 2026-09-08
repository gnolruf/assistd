//! Word expansion: the step between parsing and dispatch that turns
//! `~/notes` into an absolute path and `*.toml` into the files it names.
//!
//! Only unquoted words expand, and only tilde and globbing are
//! implemented — `$VAR`, command substitution and brace expansion stay
//! the `bash` command's job. An unmatched glob is left as written, which
//! is bash's default and keeps a pattern that names nothing from
//! silently vanishing from the argv.

use glob::{MatchOptions, glob_with};

use super::Word;

/// Glob behaviour matched to bash defaults: `*` stops at `/` only when
/// the pattern says so, and never swallows a leading dot, so `cat *`
/// can't drag hidden files into the model's context.
const MATCH_OPTIONS: MatchOptions = MatchOptions {
    case_sensitive: true,
    require_literal_separator: false,
    require_literal_leading_dot: true,
};

/// Expand every word into the flat argument list a command receives.
/// One word can yield several arguments (a matching glob) or exactly one
/// (everything else).
pub fn expand_args(words: &[Word]) -> Vec<String> {
    words.iter().flat_map(expand_word).collect()
}

fn expand_word(word: &Word) -> Vec<String> {
    if word.quoted {
        return vec![word.text.clone()];
    }
    let tilde = expand_tilde(&word.text);
    expand_glob(&tilde).unwrap_or_else(|| vec![tilde])
}

fn expand_tilde(text: &str) -> String {
    let Some(rest) = text.strip_prefix('~') else {
        return text.to_string();
    };
    if !(rest.is_empty() || rest.starts_with('/')) {
        return text.to_string();
    }
    match std::env::var("HOME") {
        Ok(home) => format!("{home}{rest}"),
        Err(_) => text.to_string(),
    }
}

fn expand_glob(pattern: &str) -> Option<Vec<String>> {
    if !pattern.contains(['*', '?', '[']) {
        return None;
    }
    let matches: Vec<String> = glob_with(pattern, MATCH_OPTIONS)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.to_string_lossy().into_owned()))
        .collect();
    (!matches.is_empty()).then_some(matches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::{TempDir, tempdir};

    fn fixture() -> TempDir {
        let dir = tempdir().expect("tempdir");
        for name in ["alpha.toml", "beta.toml", "gamma.txt", ".hidden.toml"] {
            std::fs::write(dir.path().join(name), b"x").expect("write fixture");
        }
        dir
    }

    fn joined(dir: &TempDir, name: &str) -> String {
        dir.path().join(name).to_string_lossy().into_owned()
    }

    #[test]
    fn plain_word_passes_through() {
        assert_eq!(expand_args(&[Word::bare("cat")]), vec!["cat".to_string()]);
    }

    #[test]
    fn glob_expands_to_sorted_matches() {
        let dir = fixture();
        let out = expand_args(&[Word::bare(joined(&dir, "*.toml"))]);
        assert_eq!(
            out,
            vec![joined(&dir, "alpha.toml"), joined(&dir, "beta.toml")]
        );
    }

    #[test]
    fn glob_skips_hidden_entries() {
        let dir = fixture();
        let out = expand_args(&[Word::bare(joined(&dir, "*"))]);
        assert!(
            !out.iter().any(|p| p.ends_with(".hidden.toml")),
            "hidden file leaked into {out:?}"
        );
    }

    #[test]
    fn quoted_glob_is_literal() {
        let dir = fixture();
        let pattern = joined(&dir, "*.toml");
        assert_eq!(expand_args(&[Word::quoted(&pattern)]), vec![pattern]);
    }

    #[test]
    fn unmatched_glob_stays_literal() {
        let dir = fixture();
        let pattern = joined(&dir, "*.rs");
        assert_eq!(expand_args(&[Word::bare(&pattern)]), vec![pattern]);
    }

    #[test]
    fn regex_metachars_survive_when_nothing_matches() {
        // A grep pattern is a bare word too; it must reach the command
        // untouched whenever it doesn't happen to name files.
        assert_eq!(
            expand_args(&[Word::bare(".*ERROR")]),
            vec![".*ERROR".to_string()]
        );
    }

    #[test]
    fn tilde_expands_against_home() {
        let home = std::env::var("HOME").expect("HOME set in test env");
        assert_eq!(
            expand_args(&[Word::bare("~/notes.md")]),
            vec![
                PathBuf::from(home)
                    .join("notes.md")
                    .to_string_lossy()
                    .into_owned()
            ]
        );
    }

    #[test]
    fn bare_tilde_is_home() {
        let home = std::env::var("HOME").expect("HOME set in test env");
        assert_eq!(expand_args(&[Word::bare("~")]), vec![home]);
    }

    #[test]
    fn quoted_tilde_is_literal() {
        assert_eq!(
            expand_args(&[Word::quoted("~/notes.md")]),
            vec!["~/notes.md".to_string()]
        );
    }

    #[test]
    fn named_user_tilde_is_left_alone() {
        // `~alice` needs a passwd lookup we don't do; leaving it literal
        // surfaces a clear "file not found: ~alice/x" instead of a path
        // silently pointing at the wrong home.
        assert_eq!(
            expand_args(&[Word::bare("~alice/x")]),
            vec!["~alice/x".to_string()]
        );
    }
}

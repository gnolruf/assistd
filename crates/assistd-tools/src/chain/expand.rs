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
    fn glob_expands_to_sorted_visible_matches() {
        let dir = fixture();
        assert_eq!(
            expand_args(&[Word::bare(joined(&dir, "*.toml"))]),
            [joined(&dir, "alpha.toml"), joined(&dir, "beta.toml")]
        );
        assert_eq!(
            expand_args(&[Word::bare(joined(&dir, "*"))]),
            [
                joined(&dir, "alpha.toml"),
                joined(&dir, "beta.toml"),
                joined(&dir, "gamma.txt")
            ]
        );
    }

    #[test]
    fn words_that_do_not_expand_pass_through_verbatim() {
        let dir = fixture();
        for (case, word) in [
            ("plain word", Word::bare("cat")),
            ("quoted glob", Word::quoted(joined(&dir, "*.toml"))),
            ("unmatched glob", Word::bare(joined(&dir, "*.rs"))),
            // A grep pattern is a bare word too; it must reach the
            // command untouched whenever it doesn't happen to name files.
            ("regex matching no file", Word::bare(".*ERROR")),
            ("quoted tilde", Word::quoted("~/notes.md")),
            // `~alice` needs a passwd lookup we don't do; leaving it
            // literal surfaces "file not found: ~alice/x" instead of a
            // path silently pointing at the wrong home.
            ("named-user tilde", Word::bare("~alice/x")),
        ] {
            assert_eq!(
                expand_args(std::slice::from_ref(&word)),
                [word.text.as_str()],
                "{case}"
            );
        }
    }

    #[test]
    fn tilde_expands_against_home() {
        let home = std::env::var("HOME").expect("HOME set in test env");
        assert_eq!(
            expand_args(&[Word::bare("~/notes.md"), Word::bare("~")]),
            [format!("{home}/notes.md"), home]
        );
    }
}

use tempfile::{TempDir, tempdir};

use super::*;

fn fixture() -> TempDir {
    let dir = tempdir().expect("tempdir");
    for name in ["alpha.toml", "beta.toml", "gamma.txt", ".hidden.toml"] {
        fs::write(dir.path().join(name), b"x").expect("write fixture");
    }
    fs::create_dir(dir.path().join("sub")).expect("create sub");
    fs::write(dir.path().join("sub/deep.toml"), b"x").expect("write deep");
    dir
}

fn joined(dir: &TempDir, name: &str) -> String {
    dir.path().join(name).to_string_lossy().into_owned()
}

fn expanded(word: Word) -> Vec<String> {
    expand_words(&[word]).expect("expansion within budget")
}

#[test]
fn glob_expands_to_sorted_visible_matches() {
    let dir = fixture();
    assert_eq!(
        expanded(Word::bare(joined(&dir, "*.toml"))),
        [joined(&dir, "alpha.toml"), joined(&dir, "beta.toml")]
    );
    assert_eq!(
        expanded(Word::bare(joined(&dir, "*"))),
        [
            joined(&dir, "alpha.toml"),
            joined(&dir, "beta.toml"),
            joined(&dir, "gamma.txt"),
            joined(&dir, "sub"),
        ]
    );
    assert_eq!(
        expanded(Word::bare(joined(&dir, ".*.toml"))),
        [joined(&dir, ".hidden.toml")]
    );
}

#[test]
fn words_that_do_not_expand_pass_through_verbatim() {
    let dir = fixture();
    for (case, word) in [
        ("plain word", Word::bare("cat")),
        ("quoted glob", Word::quoted(joined(&dir, "*.toml"))),
        ("unmatched glob", Word::bare(joined(&dir, "*.rs"))),
        ("invalid glob", Word::bare(joined(&dir, "[a"))),
        ("regex matching no file", Word::bare(".*ERROR")),
        ("quoted tilde", Word::quoted("~/notes.md")),
        ("named-user tilde", Word::bare("~alice/x")),
    ] {
        assert_eq!(expanded(word.clone()), [word.text.as_str()], "{case}");
    }
}

#[test]
fn glob_past_match_cap_fails() {
    let dir = tempdir().expect("tempdir");
    for index in 0..=MAX_GLOB_MATCHES {
        fs::write(dir.path().join(index.to_string()), b"").expect("write fixture");
    }
    let err = expand_words(&[Word::bare(joined(&dir, "*"))]).expect_err("over the cap");
    assert!(matches!(err, ExpandError::TooManyMatches { .. }), "{err}");
}

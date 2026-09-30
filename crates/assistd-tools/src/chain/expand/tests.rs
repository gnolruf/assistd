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
fn glob_descends_one_directory_per_component() {
    let dir = fixture();
    assert_eq!(
        expanded(Word::bare(joined(&dir, "*/*.toml"))),
        [joined(&dir, "sub/deep.toml")]
    );
    assert_eq!(
        expanded(Word::bare(joined(&dir, "**/*.toml"))),
        [joined(&dir, "sub/deep.toml")]
    );
    assert_eq!(
        expanded(Word::bare(joined(&dir, "*/"))),
        [joined(&dir, "sub")]
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
fn tilde_expands_against_home() {
    let home = std::env::var("HOME").expect("HOME set in test env");
    assert_eq!(
        expand_words(&[Word::bare("~/notes.md"), Word::bare("~")]).expect("no globs"),
        [format!("{home}/notes.md"), home]
    );
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

#[test]
fn walk_past_entry_budget_fails() {
    let dir = fixture();
    let mut walk = GlobWalk {
        pattern: "*.rs",
        entries_read: MAX_GLOB_ENTRIES - 1,
    };
    let pattern = Pattern::new("*.rs").expect("valid pattern");
    let err = walk
        .matching_children(dir.path(), &pattern)
        .expect_err("over the budget");
    assert!(matches!(err, ExpandError::TooManyEntries { .. }), "{err}");
}

#[tokio::test]
async fn expand_args_runs_off_the_runtime_thread() {
    let dir = fixture();
    assert_eq!(
        expand_args(&[Word::bare(joined(&dir, "*.txt"))])
            .await
            .expect("within budget"),
        [joined(&dir, "gamma.txt")]
    );
}

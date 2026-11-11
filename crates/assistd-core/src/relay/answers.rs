//! The startup sweep of saved sub-agent answers no stored relay step names any more.

use std::collections::HashSet;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use tracing::{info, warn};

/// Delete every answer under `answers_dir` that `kept` does not name, then each
/// session directory left empty. Must run before any relay saves an answer.
pub fn sweep_relay_answers(answers_dir: &Path, kept: &HashSet<PathBuf>) {
    let sessions = match std::fs::read_dir(answers_dir) {
        Ok(sessions) => sessions,
        Err(error) if error.kind() == ErrorKind::NotFound => return,
        Err(error) => {
            warn!(
                "cannot sweep saved relay answers in {}: {error}",
                answers_dir.display()
            );
            return;
        }
    };
    let mut removed = 0;
    for session in sessions.flatten() {
        let session_dir = session.path();
        removed += sweep_session(&session_dir, kept);
        remove_if_empty(&session_dir);
    }
    if removed > 0 {
        info!(removed, "deleted saved relay answers no stored relay names");
    }
}

/// Delete the answers in `session_dir` that `kept` does not name, returning how many.
fn sweep_session(session_dir: &Path, kept: &HashSet<PathBuf>) -> usize {
    let Ok(answers) = std::fs::read_dir(session_dir) else {
        return 0;
    };
    answers
        .flatten()
        .map(|answer| answer.path())
        .filter(|answer| !kept.contains(answer))
        .filter(|answer| {
            std::fs::remove_file(answer)
                .inspect_err(|error| {
                    warn!(
                        "cannot delete saved relay answer {}: {error}",
                        answer.display()
                    );
                })
                .is_ok()
        })
        .count()
}

/// `remove_dir` refuses a directory that still holds anything.
fn remove_if_empty(dir: &Path) {
    let _ = std::fs::remove_dir(dir);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_keeps_named_answers_and_removes_the_rest_with_emptied_sessions() {
        let answers_dir = tempfile::tempdir().unwrap();
        let live = answers_dir.path().join("live");
        let gone = answers_dir.path().join("gone");
        std::fs::create_dir(&live).unwrap();
        std::fs::create_dir(&gone).unwrap();
        let kept = live.join("relay-kept.txt");
        for answer in [
            &kept,
            &live.join("relay-orphan.txt"),
            &gone.join("relay-old.txt"),
        ] {
            std::fs::write(answer, "answer").unwrap();
        }

        sweep_relay_answers(answers_dir.path(), &HashSet::from([kept.clone()]));

        let left: Vec<_> = std::fs::read_dir(&live)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(left, [kept]);
        assert!(!gone.exists());
    }
}

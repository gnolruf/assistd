//! Whether a program is a windowed desktop application, judged by the
//! system's XDG desktop entries.

use std::fs;
use std::path::{Path, PathBuf};

use super::allowlist::user_can_modify;
use super::review::{basename, runs_another_command};

/// `$XDG_DATA_DIRS` when unset or empty, per the XDG base directory spec.
const DEFAULT_DATA_DIRS: &str = "/usr/local/share:/usr/share";

/// Whether a desktop entry the user cannot modify starts `program` outside
/// a terminal. Wrappers such as `env` and `nohup` never count.
pub(crate) async fn is_desktop_application(program: &str) -> bool {
    if runs_another_command(program) {
        return false;
    }
    let name = basename(program).to_string();
    tokio::task::spawn_blocking(move || {
        applications_dirs()
            .iter()
            .any(|dir| dir_launches(dir, &name))
    })
    .await
    .unwrap_or(false)
}

fn applications_dirs() -> Vec<PathBuf> {
    let data_dirs = std::env::var("XDG_DATA_DIRS")
        .ok()
        .filter(|dirs| !dirs.is_empty())
        .unwrap_or_else(|| DEFAULT_DATA_DIRS.to_string());
    std::env::split_paths(&data_dirs)
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join("applications"))
        .collect()
}

fn dir_launches(dir: &Path, name: &str) -> bool {
    if !fs::metadata(dir).is_ok_and(|meta| !user_can_modify(&meta)) {
        return false;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "desktop"))
        .any(|path| entry_launches(&path, name))
}

fn entry_launches(path: &Path, name: &str) -> bool {
    fs::metadata(path).is_ok_and(|meta| meta.is_file() && !user_can_modify(&meta))
        && fs::read_to_string(path)
            .is_ok_and(|entry| windowed_programs(&entry).any(|program| program == name))
}

/// The programs a desktop entry's `Exec` keys start; none when it runs in
/// a terminal.
fn windowed_programs(entry: &str) -> impl Iterator<Item = &str> {
    let in_terminal = entry.lines().any(|line| line.trim() == "Terminal=true");
    entry
        .lines()
        .filter(move |_| !in_terminal)
        .filter_map(|line| line.strip_prefix("Exec="))
        .filter_map(|exec| exec.split_whitespace().next())
        .map(|word| basename(word.trim_matches('"')))
}

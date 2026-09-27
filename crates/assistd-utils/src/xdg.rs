//! XDG base directories with the `$HOME` fallbacks the basedir spec names.
//! An empty variable counts as unset.

use std::ffi::OsString;
use std::path::PathBuf;

/// `$XDG_CONFIG_HOME`, else `$HOME/.config`.
pub fn config_home() -> Option<PathBuf> {
    base_dir("XDG_CONFIG_HOME", ".config")
}

/// `$XDG_DATA_HOME`, else `$HOME/.local/share`.
pub fn data_home() -> Option<PathBuf> {
    base_dir("XDG_DATA_HOME", ".local/share")
}

/// `$XDG_CACHE_HOME`, else `$HOME/.cache`.
pub fn cache_home() -> Option<PathBuf> {
    base_dir("XDG_CACHE_HOME", ".cache")
}

/// `$XDG_STATE_HOME`, else `$HOME/.local/state`.
pub fn state_home() -> Option<PathBuf> {
    base_dir("XDG_STATE_HOME", ".local/state")
}

/// `$XDG_RUNTIME_DIR`, which has no `$HOME` fallback.
pub fn runtime_dir() -> Option<PathBuf> {
    non_empty(std::env::var_os("XDG_RUNTIME_DIR")).map(PathBuf::from)
}

fn base_dir(var: &str, home_relative: &str) -> Option<PathBuf> {
    base_dir_from(
        std::env::var_os(var),
        std::env::var_os("HOME"),
        home_relative,
    )
}

fn base_dir_from(
    xdg: Option<OsString>,
    home: Option<OsString>,
    home_relative: &str,
) -> Option<PathBuf> {
    non_empty(xdg)
        .map(PathBuf::from)
        .or_else(|| non_empty(home).map(|home| PathBuf::from(home).join(home_relative)))
}

fn non_empty(value: Option<OsString>) -> Option<OsString> {
    value.filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn base_dir_prefers_xdg_then_home_and_ignores_empty_values() {
        let xdg = Some(OsString::from("/tmp/xdg-test"));
        let home = Some(OsString::from("/home/alice"));
        assert_eq!(
            base_dir_from(xdg, home.clone(), ".cache").as_deref(),
            Some(Path::new("/tmp/xdg-test"))
        );
        assert_eq!(
            base_dir_from(Some(OsString::new()), home.clone(), ".cache").as_deref(),
            Some(Path::new("/home/alice/.cache"))
        );
        assert_eq!(
            base_dir_from(None, home, ".cache").as_deref(),
            Some(Path::new("/home/alice/.cache"))
        );
        assert_eq!(base_dir_from(None, Some(OsString::new()), ".cache"), None);
        assert_eq!(base_dir_from(None, None, ".cache"), None);
    }
}

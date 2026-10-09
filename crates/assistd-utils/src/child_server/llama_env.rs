//! Keeps llama.cpp's environment-variable settings from reaching a spawned
//! llama-server, whose options should come only from its checked command line.

use std::ffi::{OsStr, OsString};

use tokio::process::Command;

/// `LLAMA_ARG_<FLAG>` sets any server flag.
const LLAMA_ARG_PREFIX: &[u8] = b"LLAMA_ARG_";
/// Sets `--api-key`, adding a key alongside the one assistd generates.
const LLAMA_API_KEY: &str = "LLAMA_API_KEY";

/// Stops `cmd` inheriting any llama.cpp setting from this process's environment.
pub fn remove_llama_env(cmd: &mut Command) {
    remove_llama_vars(cmd, std::env::vars_os().map(|(name, _)| name));
}

fn remove_llama_vars(cmd: &mut Command, names: impl IntoIterator<Item = OsString>) {
    for name in names.into_iter().filter(|name| is_llama_setting(name)) {
        cmd.env_remove(name);
    }
}

fn is_llama_setting(name: &OsStr) -> bool {
    name.as_encoded_bytes().starts_with(LLAMA_ARG_PREFIX) || name == OsStr::new(LLAMA_API_KEY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn removed_vars(inherited: &[&str]) -> Vec<OsString> {
        let mut cmd = Command::new("llama-server");
        remove_llama_vars(&mut cmd, inherited.iter().map(OsString::from));
        cmd.as_std()
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(name, _)| name.to_owned())
            .collect()
    }

    #[test]
    fn only_llama_settings_are_removed() {
        assert_eq!(
            removed_vars(&["LLAMA_ARG_TOOLS", "PATH", "LLAMA_CACHE", "LLAMA_API_KEY"]),
            ["LLAMA_API_KEY", "LLAMA_ARG_TOOLS"]
        );
    }
}

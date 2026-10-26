//! Home paths that hold credentials, which neither the in-process readers
//! nor sandboxed commands may see.

use std::path::{Path, PathBuf};

/// Paths under `$HOME` holding keys, tokens, password stores or browser
/// sessions.
const CREDENTIAL_PATHS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".password-store",
    ".local/share/keyrings",
    ".pki",
    ".netrc",
    ".git-credentials",
    ".config/gh",
    ".config/hub",
    ".config/git/credentials",
    ".aws",
    ".azure",
    ".config/gcloud",
    ".kube",
    ".docker",
    ".vault-token",
    ".terraform.d",
    ".config/rclone",
    ".cargo/credentials",
    ".cargo/credentials.toml",
    ".npmrc",
    ".pypirc",
    ".mozilla",
    ".thunderbird",
    ".config/google-chrome",
    ".config/chromium",
    ".config/BraveSoftware",
];

/// The credential paths under `home` that exist, as `home` spells them.
pub(crate) fn existing_credential_paths(home: &Path) -> impl Iterator<Item = PathBuf> {
    CREDENTIAL_PATHS
        .iter()
        .map(|relative| home.join(relative))
        .filter(|path| path.exists())
}

/// Whether `real_path`, with symlinks resolved, is or lies under a
/// credential path of `home`. Resolves symlinks, so it blocks.
pub(crate) fn holds_credentials(real_path: &Path, home: &Path) -> bool {
    existing_credential_paths(home)
        .filter_map(|path| std::fs::canonicalize(path).ok())
        .any(|credential| real_path.starts_with(credential))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_symlink_into_a_credential_dir_still_holds_credentials() {
        let home_dir = tempfile::tempdir().expect("tempdir");
        let home = std::fs::canonicalize(home_dir.path()).expect("canonical home");
        std::fs::create_dir_all(home.join(".ssh")).expect("ssh dir");
        std::fs::write(home.join(".ssh/id_ed25519"), "key").expect("key");
        std::fs::write(home.join("notes.txt"), "notes").expect("notes");
        std::os::unix::fs::symlink(home.join(".ssh/id_ed25519"), home.join("key")).expect("link");
        let real = |name: &str| std::fs::canonicalize(home.join(name)).expect("resolves");
        assert!(holds_credentials(&real("key"), &home));
        assert!(!holds_credentials(&real("notes.txt"), &home));
    }
}

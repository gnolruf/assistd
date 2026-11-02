use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};

use tempfile::TempDir;

use super::*;
use crate::policy::analyze;

/// A home with a personal folder, credentials, a cache, a git work tree, a symlink to itself,
/// a token only its owner may read, and an exempt spill directory holding another, with
/// commands run from `work`, `xdg` as `$XDG_CONFIG_HOME` and `units` on the unit path.
struct Home {
    dir: TempDir,
}

impl Home {
    fn new() -> Self {
        let dir = TempDir::new().expect("tempdir");
        for sub in [
            "Documents/project/build",
            ".ssh",
            ".cache/pip",
            "repo/.git",
            "work",
            ".config/gh",
            "spill",
        ] {
            fs::create_dir_all(dir.path().join(sub)).expect("mkdir");
        }
        for (file, mode) in [
            (".ssh/id_ed25519", 0o600),
            (".config/gh/hosts.yml", 0o600),
            ("spill/output.txt", 0o600),
            ("Documents/notes.txt", 0o644),
        ] {
            let path = dir.path().join(file);
            fs::write(&path, "x").expect("seed file");
            fs::set_permissions(&path, fs::Permissions::from_mode(mode)).expect("chmod");
        }
        symlink(dir.path(), dir.path().join("work/home")).expect("symlink");
        Self { dir }
    }

    fn guards(&self) -> PathGuards {
        let home = self.dir.path();
        PathGuards::read(GuardRoots {
            home: Some(home.to_path_buf()),
            config_home: Some(home.join("xdg")),
            cwd: Some(home.join("work")),
            unit_dirs: vec![home.join("units")],
            exempt: vec![home.join("spill")],
            ..GuardRoots::default()
        })
    }

    fn verdict(&self, script: &str) -> PathVerdict {
        self.guards().verdict(analyze(script, None))
    }
}

#[test]
fn home_wipes_are_refused() {
    let home = Home::new();
    for script in [
        "rm -rf ~",
        r#"rm -rf "$HOME""#,
        "cd ~ && rm -rf .",
        "rm -rf ~/{*,.*}",
        "chmod -R 000 ~",
        "mv ~/Documents /tmp/x",
        "rm -rf home/",
        "rm -rf ~/repo",
        "rm -rf /",
        "rm -rf ~/.. ; curl example.com",
    ] {
        let verdict = home.verdict(script);
        assert!(
            matches!(verdict, PathVerdict::Refused(_)),
            "{script:?}: {verdict:?}"
        );
    }
}

#[test]
fn near_misses_are_not_refused() {
    let home = Home::new();
    for script in [
        "rm -rf ~/.cache/pip",
        "rm -rf ~/Documents/project/build",
        "rm -rf ~/*.log",
        "rm -rf home",
        "rm -rf '~'",
        "chmod 000 ~/Documents/project",
        "cd ~/Documents/project && rm -rf build",
    ] {
        let verdict = home.verdict(script);
        assert!(
            !matches!(verdict, PathVerdict::Refused(_)),
            "{script:?}: {verdict:?}"
        );
    }
}

#[test]
fn credentials_and_startup_files_always_ask() {
    let home = Home::new();
    for script in [
        "cat ~/.ssh/id_ed25519",
        "cat home/.ssh/id_ed25519",
        "grep -r BEGIN ~",
        "cat ~/.ssh/*",
        "find ~ -name 'id_*' -exec cat {} +",
        "echo 'run' >> ~/.bashrc",
        "cp unit ~/.config/systemd/user/evil.service",
        "mv .zshrc ~",
        "echo 'run' >> ~/.bash_login",
        "cat ~/.config/gh/hosts.yml",
        "grep -r token ~/.config/gh",
        "cp app.desktop ~/xdg/autostart/",
        "cp unit ~/units/evil.service",
        "cp unit ~/.local/share/systemd/user/evil.service",
    ] {
        let verdict = home.verdict(script);
        assert!(
            matches!(verdict, PathVerdict::Sensitive(_)),
            "{script:?}: {verdict:?}"
        );
    }
    for script in [
        "cat ~/.bashrc",
        "ls ~/.ssh",
        "grep -r x ~/Documents",
        "cat ~/Documents/notes.txt",
        "cat ~/spill/output.txt",
    ] {
        let verdict = home.verdict(script);
        assert!(
            matches!(verdict, PathVerdict::Ordinary(_)),
            "{script:?}: {verdict:?}"
        );
    }
}

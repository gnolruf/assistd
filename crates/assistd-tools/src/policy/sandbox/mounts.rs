//! What a sandbox's bwrap flags mount: the directories a command gets empty and private,
//! and the host's that stay shared.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// The bwrap flags that mount a host path at their second argument.
const BIND_FLAGS: &[&str] = &[
    "--bind",
    "--bind-try",
    "--ro-bind",
    "--ro-bind-try",
    "--ro-bind-fd",
    "--dev-bind",
    "--dev-bind-try",
];

/// What a run of bwrap flags leaves mounted, later flags replacing what they cover.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Mounts {
    /// Writable empty tmpfs mounts no later bind covers.
    private: Vec<PathBuf>,
    /// Host binds no later tmpfs covers.
    shared: Vec<PathBuf>,
    /// The directory the command starts in, when the flags name one.
    pub(super) working_dir: Option<PathBuf>,
}

impl Mounts {
    /// The mounts `flags`, the bwrap arguments before `--`, leave in place.
    pub(super) fn of(flags: &[&OsStr]) -> Self {
        let mut mounts = Self::default();
        let mut rest = flags;
        while let [flag, tail @ ..] = rest {
            rest = match (flag.to_str(), tail) {
                (Some("--tmpfs"), [dir, tail @ ..]) => {
                    mounts.mount_tmpfs(Path::new(dir));
                    tail
                }
                (Some("--chdir"), [dir, tail @ ..]) => {
                    mounts.working_dir = Some(PathBuf::from(dir));
                    tail
                }
                (Some("--remount-ro"), [dir, tail @ ..]) => {
                    mounts.private.retain(|private| private != Path::new(dir));
                    tail
                }
                (Some(flag), [_, dir, tail @ ..]) if BIND_FLAGS.contains(&flag) => {
                    mounts.mount_bind(Path::new(dir));
                    tail
                }
                _ => tail,
            };
        }
        mounts
    }

    fn mount_tmpfs(&mut self, dir: &Path) {
        self.private.retain(|private| !private.starts_with(dir));
        self.shared.retain(|shared| !shared.starts_with(dir));
        self.private.push(dir.to_path_buf());
    }

    pub(super) fn mount_bind(&mut self, dir: &Path) {
        self.private.retain(|private| !private.starts_with(dir));
        self.shared.push(dir.to_path_buf());
    }

    /// The private mount `path` is in, unless a host bind inside that
    /// mount covers it.
    pub(super) fn private_dir_of(&self, path: &Path) -> Option<&PathBuf> {
        let shared_within = |private: &Path| {
            self.shared
                .iter()
                .any(|shared| shared.starts_with(private) && path.starts_with(shared))
        };
        self.private
            .iter()
            .find(|private| path.starts_with(private) && !shared_within(private))
    }

    pub(super) fn private_dirs_named(&self, script: &str) -> Vec<PathBuf> {
        self.private
            .iter()
            .filter(|private| {
                absolute_paths(script).any(|path| self.private_dir_of(path) == Some(*private))
            })
            .cloned()
            .collect()
    }
}

fn absolute_paths(script: &str) -> impl Iterator<Item = &Path> {
    script
        .split(|c: char| !is_path_char(c))
        .filter(|word| word.starts_with('/'))
        .map(Path::new)
}

fn is_path_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | '~')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mounts_of(flags: &[&str]) -> Mounts {
        let flags: Vec<&OsStr> = flags.iter().map(OsStr::new).collect();
        Mounts::of(&flags)
    }

    #[test]
    fn private_dirs_are_named_only_as_the_start_of_an_absolute_path() {
        let mounts = mounts_of(&["--ro-bind", "/", "/", "--tmpfs", "/tmp", "--tmpfs", "/run"]);
        for (script, expected) in [
            ("cat /tmp/out.txt", vec!["/tmp"]),
            ("ls -la /tmp/", vec!["/tmp"]),
            ("ls /tmp", vec!["/tmp"]),
            (
                "sort --output=/tmp/sorted \"/run/user/1000/x\"",
                vec!["/tmp", "/run"],
            ),
            (
                "ls /var/tmp ~/tmp ./run /tmpfiles /runner $HOME/tmp",
                vec![],
            ),
            ("cargo run", vec![]),
        ] {
            let expected: Vec<PathBuf> = expected.into_iter().map(PathBuf::from).collect();
            assert_eq!(mounts.private_dirs_named(script), expected, "{script}");
        }
    }

    #[test]
    fn a_later_mount_replaces_what_it_covers() {
        let rebound = mounts_of(&["--tmpfs", "/tmp", "--bind", "/tmp", "/tmp"]);
        assert!(rebound.private.is_empty());
        let hidden = mounts_of(&["--bind", "/srv/data", "/srv/data", "--tmpfs", "/srv"]);
        assert_eq!(
            hidden,
            Mounts {
                private: vec![PathBuf::from("/srv")],
                shared: Vec::new(),
                working_dir: None,
            }
        );
    }
}

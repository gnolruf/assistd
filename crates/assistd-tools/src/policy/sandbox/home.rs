//! `$HOME` as a host-profile sandbox lays it out: visible entries writable, and the
//! directories host shells look programs up in kept read-only beneath them.

use std::io;
use std::path::{Component, Path, PathBuf};

use rustix::fs::Access;

use super::HomeAccess;

/// A usable `$HOME`: an absolute path resolving to a directory other than
/// `/`, with its visible (non-dot) entries listed once per command.
#[derive(Debug)]
pub(super) struct SandboxHome {
    pub(super) dir: String,
    real: PathBuf,
    visible: Vec<VisibleEntry>,
}

#[derive(Debug)]
struct VisibleEntry {
    path: PathBuf,
    /// Known not to be a symlink, so the profile may bind it writable.
    bindable: bool,
}

impl SandboxHome {
    /// `None` when `home` is unset, relative, or does not resolve to a
    /// directory other than `/`.
    pub(super) fn read(home: Option<String>) -> Option<Self> {
        let dir = home.filter(|home| Path::new(home).is_absolute())?;
        let real = std::fs::canonicalize(&dir)
            .ok()
            .filter(|real| real.is_dir() && real.parent().is_some())?;
        let visible = visible_entries(Path::new(&dir));
        Some(Self { dir, real, visible })
    }

    fn writable_entries(&self) -> impl Iterator<Item = &str> {
        self.visible
            .iter()
            .filter(|entry| entry.bindable)
            .filter_map(|entry| entry.path.to_str())
    }

    pub(super) fn bind_flags(&self, access: HomeAccess) -> Vec<String> {
        let home = ["--ro-bind", self.dir.as_str(), self.dir.as_str()];
        let entries = self
            .writable_entries()
            .filter(|_| access == HomeAccess::Writable)
            .flat_map(|entry| ["--bind", entry, entry]);
        home.into_iter().chain(entries).map(String::from).collect()
    }

    /// [`SandboxHome::command_dirs`] when `access` leaves entries writable; none for a
    /// read-only home, where a guard's self-bind would be the one writable mount.
    pub(super) fn locked_command_dirs(
        &self,
        access: HomeAccess,
        host_path: &[PathBuf],
        protected: &[PathBuf],
    ) -> CommandDirs {
        match access {
            HomeAccess::Writable => self.command_dirs(host_path, protected),
            HomeAccess::ReadOnly => CommandDirs::default(),
        }
    }

    /// The directories under visible entries that host shells look programs
    /// up in, and the ancestors that keep them in place; ancestors inside a
    /// `protected` directory are left out.
    fn command_dirs(&self, host_path: &[PathBuf], protected: &[PathBuf]) -> CommandDirs {
        let existing = self.existing_command_dirs(host_path);
        let placeholders = self.command_dir_placeholders(host_path, &existing);
        let real_protected: Vec<PathBuf> = protected
            .iter()
            .filter_map(|dir| std::fs::canonicalize(dir).ok())
            .collect();
        let locked = [&existing[..], &placeholders[..]].concat();
        let mut rename_guards: Vec<PathBuf> = locked
            .iter()
            .flat_map(|dir| self.ancestors_below_entry(dir))
            .filter(|ancestor| {
                !locked
                    .iter()
                    .chain(&real_protected)
                    .any(|outer| ancestor.starts_with(outer))
            })
            .collect();
        rename_guards.sort();
        rename_guards.dedup();
        CommandDirs {
            existing,
            placeholders,
            rename_guards,
        }
    }

    /// The ancestors of `dir` strictly between it and its top-level entry.
    fn ancestors_below_entry(&self, dir: &Path) -> Vec<PathBuf> {
        let Some(entry) = dir
            .strip_prefix(&self.real)
            .ok()
            .and_then(|rest| rest.components().next())
            .map(|entry| self.real.join(entry))
        else {
            return Vec::new();
        };
        dir.ancestors()
            .skip(1)
            .take_while(|ancestor| ancestor.starts_with(&entry) && *ancestor != entry)
            .map(Path::to_path_buf)
            .collect()
    }

    /// The existing directories of `host_path`, `~/bin` and every
    /// `~/<entry>/bin`, symlinks resolved, that are home or lie under a
    /// visible entry.
    pub(super) fn existing_command_dirs(&self, host_path: &[PathBuf]) -> Vec<PathBuf> {
        let home_bins = std::iter::once(Path::new(&self.dir).join("bin"))
            .chain(self.visible.iter().map(|entry| entry.path.join("bin")));
        let mut dirs: Vec<PathBuf> = host_path
            .iter()
            .map(std::fs::canonicalize)
            .chain(home_bins.map(std::fs::canonicalize))
            .filter_map(Result::ok)
            .filter(|dir| dir.is_dir() && is_within_visible_home_entry(dir, &self.real))
            .collect();
        dirs.sort();
        dirs.dedup();
        dirs
    }

    /// The first missing components of the `host_path` directories a command
    /// could create, except those under one of `existing` or one another.
    fn command_dir_placeholders(
        &self,
        host_path: &[PathBuf],
        existing: &[PathBuf],
    ) -> Vec<PathBuf> {
        let mut missing: Vec<PathBuf> = host_path
            .iter()
            .filter_map(|dir| self.placeholder_for(dir))
            .collect();
        missing.sort();
        missing.dedup();
        let is_covered = |dir: &PathBuf| {
            existing
                .iter()
                .chain(&missing)
                .any(|outer| outer != dir && dir.starts_with(outer))
        };
        missing
            .iter()
            .filter(|dir| !is_covered(dir))
            .cloned()
            .collect()
    }

    /// The first missing directory `dir` needs created under its nearest existing ancestor,
    /// symlinks resolved; `None` unless that ancestor is creatable-in and inside a writable entry.
    fn placeholder_for(&self, dir: &Path) -> Option<PathBuf> {
        let (existing, real_existing) = dir
            .ancestors()
            .find_map(|ancestor| Some((ancestor, std::fs::canonicalize(ancestor).ok()?)))?;
        let missing = dir.strip_prefix(existing).ok()?;
        let first_missing = real_existing.join(missing.components().next()?);
        let creatable = missing
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
            && is_absent(&first_missing)
            && self.is_in_writable_entry(&real_existing)
            && can_create_in(&real_existing);
        creatable.then_some(first_missing)
    }

    fn is_in_writable_entry(&self, real_path: &Path) -> bool {
        real_path
            .strip_prefix(&self.real)
            .ok()
            .and_then(|rest| rest.components().next())
            .is_some_and(|entry| {
                self.writable_entries()
                    .any(|writable| Path::new(writable).file_name() == Some(entry.as_os_str()))
            })
    }
}

/// Directories host shells look programs up in, kept read-only inside the
/// sandbox, with all paths resolved through symlinks.
#[derive(Debug, Default)]
pub(super) struct CommandDirs {
    existing: Vec<PathBuf>,
    /// Missing directories a command could create, mounted empty instead.
    placeholders: Vec<PathBuf>,
    /// Ancestors bound onto themselves, parent first, so that a command
    /// cannot rename them and rebuild the directory beneath.
    rename_guards: Vec<PathBuf>,
}

impl CommandDirs {
    pub(super) fn rename_guard_flags(&self) -> Vec<PathBuf> {
        self.rename_guards
            .iter()
            .flat_map(|dir| [PathBuf::from("--bind"), dir.clone(), dir.clone()])
            .collect()
    }

    /// Read-only binds over the existing directories, then empty read-only
    /// mounts over the placeholders.
    pub(super) fn lock_flags(&self) -> Vec<PathBuf> {
        let binds = self
            .existing
            .iter()
            .flat_map(|dir| [PathBuf::from("--ro-bind"), dir.clone(), dir.clone()]);
        let empty_mounts = self.placeholders.iter().flat_map(|dir| {
            [
                PathBuf::from("--tmpfs"),
                dir.clone(),
                PathBuf::from("--remount-ro"),
                dir.clone(),
            ]
        });
        binds.chain(empty_mounts).collect()
    }
}

/// Whether `dir` is `real_home`, whose visible files the profile binds
/// writable, or lies under one of its visible entries.
pub(crate) fn is_within_visible_home_entry(dir: &Path, real_home: &Path) -> bool {
    dir.strip_prefix(real_home).is_ok_and(|rest| {
        rest.components()
            .next()
            .is_none_or(|entry| !entry.as_os_str().as_encoded_bytes().starts_with(b"."))
    })
}

fn visible_entries(home: &Path) -> Vec<VisibleEntry> {
    let Ok(entries) = std::fs::read_dir(home) else {
        return Vec::new();
    };
    let mut visible: Vec<VisibleEntry> = entries
        .filter_map(Result::ok)
        .filter(|entry| !entry.file_name().as_encoded_bytes().starts_with(b"."))
        .map(|entry| VisibleEntry {
            bindable: entry.file_type().is_ok_and(|kind| !kind.is_symlink()),
            path: entry.path(),
        })
        .collect();
    visible.sort_by(|a, b| a.path.cmp(&b.path));
    visible
}

fn is_absent(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_err_and(|error| error.kind() == io::ErrorKind::NotFound)
}

fn can_create_in(dir: &Path) -> bool {
    dir.is_dir() && rustix::fs::access(dir, Access::WRITE_OK | Access::EXEC_OK).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox_home(dir: &Path) -> SandboxHome {
        SandboxHome::read(Some(dir.to_string_lossy().into_owned())).expect("usable home")
    }

    fn lock_flags_for(home: &Path, host_path: &[PathBuf]) -> Vec<PathBuf> {
        sandbox_home(home).command_dirs(host_path, &[]).lock_flags()
    }

    fn read_only_binds_in(home: &Path, dirs: &[&str]) -> Vec<PathBuf> {
        dirs.iter()
            .flat_map(|dir| {
                let dir = home.join(dir);
                [PathBuf::from("--ro-bind"), dir.clone(), dir]
            })
            .collect()
    }

    #[test]
    fn home_command_dirs_on_the_host_path_stay_read_only() {
        let home_dir = tempfile::tempdir().expect("tempdir");
        let home = std::fs::canonicalize(home_dir.path()).expect("canonical home");
        let elsewhere = tempfile::tempdir().expect("tempdir");
        for dir in ["bin", "go/bin", "projects/tools/bin", ".local/bin", "docs"] {
            std::fs::create_dir_all(home.join(dir)).expect(dir);
        }
        let host_path = [
            home.join("go/bin"),
            home.join("projects/tools/bin"),
            home.join(".local/bin"),
            home.join("missing/bin"),
            elsewhere.path().to_path_buf(),
            PathBuf::from("/usr/bin"),
            home.join("go/bin"),
        ];
        assert_eq!(
            lock_flags_for(&home, &host_path),
            read_only_binds_in(&home, &["bin", "go/bin", "projects/tools/bin"])
        );
        assert_eq!(
            lock_flags_for(&home, std::slice::from_ref(&home)),
            read_only_binds_in(&home, &["", "bin", "go/bin"])
        );
    }

    #[test]
    fn home_bin_stays_read_only_off_the_host_path_and_through_a_symlink() {
        let home_dir = tempfile::tempdir().expect("tempdir");
        let home = std::fs::canonicalize(home_dir.path()).expect("canonical home");
        assert!(lock_flags_for(&home, &[]).is_empty());
        for dir in ["projects/tools/bin", ".local/bin"] {
            std::fs::create_dir_all(home.join(dir)).expect(dir);
        }
        for (target, expected) in [
            (
                home.join("projects/tools/bin"),
                read_only_binds_in(&home, &["projects/tools/bin"]),
            ),
            (home.join(".local/bin"), Vec::new()),
            (PathBuf::from("/usr/bin"), Vec::new()),
        ] {
            std::os::unix::fs::symlink(&target, home.join("bin")).expect("symlink");
            assert_eq!(lock_flags_for(&home, &[]), expected, "{target:?}");
            std::fs::remove_file(home.join("bin")).expect("remove symlink");
        }
    }

    fn empty_read_only_mounts_in(home: &Path, dirs: &[&str]) -> Vec<PathBuf> {
        dirs.iter()
            .flat_map(|dir| {
                let dir = home.join(dir);
                [
                    PathBuf::from("--tmpfs"),
                    dir.clone(),
                    PathBuf::from("--remount-ro"),
                    dir,
                ]
            })
            .collect()
    }

    #[test]
    fn missing_host_path_dirs_under_writable_entries_get_empty_read_only_mounts() {
        let home_dir = tempfile::tempdir().expect("tempdir");
        let home = std::fs::canonicalize(home_dir.path()).expect("canonical home");
        let elsewhere = tempfile::tempdir().expect("tempdir");
        for dir in ["go", "projects", "src", "tools/bin", ".local"] {
            std::fs::create_dir_all(home.join(dir)).expect(dir);
        }
        std::fs::write(home.join("notes.txt"), b"").expect("notes.txt");
        let symlink = |target: &Path, name: &str| {
            std::os::unix::fs::symlink(target, home.join(name)).expect(name);
        };
        symlink(elsewhere.path(), "away");
        symlink(&home.join("src"), "alias");
        symlink(&home.join("go/nowhere"), "go/dangling");
        let host_path = [
            "go/bin",
            "go/bin/nested",
            "projects/tool/bin",
            "alias/bin",
            "tools/bin/nested",
            "go/new/../bin",
            "go/dangling/bin",
            ".local/bin",
            "absent/bin",
            "notes.txt/bin",
            "away/bin",
            "go/bin",
        ]
        .map(|dir| home.join(dir));
        let host_path = [&host_path[..], &[elsewhere.path().join("bin")]].concat();
        let expected = [
            read_only_binds_in(&home, &["tools/bin"]),
            empty_read_only_mounts_in(&home, &["go/bin", "projects/tool", "src/bin"]),
        ]
        .concat();
        assert_eq!(lock_flags_for(&home, &host_path), expected);
    }

    fn paths_in(home: &Path, dirs: &[&str]) -> Vec<PathBuf> {
        dirs.iter().map(|dir| home.join(dir)).collect()
    }

    #[test]
    fn rename_guards_pin_every_ancestor_between_an_entry_and_a_locked_dir() {
        let home_dir = tempfile::tempdir().expect("tempdir");
        let home = std::fs::canonicalize(home_dir.path()).expect("canonical home");
        for dir in [
            "projects/tools/bin",
            "projects/a/b/bin",
            "projects/lib/sub/bin",
            "projects/.venv/bin",
            "go/bin",
            "work/conf/x/bin",
        ] {
            std::fs::create_dir_all(home.join(dir)).expect(dir);
        }
        let host_path = paths_in(
            &home,
            &[
                "projects/tools/bin",
                "projects/a/b/bin",
                "projects/a/b/bin",
                "projects/a/c/bin",
                "projects/new/x/bin",
                "projects/lib",
                "projects/lib/sub/bin",
                "projects/.venv/bin",
                "go/bin",
                "work/conf/x/bin",
            ],
        );
        let dirs = sandbox_home(&home).command_dirs(&host_path, &[home.join("work/conf")]);
        assert_eq!(
            dirs.rename_guards,
            paths_in(
                &home,
                &[
                    "projects/.venv",
                    "projects/a",
                    "projects/a/b",
                    "projects/tools"
                ]
            )
        );
        assert_eq!(
            dirs.placeholders,
            paths_in(&home, &["projects/a/c", "projects/new"])
        );
    }

    #[test]
    fn bin_dirs_directly_under_visible_entries_stay_read_only() {
        let home_dir = tempfile::tempdir().expect("tempdir");
        let home = std::fs::canonicalize(home_dir.path()).expect("canonical home");
        for dir in [
            "go/bin",
            "tools/bin",
            "projects/app/bin",
            "projects/lib/bin",
            ".cargo/bin",
            "docs",
        ] {
            std::fs::create_dir_all(home.join(dir)).expect(dir);
        }
        let symlink = |target: &str, name: &str| {
            std::os::unix::fs::symlink(home.join(target), home.join(name)).expect(name);
        };
        symlink("projects/lib", "lib");
        symlink(".cargo", "cargo");
        assert_eq!(
            lock_flags_for(&home, &[home.join("go/bin")]),
            read_only_binds_in(&home, &["go/bin", "projects/lib/bin", "tools/bin"])
        );
    }
}

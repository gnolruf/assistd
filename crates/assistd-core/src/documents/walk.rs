//! Which files under an index root are indexed: the walk that leaves out hidden,
//! ignored, excluded, oversized and sensitive ones, and follows no symlink.

use std::io;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use assistd_config::DocumentsConfig;
use assistd_memory::FileStamp;
use assistd_tools::policy::PathGuards;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use ignore::{DirEntry, WalkBuilder};

use super::pass::SkipCause;

/// What the walk makes of one entry under a root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Walked {
    /// A regular file to index unless its stamp is unchanged.
    Candidate { path: PathBuf, stamp: FileStamp },
    /// A file left out of the index, counted with why.
    Skipped(SkipCause),
    /// Part of the root could not be listed, so files the walk never reached may
    /// still be there; holds the error.
    Unlisted(String),
}

/// One index root with the `documents.exclude` patterns and size limit its files
/// are held to.
#[derive(Debug, Clone)]
pub(super) struct IndexRoot {
    path: PathBuf,
    excluded: Gitignore,
    max_file_bytes: u64,
}

impl IndexRoot {
    /// Errors on a `documents.exclude` pattern that is not a valid glob.
    pub(super) fn new(path: &Path, documents: &DocumentsConfig) -> Result<Self, ignore::Error> {
        let mut excluded = GitignoreBuilder::new(path);
        for pattern in &documents.exclude {
            excluded.add_line(None, pattern)?;
        }
        Ok(Self {
            path: path.to_path_buf(),
            excluded: excluded.build()?,
            max_file_bytes: documents.max_file_bytes(),
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn max_file_bytes(&self) -> u64 {
        self.max_file_bytes
    }

    /// Hand `visit` what the walk makes of each entry under the root until it
    /// breaks. Ignore files count from the root down, in or out of a git
    /// repository. Blocks on the filesystem.
    pub(super) fn walk(
        &self,
        guards: &PathGuards,
        mut visit: impl FnMut(Walked) -> ControlFlow<()>,
    ) {
        let excluded = self.excluded.clone();
        let entries = WalkBuilder::new(&self.path)
            .parents(false)
            .git_global(false)
            .require_git(false)
            .filter_entry(move |entry| !is_excluded(&excluded, entry))
            .build();
        for entry in entries {
            let walked = match entry {
                Ok(entry) => self.classify(guards, entry),
                Err(error) => unlisted(&error),
            };
            if let Some(walked) = walked
                && visit(walked).is_break()
            {
                return;
            }
        }
    }

    /// What `entry` is to the index; `None` for anything but a regular file.
    fn classify(&self, guards: &PathGuards, entry: DirEntry) -> Option<Walked> {
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            return None;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(error) => return unlisted(&error),
        };
        if metadata.len() > self.max_file_bytes {
            return Some(Walked::Skipped(SkipCause::TooLarge));
        }
        if guards.sensitive_read(entry.path()).is_some() {
            return Some(Walked::Skipped(SkipCause::Sensitive));
        }
        Some(Walked::Candidate {
            stamp: FileStamp::of(&metadata),
            path: entry.into_path(),
        })
    }
}

/// Why the files of the index root at `path` cannot be told from deleted ones: it is
/// missing, cannot be listed, or is empty, as the mount point of an unmounted drive
/// is. `None` when it holds something. Blocks on the filesystem.
pub(crate) fn root_unavailable(path: &Path) -> Option<String> {
    match std::fs::read_dir(path).map(|mut entries| entries.next().is_none()) {
        Ok(false) => None,
        Ok(true) => Some("it is empty".to_owned()),
        Err(error) => Some(error.to_string()),
    }
}

/// Whether a `documents.exclude` pattern leaves `entry` out; the root itself never is.
fn is_excluded(excluded: &Gitignore, entry: &DirEntry) -> bool {
    let is_dir = entry.file_type().is_some_and(|kind| kind.is_dir());
    entry.depth() > 0 && excluded.matched(entry.path(), is_dir).is_ignore()
}

/// What a failed listing means for the pass: nothing when the entry vanished
/// mid-walk, a skip when it may not be read, and otherwise an unlisted part.
fn unlisted(error: &ignore::Error) -> Option<Walked> {
    match error.io_error().map(io::Error::kind) {
        Some(io::ErrorKind::NotFound) => None,
        Some(io::ErrorKind::PermissionDenied) => Some(Walked::Skipped(SkipCause::Unreadable)),
        _ => Some(Walked::Unlisted(error.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, Permissions};
    use std::os::unix::fs::{PermissionsExt, symlink};

    use assistd_config::AbsoluteDir;
    use assistd_tools::policy::GuardRoots;

    use super::*;

    const EVERYONE_READS: u32 = 0o644;
    const OWNER_ONLY: u32 = 0o600;

    fn write(path: &Path, contents: &[u8], mode: u32) {
        fs::create_dir_all(path.parent().expect("a parent")).expect("directories");
        fs::write(path, contents).expect("file");
        fs::set_permissions(path, Permissions::from_mode(mode)).expect("mode");
    }

    #[test]
    fn the_walk_leaves_out_hidden_ignored_excluded_oversized_and_sensitive_files_and_symlinks() {
        let home = tempfile::tempdir().expect("tempdir");
        let home_path = home.path().canonicalize().expect("canonical home");
        let (root, outside) = (home_path.join("notes"), home_path.join("outside"));
        let documents = DocumentsConfig {
            enabled: true,
            roots: vec![
                root.to_str()
                    .expect("utf-8")
                    .parse::<AbsoluteDir>()
                    .unwrap(),
            ],
            exclude: vec!["drafts/**".into(), "*.log".into()],
            max_file_mb: 1.try_into().unwrap(),
            ..DocumentsConfig::default()
        };
        let note = |name: &str| write(&root.join(name), b"text", EVERYONE_READS);
        for kept in ["plan.md", "work/deep.md"] {
            note(kept);
        }
        for left_out in [
            ".hidden.md",
            ".obsidian/workspace.md",
            "ignored.md",
            "build/output.md",
            "scratch.md",
            "drafts/idea.md",
            "work/run.log",
        ] {
            note(left_out);
        }
        write(
            &root.join(".gitignore"),
            b"ignored.md\nbuild/\n",
            EVERYONE_READS,
        );
        write(&root.join(".ignore"), b"scratch.md\n", EVERYONE_READS);
        let oversized = vec![b'a'; usize::try_from(documents.max_file_bytes()).unwrap() + 1];
        write(&root.join("huge.txt"), &oversized, EVERYONE_READS);
        write(&root.join("token.md"), b"secret", OWNER_ONLY);
        write(&outside.join("elsewhere.md"), b"text", EVERYONE_READS);
        symlink(&outside, root.join("linked-dir")).expect("directory symlink");
        symlink(outside.join("elsewhere.md"), root.join("linked.md")).expect("file symlink");
        let guards = PathGuards::read(GuardRoots {
            home: Some(home_path),
            ..GuardRoots::default()
        });

        let (mut candidates, mut skipped) = (Vec::new(), Vec::new());
        IndexRoot::new(&root, &documents)
            .expect("valid patterns")
            .walk(&guards, |walked| {
                match walked {
                    Walked::Candidate { path, .. } => candidates.push(path),
                    Walked::Skipped(cause) => skipped.push(cause),
                    Walked::Unlisted(error) => panic!("unlisted: {error}"),
                }
                ControlFlow::Continue(())
            });
        candidates.sort();
        skipped.sort();

        assert_eq!(
            candidates,
            [root.join("plan.md"), root.join("work/deep.md")]
        );
        assert_eq!(skipped, [SkipCause::Sensitive, SkipCause::TooLarge]);
    }
}

//! Finding a program to spawn, and saying why one that exists on disk
//! cannot be run.

use std::ffi::OsStr;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use rustix::fs::Access;
use thiserror::Error;

/// Why [`find_program`] found nothing this user can run.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProgramError {
    #[error("not found on PATH")]
    NotOnPath,

    #[error("{} does not exist", path.display())]
    Missing { path: PathBuf },

    #[error("{} is not an executable file for this user", path.display())]
    NotExecutable { path: PathBuf },

    #[error(
        "{} links to {}, which this user cannot reach because a directory on the way \
         denies access",
        link.display(),
        target.display()
    )]
    TargetUnreachable { link: PathBuf, target: PathBuf },

    #[error("{} links to {}, which does not exist", link.display(), target.display())]
    TargetMissing { link: PathBuf, target: PathBuf },

    #[error("{} cannot be inspected: {kind}", path.display())]
    Inaccessible { path: PathBuf, kind: ErrorKind },
}

/// The executable `program` names: itself when it contains a `/`, else the
/// first match in `path_env`'s directories. When nothing runnable is found,
/// the error describes the first entry that exists under that name.
pub fn find_program(program: &Path, path_env: &OsStr) -> Result<PathBuf, ProgramError> {
    let candidates = candidate_paths(program, path_env);
    if let Some(found) = candidates.iter().find(|path| is_executable_file(path)) {
        return Ok(found.clone());
    }
    let existing = candidates
        .into_iter()
        .find(|path| path.symlink_metadata().is_ok());
    match existing {
        Some(path) => Err(why_unrunnable(path)),
        None if is_path(program) => Err(ProgramError::Missing {
            path: program.to_path_buf(),
        }),
        None => Err(ProgramError::NotOnPath),
    }
}

fn candidate_paths(program: &Path, path_env: &OsStr) -> Vec<PathBuf> {
    if is_path(program) {
        return vec![program.to_path_buf()];
    }
    std::env::split_paths(path_env)
        .filter(|dir| !dir.as_os_str().is_empty())
        .map(|dir| dir.join(program))
        .collect()
}

fn is_path(program: &Path) -> bool {
    program.as_os_str().as_encoded_bytes().contains(&b'/')
}

fn is_executable_file(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|meta| meta.is_file())
        && rustix::fs::access(path, Access::EXEC_OK).is_ok()
}

fn why_unrunnable(path: PathBuf) -> ProgramError {
    match (fs::metadata(&path), fs::read_link(&path)) {
        (Ok(_), _) => ProgramError::NotExecutable { path },
        (Err(e), Ok(target)) if e.kind() == ErrorKind::PermissionDenied => {
            ProgramError::TargetUnreachable { link: path, target }
        }
        (Err(e), Ok(target)) if e.kind() == ErrorKind::NotFound => {
            ProgramError::TargetMissing { link: path, target }
        }
        (Err(e), _) => ProgramError::Inaccessible {
            path,
            kind: e.kind(),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::fs::Permissions;
    use std::os::unix::fs::{PermissionsExt, symlink};

    use super::*;

    #[test]
    fn an_unrunnable_entry_says_why_and_a_later_runnable_one_wins() {
        let root = tempfile::tempdir().unwrap();
        let [locked, first, second] = ["locked", "first", "second"].map(|dir| {
            let dir = root.path().join(dir);
            fs::create_dir(&dir).unwrap();
            dir
        });
        for (name, mode) in [("runnable", 0o755), ("plain", 0o644)] {
            fs::write(locked.join(name), "#!/bin/sh\n").unwrap();
            fs::write(first.join(name), "#!/bin/sh\n").unwrap();
            fs::set_permissions(first.join(name), Permissions::from_mode(mode)).unwrap();
        }
        symlink(locked.join("runnable"), first.join("locked")).unwrap();
        symlink(root.path().join("gone"), first.join("dangling")).unwrap();
        fs::copy(first.join("runnable"), second.join("plain")).unwrap();
        fs::set_permissions(&locked, Permissions::from_mode(0o000)).unwrap();
        let path_env = std::env::join_paths([&first, &second]).unwrap();
        let find = |name: &str| find_program(Path::new(name), &path_env);

        assert_eq!(find("plain"), Ok(second.join("plain")));
        assert_eq!(
            find("locked"),
            Err(ProgramError::TargetUnreachable {
                link: first.join("locked"),
                target: locked.join("runnable"),
            })
        );
        assert_eq!(
            find("dangling"),
            Err(ProgramError::TargetMissing {
                link: first.join("dangling"),
                target: root.path().join("gone"),
            })
        );
        fs::set_permissions(&locked, Permissions::from_mode(0o755)).unwrap();
        assert_eq!(find("runnable"), Ok(first.join("runnable")));
        assert_eq!(find("absent"), Err(ProgramError::NotOnPath));
    }
}

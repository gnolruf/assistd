//! Where the daemon's socket lives, and the owner-only directory and lock files beside it.

use std::env;
use std::fs::{DirBuilder, File, OpenOptions};
use std::io;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use rustix::fs::OFlags;
use rustix::process::geteuid;
use thiserror::Error;

/// Mode for a socket directory the daemon has to create itself.
const SOCKET_DIR_MODE: u32 = 0o700;

/// Mode for lock files beside the socket.
const LOCK_FILE_MODE: u32 = 0o600;

/// Why the socket directory, or a lock file in it, cannot be used.
#[derive(Debug, Error)]
pub enum SocketDirError {
    #[error("failed to open lock file at {path}: {source}")]
    LockFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to prepare socket directory {path}: {source}")]
    Create {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("socket directory {path} is {problem}; refusing to listen there")]
    Unsafe {
        path: PathBuf,
        problem: &'static str,
    },
}

/// The per-user daemon socket: `$XDG_RUNTIME_DIR/assistd.sock`, else
/// `/tmp/assistd-<euid>/assistd.sock`, whose directory the daemon creates owner-only.
pub fn socket_path() -> PathBuf {
    let runtime_dir = env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute());
    socket_path_for(runtime_dir, geteuid().as_raw())
}

fn socket_path_for(runtime_dir: Option<PathBuf>, euid: u32) -> PathBuf {
    runtime_dir
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/assistd-{euid}")))
        .join("assistd.sock")
}

/// Open (creating owner-only if needed) a lock file in the socket
/// directory without following a symlink at `lock_path`. Fails when the
/// directory is not a private directory owned by this user.
pub fn open_private_lock_file(lock_path: &Path) -> Result<File, SocketDirError> {
    if let Some(dir) = lock_path.parent() {
        ensure_private_socket_dir(dir)?;
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(LOCK_FILE_MODE)
        .custom_flags(OFlags::NOFOLLOW.bits().cast_signed())
        .open(lock_path)
        .map_err(|source| SocketDirError::LockFile {
            path: lock_path.to_path_buf(),
            source,
        })
}

/// Create `dir` owner-only if it is missing. Fails unless it is then a directory owned by
/// this user that no other user may write to.
pub fn ensure_private_socket_dir(dir: &Path) -> Result<(), SocketDirError> {
    let dir_error = |source| SocketDirError::Create {
        path: dir.to_path_buf(),
        source,
    };
    DirBuilder::new()
        .recursive(true)
        .mode(SOCKET_DIR_MODE)
        .create(dir)
        .map_err(dir_error)?;
    let metadata = std::fs::symlink_metadata(dir).map_err(dir_error)?;
    let problem = if !metadata.is_dir() {
        Some("not a directory")
    } else if metadata.uid() != geteuid().as_raw() {
        Some("owned by another user")
    } else if metadata.mode() & 0o022 != 0 {
        Some("writable by other users")
    } else {
        None
    };
    problem.map_or(Ok(()), |problem| {
        Err(SocketDirError::Unsafe {
            path: dir.to_path_buf(),
            problem,
        })
    })
}

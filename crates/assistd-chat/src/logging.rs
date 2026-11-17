//! Where the chat logs while it owns the terminal: an owner-only directory
//! holding the daily trace log and the file stderr points at.

use std::fs::{DirBuilder, Permissions};
use std::io;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rustix::fs::{Mode, OFlags};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, fmt};

/// Owner-only modes for the log directory and the stderr log.
const LOG_DIR_MODE: u32 = 0o700;
const LOG_FILE_MODE: u32 = 0o600;

/// Points fd 2 at a log file, restoring the original stderr on drop so
/// errors returned after the TUI exits still reach the terminal.
pub(super) struct StderrRedirect {
    original: OwnedFd,
}

impl StderrRedirect {
    pub(super) fn to_log() -> Result<Self> {
        let path = log_dir()?.join("chat-stderr.log");
        let file = open_private_log(&path)
            .with_context(|| format!("opening stderr log {}", path.display()))?;
        let original = rustix::io::dup(io::stderr().as_fd()).context("dup stderr")?;
        rustix::stdio::dup2_stderr(&file)
            .with_context(|| format!("dup2 stderr → {}", path.display()))?;
        Ok(Self { original })
    }
}

impl Drop for StderrRedirect {
    fn drop(&mut self) {
        let _ = rustix::stdio::dup2_stderr(&self.original);
    }
}

/// The owner-only chat log directory under the XDG state dir, else the
/// runtime dir. Fails when neither is known or the directory is not a real
/// directory owned by this user.
fn log_dir() -> Result<PathBuf> {
    let dir = dirs::state_dir()
        .or_else(dirs::runtime_dir)
        .context("no XDG state or runtime directory for chat logs")?
        .join("assistd");
    DirBuilder::new()
        .recursive(true)
        .mode(LOG_DIR_MODE)
        .create(&dir)
        .with_context(|| format!("creating log dir {}", dir.display()))?;
    restrict_log_dir(&dir).with_context(|| format!("securing log dir {}", dir.display()))?;
    Ok(dir)
}

fn restrict_log_dir(dir: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(dir)?;
    anyhow::ensure!(metadata.is_dir(), "not a directory");
    anyhow::ensure!(
        metadata.uid() == rustix::process::geteuid().as_raw(),
        "owned by another user"
    );
    if metadata.mode() & 0o077 != 0 {
        std::fs::set_permissions(dir, Permissions::from_mode(LOG_DIR_MODE))?;
    }
    Ok(())
}

fn open_private_log(path: &Path) -> io::Result<OwnedFd> {
    let flags =
        OFlags::WRONLY | OFlags::CREATE | OFlags::APPEND | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let fd = rustix::fs::open(path, flags, Mode::from_raw_mode(LOG_FILE_MODE))?;
    rustix::fs::fchmod(&fd, Mode::from_raw_mode(LOG_FILE_MODE))?;
    Ok(fd)
}

pub(super) fn init_file_tracing() -> Result<WorkerGuard> {
    let file_appender = tracing_appender::rolling::daily(log_dir()?, "chat.log");
    let (writer, guard) = tracing_appender::non_blocking(file_appender);

    fmt()
        .with_writer(writer)
        .with_ansi(false)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    Ok(guard)
}

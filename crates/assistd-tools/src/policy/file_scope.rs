//! Where the in-daemon file commands resolve the paths they are given: the
//! host's whole filesystem, or only what lies beneath one directory and the
//! inputs mounted inside it.

use std::fs::{File, Metadata, ReadDir};
use std::io::{self, ErrorKind};
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustix::fs::{FileType, Mode, OFlags, ResolveFlags};
use rustix::io::Errno;

use assistd_utils::path::lexical_clean;

/// The view of the filesystem a command's file arguments resolve in.
/// Every method touches the filesystem, so call it on the blocking pool.
#[derive(Debug, Clone, Default)]
pub enum FileScope {
    /// The daemon's own view: any path the user could open.
    #[default]
    Host,
    /// Only paths beneath a [`ConfinedRoot`].
    Confined(Arc<ConfinedRoot>),
}

impl FileScope {
    /// Open the regular file or directory at `path` read-only with its size,
    /// refusing devices, FIFOs and sockets as `assistd_utils::fs::open_regular_or_dir` does.
    pub(crate) fn open_regular_or_dir(&self, path: &Path) -> io::Result<(File, u64)> {
        match self {
            Self::Host => assistd_utils::fs::open_regular_or_dir(path),
            Self::Confined(root) => {
                let fd = root.open_beneath(path, OFlags::PATH)?;
                assistd_utils::fs::open_regular_or_dir(&reopen_path(&fd))
            }
        }
    }

    /// The entries of the directory at `path`. Each entry's own path is not
    /// meaningful under [`FileScope::Confined`]: join its name onto `path`.
    pub(crate) fn read_dir(&self, path: &Path) -> io::Result<ReadDir> {
        match self {
            Self::Host => std::fs::read_dir(path),
            Self::Confined(root) => {
                let fd = root.open_beneath(path, OFlags::PATH | OFlags::DIRECTORY)?;
                std::fs::read_dir(reopen_path(&fd))
            }
        }
    }

    /// The metadata of `path` itself, not of what a symlink there points at.
    pub(crate) fn symlink_metadata(&self, path: &Path) -> io::Result<Metadata> {
        match self {
            Self::Host => std::fs::symlink_metadata(path),
            Self::Confined(root) => {
                File::from(root.open_beneath(path, OFlags::PATH | OFlags::NOFOLLOW)?).metadata()
            }
        }
    }

    /// The metadata of what `path` names, following symlinks.
    pub(crate) fn metadata(&self, path: &Path) -> io::Result<Metadata> {
        match self {
            Self::Host => std::fs::metadata(path),
            Self::Confined(root) => File::from(root.open_beneath(path, OFlags::PATH)?).metadata(),
        }
    }

    /// The directory `~` expands to: `$HOME` on the host, the root's home
    /// when confined.
    pub(crate) fn expand_tilde(&self, raw: &str) -> PathBuf {
        match self {
            Self::Host => assistd_utils::path::expand_tilde_from_env(raw),
            Self::Confined(root) => assistd_utils::path::expand_tilde(raw, &root.home),
        }
    }
}

/// A directory the kernel resolves paths beneath, refusing `..`, absolute
/// symlinks and any other route out of it, with host files or directories
/// mounted at paths inside it. Relative paths start at `home`.
#[derive(Debug)]
pub struct ConfinedRoot {
    dir: PathBuf,
    fd: OwnedFd,
    home: PathBuf,
    mounts: Vec<Mount>,
}

/// A host file or directory, held open, that paths at and beneath `at`
/// resolve in, confined to it as the root is to its directory.
#[derive(Debug)]
struct Mount {
    at: PathBuf,
    fd: OwnedFd,
    is_dir: bool,
}

impl ConfinedRoot {
    /// `dir`, which must be canonical, with relative paths resolving under
    /// `home`, one of its subdirectories.
    pub(crate) fn open(dir: PathBuf, home: PathBuf) -> io::Result<Self> {
        let fd = rustix::fs::open(
            &dir,
            OFlags::PATH | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )?;
        Ok(Self {
            dir,
            fd,
            home,
            mounts: Vec::new(),
        })
    }

    /// Resolve `at`, a path beneath the root that no other mount covers, and
    /// everything under it in the file or directory `fd` was opened on.
    pub(crate) fn mount(&mut self, at: PathBuf, fd: OwnedFd) -> io::Result<()> {
        let is_dir =
            FileType::from_raw_mode(rustix::fs::fstat(&fd)?.st_mode) == FileType::Directory;
        self.mounts.push(Mount { at, fd, is_dir });
        Ok(())
    }

    /// An `O_PATH` descriptor for `path`, which must name a place beneath the
    /// root; anything else fails with `PermissionDenied`. `..` is resolved
    /// lexically first, so a path climbs into or out of a mount by its name.
    fn open_beneath(&self, path: &Path, flags: OFlags) -> io::Result<OwnedFd> {
        let absolute = lexical_clean(&self.home.join(path));
        let mounted = self.mounts.iter().find_map(|mount| {
            let rest = absolute.strip_prefix(&mount.at).ok()?;
            Some((mount, rest))
        });
        if let Some((mount, rest)) = mounted {
            return mount.open(rest, flags, path);
        }
        let beneath = absolute
            .strip_prefix(&self.dir)
            .map_err(|_| outside_root(path))?;
        open_confined(&self.fd, beneath, flags, path)
    }
}

impl Mount {
    /// `rest` beneath the mount; for a mounted file, only the file itself.
    fn open(&self, rest: &Path, flags: OFlags, path: &Path) -> io::Result<OwnedFd> {
        if self.is_dir {
            return open_confined(&self.fd, rest, flags, path);
        }
        if !rest.as_os_str().is_empty() || flags.contains(OFlags::DIRECTORY) {
            return Err(Errno::NOTDIR.into());
        }
        self.fd.try_clone()
    }
}

/// `beneath` opened under the directory `fd`, mapping an escape to
/// [`outside_root`] for `path`.
fn open_confined(fd: &OwnedFd, beneath: &Path, flags: OFlags, path: &Path) -> io::Result<OwnedFd> {
    let beneath = if beneath.as_os_str().is_empty() {
        Path::new(".")
    } else {
        beneath
    };
    rustix::fs::openat2(
        fd,
        beneath,
        flags | OFlags::CLOEXEC,
        Mode::empty(),
        ResolveFlags::BENEATH | ResolveFlags::NO_MAGICLINKS,
    )
    .map_err(|errno| match errno {
        Errno::XDEV => outside_root(path),
        errno => errno.into(),
    })
}

fn outside_root(path: &Path) -> io::Error {
    io::Error::new(
        ErrorKind::PermissionDenied,
        format!("{} is outside the sandbox", path.display()),
    )
}

/// The procfs path that opens the very file `fd` refers to, with no path
/// resolved again.
fn reopen_path(fd: &OwnedFd) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{}", fd.as_raw_fd()))
}

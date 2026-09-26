//! Restricted Wayland sockets from the `wp-security-context-v1` protocol,
//! over which the compositor withholds privileged protocols.

use std::io::{self, PipeWriter};
#[cfg(feature = "wayland")]
use std::os::fd::AsFd;
use std::os::unix::fs::MetadataExt;
#[cfg(feature = "wayland")]
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};

use rustix::event::{PollFd, PollFlags, Timespec, poll};

#[cfg(feature = "wayland")]
use wayland_client::globals::{BindError, GlobalError, GlobalListContents, registry_queue_init};
#[cfg(feature = "wayland")]
use wayland_client::protocol::wl_registry::{self, WlRegistry};
#[cfg(feature = "wayland")]
use wayland_client::{
    ConnectError, Connection, Dispatch, DispatchError, QueueHandle, delegate_noop,
};
#[cfg(feature = "wayland")]
use wayland_protocols::wp::security_context::v1::client::{
    wp_security_context_manager_v1::WpSecurityContextManagerV1,
    wp_security_context_v1::WpSecurityContextV1,
};

/// The name the compositor records as the sandbox engine of every client
/// connecting through a [`RestrictedWaylandSocket`].
#[cfg(feature = "wayland")]
const SANDBOX_ENGINE: &str = "assistd";

/// Why a [`RestrictedWaylandSocket`] could not be created.
#[derive(thiserror::Error, Debug)]
pub enum SecurityContextError {
    #[error("assistd was built without the `wayland` feature")]
    NotBuilt,

    #[cfg(feature = "wayland")]
    #[error("cannot connect to the Wayland compositor: {0}")]
    Connect(#[from] ConnectError),

    #[error("the compositor does not support wp-security-context-v1 (sway needs 1.9+)")]
    Unsupported,

    #[error("Wayland protocol error: {0}")]
    Protocol(String),

    #[error("restricted socket: {0}")]
    Io(#[from] io::Error),
}

/// A listening socket whose clients the compositor treats as sandboxed:
/// it hides privileged globals from them, such as virtual keyboards and
/// screen capture. The compositor stops accepting on it once dropped.
#[derive(Debug)]
pub struct RestrictedWaylandSocket {
    path: PathBuf,
    bound: (u64, u64),
    stop_listening: PipeWriter,
}

#[cfg(feature = "wayland")]
struct Registry;

impl RestrictedWaylandSocket {
    /// Bind a socket at `path`, replacing any stale one, and hand it to the
    /// compositor `$WAYLAND_DISPLAY` names. Blocks for one round trip.
    ///
    /// # Errors
    /// [`SecurityContextError::Unsupported`] when the compositor lacks the
    /// protocol; the other variants when it or the socket is unreachable.
    #[cfg(feature = "wayland")]
    pub fn create(path: PathBuf) -> Result<Self, SecurityContextError> {
        let conn = Connection::connect_to_env()?;
        let (globals, mut queue) = registry_queue_init::<Registry>(&conn)?;
        let manager: WpSecurityContextManagerV1 = globals.bind(&queue.handle(), 1..=1, ())?;
        let listener = bind_listener(&path)?;
        let bound = file_identity(&path)?;
        let (stop_reader, stop_writer) = io::pipe()?;
        let context =
            manager.create_listener(listener.as_fd(), stop_reader.as_fd(), &queue.handle(), ());
        context.set_sandbox_engine(SANDBOX_ENGINE.into());
        context.commit();
        context.destroy();
        manager.destroy();
        queue.roundtrip(&mut Registry)?;
        Ok(Self {
            path,
            bound,
            stop_listening: stop_writer,
        })
    }

    /// Always [`SecurityContextError::NotBuilt`]: this build has no
    /// Wayland client.
    ///
    /// # Errors
    /// Always.
    #[cfg(not(feature = "wayland"))]
    pub fn create(_path: PathBuf) -> Result<Self, SecurityContextError> {
        Err(SecurityContextError::NotBuilt)
    }

    /// Where the socket is bound on the host.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the compositor still accepts on the socket: `false` once it
    /// has exited, as when it restarts.
    pub fn is_listening(&self) -> bool {
        !reader_hung_up(&self.stop_listening)
    }
}

impl Drop for RestrictedWaylandSocket {
    fn drop(&mut self) {
        if file_identity(&self.path).is_ok_and(|identity| identity == self.bound) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(feature = "wayland")]
impl From<GlobalError> for SecurityContextError {
    fn from(err: GlobalError) -> Self {
        Self::Protocol(err.to_string())
    }
}

#[cfg(feature = "wayland")]
impl From<BindError> for SecurityContextError {
    fn from(_: BindError) -> Self {
        Self::Unsupported
    }
}

#[cfg(feature = "wayland")]
impl From<DispatchError> for SecurityContextError {
    fn from(err: DispatchError) -> Self {
        Self::Protocol(err.to_string())
    }
}

#[cfg(feature = "wayland")]
impl Dispatch<WlRegistry, GlobalListContents> for Registry {
    fn event(
        _: &mut Self,
        _: &WlRegistry,
        _: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

#[cfg(feature = "wayland")]
delegate_noop!(Registry: WpSecurityContextManagerV1);
#[cfg(feature = "wayland")]
delegate_noop!(Registry: WpSecurityContextV1);

fn file_identity(path: &Path) -> io::Result<(u64, u64)> {
    let metadata = std::fs::symlink_metadata(path)?;
    Ok((metadata.dev(), metadata.ino()))
}

fn reader_hung_up(writer: &PipeWriter) -> bool {
    let mut fds = [PollFd::new(writer, PollFlags::empty())];
    let immediately = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    poll(&mut fds, Some(&immediately)).map_or(true, |_| {
        fds[0].revents().intersects(PollFlags::ERR | PollFlags::HUP)
    })
}

/// A non-blocking listener at `path`, so the compositor's accept never
/// stalls on a client that hung up first.
#[cfg(feature = "wayland")]
fn bind_listener(path: &Path) -> io::Result<UnixListener> {
    match std::fs::remove_file(path) {
        Err(err) if err.kind() != io::ErrorKind::NotFound => return Err(err),
        _ => {}
    }
    let listener = UnixListener::bind(path)?;
    listener.set_nonblocking(true)?;
    Ok(listener)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pipe_reads_as_hung_up_only_once_its_reader_closes() {
        let (reader, writer) = io::pipe().expect("pipe");
        assert!(!reader_hung_up(&writer));
        drop(reader);
        assert!(reader_hung_up(&writer));
    }
}

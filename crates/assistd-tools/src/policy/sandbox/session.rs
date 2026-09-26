//! What a sandboxed graphical launch reaches: a restricted Wayland socket
//! in place of the compositor's, and no abstract Unix socket.

use std::ffi::OsString;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use assistd_wm::{RestrictedWaylandSocket, SecurityContextError};
use landlock::{CompatLevel, Compatible, Ruleset, RulesetAttr, RulesetError, Scope};
use tokio::process::{Child, Command as ProcCommand};
use tokio::sync::Mutex;
use tracing::info;

use super::SandboxAccess;

/// File name of the restricted socket, under `$XDG_RUNTIME_DIR`.
const RESTRICTED_SOCKET_NAME: &str = "assistd-wayland.sock";

/// Why a graphical launch was refused or did not start.
#[derive(Debug, thiserror::Error)]
pub enum LaunchError {
    #[error(
        "no Wayland session (WAYLAND_DISPLAY or XDG_RUNTIME_DIR is unset); applications \
         launch only on Wayland, since any X11 client can inject keystrokes"
    )]
    NoWayland,
    #[error("cannot restrict the compositor's protocols: {0}")]
    SecurityContext(#[from] SecurityContextError),
    #[error(
        "the kernel cannot keep launched applications off abstract sockets such as the X \
         server's (needs Landlock scoping, Linux 6.12+): {0}"
    )]
    Landlock(#[from] RulesetError),
    #[error("spawn failed: {0}")]
    Spawn(#[source] io::Error),
}

/// A restricted socket, and the path clients look for the compositor at,
/// where the sandbox binds it.
#[derive(Debug)]
pub(super) struct SessionDisplay {
    display: PathBuf,
    socket: RestrictedWaylandSocket,
}

impl SessionDisplay {
    pub(super) fn access(&self) -> SandboxAccess<'_> {
        SandboxAccess::Session {
            restricted: self.socket.path(),
            display: &self.display,
        }
    }
}

/// The [`SessionDisplay`] every launch shares, created on first use and
/// again whenever the compositor stops listening on it, as on a restart.
#[derive(Debug, Default)]
pub(super) struct SharedDisplay(Mutex<Option<Arc<SessionDisplay>>>);

impl SharedDisplay {
    pub(super) async fn get(&self) -> Result<Arc<SessionDisplay>, LaunchError> {
        let mut slot = self.0.lock().await;
        if let Some(stale) = slot.take_if(|display| !display.socket.is_listening()) {
            info!(
                target: "assistd::policy",
                socket = %stale.socket.path().display(),
                "compositor stopped listening on the restricted Wayland socket; recreating it"
            );
        }
        if let Some(display) = slot.as_ref() {
            return Ok(display.clone());
        }
        let display = Arc::new(create_display().await?);
        *slot = Some(display.clone());
        Ok(display)
    }
}

async fn create_display() -> Result<SessionDisplay, LaunchError> {
    let runtime_dir = std::env::var_os("XDG_RUNTIME_DIR").ok_or(LaunchError::NoWayland)?;
    let display = wayland_socket(
        Some(runtime_dir.clone()),
        std::env::var_os("WAYLAND_DISPLAY"),
    )
    .ok_or(LaunchError::NoWayland)?;
    let path = PathBuf::from(runtime_dir).join(RESTRICTED_SOCKET_NAME);
    let socket = tokio::task::spawn_blocking(move || RestrictedWaylandSocket::create(path))
        .await
        .map_err(|e| SecurityContextError::Io(io::Error::other(e)))??;
    Ok(SessionDisplay { display, socket })
}

/// Where Wayland clients connect: `display` itself when absolute, else
/// under `runtime_dir`.
pub(super) fn wayland_socket(
    runtime_dir: Option<OsString>,
    display: Option<OsString>,
) -> Option<PathBuf> {
    let display = PathBuf::from(display?);
    if display.is_absolute() {
        return Some(display);
    }
    Some(PathBuf::from(runtime_dir?).join(display))
}

/// Spawn `cmd` from a new thread that Landlock bars from connecting to
/// abstract Unix sockets made outside it, such as the X server's; the child
/// inherits the restriction and the daemon's other threads keep none.
pub(super) fn spawn_without_abstract_sockets(mut cmd: ProcCommand) -> Result<Child, LaunchError> {
    let runtime = tokio::runtime::Handle::current();
    std::thread::spawn(move || {
        let _runtime = runtime.enter();
        forbid_abstract_sockets()?;
        cmd.spawn().map_err(LaunchError::Spawn)
    })
    .join()
    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

fn forbid_abstract_sockets() -> Result<(), RulesetError> {
    Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .scope(Scope::AbstractUnixSocket)?
        .create()?
        .restrict_self()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};

    use super::*;

    /// Returns early where the kernel lacks Landlock scoping.
    #[test]
    fn confined_thread_reaches_path_sockets_but_not_abstract_ones() {
        let name = format!("assistd-landlock-test-{}", std::process::id());
        let abstract_addr = SocketAddr::from_abstract_name(&name).expect("abstract address");
        let _abstract = UnixListener::bind_addr(&abstract_addr).expect("abstract listener");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("wayland-1");
        let _path_listener = UnixListener::bind(&path).expect("path listener");

        let confined_addr = abstract_addr.clone();
        let reached = std::thread::spawn(move || {
            forbid_abstract_sockets().ok()?;
            Some((
                UnixStream::connect_addr(&confined_addr).is_ok(),
                UnixStream::connect(&path).is_ok(),
            ))
        })
        .join()
        .expect("confined thread");
        let Some((abstract_reached, path_reached)) = reached else {
            return;
        };

        assert!(
            !abstract_reached,
            "confined thread reached an abstract socket"
        );
        assert!(path_reached, "confined thread lost path sockets");
        assert!(
            UnixStream::connect_addr(&abstract_addr).is_ok(),
            "the restriction leaked to another thread"
        );
    }

    #[test]
    fn wayland_socket_resolves_a_relative_display_under_the_runtime_dir() {
        let os = |text: &str| Some(OsString::from(text));
        assert_eq!(
            wayland_socket(os("/run/user/1000"), os("wayland-1")),
            Some(PathBuf::from("/run/user/1000/wayland-1"))
        );
        assert_eq!(
            wayland_socket(None, os("/run/wl.sock")),
            Some(PathBuf::from("/run/wl.sock"))
        );
        assert_eq!(wayland_socket(None, os("wayland-1")), None);
        assert_eq!(wayland_socket(os("/run/user/1000"), None), None);
    }
}

//! Window-manager subsystem wiring for the daemon.
//!
//! Thin wrapper over [`crate::wm_backend::start_backend`]: exposes the
//! started backend as an [`Arc`]`<dyn WindowManager>` for tools plus a
//! [`WmHandle`] for shutdown.

use std::sync::Arc;

use assistd_core::{Config, WindowManager};
use assistd_wm::WmHandle;
use tokio::sync::watch;

use crate::wm_backend::{WmBackend, start_backend};

/// Live handles for the window-manager subsystem, returned by [`init`].
pub struct WindowSubsystem {
    /// Window-manager trait object (no-op when no backend is available).
    pub manager: Arc<dyn WindowManager>,
    /// Backend handle used to shut down the WM connection at exit.
    pub handle: Option<WmHandle>,
}

impl WindowSubsystem {
    /// Shut down the window-manager backend gracefully.
    pub async fn shutdown(self) {
        if let Some(h) = self.handle {
            h.shutdown().await;
        }
    }
}

/// Detect and start the appropriate window-manager backend.
///
/// Falls back to a no-op backend when auto-detection finds no supported
/// compositor or when the backend fails to connect.
pub async fn init(config: &Config, shutdown_tx: &watch::Sender<bool>) -> WindowSubsystem {
    let WmBackend { manager, handle } = start_backend(config, shutdown_tx.subscribe()).await;
    WindowSubsystem { manager, handle }
}

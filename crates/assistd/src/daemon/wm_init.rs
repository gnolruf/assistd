//! Window-manager subsystem wiring for the daemon.

use std::sync::Arc;

use assistd_core::{Config, WindowManager};
use assistd_wm::{NoWindowManager, WmHandle};
use tokio::sync::watch;

pub(super) struct WindowSubsystem {
    pub manager: Arc<dyn WindowManager>,
    pub handle: Option<WmHandle>,
}

impl WindowSubsystem {
    pub(super) async fn shutdown(self) {
        if let Some(h) = self.handle {
            h.shutdown().await;
        }
    }
}

/// Start the configured (or detected) compositor backend, disconnected
/// when none is available.
pub(super) async fn init(config: &Config, shutdown_tx: &watch::Sender<bool>) -> WindowSubsystem {
    let Some(compositor) = config.compositor.resolve() else {
        tracing::info!(
            target: "assistd::wm",
            "no supported compositor configured or detected \
             (no $SWAYSOCK/$I3SOCK/$HYPRLAND_INSTANCE_SIGNATURE/$XDG_CURRENT_DESKTOP); \
             window operations disabled"
        );
        return WindowSubsystem {
            manager: Arc::new(NoWindowManager),
            handle: None,
        };
    };
    tracing::info!(target: "assistd::wm", "compositor = {compositor:?}");
    let (manager, handle) = assistd_wm::start(compositor, shutdown_tx.subscribe()).await;
    WindowSubsystem { manager, handle }
}

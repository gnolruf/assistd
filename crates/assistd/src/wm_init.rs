//! Window-manager subsystem wiring for the daemon.

use std::sync::Arc;

use assistd_core::{Config, WindowManager};
use assistd_wm::WmHandle;
use tokio::sync::watch;

use crate::wm_backend::{WmBackend, start_backend};

pub struct WindowSubsystem {
    pub manager: Arc<dyn WindowManager>,
    pub handle: Option<WmHandle>,
}

impl WindowSubsystem {
    pub async fn shutdown(self) {
        if let Some(h) = self.handle {
            h.shutdown().await;
        }
    }
}

pub async fn init(config: &Config, shutdown_tx: &watch::Sender<bool>) -> WindowSubsystem {
    let WmBackend { manager, handle } = start_backend(config, shutdown_tx.subscribe()).await;
    WindowSubsystem { manager, handle }
}

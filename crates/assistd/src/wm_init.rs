//! Window-manager subsystem wiring for the daemon.
//!
//! Resolves the configured compositor (auto-detect or explicit), starts
//! the matching backend from `assistd-wm`, and produces an
//! [`Arc<dyn WindowManager>`] for tools plus a [`WmHandle`] for shutdown.
//! Falls back to [`NoWindowManager`] when no backend is available.

use std::sync::Arc;

use assistd_core::{CompositorType, Config, NoWindowManager, WindowManager};
use assistd_wm::{I3Backend, SwayBackend, WmHandle};
use tokio::sync::watch;
use tracing::info;

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
/// Falls back to a no-op [`NoWindowManager`] when auto-detection finds
/// no supported compositor or when the backend fails to connect.
pub async fn init(config: &Config, shutdown_tx: &watch::Sender<bool>) -> WindowSubsystem {
    let resolved = match config.compositor.compositor_type {
        CompositorType::Auto => match assistd_core::config::compositor::detect_from_env(
            std::env::var_os("SWAYSOCK").is_some(),
            std::env::var_os("I3SOCK").is_some(),
            std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some(),
            std::env::var("XDG_CURRENT_DESKTOP").ok().as_deref(),
        ) {
            Some(c) => {
                info!("wm: auto-detected compositor = {:?}", c);
                c
            }
            None => {
                info!(
                    "wm: auto-detect found no supported compositor (no $SWAYSOCK/$I3SOCK/$HYPRLAND_INSTANCE_SIGNATURE/$XDG_CURRENT_DESKTOP); window ops disabled"
                );
                CompositorType::Auto
            }
        },
        explicit => explicit,
    };

    match resolved {
        CompositorType::I3 => match I3Backend::start(shutdown_tx.subscribe()).await {
            Ok(handle) => {
                info!("wm: i3 backend connected");
                WindowSubsystem {
                    manager: handle.backend.clone(),
                    handle: Some(WmHandle::I3(handle)),
                }
            }
            Err(e) => {
                tracing::warn!("wm: i3 backend unavailable ({e:#}); window ops disabled");
                WindowSubsystem {
                    manager: Arc::new(NoWindowManager),
                    handle: None,
                }
            }
        },
        CompositorType::Sway => match SwayBackend::start(shutdown_tx.subscribe()).await {
            Ok(handle) => {
                info!("wm: sway backend connected");
                WindowSubsystem {
                    manager: handle.backend.clone(),
                    handle: Some(WmHandle::Sway(handle)),
                }
            }
            Err(e) => {
                tracing::warn!("wm: sway backend unavailable ({e:#}); window ops disabled");
                WindowSubsystem {
                    manager: Arc::new(NoWindowManager),
                    handle: None,
                }
            }
        },
        CompositorType::Hyprland => {
            info!("wm: hyprland backend not yet implemented; window ops disabled");
            WindowSubsystem {
                manager: Arc::new(NoWindowManager),
                handle: None,
            }
        }
        CompositorType::Auto => WindowSubsystem {
            manager: Arc::new(NoWindowManager),
            handle: None,
        },
    }
}

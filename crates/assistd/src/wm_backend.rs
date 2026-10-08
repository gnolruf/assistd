//! Window-manager backend construction from config.

use std::sync::Arc;

use assistd_config::{
    CompositorType, Config,
    compositor::{SessionEnv, detect_from_env},
};
#[cfg(feature = "i3")]
use assistd_wm::I3Backend;
#[cfg(feature = "sway")]
use assistd_wm::SwayBackend;
use assistd_wm::{NoWindowManager, WindowManager, WmHandle};
use tokio::sync::watch;

pub(crate) struct WmBackend {
    pub manager: Arc<dyn WindowManager>,
    pub handle: Option<WmHandle>,
}

impl WmBackend {
    fn disconnected() -> Self {
        Self {
            manager: Arc::new(NoWindowManager),
            handle: None,
        }
    }
}

/// Start the configured (or detected) compositor backend, disconnected
/// when none is available.
pub(crate) async fn start_backend(
    config: &Config,
    shutdown_rx: watch::Receiver<bool>,
) -> WmBackend {
    connect(
        resolve_compositor(config.compositor.compositor_type),
        shutdown_rx,
    )
    .await
}

/// Resolve `Auto` from the session environment; stays `Auto` when no
/// supported compositor is detected.
fn resolve_compositor(configured: CompositorType) -> CompositorType {
    match configured {
        CompositorType::Auto => match detect_from_env(&SessionEnv::from_process()) {
            Some(c) => {
                tracing::info!(target: "assistd::wm", "auto-detected compositor = {c:?}");
                c
            }
            None => {
                tracing::info!(
                    target: "assistd::wm",
                    "auto-detect found no supported compositor \
                     (no $SWAYSOCK/$I3SOCK/$HYPRLAND_INSTANCE_SIGNATURE/$XDG_CURRENT_DESKTOP); \
                     window operations disabled"
                );
                CompositorType::Auto
            }
        },
        explicit => explicit,
    }
}

#[cfg(any(feature = "i3", feature = "sway"))]
async fn connect(compositor: CompositorType, shutdown_rx: watch::Receiver<bool>) -> WmBackend {
    match compositor {
        #[cfg(feature = "i3")]
        CompositorType::I3 => match I3Backend::start(shutdown_rx).await {
            Ok(handle) => {
                tracing::info!(target: "assistd::wm", "i3 backend connected");
                WmBackend {
                    manager: handle.backend.clone(),
                    handle: Some(WmHandle::I3(handle)),
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "assistd::wm",
                    "i3 backend unavailable ({e:#}); window operations disabled"
                );
                WmBackend::disconnected()
            }
        },
        #[cfg(feature = "sway")]
        CompositorType::Sway => match SwayBackend::start(shutdown_rx).await {
            Ok(handle) => {
                tracing::info!(target: "assistd::wm", "sway backend connected");
                WmBackend {
                    manager: handle.backend.clone(),
                    handle: Some(WmHandle::Sway(handle)),
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "assistd::wm",
                    "sway backend unavailable ({e:#}); window operations disabled"
                );
                WmBackend::disconnected()
            }
        },
        other => without_backend(other),
    }
}

#[cfg(not(any(feature = "i3", feature = "sway")))]
fn connect(
    compositor: CompositorType,
    _shutdown_rx: watch::Receiver<bool>,
) -> std::future::Ready<WmBackend> {
    std::future::ready(without_backend(compositor))
}

fn without_backend(compositor: CompositorType) -> WmBackend {
    match compositor {
        CompositorType::I3 => {
            tracing::warn!(
                target: "assistd::wm",
                "i3 backend not compiled into this build (feature `i3`); window operations disabled"
            );
        }
        CompositorType::Sway => {
            tracing::warn!(
                target: "assistd::wm",
                "sway backend not compiled into this build (feature `sway`); window operations disabled"
            );
        }
        CompositorType::Hyprland => {
            tracing::info!(
                target: "assistd::wm",
                "no hyprland backend; window operations disabled"
            );
        }
        CompositorType::Auto => {}
    }
    WmBackend::disconnected()
}

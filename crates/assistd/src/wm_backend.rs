//! Shared window-manager backend construction.
//!
//! The daemon ([`crate::wm_init`]) and the tray popup
//! ([`crate::tray::popup`]) both resolve the configured compositor and
//! start the matching `assistd-wm` backend. The logic lives here so the
//! two feature-gated call sites can't drift.

use std::sync::Arc;

use assistd_config::{CompositorType, Config, compositor::detect_from_env};
use assistd_wm::{I3Backend, NoWindowManager, SwayBackend, WindowManager, WmHandle};
use tokio::sync::watch;

/// A started window-manager backend: the trait object callers issue
/// operations through, plus the supervisor handle used to shut it down.
/// `handle` is `None` when no compositor backend connected, in which
/// case `manager` is a [`NoWindowManager`] that rejects every operation.
pub struct WmBackend {
    pub manager: Arc<dyn WindowManager>,
    pub handle: Option<WmHandle>,
}

impl WmBackend {
    /// A disconnected backend: every window operation fails fast.
    fn disconnected() -> Self {
        Self {
            manager: Arc::new(NoWindowManager),
            handle: None,
        }
    }
}

/// Resolve the configured compositor (explicit or auto-detected) and
/// start the matching backend.
///
/// Falls back to a disconnected [`WmBackend`] when no supported
/// compositor is found or the backend fails to connect, so callers can
/// degrade gracefully instead of aborting startup.
pub async fn start_backend(config: &Config, shutdown_rx: watch::Receiver<bool>) -> WmBackend {
    let resolved = match config.compositor.compositor_type {
        CompositorType::Auto => match detect_from_env(
            std::env::var_os("SWAYSOCK").is_some(),
            std::env::var_os("I3SOCK").is_some(),
            std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some(),
            std::env::var("XDG_CURRENT_DESKTOP").ok().as_deref(),
        ) {
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
    };

    match resolved {
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
        CompositorType::Hyprland => {
            tracing::info!(
                target: "assistd::wm",
                "hyprland backend not yet implemented; window operations disabled"
            );
            WmBackend::disconnected()
        }
        CompositorType::Auto => WmBackend::disconnected(),
    }
}

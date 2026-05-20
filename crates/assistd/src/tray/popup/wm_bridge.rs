//! Window-manager glue for the tray popup.
//!
//! Builds the appropriate backend at startup (using the same
//! auto-detect / explicit-config logic the daemon uses, copied here
//! because `wm_init.rs` is gated behind the `daemon` feature) and
//! drives a small placement worker that drains [`PlaceRequest`]s and
//! issues [`WindowManager::place_floating`] calls. All errors are
//! logged at warn; placement is best-effort.

use std::sync::Arc;

use assistd_config::{CompositorType, Config, compositor::detect_from_env};
use assistd_wm::{
    AnchorCorner, I3Backend, NoWindowManager, PlacementAnchor, PlacementCriteria, SwayBackend,
    WindowManager, WmHandle,
};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::watch;

use super::visibility::PlaceRequest;

/// Pair returned by [`build_wm_backend`]: the trait object the
/// placement worker calls, plus the supervisor handle the popup module
/// drives on shutdown.
pub struct WmBackendBundle {
    pub manager: Arc<dyn WindowManager>,
    pub handle: Option<WmHandle>,
}

/// Detect (or read from config) the compositor and start the matching
/// backend. Mirrors `crates/assistd/src/wm_init.rs` so the tray can
/// construct a backend without depending on the `daemon` feature.
pub async fn build_wm_backend(cfg: &Config, shutdown_rx: watch::Receiver<bool>) -> WmBackendBundle {
    let resolved = match cfg.compositor.compositor_type {
        CompositorType::Auto => match detect_from_env(
            std::env::var_os("SWAYSOCK").is_some(),
            std::env::var_os("I3SOCK").is_some(),
            std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some(),
            std::env::var("XDG_CURRENT_DESKTOP").ok().as_deref(),
        ) {
            Some(c) => {
                tracing::info!(target: "tray", "popup: auto-detected compositor = {:?}", c);
                c
            }
            None => {
                tracing::info!(
                    target: "tray",
                    "popup: no supported compositor detected; placement disabled"
                );
                CompositorType::Auto
            }
        },
        explicit => explicit,
    };

    match resolved {
        CompositorType::I3 => match I3Backend::start(shutdown_rx).await {
            Ok(handle) => {
                tracing::info!(target: "tray", "popup: i3 backend connected");
                WmBackendBundle {
                    manager: handle.backend.clone(),
                    handle: Some(WmHandle::I3(handle)),
                }
            }
            Err(e) => {
                tracing::warn!(target: "tray", "popup: i3 backend unavailable ({e:#}); placement disabled");
                WmBackendBundle {
                    manager: Arc::new(NoWindowManager),
                    handle: None,
                }
            }
        },
        CompositorType::Sway => match SwayBackend::start(shutdown_rx).await {
            Ok(handle) => {
                tracing::info!(target: "tray", "popup: sway backend connected");
                WmBackendBundle {
                    manager: handle.backend.clone(),
                    handle: Some(WmHandle::Sway(handle)),
                }
            }
            Err(e) => {
                tracing::warn!(target: "tray", "popup: sway backend unavailable ({e:#}); placement disabled");
                WmBackendBundle {
                    manager: Arc::new(NoWindowManager),
                    handle: None,
                }
            }
        },
        CompositorType::Hyprland => {
            tracing::info!(target: "tray", "popup: hyprland backend not yet implemented; placement disabled");
            WmBackendBundle {
                manager: Arc::new(NoWindowManager),
                handle: None,
            }
        }
        CompositorType::Auto => WmBackendBundle {
            manager: Arc::new(NoWindowManager),
            handle: None,
        },
    }
}

/// Drain [`PlaceRequest`]s and call `place_floating` for each. The
/// criteria + anchor are fixed at spawn time (config doesn't change
/// while the tray is running); the only thing each call carries is
/// "please place me now."
pub async fn place_worker(
    mut rx: UnboundedReceiver<PlaceRequest>,
    backend: Arc<dyn WindowManager>,
    criteria: PlacementCriteria,
    anchor: PlacementAnchor,
) {
    while let Some(_req) = rx.recv().await {
        match backend.place_floating(&criteria, anchor).await {
            Ok(()) => {}
            Err(e) => {
                tracing::warn!(target: "tray", "popup: place_floating failed: {e}");
            }
        }
    }
}

/// Build a [`PlacementAnchor`] from the user's [`TrayPopupConfig`]
/// using the configured corner + offsets + width/height.
pub fn anchor_from_config(cfg: &assistd_config::TrayPopupConfig) -> PlacementAnchor {
    PlacementAnchor {
        corner: map_anchor(cfg.anchor),
        offset_x: cfg.offset_x,
        offset_y: cfg.offset_y,
        width: cfg.width,
        height: cfg.height,
    }
}

/// Construct a [`PlacementCriteria::AppId`] targeting the constant
/// app_id the popup window sets. Centralised so the GUI and placement
/// references stay in sync.
pub fn popup_criteria() -> PlacementCriteria {
    PlacementCriteria::AppId(assistd_config::defaults::DEFAULT_TRAY_POPUP_APP_ID.to_string())
}

/// Map the i3/sway anchor enum that `assistd-wm` expects from the
/// equivalent shape in `assistd-config`. Helper indirected through
/// [`AnchorCorner`] so neither crate needs to depend on the other for
/// this enum.
pub fn map_anchor(corner: assistd_config::PopupAnchor) -> AnchorCorner {
    match corner {
        assistd_config::PopupAnchor::TopLeft => AnchorCorner::TopLeft,
        assistd_config::PopupAnchor::TopRight => AnchorCorner::TopRight,
        assistd_config::PopupAnchor::BottomLeft => AnchorCorner::BottomLeft,
        assistd_config::PopupAnchor::BottomRight => AnchorCorner::BottomRight,
        assistd_config::PopupAnchor::Center => AnchorCorner::Center,
    }
}

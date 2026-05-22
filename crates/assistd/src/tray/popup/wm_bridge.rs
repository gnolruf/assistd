//! Window-manager glue for the tray popup.
//!
//! Starts the compositor backend via [`crate::wm_backend`] and drives a
//! small placement worker that drains [`PlaceRequest`]s and issues
//! [`WindowManager::place_floating`] calls. All errors are logged at
//! warn; placement is best-effort.

use std::sync::Arc;

use assistd_config::Config;
use assistd_wm::{AnchorCorner, PlacementAnchor, PlacementCriteria, WindowManager, WmHandle};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::watch;

use crate::wm_backend::{WmBackend, start_backend};

use super::visibility::PlaceRequest;

/// Pair returned by [`build_wm_backend`]: the trait object the
/// placement worker calls, plus the supervisor handle the popup module
/// drives on shutdown.
pub struct WmBackendBundle {
    pub manager: Arc<dyn WindowManager>,
    pub handle: Option<WmHandle>,
}

/// Start the compositor backend for the tray popup, adapting
/// [`crate::wm_backend::start_backend`]'s result into a
/// [`WmBackendBundle`].
pub async fn build_wm_backend(cfg: &Config, shutdown_rx: watch::Receiver<bool>) -> WmBackendBundle {
    let WmBackend { manager, handle } = start_backend(cfg, shutdown_rx).await;
    WmBackendBundle { manager, handle }
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

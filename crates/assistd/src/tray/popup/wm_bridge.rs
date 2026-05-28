//! Window-manager glue for the tray popup.

use std::sync::Arc;

use assistd_config::Config;
use assistd_wm::{AnchorCorner, PlacementAnchor, PlacementCriteria, WindowManager, WmHandle};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::watch;

use crate::wm_backend::{WmBackend, start_backend};

use super::visibility::PlaceRequest;

pub struct WmBackendBundle {
    pub manager: Arc<dyn WindowManager>,
    pub handle: Option<WmHandle>,
}

pub async fn build_wm_backend(cfg: &Config, shutdown_rx: watch::Receiver<bool>) -> WmBackendBundle {
    let WmBackend { manager, handle } = start_backend(cfg, shutdown_rx).await;
    WmBackendBundle { manager, handle }
}

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

pub fn anchor_from_config(cfg: &assistd_config::TrayPopupConfig) -> PlacementAnchor {
    PlacementAnchor {
        corner: map_anchor(cfg.anchor),
        offset_x: cfg.offset_x,
        offset_y: cfg.offset_y,
        width: cfg.width,
        height: cfg.height,
    }
}

pub fn popup_criteria() -> PlacementCriteria {
    PlacementCriteria::AppId(assistd_config::defaults::DEFAULT_TRAY_POPUP_APP_ID.to_string())
}

pub fn map_anchor(corner: assistd_config::PopupAnchor) -> AnchorCorner {
    match corner {
        assistd_config::PopupAnchor::TopLeft => AnchorCorner::TopLeft,
        assistd_config::PopupAnchor::TopRight => AnchorCorner::TopRight,
        assistd_config::PopupAnchor::BottomLeft => AnchorCorner::BottomLeft,
        assistd_config::PopupAnchor::BottomRight => AnchorCorner::BottomRight,
        assistd_config::PopupAnchor::Center => AnchorCorner::Center,
    }
}

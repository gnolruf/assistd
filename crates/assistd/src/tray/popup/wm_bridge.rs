//! Window-manager glue for the tray popup.

use std::sync::Arc;

use assistd_config::defaults::DEFAULT_TRAY_POPUP_APP_ID;
use assistd_config::{PopupAnchor, TrayPopupConfig};
use assistd_wm::{AnchorCorner, PlacementAnchor, PlacementCriteria, WindowManager};
use tokio::sync::mpsc::UnboundedReceiver;

use super::visibility::PlaceRequest;

pub async fn place_worker(
    mut rx: UnboundedReceiver<PlaceRequest>,
    backend: Arc<dyn WindowManager>,
    criteria: PlacementCriteria,
    anchor: PlacementAnchor,
) {
    while rx.recv().await.is_some() {
        if let Err(e) = backend.place_floating(&criteria, anchor).await {
            tracing::warn!(target: "tray", "popup: place_floating failed: {e}");
        }
    }
}

pub fn anchor_from_config(cfg: &TrayPopupConfig) -> PlacementAnchor {
    PlacementAnchor {
        corner: map_anchor(cfg.anchor),
        offset_x: cfg.offset_x,
        offset_y: cfg.offset_y,
        width: cfg.width,
        height: cfg.height,
    }
}

pub fn popup_criteria() -> PlacementCriteria {
    PlacementCriteria::AppId(DEFAULT_TRAY_POPUP_APP_ID.to_string())
}

pub fn map_anchor(corner: PopupAnchor) -> AnchorCorner {
    match corner {
        PopupAnchor::TopLeft => AnchorCorner::TopLeft,
        PopupAnchor::TopRight => AnchorCorner::TopRight,
        PopupAnchor::BottomLeft => AnchorCorner::BottomLeft,
        PopupAnchor::BottomRight => AnchorCorner::BottomRight,
        PopupAnchor::Center => AnchorCorner::Center,
    }
}

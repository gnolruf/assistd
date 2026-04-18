use serde::{Deserialize, Serialize};

/// Supported tiling compositors.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CompositorType {
    I3,
    Sway,
    Hyprland,
}

/// Compositor integration settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositorConfig {
    /// Which compositor to integrate with.
    #[serde(rename = "type")]
    pub compositor_type: CompositorType,
}

impl Default for CompositorConfig {
    fn default() -> Self {
        Self {
            compositor_type: CompositorType::Sway,
        }
    }
}

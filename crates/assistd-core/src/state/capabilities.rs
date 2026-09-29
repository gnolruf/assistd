//! `GetCapabilities` handler.

use std::sync::Arc;

use tokio::sync::mpsc;

use assistd_ipc::{Component, Event, StatusKind, StatusSeverity};
use assistd_llm::VisionState;

use super::AppState;

impl AppState {
    /// Report disabled tools and MCP startup failures, then the model name
    /// and whether llama-server supports vision, probed live rather than
    /// read from the vision gate.
    pub(super) async fn handle_get_capabilities(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) {
        if let Some(disabled) = self.subsystems.tools_disabled {
            let _ = tx
                .send(Event::Status {
                    id: id.clone(),
                    severity: StatusSeverity::Error,
                    component: Component::Agent,
                    event: StatusKind::StartupFailed,
                    message: disabled.to_string(),
                })
                .await;
        }
        for failure in &self.subsystems.mcp_startup_failures {
            let _ = tx
                .send(Event::Status {
                    id: id.clone(),
                    severity: StatusSeverity::Warning,
                    component: Component::Mcp,
                    event: StatusKind::StartupFailed,
                    message: format!(
                        "MCP server '{}' is not available: {}",
                        failure.server_name, failure.reason
                    ),
                })
                .await;
        }

        let probe = match &self.subsystems.vision_revalidator {
            Some(revalidator) => revalidator.probe().await,
            None => VisionState::default(),
        };
        let model_name = self.config.model.name.rsplit_once('/').map_or_else(
            || self.config.model.name.clone(),
            |(_, rest)| rest.to_string(),
        );
        let _ = tx
            .send(Event::Capabilities {
                id: id.clone(),
                vision: probe.vision_supported,
                model_name,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
    }
}

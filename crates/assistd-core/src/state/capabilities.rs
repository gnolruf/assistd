//! `GetCapabilities` handler.

use super::AppState;
use assistd_ipc::{Component, Event, StatusKind, StatusSeverity};
use std::sync::Arc;
use tokio::sync::mpsc;

impl AppState {
    /// Report MCP startup failures, then probe llama-server for vision
    /// support and the model name. Probes live rather than reading the
    /// vision gate, so the answer describes the server as it is now.
    pub(super) async fn handle_get_capabilities(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) {
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
            Some(rev) => rev.probe().await,
            None => assistd_llm::VisionState::default(),
        };
        let model_name = self
            .config
            .model
            .name
            .rsplit_once('/')
            .map(|(_, rest)| rest.to_string())
            .unwrap_or_else(|| self.config.model.name.clone());
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

//! `GetCapabilities` handler.

use assistd_ipc::{Component, ComponentReadiness, Event, EventBody, StatusKind, StatusSeverity};
use assistd_llm::VisionState;

use super::status_service::StatusService;
use crate::bus::EventSink;

impl StatusService {
    /// Report disabled tools, MCP servers that failed to start, the model
    /// name, and whether llama-server supports vision, probed live rather
    /// than read from the vision gate.
    pub(super) async fn handle_get_capabilities(&self, id: String, tx: EventSink) {
        if let Some(disabled) = &self.tools_disabled {
            let _ = tx
                .send(Event::new(
                    id.clone(),
                    EventBody::Status {
                        severity: StatusSeverity::Error,
                        component: Component::Agent,
                        event: StatusKind::StartupFailed,
                        message: disabled.to_string(),
                    },
                ))
                .await;
        }
        for server in self.mcp_servers.iter() {
            let ComponentReadiness::Unavailable { reason } = server.readiness() else {
                continue;
            };
            let _ = tx
                .send(Event::new(
                    id.clone(),
                    EventBody::Status {
                        severity: StatusSeverity::Warning,
                        component: Component::Mcp,
                        event: StatusKind::StartupFailed,
                        message: format!(
                            "MCP server '{}' is not available: {reason}",
                            server.name()
                        ),
                    },
                ))
                .await;
        }

        let probe = match &self.vision_revalidator {
            Some(revalidator) => revalidator.probe().await,
            None => VisionState::default(),
        };
        let model_name = self.config.model.name.rsplit_once('/').map_or_else(
            || self.config.model.name.clone(),
            |(_, rest)| rest.to_string(),
        );
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::Capabilities {
                    vision: probe.vision_supported,
                    model_name,
                    auto_mode: self.config.tools.allow_auto_mode,
                },
            ))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }
}

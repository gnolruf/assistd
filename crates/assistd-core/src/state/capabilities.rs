//! `GetCapabilities` handler: probes the running llama-server and
//! reports vision support + model name.

use super::AppState;
use crate::recovery::{Component, RecoverySeverity};
use anyhow::Result;
use assistd_ipc::Event;
use std::sync::Arc;
use tokio::sync::mpsc;

impl AppState {
    /// Probe the running llama-server's capabilities and surface the
    /// model name in one shot, letting clients render `vision: on/off`
    /// without reaching into the HTTP API directly. Re-probes per
    /// request because a model swap on a long-lived daemon can flip
    /// the vision flag between calls.
    pub(super) async fn handle_get_capabilities(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        for failure in &self.subsystems.mcp_startup_failures {
            let _ = tx
                .send(Event::Status {
                    id: id.clone(),
                    severity: RecoverySeverity::Warning.as_str().to_string(),
                    component: Component::Mcp.as_str().to_string(),
                    event: "startup_failed".to_string(),
                    message: format!(
                        "MCP server '{}' is not available: {}",
                        failure.server_name, failure.reason
                    ),
                })
                .await;
        }

        let probe = match assistd_llm::LlamaServerControl::new(
            &self.config.llama_server.host,
            self.config.llama_server.port,
        ) {
            Ok(control) => {
                assistd_llm::probe_capabilities_routed(
                    &self.config.llama_server.host,
                    self.config.llama_server.port,
                    &self.config.model.name,
                    &control,
                )
                .await
            }
            Err(e) => {
                tracing::warn!(
                    target: "assistd::vision",
                    "GetCapabilities: failed to build control client: {e}"
                );
                assistd_llm::VisionState::default()
            }
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
        Ok(())
    }
}

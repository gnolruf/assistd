//! How far each subsystem that starts in the background has got, as
//! answered to `GetReadiness` and broadcast as each one settles.

use std::sync::Arc;

use tokio::sync::mpsc;

use assistd_ipc::{ComponentReadiness, Event, StartupComponent};

use super::AppState;

const STARTUP_EVENT_ID: &str = "startup";

impl AppState {
    /// Every background subsystem with its current readiness: voice,
    /// embedding, then each configured MCP server.
    pub fn startup_readiness(&self) -> Vec<(StartupComponent, ComponentReadiness)> {
        let embedding = (
            StartupComponent::Embedding,
            self.memory.embedder.readiness().into(),
        );
        let mcp = self.subsystems.mcp_servers.iter().map(|server| {
            (
                StartupComponent::Mcp {
                    server: server.name().to_string(),
                },
                server.readiness().into(),
            )
        });
        self.subsystems
            .voice
            .readiness()
            .into_iter()
            .chain([embedding])
            .chain(mcp)
            .collect()
    }

    /// Broadcast `component`'s readiness to bus subscribers.
    pub fn publish_readiness(&self, component: StartupComponent, state: ComponentReadiness) {
        self.runtime.publish(&Event::Readiness {
            id: STARTUP_EVENT_ID.to_string(),
            component,
            state,
        });
    }

    pub(super) async fn handle_get_readiness(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) {
        for (component, state) in self.startup_readiness() {
            let _ = tx
                .send(Event::Readiness {
                    id: id.clone(),
                    component,
                    state,
                })
                .await;
        }
        let _ = tx.send(Event::Done { id }).await;
    }
}

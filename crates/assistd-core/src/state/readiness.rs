//! How far each subsystem that starts in the background has got, as
//! answered to `GetReadiness` and broadcast as each one settles.

use assistd_ipc::{ComponentReadiness, Event, EventBody, StartupComponent};

use super::status_service::StatusService;
use crate::bus::{EventBus, EventSink};

const STARTUP_EVENT_ID: &str = "startup";

impl StatusService {
    /// Every background subsystem with its current readiness: voice,
    /// embedding, web search, then each configured MCP server.
    pub fn startup_readiness(&self) -> Vec<(StartupComponent, ComponentReadiness)> {
        let embedding = (
            StartupComponent::Embedding,
            self.embedder.readiness().into(),
        );
        let web_search = (StartupComponent::WebSearch, self.web_search.get().into());
        let mcp = self.mcp_servers.iter().map(|server| {
            (
                StartupComponent::Mcp {
                    server: server.name().to_string(),
                },
                server.readiness(),
            )
        });
        self.voice
            .readiness()
            .into_iter()
            .chain([embedding, web_search])
            .chain(mcp)
            .collect()
    }

    /// Broadcast `component`'s readiness to bus subscribers.
    pub fn publish_readiness(&self, component: StartupComponent, state: ComponentReadiness) {
        publish_readiness(&self.bus, component, state);
    }

    pub(super) async fn handle_get_readiness(&self, id: String, tx: EventSink) {
        for (component, state) in self.startup_readiness() {
            let _ = tx
                .send(Event::new(
                    id.clone(),
                    EventBody::Readiness { component, state },
                ))
                .await;
        }
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
    }
}

/// Broadcast `component`'s readiness on `bus`.
pub(super) fn publish_readiness(
    bus: &EventBus,
    component: StartupComponent,
    state: ComponentReadiness,
) {
    bus.publish(&Event::new(
        STARTUP_EVENT_ID.to_string(),
        EventBody::Readiness { component, state },
    ));
}

//! A daemon `AppState` over stub subsystems, for tests of the startup wiring.

use std::sync::Arc;

use assistd_core::{AppState, Config, PresenceManager, ToolRegistry, VoiceManager};
use assistd_llm::EchoBackend;
use tokio::sync::watch;

/// An `AppState` for `config` with an echo model that never starts, no
/// tools, and voice still starting.
pub(super) fn app_state(config: Config) -> AppState {
    let presence = PresenceManager::new_sleeping(
        config.model.clone(),
        config.timeouts.clone(),
        watch::channel(false).1,
    )
    .expect("control client for the configured model address");
    AppState::new(
        config,
        Arc::new(EchoBackend::new()),
        presence,
        Arc::new(ToolRegistry::new()),
        VoiceManager::new(true),
    )
}

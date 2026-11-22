//! Voice for the daemon: once the model has loaded, the voice worker runs
//! under supervision from the daemon's own binary, and voice readiness is
//! published whenever it changes.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

use assistd_core::AppState;
use assistd_utils::readiness::Readiness;
use assistd_voice::worker::{VoiceWorkerSpec, supervise};
use tokio::sync::watch;
use tracing::info;

use super::SELF_EXE;
use super::background_service::{BackgroundService, publish_changes};

/// The hidden subcommand the worker runs as.
const WORKER_SUBCOMMAND: &str = "voice-worker";

/// Start the worker once `model_settled` flips, and stop it when `shutdown`
/// does.
pub(super) fn spawn(
    state: Arc<AppState>,
    model_settled: watch::Receiver<bool>,
    shutdown: watch::Receiver<bool>,
) -> BackgroundService<()> {
    BackgroundService::spawn("voice worker", run(state, model_settled, shutdown))
}

async fn run(
    state: Arc<AppState>,
    mut model_settled: watch::Receiver<bool>,
    mut shutdown: watch::Receiver<bool>,
) {
    let settled = tokio::select! {
        biased;
        _ = shutdown.wait_for(|stop| *stop) => false,
        settled = model_settled.wait_for(|settled| *settled) => settled.is_ok(),
    };
    if !settled {
        return;
    }
    let voice = &state.subsystems.voice;
    let config = &state.config.voice;
    if !config.enabled && !config.synthesis.enabled {
        info!("voice: input and speech both disabled in config; no voice worker");
        voice.set_capture(Readiness::Unavailable(
            "disabled in config (voice.enabled = false)".into(),
        ));
        voice.set_speech(Readiness::Unavailable(
            "disabled in config (voice.synthesis.enabled = false)".into(),
        ));
        publish_readiness(&state);
        return;
    }
    let spec = VoiceWorkerSpec {
        program: PathBuf::from(SELF_EXE),
        args: vec![OsString::from(WORKER_SUBCOMMAND)],
        config: config.clone(),
    };
    let changes = voice.subscribe_readiness();
    tokio::join!(
        supervise(spec, Arc::clone(voice), shutdown.clone()),
        publish_changes(changes, shutdown, |()| publish_readiness(&state)),
    );
}

fn publish_readiness(state: &AppState) {
    for (component, readiness) in state.subsystems.voice.readiness() {
        state.status.publish_readiness(component, readiness);
    }
}

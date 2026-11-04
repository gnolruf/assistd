//! Voice for the daemon: once the model has loaded, the voice worker runs
//! under supervision from the daemon's own binary, and voice readiness is
//! published whenever it changes.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

use assistd_core::{AppState, Component, spawn_supervised};
use assistd_utils::readiness::Readiness;
use assistd_voice::worker::{VoiceWorkerSpec, supervise};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::info;

use super::voice_probe::PresenceGpuProbe;
use super::vram::VramLedger;

/// The running daemon's own image, which stays this build even after the
/// file on disk is replaced.
const SELF_EXE: &str = "/proc/self/exe";

/// The hidden subcommand the worker runs as.
const WORKER_SUBCOMMAND: &str = "voice-worker";

/// The background task that runs the voice worker.
pub(super) struct VoiceService {
    task: JoinHandle<()>,
}

impl VoiceService {
    /// Start the worker once `model_settled` flips, and stop it when
    /// `shutdown` does.
    pub(super) fn spawn(
        state: Arc<AppState>,
        vram: Option<Arc<VramLedger>>,
        model_settled: watch::Receiver<bool>,
        shutdown: watch::Receiver<bool>,
    ) -> Self {
        let task = spawn_supervised(
            "voice_worker",
            Component::Voice,
            run(state, vram, model_settled, shutdown),
        );
        Self { task }
    }

    /// Wait for the worker to stop. The `voice` shutdown stage must already
    /// have flipped.
    pub(super) async fn shutdown(self) {
        if let Err(e) = self.task.await {
            tracing::error!("voice worker task failed: {e}");
        }
    }
}

async fn run(
    state: Arc<AppState>,
    vram: Option<Arc<VramLedger>>,
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
    let gpu = Arc::new(PresenceGpuProbe::new(
        state.subsystems.presence.clone(),
        vram,
    ));
    let changes = voice.subscribe_readiness();
    tokio::join!(
        supervise(spec, Arc::clone(voice), gpu, shutdown.clone()),
        publish_changes(&state, changes, shutdown),
    );
}

async fn publish_changes(
    state: &AppState,
    mut changes: watch::Receiver<()>,
    mut shutdown: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            biased;
            _ = shutdown.wait_for(|stop| *stop) => return,
            changed = changes.changed() => {
                if changed.is_err() {
                    return;
                }
                publish_readiness(state);
            }
        }
    }
}

fn publish_readiness(state: &AppState) {
    for (component, readiness) in state.subsystems.voice.readiness() {
        state.publish_readiness(component, readiness);
    }
}

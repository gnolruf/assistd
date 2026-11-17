//! Startup work that runs while the socket already serves: load the model,
//! then, once voice capture is up, start continuous listening.

use std::net::SocketAddr;
use std::sync::Arc;

use assistd_core::{
    AppState, Component, PresenceManager, VisionRevalidator, listen_dispatcher, spawn_supervised,
};
use assistd_voice::{ContinuousListener, VoiceCapture, VoiceManager};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

/// What warmup brings up behind the managers `state` already serves.
pub(super) struct Warmup {
    pub state: Arc<AppState>,
    pub vision: Arc<VisionRevalidator>,
    /// Flipped once the first model load has finished, whether or not it
    /// succeeded.
    pub model_settled: watch::Sender<bool>,
}

/// Run warmup until it finishes or `shutdown` flips. The model load runs
/// on its own task, so only the LLM shutdown stage cancels it.
pub(super) fn spawn(warmup: Warmup, shutdown: watch::Receiver<bool>) -> JoinHandle<()> {
    spawn_supervised("warmup", Component::Daemon, run(warmup, shutdown))
}

async fn run(warmup: Warmup, mut shutdown: watch::Receiver<bool>) {
    let Warmup {
        state,
        vision,
        model_settled,
    } = warmup;
    let presence = state.subsystems.presence.clone();
    let model_addr = state.config.model.listen_addr();
    let model_load = spawn_supervised(
        "initial_wake",
        Component::Llm,
        load_model(presence.clone(), vision, model_addr),
    );
    if until_shutdown(&mut shutdown, model_load).await.is_none() {
        return;
    }
    model_settled.send_replace(true);
    let voice = &state.config.voice;
    if !(voice.enabled && voice.wakeword.enabled) {
        return;
    }
    let capture = until_shutdown(&mut shutdown, started_capture(&state.subsystems.voice)).await;
    if let Some(Some(capture)) = capture {
        run_listen_dispatcher(&state, capture.listener, shutdown).await;
    }
}

/// The capture handles once capture is up, waiting through any time it is
/// unavailable first; `None` once `voice` reports nothing more.
async fn started_capture(voice: &VoiceManager) -> Option<VoiceCapture> {
    let mut changes = voice.subscribe_readiness();
    loop {
        if let Ok(capture) = voice.capture() {
            return Some(capture);
        }
        changes.changed().await.ok()?;
    }
}

/// `None` when `shutdown` flips before `work` completes.
async fn until_shutdown<T>(
    shutdown: &mut watch::Receiver<bool>,
    work: impl Future<Output = T>,
) -> Option<T> {
    tokio::select! {
        biased;
        _ = shutdown.wait_for(|v| *v) => None,
        out = work => Some(out),
    }
}

/// Wake the model, then seed the vision gate from it. A failed load
/// leaves the daemon `Sleeping`; the next query or wake retries it.
async fn load_model(
    presence: Arc<PresenceManager>,
    vision: Arc<VisionRevalidator>,
    model_addr: SocketAddr,
) {
    if let Err(e) = presence.wake().await {
        error!("presence: initial model load failed: {e:#}; the next query or wake retries it");
        return;
    }
    info!("presence: Active (llama-server ready on {model_addr})");
    vision.revalidate_if_stale(&presence).await;
    if vision.gate().supported() {
        info!("vision: enabled (model has mmproj)");
    } else {
        warn!("Vision not available: mmproj not loaded.");
    }
}

async fn run_listen_dispatcher(
    state: &Arc<AppState>,
    listener: Arc<dyn ContinuousListener>,
    shutdown: watch::Receiver<bool>,
) {
    let forwarder = listen_dispatcher::spawn_dispatcher(
        state.clone(),
        listener,
        state.config.voice.wakeword.start_on_launch,
        shutdown,
    );
    let _ = forwarder.await;
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::time::Duration;

    use assistd_config::VoiceConfig;
    use assistd_voice::worker::{VoiceWorkerSpec, supervise};

    use super::*;

    /// Reports the mic missing, then capture ready once the file named by its
    /// first argument exists.
    const STAND_IN_WORKER: &str = r#"
read -r _configure
echo '{"type":"capture_readiness","startup":{"status":"unavailable","reason":"mic not found"}}'
echo '{"type":"speech_readiness","startup":{"status":"ready"}}'
while [ ! -e "$1" ]; do sleep 0.05; done
echo '{"type":"capture_readiness","startup":{"status":"ready"}}'
while read -r _line; do :; done
"#;

    #[tokio::test]
    async fn capture_is_awaited_through_a_mic_missing_at_startup() {
        let dir = tempfile::tempdir().unwrap();
        let plugged_in = dir.path().join("plugged-in");
        let spec = VoiceWorkerSpec {
            program: "/bin/sh".into(),
            args: ["-c", STAND_IN_WORKER, "stand-in-worker"]
                .into_iter()
                .map(OsString::from)
                .chain([plugged_in.clone().into_os_string()])
                .collect(),
            config: VoiceConfig::default(),
        };
        let voice = VoiceManager::new(true);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let supervisor = tokio::spawn(supervise(spec, Arc::clone(&voice), shutdown_rx));
        let mut readiness = voice.subscribe_readiness();
        while voice.speech().output().is_err() {
            readiness.changed().await.unwrap();
        }

        let capture = started_capture(&voice);
        tokio::pin!(capture);
        let while_missing = tokio::time::timeout(Duration::from_millis(200), &mut capture).await;
        assert!(
            while_missing.is_err(),
            "a missing mic must not end the wait"
        );

        std::fs::write(&plugged_in, "").unwrap();
        let plugged = tokio::time::timeout(Duration::from_secs(10), capture).await;
        assert!(matches!(plugged, Ok(Some(_))), "capture never came up");

        shutdown_tx.send_replace(true);
        supervisor.await.unwrap();
    }
}

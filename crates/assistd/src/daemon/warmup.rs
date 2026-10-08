//! Startup work that runs while the socket already serves: load the model,
//! then voice, then start continuous listening.

use std::net::SocketAddr;
use std::sync::Arc;

use assistd_core::{
    AppState, Component, ContinuousListener, PresenceManager, VisionRevalidator, spawn_supervised,
};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use super::{listen_dispatcher, voice_init};

const VOICE_STARTUP_EVENT_ID: &str = "voice-startup";

/// What warmup brings up behind the managers `state` already serves.
pub(super) struct Warmup {
    pub state: Arc<AppState>,
    pub vision: Arc<VisionRevalidator>,
}

/// Run warmup until it finishes or `shutdown` flips. The model load runs
/// on its own task, so only the LLM shutdown stage cancels it.
pub(super) fn spawn(warmup: Warmup, shutdown: watch::Receiver<bool>) -> JoinHandle<()> {
    spawn_supervised("warmup", Component::Daemon, run(warmup, shutdown))
}

async fn run(warmup: Warmup, mut shutdown: watch::Receiver<bool>) {
    let Warmup { state, vision } = warmup;
    let presence = state.subsystems.presence.clone();
    let model_addr = SocketAddr::new(state.config.model.host, state.config.model.port.get());
    let model_load = spawn_supervised(
        "initial_wake",
        Component::Llm,
        load_model(presence.clone(), vision, model_addr),
    );
    if until_shutdown(&mut shutdown, model_load).await.is_none() {
        return;
    }
    let voice = &state.subsystems.voice;
    let init_voice = Box::pin(voice_init::init(voice, &state.config, &presence));
    if until_shutdown(&mut shutdown, init_voice).await.is_none() {
        return;
    }
    state
        .runtime
        .publish(&voice.readiness_event(VOICE_STARTUP_EVENT_ID.into()));
    if let Ok(capture) = voice.capture() {
        run_listen_dispatcher(&state, capture.listener, presence, shutdown).await;
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
    presence: Arc<PresenceManager>,
    shutdown: watch::Receiver<bool>,
) {
    let voice = &state.config.voice;
    if !(voice.enabled && voice.continuous.enabled) {
        return;
    }
    let handles = listen_dispatcher::spawn_dispatcher(
        state.clone(),
        listener,
        presence,
        voice.continuous.start_on_launch,
        shutdown,
    );
    let _ = handles.forwarder.await;
    let _ = handles.presence_gate.await;
}

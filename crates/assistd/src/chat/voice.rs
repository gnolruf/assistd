//! Chat-TUI-side glue for the push-to-talk voice pipeline.
//!
//! The chat TUI doesn't connect to the daemon's Unix socket — it
//! runs the whole stack in-process. So the voice hotkey listener
//! (shared with the daemon) drives a `MicVoiceInput` directly, and a
//! watch-channel subscriber forwards state transitions to the App
//! via the TUI's `VoiceEvent` channel.

use std::sync::Arc;

use assistd_core::{Config, NoVoiceInput, VoiceCaptureState, VoiceInput};
use assistd_voice::MicVoiceInput;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::hotkey;

/// Messages emitted by the voice pipeline into the TUI event loop.
#[derive(Debug, Clone)]
pub enum VoiceEvent {
    /// Capture state moved to a new value. The TUI reducer updates
    /// the "Listening…" / "transcribing…" indicator.
    State(VoiceCaptureState),
    /// Final transcription from the last PTT cycle. Empty string
    /// means VAD trimmed the audio down to silence.
    Transcription(String),
    /// Non-fatal voice pipeline error — surfaced as a TUI notice,
    /// does not terminate the listener.
    Error(String),
}

/// Handle to the background pieces of the TUI's voice pipeline.
/// Fields are retained (not dropped) so the hotkey listener and
/// state-forwarder Arc references stay alive for the TUI lifetime.
/// `voice` is exposed in case the chat TUI ever needs to reach into
/// the trait directly (e.g. a keyboard-bound "mute" toggle).
#[allow(dead_code)]
pub struct VoicePipeline {
    pub voice: Arc<dyn VoiceInput>,
    pub hotkey_handle: Option<JoinHandle<()>>,
    pub state_forwarder: JoinHandle<()>,
}

/// Build the voice subsystem for the chat TUI.
///
/// - If `voice.enabled = false`, returns a `NoVoiceInput` stub so
///   downstream code doesn't need to special-case absence.
/// - On success, spawns:
///     1. The shared `hotkey::spawn_listener` (press = start record,
///        release = stop + transcribe). Skipped when
///        `voice.hotkey` is empty or the WM is Wayland-only.
///     2. A state forwarder that watches the trait's `subscribe()`
///        channel and pushes `VoiceEvent::State` into the TUI.
/// - The worker that awaits a completed transcription and emits
///   `VoiceEvent::Transcription` is attached inside the hotkey's
///   release handler (via a custom release_callback).
pub async fn spawn(
    config: &Config,
    tx: mpsc::Sender<VoiceEvent>,
    shutdown_rx: watch::Receiver<bool>,
) -> VoicePipeline {
    let voice: Arc<dyn VoiceInput> = if config.voice.enabled {
        info!(
            "voice: building mic input ({})",
            config.voice.transcription.model
        );
        match MicVoiceInput::from_config(&config.voice).await {
            Ok(v) => Arc::new(v),
            Err(e) => {
                warn!("voice input failed to initialize: {e:#}");
                let _ = tx
                    .send(VoiceEvent::Error(format!("init failed: {e}")))
                    .await;
                Arc::new(NoVoiceInput::new())
            }
        }
    } else {
        Arc::new(NoVoiceInput::new())
    };

    let state_forwarder = spawn_state_forwarder(voice.clone(), tx.clone(), shutdown_rx.clone());

    // For the TUI we replace the hotkey's bare `start/stop` calls
    // with a wrapper that also forwards the transcription result as
    // a VoiceEvent — so the TUI can auto-submit without reaching
    // back into the voice trait.
    let tui_voice = Arc::new(TuiVoiceAdapter {
        inner: voice.clone(),
        tx: tx.clone(),
    });
    let hotkey_handle = hotkey::spawn_listener(
        &config.presence,
        &config.voice,
        None, // chat TUI doesn't own a PresenceManager for hotkey use
        tui_voice,
        None, // chat TUI doesn't run the continuous listener
        None, // chat TUI doesn't own a VoiceOutputController either
        shutdown_rx,
    );

    VoicePipeline {
        voice,
        hotkey_handle,
        state_forwarder,
    }
}

fn spawn_state_forwarder(
    voice: Arc<dyn VoiceInput>,
    tx: mpsc::Sender<VoiceEvent>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    let mut rx = voice.subscribe();
    tokio::spawn(async move {
        // Seed with the initial state so the indicator reflects
        // reality even if the first real transition is far off.
        let initial = *rx.borrow_and_update();
        let _ = tx.send(VoiceEvent::State(initial)).await;
        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        break;
                    }
                }
                changed = rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    let s = *rx.borrow_and_update();
                    if tx.send(VoiceEvent::State(s)).await.is_err() {
                        break;
                    }
                }
            }
        }
    })
}

/// Wraps the real VoiceInput so that when the hotkey listener calls
/// `stop_and_transcribe`, the result is also forwarded into the TUI
/// event channel as a `VoiceEvent::Transcription`. Without this,
/// the transcription text would only be visible to the listener's
/// log line.
struct TuiVoiceAdapter {
    inner: Arc<dyn VoiceInput>,
    tx: mpsc::Sender<VoiceEvent>,
}

#[async_trait::async_trait]
impl VoiceInput for TuiVoiceAdapter {
    async fn start_recording(&self) -> anyhow::Result<()> {
        self.inner.start_recording().await
    }

    async fn stop_and_transcribe(&self) -> anyhow::Result<String> {
        match self.inner.stop_and_transcribe().await {
            Ok(text) => {
                let _ = self.tx.send(VoiceEvent::Transcription(text.clone())).await;
                Ok(text)
            }
            Err(e) => {
                let msg = format!("{e:#}");
                let _ = self.tx.send(VoiceEvent::Error(msg.clone())).await;
                Err(e)
            }
        }
    }

    fn state(&self) -> VoiceCaptureState {
        self.inner.state()
    }

    fn subscribe(&self) -> watch::Receiver<VoiceCaptureState> {
        self.inner.subscribe()
    }
}

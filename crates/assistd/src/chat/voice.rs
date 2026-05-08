//! Chat-TUI voice glue, daemon-window edition.
//!
//! The TUI no longer owns a `MicVoiceInput` — the daemon does. We
//! still spawn the global hotkey listener locally because PTT
//! keystrokes need to arrive at the foreground process. The
//! press/release callbacks dispatch `Request::PttStart` and
//! `Request::PttStop` to the daemon over IPC; the daemon's
//! response stream (`VoiceState` → `Transcription` → `Delta`s →
//! `Done`) is forwarded onto the App's `ChatEvent` channel as
//! `ChatEvent::Wire(_)` so the same reducer that handles
//! query-driven streaming updates the listening indicator and the
//! output pane uniformly.

use std::sync::Arc;

use assistd_core::Config;
use assistd_ipc::{IpcClient, Request, VoiceCaptureState};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{info, warn};
use uuid::Uuid;

use super::app::ChatEvent;
use crate::hotkey;

/// Handles owned by the voice glue. Held by the caller for the
/// lifetime of the TUI session so the hotkey listener doesn't drop
/// its `Arc` references mid-run. Voice itself runs in the daemon —
/// this struct exists only to keep the local hotkey thread alive.
#[allow(dead_code)]
pub struct VoicePipeline {
    pub hotkey_handle: Option<JoinHandle<()>>,
}

/// Build the chat-TUI voice pipeline.
///
/// - When `voice.enabled = false`, returns a placeholder with no
///   hotkey bound — the daemon's voice IPC remains reachable for
///   anything that wants to drive PTT manually (`assistd ptt-start`).
/// - Otherwise spawns `crate::hotkey::spawn_listener` against an
///   IPC-shimmed [`assistd_voice::VoiceInput`] proxy whose
///   `start_recording` and `stop_and_transcribe` issue Unix-socket
///   requests to the daemon. The daemon's streaming response flows
///   into `chat_tx` so the UI reducer sees Whisper transitions and
///   the auto-dispatched query response.
pub async fn spawn(
    config: &Config,
    ipc: Arc<IpcClient>,
    chat_tx: mpsc::Sender<ChatEvent>,
    shutdown_rx: watch::Receiver<bool>,
) -> VoicePipeline {
    if !config.voice.enabled {
        info!("voice: disabled in config; PTT hotkey will not bind");
        return VoicePipeline {
            hotkey_handle: None,
        };
    }

    info!(
        "voice: routing PTT through daemon IPC (hotkey={:?})",
        config.voice.hotkey
    );

    let proxy: Arc<dyn assistd_voice::VoiceInput> = Arc::new(IpcVoiceProxy::new(ipc, chat_tx));
    let hotkey_handle = hotkey::spawn_listener(
        &config.presence,
        &config.voice,
        hotkey::Subsystems {
            presence: None, // chat TUI doesn't own a PresenceManager
            voice: proxy,
            listener: None,     // continuous-listen runs in the daemon
            voice_output: None, // VoiceOutputController also lives in the daemon
        },
        shutdown_rx,
    );

    VoicePipeline { hotkey_handle }
}

/// `VoiceInput` adapter that forwards calls to the daemon's IPC
/// surface. `start_recording` does a one-shot `PttStart`. The hotkey
/// listener calls `stop_and_transcribe` when the user releases the
/// PTT key — that goes through `PttStop`, whose response stream the
/// daemon dispatches as a Query. We forward every event onto the
/// app's `ChatEvent::Wire` channel and return the transcription text
/// so the listener's existing log line doesn't go silent. Reducer
/// logic in `App::on_wire_event` handles the actual UI updates.
struct IpcVoiceProxy {
    ipc: Arc<IpcClient>,
    chat_tx: mpsc::Sender<ChatEvent>,
    state: watch::Sender<VoiceCaptureState>,
    state_rx: watch::Receiver<VoiceCaptureState>,
}

impl IpcVoiceProxy {
    fn new(ipc: Arc<IpcClient>, chat_tx: mpsc::Sender<ChatEvent>) -> Self {
        let (state, state_rx) = watch::channel(VoiceCaptureState::Idle);
        Self {
            ipc,
            chat_tx,
            state,
            state_rx,
        }
    }

    fn set_state(&self, s: VoiceCaptureState) {
        let _ = self.state.send(s);
    }
}

#[async_trait::async_trait]
impl assistd_voice::VoiceInput for IpcVoiceProxy {
    async fn start_recording(&self) -> anyhow::Result<()> {
        self.set_state(VoiceCaptureState::Recording);
        let req = Request::PttStart {
            id: Uuid::new_v4().to_string(),
        };
        let mut stream = self.ipc.one_shot(req).await.map_err(anyhow::Error::from)?;

        while let Some(ev) = stream.next_event().await? {
            let terminal = ev.is_terminal();
            let _ = self.chat_tx.send(ChatEvent::Wire(ev)).await;
            if terminal {
                break;
            }
        }
        Ok(())
    }

    async fn stop_and_transcribe(&self) -> anyhow::Result<String> {
        self.set_state(VoiceCaptureState::Transcribing);
        let req = Request::PttStop {
            id: Uuid::new_v4().to_string(),
        };
        let mut stream = self.ipc.one_shot(req).await.map_err(anyhow::Error::from)?;
        let mut transcript = String::new();
        while let Some(ev) = stream.next_event().await? {
            if let assistd_ipc::Event::Transcription { text, .. } = &ev {
                transcript = text.clone();
            }
            if let assistd_ipc::Event::VoiceState { state, .. } = &ev {
                self.set_state(*state);
            }
            let terminal = ev.is_terminal();
            let _ = self.chat_tx.send(ChatEvent::Wire(ev)).await;
            if terminal {
                break;
            }
        }
        self.set_state(VoiceCaptureState::Idle);
        Ok(transcript)
    }

    fn state(&self) -> VoiceCaptureState {
        *self.state_rx.borrow()
    }

    fn subscribe(&self) -> watch::Receiver<VoiceCaptureState> {
        self.state_rx.clone()
    }
}

#[allow(dead_code)] // used only when the `chat` feature compiles voice off
fn _suppress_unused_warn() {
    warn!("voice glue unused");
}

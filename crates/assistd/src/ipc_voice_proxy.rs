//! `VoiceInput` adapter that forwards `start_recording` /
//! `stop_and_transcribe` to the daemon's IPC surface as
//! `Request::PttStart` / `Request::PttStop`.
//!
//! Used in two places:
//!
//! - the chat TUI, where the local hotkey listener needs to route
//!   press/release into the daemon (no local voice subsystem);
//! - the daemon itself, where the hotkey listener routes back through
//!   the daemon's own Unix socket so the PTT flow goes through the
//!   canonical `handle_ptt_start` / `handle_ptt_stop` path. That gets
//!   us the presence warmup (so Whisper takes the GPU path instead of
//!   falling back to CPU) and the per-connection bus tee in
//!   `socket.rs` for free, instead of duplicating both inside the
//!   hotkey listener.

use std::sync::Arc;

use assistd_ipc::{Event, IpcClient, Request, VoiceCaptureState};
use async_trait::async_trait;
use tokio::sync::{mpsc, watch};
use uuid::Uuid;

/// Adapter that satisfies [`assistd_voice::VoiceInput`] by talking IPC.
///
/// `event_sink`, when `Some`, receives every event the daemon emits
/// over the PTT connection so a UI (or test harness) can react to
/// transcription / streaming reply / tool calls. `None` is the daemon's
/// own configuration: events still tee onto the daemon's broadcast bus
/// inside `socket.rs`, so the popup and other passive subscribers see
/// them without the proxy needing to re-forward.
pub struct IpcVoiceProxy {
    ipc: Arc<IpcClient>,
    event_sink: Option<mpsc::Sender<Event>>,
    state: watch::Sender<VoiceCaptureState>,
    state_rx: watch::Receiver<VoiceCaptureState>,
}

impl IpcVoiceProxy {
    pub fn new(ipc: Arc<IpcClient>, event_sink: Option<mpsc::Sender<Event>>) -> Self {
        let (state, state_rx) = watch::channel(VoiceCaptureState::Idle);
        Self {
            ipc,
            event_sink,
            state,
            state_rx,
        }
    }

    fn set_state(&self, s: VoiceCaptureState) {
        let _ = self.state.send(s);
    }

    async fn forward(&self, ev: Event) {
        if let Some(sink) = self.event_sink.as_ref() {
            let _ = sink.send(ev).await;
        }
    }
}

#[async_trait]
impl assistd_voice::VoiceInput for IpcVoiceProxy {
    async fn start_recording(&self) -> anyhow::Result<()> {
        self.set_state(VoiceCaptureState::Recording);
        let req = Request::PttStart {
            id: Uuid::new_v4().to_string(),
        };
        let mut stream = self.ipc.one_shot(req).await.map_err(anyhow::Error::from)?;

        while let Some(ev) = stream.next_event().await? {
            let terminal = ev.is_terminal();
            self.forward(ev).await;
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
            if let Event::Transcription { text, .. } = &ev {
                transcript = text.clone();
            }
            if let Event::VoiceState { state, .. } = &ev {
                self.set_state(*state);
            }
            let terminal = ev.is_terminal();
            self.forward(ev).await;
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

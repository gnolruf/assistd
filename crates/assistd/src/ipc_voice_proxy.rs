//! [`assistd_voice::VoiceInput`] over IPC: press and release become
//! `Request::PttStart` and `Request::PttStop`. Both the chat TUI and the
//! daemon's own hotkey listener use it, so every push-to-talk turn takes
//! the daemon's one PTT path.

use std::sync::Arc;

use assistd_ipc::{Event, IpcClient, Request, VoiceCaptureState};
use assistd_voice::VoiceInputError;
use async_trait::async_trait;
use tokio::sync::{mpsc, watch};
use uuid::Uuid;

/// `event_sink`, when `Some`, receives every event the daemon emits on
/// the PTT connection.
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
    async fn start_recording(&self) -> Result<(), VoiceInputError> {
        self.set_state(VoiceCaptureState::Recording);
        let req = Request::PttStart {
            id: Uuid::new_v4().to_string(),
        };
        let mut stream = self.ipc.one_shot(req).await?;

        while let Some(ev) = stream.next_event().await? {
            let terminal = ev.is_terminal();
            self.forward(ev).await;
            if terminal {
                break;
            }
        }
        Ok(())
    }

    async fn stop_and_transcribe(&self) -> Result<String, VoiceInputError> {
        self.set_state(VoiceCaptureState::Transcribing);
        let req = Request::PttStop {
            id: Uuid::new_v4().to_string(),
        };
        let mut stream = self.ipc.one_shot(req).await?;
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

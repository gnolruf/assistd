//! [`assistd_voice::VoiceInput`] over IPC: press and release become
//! `Request::PttStart` and `Request::PttStop`, so every push-to-talk turn
//! takes the daemon's one PTT path.

use std::sync::Arc;

use assistd_ipc::{Event, IpcClient, Request, VoiceCaptureState};
use assistd_voice::{VoiceInput, VoiceInputError};
use async_trait::async_trait;
use tokio::sync::watch;
use uuid::Uuid;

/// Push-to-talk over the daemon socket.
#[derive(Debug)]
pub(crate) struct IpcVoiceProxy {
    ipc: Arc<IpcClient>,
    state: watch::Sender<VoiceCaptureState>,
    state_rx: watch::Receiver<VoiceCaptureState>,
}

impl IpcVoiceProxy {
    pub(crate) fn new(ipc: Arc<IpcClient>) -> Self {
        let (state, state_rx) = watch::channel(VoiceCaptureState::Idle);
        Self {
            ipc,
            state,
            state_rx,
        }
    }

    fn set_state(&self, state: VoiceCaptureState) {
        let _ = self.state.send(state);
    }
}

#[async_trait]
impl VoiceInput for IpcVoiceProxy {
    async fn start_recording(&self) -> Result<(), VoiceInputError> {
        self.set_state(VoiceCaptureState::Recording);
        let req = Request::PttStart {
            id: Uuid::new_v4().to_string(),
        };
        let mut stream = self.ipc.one_shot(req).await?;

        while let Some(ev) = stream.next_event().await? {
            if ev.is_terminal() {
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
            match ev {
                Event::Transcription { text, .. } => transcript = text,
                Event::VoiceState { state, .. } => self.set_state(state),
                ev if ev.is_terminal() => break,
                _ => {}
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

//! Owner of the voice subsystem: capture and speech come up in the
//! background, and requests are answered by how far each has got.

use std::sync::Arc;

use assistd_ipc::{ComponentReadiness, ListenPhase, StartupComponent};
use assistd_utils::readiness::{NotReady, Readiness, ReadinessCell};
use thiserror::Error;
use tokio::sync::watch;

use crate::{ContinuousListener, VoiceInput, VoiceOutput, VoiceOutputController};

/// Push-to-talk input and the continuous listener, which share one mic and
/// one transcriber and so come up together.
#[derive(Debug, Clone)]
pub struct VoiceCapture {
    pub input: Arc<dyn VoiceInput>,
    pub listener: Arc<dyn ContinuousListener>,
}

/// Voice capture is still starting, or unavailable this run.
#[derive(Debug, Clone, Error)]
#[error("voice capture is {0}")]
pub struct CaptureUnavailable(pub NotReady);

/// Owns voice capture and speech output, each of which may still be
/// starting or may have failed to start.
#[derive(Debug)]
pub struct VoiceManager {
    capture: ReadinessCell<VoiceCapture>,
    speech: Arc<VoiceOutputController>,
    readiness_changed: watch::Sender<()>,
}

impl VoiceManager {
    /// Capture and speech both still starting; speech is muted unless
    /// `speech_enabled`.
    pub fn new(speech_enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            capture: ReadinessCell::starting(),
            speech: VoiceOutputController::new(speech_enabled),
            readiness_changed: watch::channel(()).0,
        })
    }

    /// Capture and speech already up.
    pub fn ready(
        capture: VoiceCapture,
        output: Arc<dyn VoiceOutput>,
        speech_enabled: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            capture: ReadinessCell::new(Readiness::Ready(capture)),
            speech: VoiceOutputController::ready(output, speech_enabled),
            readiness_changed: watch::channel(()).0,
        })
    }

    /// Record how far capture's startup has got.
    pub fn set_capture(&self, capture: Readiness<VoiceCapture>) {
        self.capture.set(capture);
        self.readiness_changed.send_replace(());
    }

    /// Record how far speech output's startup has got.
    pub fn set_speech(&self, output: Readiness<Arc<dyn VoiceOutput>>) {
        self.speech.set_output(output);
        self.readiness_changed.send_replace(());
    }

    /// Marked changed on every [`Self::set_capture`] and [`Self::set_speech`].
    pub fn subscribe_readiness(&self) -> watch::Receiver<()> {
        self.readiness_changed.subscribe()
    }

    /// The capture handles, or why there are none.
    pub fn capture(&self) -> Result<VoiceCapture, CaptureUnavailable> {
        self.capture.get().map_err(CaptureUnavailable)
    }

    /// Whether continuous listening is on; never before capture is up.
    pub fn listening(&self) -> bool {
        self.capture()
            .is_ok_and(|capture| capture.listener.is_active())
    }

    /// The wake-word listener's phase; `Off` before capture is up.
    pub fn listen_phase(&self) -> ListenPhase {
        self.capture().map_or(ListenPhase::Off, |capture| {
            *capture.listener.subscribe_phase().borrow()
        })
    }

    /// Speech output and its mute, skip and speaking controls.
    pub fn speech(&self) -> &Arc<VoiceOutputController> {
        &self.speech
    }

    /// How far capture and speech have started.
    pub fn readiness(&self) -> [(StartupComponent, ComponentReadiness); 2] {
        [
            (StartupComponent::VoiceInput, self.capture.get().into()),
            (StartupComponent::Speech, self.speech.output().into()),
        ]
    }
}

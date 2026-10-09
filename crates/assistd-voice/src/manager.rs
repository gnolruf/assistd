//! Owner of the voice subsystem: capture and speech come up in the
//! background, and requests are answered by how far each has got.

use std::sync::Arc;

use assistd_ipc::{ComponentReadiness, StartupComponent};
use assistd_utils::readiness::{NotReady, Readiness, ReadinessCell};
use thiserror::Error;

use crate::{ContinuousListener, VoiceInput, VoiceOutput, VoiceOutputController};

/// Push-to-talk input and the continuous listener, which share one
/// transcriber and so come up together.
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
}

impl VoiceManager {
    /// Capture and speech both still starting; speech is muted unless
    /// `speech_enabled`.
    pub fn new(speech_enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            capture: ReadinessCell::starting(),
            speech: VoiceOutputController::new(speech_enabled),
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
        })
    }

    /// Record how far capture's startup has got.
    pub fn set_capture(&self, capture: Readiness<VoiceCapture>) {
        self.capture.set(capture);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NoContinuousListener, NoVoiceInput};

    #[test]
    fn capture_reports_why_it_is_missing_until_it_is_up() {
        let manager = VoiceManager::new(true);
        assert_eq!(
            manager.capture().unwrap_err().to_string(),
            "voice capture is still starting"
        );
        assert!(!manager.listening());

        manager.set_capture(Readiness::Unavailable("whisper model missing".into()));
        assert_eq!(
            manager.capture().unwrap_err().to_string(),
            "voice capture is unavailable: whisper model missing"
        );

        manager.set_capture(Readiness::Ready(VoiceCapture {
            input: Arc::new(NoVoiceInput::new()),
            listener: Arc::new(NoContinuousListener::new()),
        }));
        assert!(manager.capture().is_ok());
    }

    #[test]
    fn readiness_reports_capture_and_speech_separately() {
        let manager = VoiceManager::new(true);
        manager
            .speech()
            .set_output(Readiness::Unavailable("disabled in config".into()));
        assert_eq!(
            manager.readiness(),
            [
                (StartupComponent::VoiceInput, ComponentReadiness::Starting),
                (
                    StartupComponent::Speech,
                    ComponentReadiness::Unavailable {
                        reason: "disabled in config".into(),
                    }
                ),
            ]
        );
    }
}

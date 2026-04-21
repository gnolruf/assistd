//! Voice subsystem: input (speech-to-text) and output (text-to-speech) traits.
//!
//! Milestone 4 provides a [`WhisperTranscriber`] (whisper-rs backed)
//! implementation of [`Transcriber`]; the higher-level [`VoiceInput`]
//! trait (hotkey + mic capture) still uses the placeholder until the
//! microphone layer lands. Milestone 5 provides a `Piper` implementation
//! of [`VoiceOutput`]. The traits are defined here so the daemon can
//! hold them as `Arc<dyn …>` from the moment the feature lands, without
//! touching the protocol or handler shape.

use anyhow::Result;
use async_trait::async_trait;

#[cfg(feature = "whisper")]
pub mod gpu;
#[cfg(feature = "whisper")]
pub mod model_cache;
#[cfg(feature = "whisper")]
pub mod transcribe;
#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "whisper")]
pub use transcribe::{Transcriber, TranscriptionError};
#[cfg(feature = "whisper")]
pub use whisper::{WhisperTranscriber, WhisperTranscriberBuilder};

/// Capture a single hotkey-triggered speech utterance and transcribe it.
#[async_trait]
pub trait VoiceInput: Send + Sync + 'static {
    /// Listen for speech until the user releases the hotkey, then return
    /// the transcribed text.
    async fn capture_utterance(&self) -> Result<String>;
}

/// Speak the given text aloud.
#[async_trait]
pub trait VoiceOutput: Send + Sync + 'static {
    /// Synthesize and play `text`, returning when playback finishes.
    async fn speak(&self, text: String) -> Result<()>;
}

/// Placeholder [`VoiceInput`] that refuses to capture — used until the
/// milestone-4 mic-capture layer lands.
pub struct NoVoiceInput;

#[async_trait]
impl VoiceInput for NoVoiceInput {
    async fn capture_utterance(&self) -> Result<String> {
        anyhow::bail!("voice input is not enabled in this build")
    }
}

/// Placeholder [`VoiceOutput`] that drops speech requests silently — used
/// until the milestone-5 implementation lands.
pub struct NoVoiceOutput;

#[async_trait]
impl VoiceOutput for NoVoiceOutput {
    async fn speak(&self, _text: String) -> Result<()> {
        Ok(())
    }
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_voice_input_returns_error() {
        assert!(NoVoiceInput.capture_utterance().await.is_err());
    }

    #[tokio::test]
    async fn no_voice_output_is_silent_success() {
        NoVoiceOutput.speak("hi".into()).await.unwrap();
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}

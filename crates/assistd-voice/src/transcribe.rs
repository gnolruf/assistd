//! The [`Transcriber`] trait and the GPU-or-CPU [`QueuedTranscriber`].

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{OnceCell, watch};

use crate::VoiceCaptureState;

/// Transcribe 16 kHz mono PCM audio into text.
#[async_trait]
pub trait Transcriber: fmt::Debug + Send + Sync + 'static {
    /// Transcribe signed 16-bit mono samples at 16 kHz. Returns trimmed text;
    /// an empty string means the input was silence, not an error.
    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError>;

    /// State transitions the transcriber drives itself (`Queued`,
    /// `Transcribing`). `None` when the implementation has no internal
    /// states to report.
    fn subscribe_state(&self) -> Option<watch::Receiver<VoiceCaptureState>> {
        None
    }

    /// Whether inference runs on a GPU, and so competes with the LLM for it.
    fn is_gpu(&self) -> bool {
        false
    }
}

/// Errors surfaced by a [`Transcriber`] implementation.
#[derive(Debug, thiserror::Error)]
pub enum TranscriptionError {
    #[error("empty audio buffer")]
    EmptyAudio,

    #[error(transparent)]
    Download(#[from] assistd_utils::hf::download::DownloadError),

    #[error("failed to load the speech-to-text model: {0}")]
    ModelLoad(String),

    #[error("transcription failed: {0}")]
    Inference(String),

    #[error("failed to load Silero VAD: {0}")]
    VadInit(String),

    #[error("tokio join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Answers "is the GPU available for transcription right now?". An LLM stream
/// in flight does not count: transcription shares the GPU with it.
pub trait BusyProbe: fmt::Debug + Send + Sync + 'static {
    /// True when a process outside assistd holds meaningful VRAM, so
    /// waiting would not help.
    fn foreign_gpu_busy(&self) -> bool;

    /// True when the GPU transcriber is safe to use. When false the
    /// context may be torn down concurrently and must not be touched.
    fn presence_active(&self) -> bool;
}

/// Probe that always reports the GPU free.
#[derive(Debug)]
pub struct NullBusyProbe;

impl BusyProbe for NullBusyProbe {
    fn foreign_gpu_busy(&self) -> bool {
        false
    }
    fn presence_active(&self) -> bool {
        true
    }
}

/// Async factory for the CPU fallback transcriber, invoked the first
/// time a fallback is needed.
pub type CpuFallbackFactory = Arc<
    dyn Fn()
            -> Pin<Box<dyn Future<Output = Result<Arc<dyn Transcriber>, TranscriptionError>> + Send>>
        + Send
        + Sync,
>;

/// Wraps a GPU transcriber with queue-and-fallback: publish `Queued`,
/// consult the [`BusyProbe`], and route to a lazily built CPU
/// transcriber when the GPU is unavailable. A CPU-backed primary runs directly.
pub struct QueuedTranscriber {
    primary: Arc<dyn Transcriber>,
    cpu: Arc<OnceCell<Arc<dyn Transcriber>>>,
    cpu_factory: CpuFallbackFactory,
    busy: Arc<dyn BusyProbe>,
    state_tx: watch::Sender<VoiceCaptureState>,
}

impl fmt::Debug for QueuedTranscriber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueuedTranscriber")
            .field("primary", &self.primary)
            .field("busy", &self.busy)
            .finish_non_exhaustive()
    }
}

impl QueuedTranscriber {
    /// Wraps `primary`. `cpu_factory` is invoked on the first fallback
    /// and a successfully built transcriber is reused.
    pub fn new(
        primary: Arc<dyn Transcriber>,
        cpu_factory: CpuFallbackFactory,
        busy: Arc<dyn BusyProbe>,
    ) -> Self {
        let (state_tx, _) = watch::channel(VoiceCaptureState::Idle);
        Self {
            primary,
            cpu: Arc::new(OnceCell::new()),
            cpu_factory,
            busy,
            state_tx,
        }
    }

    /// Consults the [`BusyProbe`], logging the reason whenever the GPU is ruled out.
    fn should_fall_back_to_cpu(&self) -> bool {
        if !self.busy.presence_active() {
            tracing::info!(
                target: "assistd::voice::queued",
                "falling back to CPU: presence is not Active"
            );
            return true;
        }
        if self.busy.foreign_gpu_busy() {
            tracing::info!(
                target: "assistd::voice::queued",
                "falling back to CPU: foreign process holds VRAM"
            );
            return true;
        }
        false
    }

    async fn cpu_transcriber(&self) -> Result<Arc<dyn Transcriber>, TranscriptionError> {
        let factory = self.cpu_factory.clone();
        let cpu = self
            .cpu
            .get_or_try_init(|| async move { factory().await })
            .await?;
        Ok(cpu.clone())
    }
}

#[async_trait]
impl Transcriber for QueuedTranscriber {
    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError> {
        let _idle = IdleOnDrop(&self.state_tx);
        if !self.primary.is_gpu() {
            let _ = self.state_tx.send(VoiceCaptureState::Transcribing);
            return self.primary.transcribe(pcm_i16_16k_mono).await;
        }

        let _ = self.state_tx.send(VoiceCaptureState::Queued);
        let use_cpu = self.should_fall_back_to_cpu();
        let _ = self.state_tx.send(VoiceCaptureState::Transcribing);

        if use_cpu {
            self.cpu_transcriber()
                .await?
                .transcribe(pcm_i16_16k_mono)
                .await
        } else {
            self.primary.transcribe(pcm_i16_16k_mono).await
        }
    }

    fn subscribe_state(&self) -> Option<watch::Receiver<VoiceCaptureState>> {
        Some(self.state_tx.subscribe())
    }
}

/// Publishes `Idle` when dropped, so a transcription that fails or is
/// cancelled still ends its `Transcribing` state.
struct IdleOnDrop<'a>(&'a watch::Sender<VoiceCaptureState>);

impl Drop for IdleOnDrop<'_> {
    fn drop(&mut self) {
        let _ = self.0.send(VoiceCaptureState::Idle);
    }
}

#[cfg(test)]
mod tests;

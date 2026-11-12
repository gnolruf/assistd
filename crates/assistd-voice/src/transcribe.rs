//! The [`Transcriber`] trait and the GPU-or-CPU [`QueuedTranscriber`].

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::{OnceCell, watch};

use crate::VoiceCaptureState;

/// Transcribe 16 kHz mono PCM audio into text.
#[async_trait]
pub trait Transcriber: fmt::Debug + Send + Sync + 'static {
    /// Transcribe signed 16-bit mono samples at 16 kHz. Returns trimmed text;
    /// an empty string means the input was silence, not an error.
    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError>;

    /// `Transcribing` while any transcription is in flight, `Idle` otherwise.
    /// `None` when the implementation does not report it.
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

    /// True while presence is `Active`. Otherwise the daemon is handing the
    /// GPU back, and transcription moves to the CPU.
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

/// Wraps a GPU transcriber with a fallback: consult the [`BusyProbe`] and
/// route to a lazily built CPU transcriber when the GPU is unavailable.
/// A CPU-backed primary runs directly.
pub struct QueuedTranscriber {
    primary: Arc<dyn Transcriber>,
    cpu: Arc<OnceCell<Arc<dyn Transcriber>>>,
    cpu_factory: CpuFallbackFactory,
    busy: Arc<dyn BusyProbe>,
    in_flight: Mutex<usize>,
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
            in_flight: Mutex::new(0),
            state_tx,
        }
    }

    fn begin_transcription(&self) -> InFlightTranscription<'_> {
        let mut in_flight = self.in_flight.lock();
        *in_flight += 1;
        self.publish(VoiceCaptureState::Transcribing);
        InFlightTranscription(self)
    }

    fn publish(&self, state: VoiceCaptureState) {
        self.state_tx
            .send_if_modified(|current| std::mem::replace(current, state) != state);
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
        let _in_flight = self.begin_transcription();
        if self.primary.is_gpu() && self.should_fall_back_to_cpu() {
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

/// One running transcription. Dropping the last one publishes `Idle`, so a
/// transcription that fails or is cancelled still ends `Transcribing`.
struct InFlightTranscription<'a>(&'a QueuedTranscriber);

impl Drop for InFlightTranscription<'_> {
    fn drop(&mut self) {
        let mut in_flight = self.0.in_flight.lock();
        *in_flight -= 1;
        if *in_flight == 0 {
            self.0.publish(VoiceCaptureState::Idle);
        }
    }
}

#[cfg(test)]
mod tests;

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
    /// Transcribe signed 16-bit mono samples at 16 kHz, biased toward the
    /// spellings in `prompt` when given. Returns trimmed text; an empty
    /// string means the input was silence, not an error.
    async fn transcribe(
        &self,
        pcm_i16_16k_mono: &[i16],
        prompt: Option<&str>,
    ) -> Result<String, TranscriptionError>;

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
    async fn transcribe(
        &self,
        pcm_i16_16k_mono: &[i16],
        prompt: Option<&str>,
    ) -> Result<String, TranscriptionError> {
        if !self.primary.is_gpu() {
            let _ = self.state_tx.send(VoiceCaptureState::Transcribing);
            let result = self.primary.transcribe(pcm_i16_16k_mono, prompt).await;
            let _ = self.state_tx.send(VoiceCaptureState::Idle);
            return result;
        }

        let _ = self.state_tx.send(VoiceCaptureState::Queued);
        let use_cpu = self.should_fall_back_to_cpu();
        let _ = self.state_tx.send(VoiceCaptureState::Transcribing);

        let result = if use_cpu {
            self.cpu_transcriber()
                .await?
                .transcribe(pcm_i16_16k_mono, prompt)
                .await
        } else {
            self.primary.transcribe(pcm_i16_16k_mono, prompt).await
        };

        let _ = self.state_tx.send(VoiceCaptureState::Idle);
        result
    }

    fn subscribe_state(&self) -> Option<watch::Receiver<VoiceCaptureState>> {
        Some(self.state_tx.subscribe())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use tokio::sync::Notify;

    use super::*;

    #[derive(Debug)]
    struct StubTranscriber {
        text: String,
        gpu: bool,
        calls: AtomicUsize,
    }

    impl StubTranscriber {
        fn with_text(text: impl Into<String>) -> Arc<Self> {
            Self::build(text, false)
        }

        fn on_gpu(text: impl Into<String>) -> Arc<Self> {
            Self::build(text, true)
        }

        fn build(text: impl Into<String>, gpu: bool) -> Arc<Self> {
            Arc::new(Self {
                text: text.into(),
                gpu,
                calls: AtomicUsize::new(0),
            })
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Transcriber for StubTranscriber {
        async fn transcribe(
            &self,
            _pcm: &[i16],
            _prompt: Option<&str>,
        ) -> Result<String, TranscriptionError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.text.clone())
        }

        fn is_gpu(&self) -> bool {
            self.gpu
        }
    }

    #[derive(Debug)]
    struct ScriptedProbe {
        foreign: AtomicBool,
        active: AtomicBool,
    }

    impl ScriptedProbe {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                foreign: AtomicBool::new(false),
                active: AtomicBool::new(true),
            })
        }
        fn set_foreign(&self, foreign: bool) {
            self.foreign.store(foreign, Ordering::SeqCst);
        }
        fn set_active(&self, active: bool) {
            self.active.store(active, Ordering::SeqCst);
        }
    }

    impl BusyProbe for ScriptedProbe {
        fn foreign_gpu_busy(&self) -> bool {
            self.foreign.load(Ordering::SeqCst)
        }
        fn presence_active(&self) -> bool {
            self.active.load(Ordering::SeqCst)
        }
    }

    fn cpu_factory_for<T: Transcriber>(cpu: Arc<T>) -> CpuFallbackFactory {
        Arc::new(move || {
            let cpu = cpu.clone();
            Box::pin(async move { Ok(cpu as Arc<dyn Transcriber>) })
        })
    }

    #[tokio::test]
    async fn queued_routes_between_primary_and_cpu_fallback() {
        #[derive(Clone, Copy)]
        struct Case {
            name: &'static str,
            primary_gpu: bool,
            foreign: bool,
            active: bool,
            uses_cpu: bool,
        }
        let base = Case {
            name: "gpu idle",
            primary_gpu: true,
            foreign: false,
            active: true,
            uses_cpu: false,
        };
        for case in [
            base,
            Case {
                name: "foreign process holds vram",
                foreign: true,
                uses_cpu: true,
                ..base
            },
            Case {
                name: "presence not active",
                active: false,
                uses_cpu: true,
                ..base
            },
            Case {
                name: "cpu primary skips the queue",
                primary_gpu: false,
                active: false,
                ..base
            },
        ] {
            let primary = if case.primary_gpu {
                StubTranscriber::on_gpu("primary")
            } else {
                StubTranscriber::with_text("primary")
            };
            let cpu = StubTranscriber::with_text("cpu");
            let probe = ScriptedProbe::new();
            probe.set_foreign(case.foreign);
            probe.set_active(case.active);
            let queued =
                QueuedTranscriber::new(primary.clone(), cpu_factory_for(cpu.clone()), probe);
            let text = queued.transcribe(&[0i16; 16], None).await.unwrap();
            let (expected, primary_calls, cpu_calls) = if case.uses_cpu {
                ("cpu", 0, 1)
            } else {
                ("primary", 1, 0)
            };
            assert_eq!(text, expected, "{}", case.name);
            assert_eq!(
                primary.calls(),
                primary_calls,
                "{}: primary calls",
                case.name
            );
            assert_eq!(cpu.calls(), cpu_calls, "{}: cpu calls", case.name);
        }
    }

    #[derive(Debug)]
    struct GatedTranscriber {
        label: &'static str,
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    #[async_trait]
    impl Transcriber for GatedTranscriber {
        async fn transcribe(
            &self,
            _pcm: &[i16],
            _prompt: Option<&str>,
        ) -> Result<String, TranscriptionError> {
            self.started.notify_one();
            self.release.notified().await;
            Ok(self.label.to_string())
        }
    }

    #[tokio::test]
    async fn queued_publishes_transcribing_while_running_and_idle_when_done() {
        let primary = StubTranscriber::on_gpu("GPU");
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let cpu: Arc<GatedTranscriber> = Arc::new(GatedTranscriber {
            label: "CPU",
            started: started.clone(),
            release: release.clone(),
        });
        let probe = ScriptedProbe::new();
        probe.set_active(false);

        let queued = Arc::new(QueuedTranscriber::new(
            primary as Arc<dyn Transcriber>,
            cpu_factory_for(cpu),
            probe,
        ));
        let rx = queued.subscribe_state().expect("state is exposed");
        assert_eq!(*rx.borrow(), VoiceCaptureState::Idle);

        let transcription = {
            let queued = queued.clone();
            tokio::spawn(async move { queued.transcribe(&[0i16; 16], None).await })
        };

        started.notified().await;
        assert_eq!(*rx.borrow(), VoiceCaptureState::Transcribing);

        release.notify_one();
        let text = transcription.await.unwrap().unwrap();
        assert_eq!(text, "CPU");
        assert_eq!(*rx.borrow(), VoiceCaptureState::Idle);
    }
}

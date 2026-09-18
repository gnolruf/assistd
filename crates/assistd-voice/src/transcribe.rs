//! The [`Transcriber`] trait and the GPU-or-CPU [`QueuedTranscriber`].

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{OnceCell, watch};

use crate::VoiceCaptureState;

/// Transcribe 16 kHz mono PCM audio into text.
#[async_trait]
pub trait Transcriber: Send + Sync + 'static {
    /// Transcribe signed 16-bit mono samples at 16 kHz. Returns trimmed
    /// text; an empty string means the input was silence, not an error.
    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError>;

    /// State transitions the transcriber drives itself (`Queued`,
    /// `Transcribing`). `None` when the implementation has no internal
    /// states to report.
    fn subscribe_state(&self) -> Option<watch::Receiver<VoiceCaptureState>> {
        None
    }
}

/// Errors surfaced by a [`Transcriber`] implementation.
#[derive(Debug, thiserror::Error)]
pub enum TranscriptionError {
    #[error("empty audio buffer")]
    EmptyAudio,

    #[error("invalid model identifier {id:?}: {reason}")]
    ModelParse { id: String, reason: String },

    #[error(transparent)]
    Download(#[from] crate::hf_download::DownloadError),

    #[error("failed to initialize whisper context: {0}")]
    WhisperInit(String),

    #[error("whisper inference failed: {0}")]
    WhisperInference(String),

    #[error("tokio join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Answers "is the GPU available for Whisper right now?".
#[async_trait]
pub trait BusyProbe: Send + Sync + 'static {
    /// Wait up to `timeout` for in-flight LLM streams to drain. `true`
    /// when the GPU became free in time.
    async fn wait_until_llm_idle(&self, timeout: Duration) -> bool;

    /// True when a process outside assistd holds meaningful VRAM, so
    /// waiting would not help.
    fn foreign_gpu_busy(&self) -> bool;

    /// True when the GPU whisper context is safe to use. When false the
    /// context may be torn down concurrently and must not be touched.
    fn presence_active(&self) -> bool;
}

/// Probe that always reports the GPU free.
pub struct NullBusyProbe;

#[async_trait]
impl BusyProbe for NullBusyProbe {
    async fn wait_until_llm_idle(&self, _timeout: Duration) -> bool {
        true
    }
    fn foreign_gpu_busy(&self) -> bool {
        false
    }
    fn presence_active(&self) -> bool {
        true
    }
}

/// Runtime knobs for [`QueuedTranscriber`].
#[derive(Debug, Clone, Copy)]
pub struct QueueConfig {
    /// How long to wait for the LLM to finish streaming before falling
    /// back to CPU. `0` falls back the moment a stream is in flight.
    pub gpu_busy_timeout_ms: u32,
    /// When false, the primary is used unconditionally.
    pub cpu_fallback_enabled: bool,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            gpu_busy_timeout_ms: 300,
            cpu_fallback_enabled: true,
        }
    }
}

/// Async factory for the CPU fallback transcriber, invoked the first
/// time a fallback is needed.
pub type CpuFallbackFactory = Arc<
    dyn Fn() -> std::pin::Pin<
            Box<dyn Future<Output = Result<Arc<dyn Transcriber>, TranscriptionError>> + Send>,
        > + Send
        + Sync,
>;

/// Wraps a GPU transcriber with queue-and-fallback: publish `Queued`,
/// consult the [`BusyProbe`], and route to a lazily built CPU
/// transcriber when the GPU is unavailable or busy past the timeout.
/// A CPU-backed primary, or fallback disabled, runs directly.
pub struct QueuedTranscriber {
    primary: Arc<dyn Transcriber>,
    primary_is_gpu: bool,
    cpu: Arc<OnceCell<Arc<dyn Transcriber>>>,
    cpu_factory: CpuFallbackFactory,
    busy: Arc<dyn BusyProbe>,
    state_tx: watch::Sender<VoiceCaptureState>,
    cfg: QueueConfig,
}

impl QueuedTranscriber {
    /// `primary_is_gpu = false` disables the queue-and-fallback path.
    pub fn new(
        primary: Arc<dyn Transcriber>,
        primary_is_gpu: bool,
        cpu_factory: CpuFallbackFactory,
        busy: Arc<dyn BusyProbe>,
        cfg: QueueConfig,
    ) -> Self {
        let (state_tx, _) = watch::channel(VoiceCaptureState::Idle);
        Self {
            primary,
            primary_is_gpu,
            cpu: Arc::new(OnceCell::new()),
            cpu_factory,
            busy,
            state_tx,
            cfg,
        }
    }
}

#[async_trait]
impl Transcriber for QueuedTranscriber {
    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError> {
        if !self.primary_is_gpu || !self.cfg.cpu_fallback_enabled {
            let _ = self.state_tx.send(VoiceCaptureState::Transcribing);
            let result = self.primary.transcribe(pcm_i16_16k_mono).await;
            let _ = self.state_tx.send(VoiceCaptureState::Idle);
            return result;
        }

        let _ = self.state_tx.send(VoiceCaptureState::Queued);

        let use_cpu = if !self.busy.presence_active() {
            tracing::info!(
                target: "assistd::voice::queued",
                "falling back to CPU: presence is not Active"
            );
            true
        } else if self.busy.foreign_gpu_busy() {
            tracing::info!(
                target: "assistd::voice::queued",
                "falling back to CPU: foreign process holds VRAM"
            );
            true
        } else if self.cfg.gpu_busy_timeout_ms == 0 {
            let idle_now = self
                .busy
                .wait_until_llm_idle(Duration::from_millis(0))
                .await;
            if !idle_now {
                tracing::info!(
                    target: "assistd::voice::queued",
                    "falling back to CPU: gpu_busy_timeout_ms = 0 and an LLM stream is in flight"
                );
            }
            !idle_now
        } else {
            let timeout = Duration::from_millis(self.cfg.gpu_busy_timeout_ms as u64);
            let idle = self.busy.wait_until_llm_idle(timeout).await;
            if !idle {
                tracing::info!(
                    target: "assistd::voice::queued",
                    timeout_ms = self.cfg.gpu_busy_timeout_ms,
                    "falling back to CPU: LLM stream did not drain within timeout"
                );
            }
            !idle
        };

        let _ = self.state_tx.send(VoiceCaptureState::Transcribing);

        let result = if use_cpu {
            let factory = self.cpu_factory.clone();
            let cpu_cell = self.cpu.clone();
            let cpu = cpu_cell
                .get_or_try_init(|| async move { factory().await })
                .await?
                .clone();
            cpu.transcribe(pcm_i16_16k_mono).await
        } else {
            self.primary.transcribe(pcm_i16_16k_mono).await
        };

        let _ = self.state_tx.send(VoiceCaptureState::Idle);
        result
    }

    fn subscribe_state(&self) -> Option<watch::Receiver<VoiceCaptureState>> {
        Some(self.state_tx.subscribe())
    }
}

/// Test transcriber returning a fixed string and counting calls.
#[cfg(any(test, feature = "test-support"))]
pub struct StubTranscriber {
    text: String,
    calls: std::sync::atomic::AtomicUsize,
}

#[cfg(any(test, feature = "test-support"))]
impl StubTranscriber {
    pub fn with_text(text: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            text: text.into(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl Transcriber for StubTranscriber {
    async fn transcribe(&self, _pcm: &[i16]) -> Result<String, TranscriptionError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(self.text.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct ScriptedProbe {
        idle: AtomicBool,
        foreign: AtomicBool,
        active: AtomicBool,
    }

    impl ScriptedProbe {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                idle: AtomicBool::new(true),
                foreign: AtomicBool::new(false),
                active: AtomicBool::new(true),
            })
        }
        fn set_idle(&self, v: bool) {
            self.idle.store(v, Ordering::SeqCst);
        }
        fn set_foreign(&self, v: bool) {
            self.foreign.store(v, Ordering::SeqCst);
        }
        fn set_active(&self, v: bool) {
            self.active.store(v, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl BusyProbe for ScriptedProbe {
        async fn wait_until_llm_idle(&self, _t: Duration) -> bool {
            self.idle.load(Ordering::SeqCst)
        }
        fn foreign_gpu_busy(&self) -> bool {
            self.foreign.load(Ordering::SeqCst)
        }
        fn presence_active(&self) -> bool {
            self.active.load(Ordering::SeqCst)
        }
    }

    fn cpu_factory_for(stub: Arc<StubTranscriber>) -> CpuFallbackFactory {
        Arc::new(move || {
            let s = stub.clone();
            Box::pin(async move { Ok(s as Arc<dyn Transcriber>) })
        })
    }

    fn default_cfg() -> QueueConfig {
        QueueConfig {
            gpu_busy_timeout_ms: 50,
            cpu_fallback_enabled: true,
        }
    }

    #[tokio::test]
    async fn queued_uses_primary_when_busy_probe_idle() {
        let primary = StubTranscriber::with_text("GPU");
        let cpu = StubTranscriber::with_text("CPU");
        let probe = ScriptedProbe::new();
        let q = QueuedTranscriber::new(
            primary.clone() as Arc<dyn Transcriber>,
            true,
            cpu_factory_for(cpu.clone()),
            probe,
            default_cfg(),
        );
        let text = q.transcribe(&[0i16; 16]).await.unwrap();
        assert_eq!(text, "GPU");
        assert_eq!(primary.calls(), 1);
        assert_eq!(cpu.calls(), 0);
    }

    #[tokio::test]
    async fn queued_falls_back_to_cpu_on_timeout() {
        let primary = StubTranscriber::with_text("GPU");
        let cpu = StubTranscriber::with_text("CPU");
        let probe = ScriptedProbe::new();
        probe.set_idle(false);
        let q = QueuedTranscriber::new(
            primary.clone() as Arc<dyn Transcriber>,
            true,
            cpu_factory_for(cpu.clone()),
            probe,
            default_cfg(),
        );
        let text = q.transcribe(&[0i16; 16]).await.unwrap();
        assert_eq!(text, "CPU");
        assert_eq!(primary.calls(), 0);
        assert_eq!(cpu.calls(), 1);
    }

    #[tokio::test]
    async fn queued_falls_back_to_cpu_on_foreign_gpu() {
        let primary = StubTranscriber::with_text("GPU");
        let cpu = StubTranscriber::with_text("CPU");
        let probe = ScriptedProbe::new();
        probe.set_foreign(true);
        let q = QueuedTranscriber::new(
            primary.clone() as Arc<dyn Transcriber>,
            true,
            cpu_factory_for(cpu.clone()),
            probe,
            default_cfg(),
        );
        let text = q.transcribe(&[0i16; 16]).await.unwrap();
        assert_eq!(text, "CPU");
        assert_eq!(cpu.calls(), 1);
    }

    #[tokio::test]
    async fn queued_falls_back_to_cpu_when_presence_not_active() {
        let primary = StubTranscriber::with_text("GPU");
        let cpu = StubTranscriber::with_text("CPU");
        let probe = ScriptedProbe::new();
        probe.set_active(false);
        let q = QueuedTranscriber::new(
            primary.clone() as Arc<dyn Transcriber>,
            true,
            cpu_factory_for(cpu.clone()),
            probe,
            default_cfg(),
        );
        let text = q.transcribe(&[0i16; 16]).await.unwrap();
        assert_eq!(text, "CPU");
        assert_eq!(cpu.calls(), 1);
    }

    #[tokio::test]
    async fn queued_skips_queue_when_primary_is_cpu() {
        let primary = StubTranscriber::with_text("CPU-primary");
        let cpu = StubTranscriber::with_text("CPU-fallback");
        let probe = ScriptedProbe::new();
        probe.set_idle(false); // would normally force fallback
        let q = QueuedTranscriber::new(
            primary.clone() as Arc<dyn Transcriber>,
            /* primary_is_gpu = */ false,
            cpu_factory_for(cpu.clone()),
            probe,
            default_cfg(),
        );
        let text = q.transcribe(&[0i16; 16]).await.unwrap();
        assert_eq!(text, "CPU-primary");
        assert_eq!(cpu.calls(), 0);
    }

    #[tokio::test]
    async fn queued_skips_queue_when_fallback_disabled() {
        let primary = StubTranscriber::with_text("GPU");
        let cpu = StubTranscriber::with_text("CPU");
        let probe = ScriptedProbe::new();
        probe.set_idle(false);
        let q = QueuedTranscriber::new(
            primary.clone() as Arc<dyn Transcriber>,
            true,
            cpu_factory_for(cpu.clone()),
            probe,
            QueueConfig {
                gpu_busy_timeout_ms: 50,
                cpu_fallback_enabled: false,
            },
        );
        let text = q.transcribe(&[0i16; 16]).await.unwrap();
        assert_eq!(text, "GPU");
        assert_eq!(cpu.calls(), 0);
    }

    struct GatedTranscriber {
        label: &'static str,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl Transcriber for GatedTranscriber {
        async fn transcribe(&self, _pcm: &[i16]) -> Result<String, TranscriptionError> {
            self.started.notify_one();
            self.release.notified().await;
            Ok(self.label.to_string())
        }
    }

    #[tokio::test]
    async fn queued_publishes_transcribing_while_running_and_idle_when_done() {
        // A watch channel collapses same-tick updates, so the Queued
        // edge is not observable; only Transcribing-while-held and the
        // final Idle are asserted.
        let primary = StubTranscriber::with_text("GPU");
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let cpu: Arc<GatedTranscriber> = Arc::new(GatedTranscriber {
            label: "CPU",
            started: started.clone(),
            release: release.clone(),
        });
        let probe = ScriptedProbe::new();
        probe.set_idle(false);

        let factory: CpuFallbackFactory = {
            let cpu = cpu.clone();
            Arc::new(move || {
                let cpu = cpu.clone();
                Box::pin(async move { Ok(cpu as Arc<dyn Transcriber>) })
            })
        };

        let q = Arc::new(QueuedTranscriber::new(
            primary as Arc<dyn Transcriber>,
            true,
            factory,
            probe,
            default_cfg(),
        ));
        let rx = q.subscribe_state().expect("state is exposed");
        assert_eq!(*rx.borrow(), VoiceCaptureState::Idle);

        let q2 = q.clone();
        let handle = tokio::spawn(async move { q2.transcribe(&[0i16; 16]).await });

        started.notified().await;
        assert_eq!(*rx.borrow(), VoiceCaptureState::Transcribing);

        release.notify_one();
        let text = handle.await.unwrap().unwrap();
        assert_eq!(text, "CPU");

        for _ in 0..20 {
            if *rx.borrow() == VoiceCaptureState::Idle {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("state never returned to Idle, current = {:?}", *rx.borrow());
    }

    #[tokio::test]
    async fn null_busy_probe_reports_idle() {
        let probe = NullBusyProbe;
        assert!(probe.wait_until_llm_idle(Duration::from_millis(1)).await);
        assert!(!probe.foreign_gpu_busy());
        assert!(probe.presence_active());
    }
}

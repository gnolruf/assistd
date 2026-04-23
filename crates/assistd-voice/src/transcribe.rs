//! Speech-to-text primitive. The `Transcriber` trait is the stable
//! boundary: callers hand over a buffer of 16 kHz mono PCM samples and
//! get back a transcribed string, regardless of which backend is
//! running underneath.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{OnceCell, watch};

use crate::VoiceCaptureState;

/// Transcribe 16 kHz mono PCM audio into text.
#[async_trait]
pub trait Transcriber: Send + Sync + 'static {
    /// `pcm_i16_16k_mono` must be signed 16-bit samples at 16 kHz, single
    /// channel. Returns the transcribed text with leading/trailing
    /// whitespace trimmed. An empty string is returned when the input
    /// contains only silence (per VAD) — it is not an error.
    async fn transcribe(&self, pcm_i16_16k_mono: &[i16]) -> Result<String, TranscriptionError>;

    /// Optional stream of capture-state transitions driven by the
    /// transcriber itself. Implementations that internally move through
    /// `Queued`/`Transcribing` (e.g. [`crate::whisper::QueuedTranscriber`])
    /// override this so owners can surface those states without
    /// polling. The default returns `None` — owners publish their own
    /// `Transcribing` → `Idle` transition around the call instead.
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

    #[error("failed to download model from {url}: {source}")]
    ModelDownload {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("model download returned HTTP {status} from {url}")]
    ModelHttp { url: String, status: u16 },

    #[error("model I/O error at {path}: {source}")]
    ModelIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to initialize whisper context: {0}")]
    WhisperInit(String),

    #[error("whisper inference failed: {0}")]
    WhisperInference(String),

    #[error("tokio join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Injected "is the GPU available for Whisper?" probe. Concrete impls
/// live in the daemon crate (which wires them to `PresenceManager` and
/// optional NVML state); the voice crate stays NVML-free and just calls
/// into the trait.
#[async_trait]
pub trait BusyProbe: Send + Sync + 'static {
    /// Blocks up to `timeout` for the LLM stream count to reach zero.
    /// Returns `true` when the GPU became free in time, `false` on
    /// timeout. Implementations should short-circuit when no stream is
    /// in flight.
    async fn wait_until_llm_idle(&self, timeout: Duration) -> bool;

    /// True if a non-assistd process is currently holding meaningful
    /// VRAM (e.g. a game or another local model runner). When true,
    /// Whisper skips the GPU path entirely — waiting wouldn't help
    /// because the contender isn't ours to schedule around.
    fn foreign_gpu_busy(&self) -> bool;

    /// True when the daemon is confidently `Active` — i.e. llama-server
    /// is running and the GPU whisper context is safe to use. When
    /// false (Drowsy/Sleeping/mid-transition), Whisper must not touch
    /// the GPU because the context may be torn down concurrently.
    fn presence_active(&self) -> bool;
}

/// Null probe used in tests and when the daemon feature is compiled
/// out. Reports "GPU is always free" so orchestration falls back to
/// the primary synchronous path.
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

/// Runtime knobs for [`QueuedTranscriber`]. Populated from
/// [`assistd_config::TranscriptionConfig`].
#[derive(Debug, Clone, Copy)]
pub struct QueueConfig {
    /// How long to wait for the LLM to finish streaming before falling
    /// back to CPU. `0` forces CPU the moment a stream is inflight.
    pub gpu_busy_timeout_ms: u32,
    /// Whether a CPU fallback is permitted at all. When false, the
    /// transcriber waits indefinitely (well — up to the timeout) and
    /// then runs on the primary anyway.
    pub cpu_fallback_enabled: bool,
}

/// Async factory for the CPU fallback context. Built once at daemon
/// startup (from the same TranscriptionConfig that built the primary)
/// and stored inside [`QueuedTranscriber`], which invokes it the first
/// time a fallback is needed. Using a boxed future keeps the voice
/// crate free of a concrete builder type.
pub type CpuFallbackFactory = Arc<
    dyn Fn() -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<Arc<dyn Transcriber>, TranscriptionError>>
                    + Send,
            >,
        > + Send
        + Sync,
>;

/// Wraps a primary (GPU) transcriber with queue-and-fallback logic:
///
/// 1. If the primary is already CPU-backed or fallback is disabled →
///    run directly on the primary.
/// 2. Otherwise publish `Queued`, then consult the injected
///    [`BusyProbe`]. Route to CPU when presence isn't Active, a foreign
///    process is holding VRAM, or the LLM stream doesn't drain within
///    the configured timeout.
/// 3. Publish `Transcribing`, run inference on the chosen context,
///    publish `Idle` and return the text.
///
/// The CPU context is built lazily on first fallback via
/// [`CpuFallbackFactory`] and retained for subsequent fallbacks. Owners
/// subscribe to state transitions via [`Transcriber::subscribe_state`].
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
        // Fast path: the primary is already CPU-backed, or the user has
        // opted out of the fallback entirely. No contention window, no
        // Queued state — straight to Transcribing.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Test-only transcriber that records how many times it was
    /// invoked and returns a configurable string. Mock primary or CPU
    /// context depending on label.
    struct StubTranscriber {
        label: &'static str,
        calls: AtomicUsize,
    }

    impl StubTranscriber {
        fn new(label: &'static str) -> Arc<Self> {
            Arc::new(Self {
                label,
                calls: AtomicUsize::new(0),
            })
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Transcriber for StubTranscriber {
        async fn transcribe(&self, _pcm: &[i16]) -> Result<String, TranscriptionError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.label.to_string())
        }
    }

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
        let primary = StubTranscriber::new("GPU");
        let cpu = StubTranscriber::new("CPU");
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
        let primary = StubTranscriber::new("GPU");
        let cpu = StubTranscriber::new("CPU");
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
        let primary = StubTranscriber::new("GPU");
        let cpu = StubTranscriber::new("CPU");
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
        let primary = StubTranscriber::new("GPU");
        let cpu = StubTranscriber::new("CPU");
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
        let primary = StubTranscriber::new("CPU-primary");
        let cpu = StubTranscriber::new("CPU-fallback");
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
        let primary = StubTranscriber::new("GPU");
        let cpu = StubTranscriber::new("CPU");
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

    /// A stub that gates on a Notify so the test can observe state
    /// while transcription is in flight rather than racing against
    /// fast synchronous stubs collapsing values in the watch channel.
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
        // The intermediate Queued → Transcribing edge is not
        // deterministically observable through a watch channel because
        // watch collapses same-tick updates to "latest value". What we
        // CAN assert is: while the CPU stub is held inside transcribe,
        // the visible state is Transcribing (not Idle), and when the
        // transcriber finishes the state returns to Idle. That's the
        // invariant the TUI relies on for its indicator — a fast
        // cold-path flash of Queued is acceptable.
        let primary = StubTranscriber::new("GPU");
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

        // Wait for CPU stub to be inside transcribe — state must now be
        // Transcribing (never Idle while inference is running).
        started.notified().await;
        assert_eq!(*rx.borrow(), VoiceCaptureState::Transcribing);

        // Release CPU inference; the final Idle must be published.
        release.notify_one();
        let text = handle.await.unwrap().unwrap();
        assert_eq!(text, "CPU");

        // Final state is Idle. Small yield in case the outer send lands
        // just after we drop back into the scheduler.
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

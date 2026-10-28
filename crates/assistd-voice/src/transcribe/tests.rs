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
    async fn transcribe(&self, _pcm: &[i16]) -> Result<String, TranscriptionError> {
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
        let queued = QueuedTranscriber::new(primary.clone(), cpu_factory_for(cpu.clone()), probe);
        let text = queued.transcribe(&[0i16; 16]).await.unwrap();
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
    async fn transcribe(&self, _pcm: &[i16]) -> Result<String, TranscriptionError> {
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
        tokio::spawn(async move { queued.transcribe(&[0i16; 16]).await })
    };

    started.notified().await;
    assert_eq!(*rx.borrow(), VoiceCaptureState::Transcribing);

    release.notify_one();
    let text = transcription.await.unwrap().unwrap();
    assert_eq!(text, "CPU");
    assert_eq!(*rx.borrow(), VoiceCaptureState::Idle);
}

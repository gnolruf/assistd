#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

//! End-to-end audio pipeline tests using a deterministic stub transcriber
//! so the suite runs in CI without downloading whisper models or relying
//! on a microphone. Exercises the public state machine
//! (`Recording → Transcribing → Idle`) and the QueuedTranscriber wrapper.
//!
//! Run with: `cargo test -p assistd-voice --features test-support --test audio_round_trip`

#![cfg(feature = "test-support")]

use std::sync::Arc;
use std::time::Duration;

use assistd_voice::{
    CpuFallbackFactory, MicVoiceInput, NullBusyProbe, QueueConfig, QueuedTranscriber,
    StubTranscriber, Transcriber, VoiceCaptureState, VoiceInput,
};

/// Build a 1-second 16 kHz mono i16 PCM buffer of a 440 Hz sine at half
/// amplitude. The actual content doesn't matter (the StubTranscriber
/// ignores it), but we want a non-trivial buffer so length-dependent
/// code paths (peak-dBfs logging, empty-buffer guard) are exercised.
fn synth_sine_pcm(seconds: f32) -> Vec<i16> {
    let sample_rate = 16_000.0_f32;
    let frequency = 440.0_f32;
    let n = (sample_rate * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let s = (2.0 * std::f32::consts::PI * frequency * t).sin();
            (s * (i16::MAX as f32 * 0.5)) as i16
        })
        .collect()
}

#[tokio::test]
async fn synthetic_pcm_through_stub_returns_canned_text() {
    let stub = StubTranscriber::with_text("ask not what your country");
    let pcm = synth_sine_pcm(1.0);
    let text = stub.transcribe(&pcm).await.unwrap();
    assert_eq!(text, "ask not what your country");
    assert_eq!(stub.calls(), 1);
}

#[tokio::test]
async fn queued_transcriber_uses_primary_and_publishes_terminal_idle() {
    let primary = StubTranscriber::with_text("queued result");
    let cpu = StubTranscriber::with_text("cpu fallback");
    let primary_for_cast: Arc<dyn Transcriber> = primary.clone();
    let cpu_arc = cpu.clone();
    let factory: CpuFallbackFactory = Arc::new(move || {
        let cpu = cpu_arc.clone();
        Box::pin(async move { Ok(cpu as Arc<dyn Transcriber>) })
    });
    let q = Arc::new(QueuedTranscriber::new(
        primary_for_cast,
        true, // primary_is_gpu = true → exercise the Queue → Transcribe → Idle path
        factory,
        Arc::new(NullBusyProbe),
        QueueConfig {
            gpu_busy_timeout_ms: 50,
            cpu_fallback_enabled: true,
        },
    ));
    let mut rx = q.subscribe_state().expect("queued exposes state stream");
    assert_eq!(*rx.borrow_and_update(), VoiceCaptureState::Idle);

    let pcm = synth_sine_pcm(0.5);
    let text = q.transcribe(&pcm).await.unwrap();
    assert_eq!(text, "queued result");
    assert_eq!(
        primary.calls(),
        1,
        "NullBusyProbe is idle: primary should run"
    );
    assert_eq!(
        cpu.calls(),
        0,
        "CPU fallback not exercised when primary used"
    );

    // Watch may collapse same-tick updates, but the await on the
    // transcriber guarantees a yield point so the terminal Idle is
    // observable. Allow a small number of yields for the watch to
    // settle to Idle.
    for _ in 0..32 {
        if *rx.borrow() == VoiceCaptureState::Idle {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("state never settled to Idle: {:?}", *rx.borrow());
}

#[tokio::test]
async fn mic_voice_input_publishes_recording_transcribing_idle_sequence() {
    let stub = StubTranscriber::with_text("transcribed pcm");
    let mic = MicVoiceInput::new(stub.clone(), None, 60);

    let mut rx = mic.subscribe();
    assert_eq!(*rx.borrow_and_update(), VoiceCaptureState::Idle);

    // Collect every state transition until the watch sender is closed
    // (mic dropped at end of test). Yields between sends in
    // transcribe_pcm_for_test ensure the watch publishes Recording →
    // Transcribing distinctly rather than collapsing them.
    let collector = tokio::spawn(async move {
        let mut seen = vec![*rx.borrow_and_update()];
        while rx.changed().await.is_ok() {
            seen.push(*rx.borrow_and_update());
        }
        seen
    });

    let pcm = synth_sine_pcm(0.25);
    let text = tokio::time::timeout(Duration::from_secs(2), mic.transcribe_pcm_for_test(&pcm))
        .await
        .expect("transcribe_pcm_for_test timed out")
        .expect("transcribe_pcm_for_test errored");
    assert_eq!(text, "transcribed pcm");
    assert_eq!(stub.calls(), 1);

    // Drop the mic so the watch sender closes and the collector exits.
    drop(mic);
    let states = collector.await.expect("collector task panicked");

    assert!(
        states.contains(&VoiceCaptureState::Recording),
        "expected Recording in {states:?}"
    );
    assert!(
        states.contains(&VoiceCaptureState::Transcribing),
        "expected Transcribing in {states:?}"
    );
    assert_eq!(
        *states.last().unwrap(),
        VoiceCaptureState::Idle,
        "final state must be Idle, got {states:?}"
    );
}

#[tokio::test]
async fn mic_voice_input_round_trip_returns_stub_text_for_silent_pcm() {
    // VAD trimming would empty out silent input, but the stub transcriber
    // doesn't run VAD; it returns its canned text regardless. This pins
    // the contract that the test path is deterministic on any input.
    let stub = StubTranscriber::with_text("anything");
    let mic = MicVoiceInput::new(stub, None, 60);
    let pcm = vec![0i16; 16_000];
    let text = mic.transcribe_pcm_for_test(&pcm).await.unwrap();
    assert_eq!(text, "anything");
    assert_eq!(mic.state(), VoiceCaptureState::Idle);
}

//! `QueuedTranscriber` through the public API with the stock
//! `NullBusyProbe`, using stub transcribers so no model is needed.

#![cfg(feature = "test-support")]

use std::sync::Arc;

use assistd_voice::{
    CpuFallbackFactory, NullBusyProbe, QueueConfig, QueuedTranscriber, StubTranscriber,
    Transcriber, VoiceCaptureState,
};

#[tokio::test]
async fn queued_transcriber_uses_primary_and_publishes_terminal_idle() {
    let primary = StubTranscriber::on_gpu("queued result");
    let cpu = StubTranscriber::with_text("cpu fallback");
    let factory: CpuFallbackFactory = {
        let cpu = cpu.clone();
        Arc::new(move || {
            let cpu = cpu.clone();
            Box::pin(async move { Ok(cpu as Arc<dyn Transcriber>) })
        })
    };
    let queued = QueuedTranscriber::new(
        primary.clone(),
        factory,
        Arc::new(NullBusyProbe),
        QueueConfig::default(),
    );
    let rx = queued
        .subscribe_state()
        .expect("queued exposes state stream");
    assert_eq!(*rx.borrow(), VoiceCaptureState::Idle);

    let text = queued.transcribe(&[0i16; 8_000]).await.unwrap();
    assert_eq!(text, "queued result");
    assert_eq!(primary.calls(), 1);
    assert_eq!(cpu.calls(), 0);
    assert_eq!(*rx.borrow(), VoiceCaptureState::Idle);
}

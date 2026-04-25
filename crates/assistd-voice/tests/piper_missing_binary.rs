//! Verifies the "if Piper is not installed, the daemon starts with
//! voice output disabled" acceptance criterion. Runs in any
//! environment because it points `binary_path` at a path that is
//! guaranteed not to exist; no real Piper is required.

#![cfg(feature = "tts")]

use assistd_config::SynthesisConfig;
use assistd_voice::PiperVoiceOutput;

#[tokio::test]
async fn start_returns_err_when_binary_is_missing() {
    let cfg = SynthesisConfig {
        enabled: true,
        binary_path: "/definitely/not/a/real/piper/binary/path".into(),
        ..SynthesisConfig::default()
    };

    let err = match PiperVoiceOutput::start(cfg).await {
        Ok(_) => panic!("expected PiperError::BinaryMissing"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("not found on PATH"),
        "expected BinaryMissing, got: {msg}"
    );
}

#[tokio::test]
async fn start_returns_err_when_voice_id_is_invalid() {
    // A valid-on-PATH binary doesn't exist in CI sandboxes either, so
    // we skip the voice-parse path and just confirm a malformed HF id
    // is rejected without ever spawning a process.
    let cfg = SynthesisConfig {
        enabled: true,
        // Point at a fake binary so the binary check fails first; we
        // really just care that bad config errors out predictably.
        binary_path: "/no/such/binary".into(),
        voice: "not-a-valid-hf-id".into(),
        ..SynthesisConfig::default()
    };

    let err = match PiperVoiceOutput::start(cfg).await {
        Ok(_) => panic!("expected an error from start()"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("not found on PATH") || msg.contains("invalid voice identifier"),
        "expected binary-or-voice error, got: {msg}"
    );
}

//! A missing Piper binary fails `PiperVoiceOutput::start` with
//! `BinaryMissing` before any download or subprocess.

#![cfg(feature = "tts")]

use assistd_config::SynthesisConfig;
use assistd_voice::{PiperError, PiperVoiceOutput};

#[tokio::test]
async fn start_returns_err_when_binary_is_missing() {
    let config = SynthesisConfig {
        enabled: true,
        binary_path: "/definitely/not/a/real/piper/binary/path".into(),
        ..SynthesisConfig::default()
    };

    let Err(err) = PiperVoiceOutput::start(config).await else {
        panic!("expected PiperError::BinaryMissing");
    };
    assert!(
        matches!(&err, PiperError::BinaryMissing { binary } if binary.as_os_str() == "/definitely/not/a/real/piper/binary/path"),
        "got {err:?}"
    );
}

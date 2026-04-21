//! End-to-end transcription test. Downloads a small Whisper model and a
//! known speech clip from public URLs, then asserts:
//!   - the 5+ second clip of clear English is transcribed accurately;
//!   - silence padding is trimmed by whisper.cpp's built-in VAD so the
//!     padded clip's transcript matches the unpadded one;
//!   - the run completes within a generous latency ceiling.
//!
//! Gated behind the `test-support` feature so a plain
//! `cargo test -p assistd-voice` stays offline.

#![cfg(feature = "test-support")]

use std::path::PathBuf;
use std::sync::Once;
use std::time::{Duration, Instant};

use assistd_voice::{Transcriber, WhisperTranscriber};

const TINY_MODEL_ID: &str = "ggerganov/whisper.cpp:ggml-tiny.en-q5_1.bin";
const VAD_MODEL_ID: &str = "ggml-org/whisper-vad:ggml-silero-v6.2.0.bin";
const SPEECH_URL: &str =
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav";

static TRACING_INIT: Once = Once::new();

fn init_tracing() {
    TRACING_INIT.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_test_writer()
            .try_init();
    });
}

async fn fetch_to(path: &std::path::Path, url: &str) {
    let bytes = reqwest::get(url)
        .await
        .expect("download speech clip")
        .bytes()
        .await
        .expect("read speech clip");
    tokio::fs::write(path, bytes)
        .await
        .expect("write speech clip");
}

fn load_pcm_16k_mono(path: &std::path::Path) -> Vec<i16> {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16000, "fixture must be 16 kHz");
    assert_eq!(spec.channels, 1, "fixture must be mono");
    assert_eq!(spec.bits_per_sample, 16, "fixture must be 16-bit");
    reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .expect("decode wav samples")
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

async fn make_transcriber(cache: &std::path::Path, vad: bool) -> WhisperTranscriber {
    let mut b = WhisperTranscriber::builder()
        .model(TINY_MODEL_ID)
        .cache_dir(Some(cache.to_path_buf()))
        .prefer_gpu(true)
        .beams(1)
        .vad_enabled(vad);
    if vad {
        b = b.vad_model(VAD_MODEL_ID).vad_silence_secs(0.5);
    }
    b.build().await.expect("transcriber builds")
}

struct Fixture {
    _tmp: tempfile::TempDir,
    cache: PathBuf,
    wav: PathBuf,
}

async fn setup() -> Fixture {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cache = tmp.path().join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let wav = tmp.path().join("jfk.wav");
    fetch_to(&wav, SPEECH_URL).await;
    Fixture {
        _tmp: tmp,
        cache,
        wav,
    }
}

#[tokio::test]
async fn transcribes_clear_english_speech() {
    init_tracing();
    let fx = setup().await;
    let pcm = load_pcm_16k_mono(&fx.wav);
    assert!(pcm.len() >= 16000 * 5, "fixture should be at least 5s");

    let transcriber = make_transcriber(&fx.cache, false).await;
    let started = Instant::now();
    let text = transcriber.transcribe(&pcm).await.expect("transcribe");
    let elapsed = started.elapsed();

    let norm = normalize(&text);
    assert!(
        norm.contains("ask not"),
        "expected 'ask not' in transcript: {norm:?}"
    );
    assert!(
        norm.contains("country"),
        "expected 'country' in transcript: {norm:?}"
    );
    assert!(
        elapsed < Duration::from_secs(60),
        "transcription took {:?} — latency regression",
        elapsed
    );
}

#[tokio::test]
async fn vad_trims_silence_padding() {
    init_tracing();
    let fx = setup().await;
    let pcm = load_pcm_16k_mono(&fx.wav);
    let silence: Vec<i16> = vec![0; 16000];
    let mut padded = Vec::with_capacity(silence.len() * 2 + pcm.len());
    padded.extend_from_slice(&silence);
    padded.extend_from_slice(&pcm);
    padded.extend_from_slice(&silence);

    let transcriber = make_transcriber(&fx.cache, true).await;
    let plain = transcriber
        .transcribe(&pcm)
        .await
        .expect("plain transcribe");
    let padded_out = transcriber
        .transcribe(&padded)
        .await
        .expect("padded transcribe");

    let plain_norm = normalize(&plain);
    let padded_norm = normalize(&padded_out);

    for needle in ["ask not", "country"] {
        assert!(
            padded_norm.contains(needle),
            "padded result missing {needle:?}: {padded_norm:?}"
        );
    }

    let plain_words = plain_norm.split_whitespace().count();
    let padded_words = padded_norm.split_whitespace().count();
    assert!(
        padded_words <= plain_words.saturating_add(3),
        "VAD should have trimmed silence: plain={plain_words} padded={padded_words}"
    );
}

#[tokio::test]
async fn empty_audio_is_rejected() {
    init_tracing();
    let fx = setup().await;
    let transcriber = make_transcriber(&fx.cache, false).await;
    let err = transcriber
        .transcribe(&[])
        .await
        .expect_err("empty audio should fail");
    assert!(matches!(err, assistd_voice::TranscriptionError::EmptyAudio));
}

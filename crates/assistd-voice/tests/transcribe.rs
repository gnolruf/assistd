//! End-to-end transcription against Parakeet on the CPU and a known
//! speech clip, both downloaded: checks accuracy, that Silero VAD
//! trims silence padding, and a generous latency ceiling. Gated behind
//! `test-support` so a plain `cargo test -p assistd-voice` stays offline.

#![cfg(feature = "test-support")]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use assistd_config::TranscriptionConfig;
use assistd_voice::{SherpaTranscriber, Transcriber};

const CPU_MODEL_ID: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8\
     @1ab9323565ddb038682214b292f588070a538ce2:encoder.int8.onnx";
const VAD_MODEL_ID: &str =
    "Tinysoft/silero-vad-6.2@3034960e5d3beb943b860ec0ea1a857bdae4ae96:silero_vad.onnx";
const SPEECH_URL: &str =
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav";

fn init_tracing() {
    assistd_utils::tracing_init::init_test_tracing("info");
}

async fn fetch_to(path: &Path, url: &str) {
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

fn load_pcm_16k_mono(path: &Path) -> Vec<i16> {
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

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Models download to the user's cache, so only the first run fetches them.
async fn make_transcriber(vad: bool) -> SherpaTranscriber {
    let config = TranscriptionConfig {
        model: CPU_MODEL_ID.into(),
        vad_enabled: vad,
        vad_model: VAD_MODEL_ID.into(),
        ..TranscriptionConfig::default()
    };
    SherpaTranscriber::load_cpu(&config)
        .await
        .expect("transcriber loads")
}

struct Fixture {
    _tmp: tempfile::TempDir,
    wav: PathBuf,
}

async fn setup() -> Fixture {
    let tmp = tempfile::tempdir().expect("tempdir");
    let wav = tmp.path().join("jfk.wav");
    fetch_to(&wav, SPEECH_URL).await;
    Fixture { _tmp: tmp, wav }
}

#[tokio::test]
#[ignore = "downloads ~650MB Parakeet model + JFK clip; run with --ignored for real-fidelity coverage"]
async fn transcribes_clear_english_speech() {
    init_tracing();
    let fixture = setup().await;
    let pcm = load_pcm_16k_mono(&fixture.wav);
    assert!(pcm.len() >= 16000 * 5, "fixture should be at least 5s");

    let transcriber = make_transcriber(false).await;
    let started = Instant::now();
    let text = transcriber.transcribe(&pcm).await.expect("transcribe");
    let elapsed = started.elapsed();

    let normalized = normalize(&text);
    assert!(
        normalized.contains("ask not"),
        "expected 'ask not' in transcript: {normalized:?}"
    );
    assert!(
        normalized.contains("country"),
        "expected 'country' in transcript: {normalized:?}"
    );
    assert!(
        elapsed < Duration::from_secs(60),
        "transcription took {elapsed:?}: latency regression"
    );
}

#[tokio::test]
#[ignore = "downloads Parakeet + Silero VAD models; run with --ignored for real-fidelity coverage"]
async fn vad_trims_silence_padding() {
    init_tracing();
    let fixture = setup().await;
    let pcm = load_pcm_16k_mono(&fixture.wav);
    let silence: Vec<i16> = vec![0; 16000];
    let mut padded = Vec::with_capacity(silence.len() * 2 + pcm.len());
    padded.extend_from_slice(&silence);
    padded.extend_from_slice(&pcm);
    padded.extend_from_slice(&silence);

    let transcriber = make_transcriber(true).await;
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

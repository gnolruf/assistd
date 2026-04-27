//! cpal stream construction and the audio-thread callback.
//!
//! `cpal::Stream` is `!Send` on most platforms (the ALSA backend holds a
//! raw ALSA handle whose safety invariants are tied to its creating
//! thread). To avoid poisoning `MicVoiceInput: Send + Sync`, the stream
//! lives entirely inside a single `spawn_blocking` worker — the outer
//! `MicVoiceInput` only holds a `JoinHandle` plus shared atomics, all of
//! which are trivially `Send`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig, StreamError};
use ringbuf::HeapRb;
use ringbuf::traits::{Producer, Split};
use thiserror::Error;
use tokio::task::JoinHandle;
use tracing::warn;

use super::consumer;

#[derive(Debug, Error)]
pub enum AudioCaptureError {
    #[error("no default input device available")]
    NoDefaultDevice,
    #[error("no input device matched {0:?}")]
    NoMatchingDevice(String),
    #[error("failed to query default input config: {0}")]
    DefaultConfig(String),
    #[error("device does not support any standard sample format")]
    UnsupportedFormat,
    #[error("cpal error building stream: {0}")]
    BuildStream(String),
    #[error("cpal error starting stream: {0}")]
    PlayStream(String),
    #[error("device error during capture: {0}")]
    DeviceError(String),
}

/// Target sample rate for whisper. The consumer resamples from native
/// device rate to this before conversion to i16.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Handles to a running capture session. The cpal stream is owned by
/// the consumer task — callers only see the atomics and the join
/// handle.
pub struct CaptureSession {
    pub stop_flag: Arc<AtomicBool>,
    pub overrun: Arc<AtomicU64>,
    pub handle: JoinHandle<Result<Vec<i16>, AudioCaptureError>>,
}

/// Pieces returned by [`open_producer_stream`] to a blocking worker.
/// The `Stream` must stay alive on the thread that built it (cpal's
/// `Stream` is `!Send` on ALSA); the caller drops it last.
pub struct ProducerStream {
    pub consumer: ringbuf::HeapCons<f32>,
    pub native_rate: u32,
    pub stream: Stream,
}

/// Open the cpal input device and start a running stream that pushes
/// mono f32 samples at the native device rate into a ring buffer.
///
/// Runs synchronously on the caller's thread because cpal's `Stream`
/// is `!Send` on ALSA — both PTT and continuous-listen callers invoke
/// this from inside a `tokio::task::spawn_blocking` worker.
///
/// `ring_capacity_samples` sizes the SPSC ring (in mono samples at
/// native rate). PTT sizes this to cover `max_recording_secs`;
/// continuous-listen can use a few seconds of headroom since its
/// consumer drains continuously.
///
/// `overrun` is the counter the cpal callback bumps when the ring is
/// full and samples are dropped. The caller keeps a clone to read it.
pub fn open_producer_stream(
    device_hint: Option<&str>,
    ring_capacity_samples: usize,
    overrun: Arc<AtomicU64>,
) -> Result<ProducerStream, AudioCaptureError> {
    let host = cpal::default_host();
    let device = select_device(&host, device_hint)?;
    let device_name = device.name().unwrap_or_else(|_| "<unknown>".to_string());

    let supported = device
        .default_input_config()
        .map_err(|e| AudioCaptureError::DefaultConfig(e.to_string()))?;
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();

    let rb = HeapRb::<f32>::new(ring_capacity_samples.max(sample_rate as usize));
    let (prod, cons) = rb.split();

    let stream = build_stream(&device, &config, sample_format, channels, prod, overrun)?;
    stream
        .play()
        .map_err(|e| AudioCaptureError::PlayStream(e.to_string()))?;

    tracing::info!(
        target: "assistd::voice::mic",
        device = %device_name,
        sample_rate,
        channels,
        format = ?sample_format,
        ring_capacity = ring_capacity_samples,
        "audio stream started"
    );

    Ok(ProducerStream {
        consumer: cons,
        native_rate: sample_rate,
        stream,
    })
}

/// Spawn a blocking worker that opens cpal, records, and returns the
/// resampled 16 kHz mono i16 PCM when stopped. Device enumeration
/// happens inside the worker so `!Send` cpal types never cross threads.
pub fn start(device_hint: Option<&str>, max_recording_secs: u32) -> CaptureSession {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let overrun = Arc::new(AtomicU64::new(0));
    let device_hint_owned = device_hint.map(|s| s.to_string());

    let worker_stop = Arc::clone(&stop_flag);
    let worker_overrun = Arc::clone(&overrun);
    let handle = tokio::task::spawn_blocking(move || {
        capture_worker(
            device_hint_owned.as_deref(),
            max_recording_secs,
            worker_stop,
            worker_overrun,
        )
    });

    CaptureSession {
        stop_flag,
        overrun,
        handle,
    }
}

fn capture_worker(
    device_hint: Option<&str>,
    max_recording_secs: u32,
    stop_flag: Arc<AtomicBool>,
    overrun: Arc<AtomicU64>,
) -> Result<Vec<i16>, AudioCaptureError> {
    // Ring capacity: 1 s per second of recording + 1 s headroom. Sized
    // in mono samples at native rate; we over-provision slightly using
    // a conservative 48 kHz assumption — `open_producer_stream`
    // enforces a per-open floor regardless.
    let conservative_rate = 48_000usize;
    let ring_cap = conservative_rate
        .saturating_mul((max_recording_secs as usize).saturating_add(1))
        .max(conservative_rate * 2);
    let ProducerStream {
        consumer: cons,
        native_rate,
        stream,
    } = open_producer_stream(device_hint, ring_cap, overrun)?;

    let max_pcm_samples = (TARGET_SAMPLE_RATE as usize).saturating_mul(max_recording_secs as usize);
    let pcm = consumer::drain_loop(cons, native_rate, max_pcm_samples, stop_flag)?;

    // Drop cpal stream on this same thread. On ALSA this joins the
    // worker thread; doing it here avoids any `!Send` drop problem.
    drop(stream);
    Ok(pcm)
}

/// Pre-flight check called from daemon startup, mirroring
/// [`crate::gpu`]'s graceful-degradation idiom. When
/// [`assistd_config::VoiceConfig::enabled`] is false, succeeds
/// unconditionally. When `mic_device` is `None`, the system default
/// will be selected at PTT start — missing-default is intentionally a
/// soft failure (we don't want a headless CI without ALSA to fail
/// daemon startup). Only the `Some(name)` case where the configured
/// name cannot be found is hard-rejected.
///
/// The error message includes the configured name, every available
/// input device name enumerated via cpal, and a hint about
/// `mic_device = null`. If cpal itself refuses to enumerate devices,
/// that is wrapped verbatim.
pub fn validate(cfg: &assistd_config::VoiceConfig) -> anyhow::Result<()> {
    if !cfg.enabled {
        return Ok(());
    }

    // Always enumerate input devices at startup so the log captures
    // which cpal-visible devices exist — indispensable for diagnosing
    // "wrong mic picked" problems on systems with many PipeWire sinks.
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_else(|| "<none>".to_string());
    if let Ok(devices) = host.input_devices() {
        let names: Vec<String> = devices.filter_map(|d| d.name().ok()).collect();
        tracing::info!(
            target: "assistd::voice::mic",
            default = %default_name,
            available = ?names,
            "cpal input devices"
        );
    }

    let Some(requested) = cfg.mic_device.as_deref() else {
        return Ok(());
    };

    let devices = match host.input_devices() {
        Ok(d) => d,
        Err(e) => {
            anyhow::bail!(
                "failed to enumerate cpal input devices while validating voice.mic_device \
                 = {requested:?}: {e}"
            );
        }
    };
    let mut names: Vec<String> = Vec::new();
    let mut matched = false;
    for d in devices {
        let name = d.name().unwrap_or_else(|_| "<unknown>".to_string());
        if name == requested {
            matched = true;
        }
        names.push(name);
    }
    if matched {
        return Ok(());
    }

    let listing = if names.is_empty() {
        "<no input devices reported by cpal>".to_string()
    } else {
        names
            .iter()
            .map(|n| format!("{n:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    anyhow::bail!(
        "voice.mic_device = {requested:?} not found. Available input devices: [{listing}]. \
         Set voice.mic_device = null to use the system default."
    );
}

pub fn select_device(host: &cpal::Host, hint: Option<&str>) -> Result<Device, AudioCaptureError> {
    match hint {
        None => host
            .default_input_device()
            .ok_or(AudioCaptureError::NoDefaultDevice),
        Some(name) => {
            let devices = host
                .input_devices()
                .map_err(|e| AudioCaptureError::DefaultConfig(e.to_string()))?;
            for d in devices {
                if d.name().map(|n| n == name).unwrap_or(false) {
                    return Ok(d);
                }
            }
            Err(AudioCaptureError::NoMatchingDevice(name.to_string()))
        }
    }
}

type RbProd = ringbuf::HeapProd<f32>;

fn build_stream(
    device: &Device,
    config: &StreamConfig,
    format: SampleFormat,
    channels: usize,
    producer: RbProd,
    overrun: Arc<AtomicU64>,
) -> Result<Stream, AudioCaptureError> {
    let err_cb = |e: StreamError| {
        warn!(target: "assistd::voice::mic", "cpal stream error: {e}");
    };

    let stream = match format {
        SampleFormat::F32 => {
            let state = CallbackState::new(producer, channels, overrun);
            device
                .build_input_stream(
                    config,
                    move |data: &[f32], _| state.push_f32(data),
                    err_cb,
                    None,
                )
                .map_err(|e| AudioCaptureError::BuildStream(e.to_string()))?
        }
        SampleFormat::I16 => {
            let state = CallbackState::new(producer, channels, overrun);
            device
                .build_input_stream(
                    config,
                    move |data: &[i16], _| state.push_i16(data),
                    err_cb,
                    None,
                )
                .map_err(|e| AudioCaptureError::BuildStream(e.to_string()))?
        }
        SampleFormat::U16 => {
            let state = CallbackState::new(producer, channels, overrun);
            device
                .build_input_stream(
                    config,
                    move |data: &[u16], _| state.push_u16(data),
                    err_cb,
                    None,
                )
                .map_err(|e| AudioCaptureError::BuildStream(e.to_string()))?
        }
        _ => return Err(AudioCaptureError::UnsupportedFormat),
    };
    Ok(stream)
}

/// Mutable state shared with the audio callback. cpal calls the
/// closure from its own worker thread; the inner `Mutex`es guard
/// scratch buffers and the ring producer, while the `AtomicU64`
/// counts dropped samples on overrun.
struct CallbackState {
    producer: std::sync::Mutex<RbProd>,
    channels: usize,
    overrun: Arc<AtomicU64>,
    /// Scratch buffer for downmixed mono samples. Pre-sized to avoid
    /// allocation in the hot path; a reasonable cpal callback delivers
    /// a few thousand frames at most.
    scratch: std::sync::Mutex<Vec<f32>>,
}

impl CallbackState {
    fn new(producer: RbProd, channels: usize, overrun: Arc<AtomicU64>) -> Self {
        Self {
            producer: std::sync::Mutex::new(producer),
            channels,
            overrun,
            scratch: std::sync::Mutex::new(Vec::with_capacity(8192)),
        }
    }

    fn push_f32(&self, data: &[f32]) {
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.clear();
        if self.channels <= 1 {
            scratch.extend_from_slice(data);
        } else {
            let ch = self.channels;
            for frame in data.chunks_exact(ch) {
                let sum: f32 = frame.iter().copied().sum();
                scratch.push(sum / ch as f32);
            }
        }
        self.push_mono(&scratch);
    }

    fn push_i16(&self, data: &[i16]) {
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.clear();
        let scale = i16::MAX as f32;
        if self.channels <= 1 {
            for &s in data {
                scratch.push(s as f32 / scale);
            }
        } else {
            let ch = self.channels;
            for frame in data.chunks_exact(ch) {
                let sum: f32 = frame.iter().map(|&s| s as f32 / scale).sum();
                scratch.push(sum / ch as f32);
            }
        }
        self.push_mono(&scratch);
    }

    fn push_u16(&self, data: &[u16]) {
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.clear();
        // u16 is offset-binary — 32768 = silence.
        if self.channels <= 1 {
            for &s in data {
                scratch.push((s as f32 - 32768.0) / 32768.0);
            }
        } else {
            let ch = self.channels;
            for frame in data.chunks_exact(ch) {
                let sum: f32 = frame.iter().map(|&s| (s as f32 - 32768.0) / 32768.0).sum();
                scratch.push(sum / ch as f32);
            }
        }
        self.push_mono(&scratch);
    }

    fn push_mono(&self, mono: &[f32]) {
        let mut prod = self.producer.lock().unwrap_or_else(|e| e.into_inner());
        let pushed = prod.push_slice(mono);
        let dropped = mono.len().saturating_sub(pushed);
        if dropped > 0 {
            self.overrun.fetch_add(dropped as u64, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assistd_config::VoiceConfig;

    #[test]
    fn validate_ok_when_voice_disabled() {
        let cfg = VoiceConfig {
            enabled: false,
            mic_device: Some("never-checked".to_string()),
            ..VoiceConfig::default()
        };
        validate(&cfg).expect("disabled voice should skip mic validation");
    }

    #[test]
    fn validate_ok_when_no_device_name() {
        let cfg = VoiceConfig {
            enabled: true,
            mic_device: None,
            ..VoiceConfig::default()
        };
        validate(&cfg).expect("missing mic_device should accept the system default");
    }

    #[test]
    fn validate_errors_with_listing_for_unknown_device() {
        let cfg = VoiceConfig {
            enabled: true,
            mic_device: Some("assistd-test-definitely-missing-device".to_string()),
            ..VoiceConfig::default()
        };
        let err = validate(&cfg).expect_err("unknown device must not pass validation");
        let msg = format!("{err}");
        assert!(
            msg.contains("assistd-test-definitely-missing-device"),
            "error should echo the configured name: {msg}"
        );
        assert!(
            msg.contains("mic_device"),
            "error should mention mic_device: {msg}"
        );
        assert!(
            msg.contains("Set voice.mic_device = null"),
            "error should hint at the null default: {msg}"
        );
    }
}

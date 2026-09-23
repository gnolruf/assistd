//! cpal stream construction and the audio-thread callback.
//!
//! `cpal::Stream` is `!Send` on ALSA, so every stream in this crate is
//! opened, held, and dropped on one `spawn_blocking` worker; only
//! atomics and join handles cross threads.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Error as CpalError, SampleFormat, Stream, StreamConfig};
use ringbuf::HeapRb;
use ringbuf::traits::{Producer, Split};
use thiserror::Error;
use tokio::task::JoinHandle;
use tracing::warn;

use super::consumer;

/// Errors produced during cpal audio capture.
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

/// Errors from [`validate`].
#[derive(Debug, Error)]
pub enum DeviceValidationError {
    #[error(
        "failed to enumerate cpal input devices while validating voice.mic_device \
         = {requested:?}: {source}"
    )]
    Enumerate {
        requested: String,
        #[source]
        source: CpalError,
    },
    /// `listing` is the comma-separated, quoted names cpal reported.
    #[error(
        "voice.mic_device = {requested:?} not found. Available input devices: [{listing}]. \
         Set voice.mic_device = null to use the system default."
    )]
    NotFound { requested: String, listing: String },
}

/// Whisper's input rate; the consumer resamples the device rate to it.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Handles to a running push-to-talk capture.
pub struct CaptureSession {
    pub stop_flag: Arc<AtomicBool>,
    pub overrun: Arc<AtomicU64>,
    pub handle: JoinHandle<Result<Vec<i16>, AudioCaptureError>>,
}

/// A running input stream and the ring consumer it feeds. The caller
/// drops `stream` last, on the thread that built it.
pub struct ProducerStream {
    pub consumer: ringbuf::HeapCons<f32>,
    pub native_rate: u32,
    pub stream: Stream,
}

/// Open the input device and start a stream pushing mono f32 samples
/// at the native rate into a ring of `ring_capacity_samples`. The
/// callback bumps `overrun` for every sample dropped on a full ring.
pub fn open_producer_stream(
    device_hint: Option<&str>,
    ring_capacity_samples: usize,
    overrun: Arc<AtomicU64>,
) -> Result<ProducerStream, AudioCaptureError> {
    let host = cpal::default_host();
    let device = select_device(&host, device_hint)?;
    let device_name = device_name(&device).unwrap_or_else(|_| "<unknown>".to_string());

    let supported = device
        .default_input_config()
        .map_err(|e| AudioCaptureError::DefaultConfig(e.to_string()))?;
    let sample_rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();

    let ring = HeapRb::<f32>::new(ring_capacity_samples.max(sample_rate as usize));
    let (producer, consumer) = ring.split();

    let stream = build_stream(&device, &config, sample_format, channels, producer, overrun)?;
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
        consumer,
        native_rate: sample_rate,
        stream,
    })
}

/// Spawn a blocking worker that records until stopped and returns the
/// 16 kHz mono i16 PCM.
pub fn start(device_hint: Option<&str>, max_recording_secs: u32) -> CaptureSession {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let overrun = Arc::new(AtomicU64::new(0));
    let device_hint_owned = device_hint.map(|s| s.to_string());

    let worker_stop = Arc::clone(&stop_flag);
    let worker_overrun = Arc::clone(&overrun);
    let handle = tokio::task::spawn_blocking(move || {
        capture_ptt(
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

fn capture_ptt(
    device_hint: Option<&str>,
    max_recording_secs: u32,
    stop_flag: Arc<AtomicBool>,
    overrun: Arc<AtomicU64>,
) -> Result<Vec<i16>, AudioCaptureError> {
    // Sized before the native rate is known, so assume 48 kHz.
    let conservative_rate = 48_000usize;
    let ring_cap = conservative_rate
        .saturating_mul((max_recording_secs as usize).saturating_add(1))
        .max(conservative_rate * 2);
    let ProducerStream {
        consumer,
        native_rate,
        stream,
    } = open_producer_stream(device_hint, ring_cap, overrun)?;

    let max_pcm_samples = (TARGET_SAMPLE_RATE as usize).saturating_mul(max_recording_secs as usize);
    let pcm = consumer::drain_to_pcm(consumer, native_rate, max_pcm_samples, stop_flag)?;
    drop(stream);
    Ok(pcm)
}

/// Startup check that a configured `mic_device` exists. A missing
/// system default is not an error, so a headless host still starts;
/// only a named device that cannot be found is rejected. Always logs
/// the available input devices.
pub fn validate(cfg: &assistd_config::VoiceConfig) -> Result<(), DeviceValidationError> {
    if !cfg.enabled {
        return Ok(());
    }

    let host = cpal::default_host();
    let requested = cfg.mic_device.as_deref();
    let names: Vec<String> = match host.input_devices() {
        Ok(devices) => devices
            .map(|d| device_name(&d).unwrap_or_else(|_| "<unknown>".to_string()))
            .collect(),
        Err(source) => match requested {
            Some(requested) => {
                return Err(DeviceValidationError::Enumerate {
                    requested: requested.to_string(),
                    source,
                });
            }
            None => return Ok(()),
        },
    };
    let default_name = host
        .default_input_device()
        .and_then(|d| device_name(&d).ok())
        .unwrap_or_else(|| "<none>".to_string());
    tracing::info!(
        target: "assistd::voice::mic",
        default = %default_name,
        available = ?names,
        "cpal input devices"
    );

    let Some(requested) = requested else {
        return Ok(());
    };
    if names.iter().any(|n| n == requested) {
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
    Err(DeviceValidationError::NotFound {
        requested: requested.to_string(),
        listing,
    })
}

fn device_name(device: &Device) -> Result<String, CpalError> {
    device.description().map(|d| {
        d.driver()
            .map(str::to_string)
            .unwrap_or_else(|| d.name().to_string())
    })
}

/// The named input device, or the system default when `hint` is `None`.
pub fn select_device(host: &cpal::Host, hint: Option<&str>) -> Result<Device, AudioCaptureError> {
    match hint {
        None => host
            .default_input_device()
            .ok_or(AudioCaptureError::NoDefaultDevice),
        Some(name) => {
            let mut devices = host
                .input_devices()
                .map_err(|e| AudioCaptureError::DefaultConfig(e.to_string()))?;
            devices
                .find(|d| device_name(d).is_ok_and(|n| n == name))
                .ok_or_else(|| AudioCaptureError::NoMatchingDevice(name.to_string()))
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
    let state = CallbackState::new(producer, channels, overrun);
    match format {
        SampleFormat::F32 => build_input_stream(device, config, state, |s: f32| s),
        SampleFormat::I16 => build_input_stream(device, config, state, |s: i16| {
            f32::from(s) / f32::from(i16::MAX)
        }),
        // u16 is offset-binary: 32768 is silence.
        SampleFormat::U16 => build_input_stream(device, config, state, |s: u16| {
            (f32::from(s) - 32768.0) / 32768.0
        }),
        _ => Err(AudioCaptureError::UnsupportedFormat),
    }
}

fn build_input_stream<T: cpal::SizedSample>(
    device: &Device,
    config: &StreamConfig,
    state: CallbackState,
    to_f32: impl Fn(T) -> f32 + Send + 'static,
) -> Result<Stream, AudioCaptureError> {
    device
        .build_input_stream(
            *config,
            move |data: &[T], _| state.push(data, &to_f32),
            |e: CpalError| warn!(target: "assistd::voice::mic", "cpal stream error: {e}"),
            None,
        )
        .map_err(|e| AudioCaptureError::BuildStream(e.to_string()))
}

struct CallbackState {
    producer: parking_lot::Mutex<RbProd>,
    channels: usize,
    overrun: Arc<AtomicU64>,
    /// Pre-sized so the audio callback never allocates.
    scratch: parking_lot::Mutex<Vec<f32>>,
}

impl CallbackState {
    fn new(producer: RbProd, channels: usize, overrun: Arc<AtomicU64>) -> Self {
        Self {
            producer: parking_lot::Mutex::new(producer),
            channels,
            overrun,
            scratch: parking_lot::Mutex::new(Vec::with_capacity(8192)),
        }
    }

    /// Downmix `data` to mono f32 and push it onto the ring.
    fn push<T: Copy>(&self, data: &[T], to_f32: impl Fn(T) -> f32) {
        let mut scratch = self.scratch.lock();
        scratch.clear();
        if self.channels <= 1 {
            scratch.extend(data.iter().map(|&s| to_f32(s)));
        } else {
            let ch = self.channels;
            for frame in data.chunks_exact(ch) {
                let sum: f32 = frame.iter().map(|&s| to_f32(s)).sum();
                scratch.push(sum / ch as f32);
            }
        }
        self.push_mono(&scratch);
    }

    fn push_mono(&self, mono: &[f32]) {
        let mut prod = self.producer.lock();
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

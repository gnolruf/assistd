//! cpal stream construction and the audio-thread callback.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use assistd_config::VoiceConfig;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Error as CpalError, SampleFormat, Stream, StreamConfig};
use parking_lot::Mutex;
use ringbuf::traits::{Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use thiserror::Error;
use tokio::task::JoinHandle;
use tracing::warn;

use super::consumer;

/// Whisper's input rate; the consumer resamples the device rate to it.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Native rate assumed when sizing the push-to-talk ring before the device is opened.
const ASSUMED_NATIVE_RATE: usize = 48_000;

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

/// Handles to a running push-to-talk capture.
pub struct CaptureSession {
    pub stop_flag: Arc<AtomicBool>,
    pub overrun: Arc<AtomicU64>,
    pub handle: JoinHandle<Result<Vec<i16>, AudioCaptureError>>,
}

/// A running input stream and the ring consumer it feeds. `stream` is
/// `!Send` on ALSA, so drop it last, on the thread that built it.
pub struct ProducerStream {
    pub consumer: HeapCons<f32>,
    pub native_rate: u32,
    pub stream: Stream,
}

type RingProducer = HeapProd<f32>;

struct CallbackState {
    producer: Mutex<RingProducer>,
    channels: usize,
    overrun: Arc<AtomicU64>,
    /// Pre-sized so the audio callback never allocates.
    scratch: Mutex<Vec<f32>>,
}

impl CallbackState {
    fn new(producer: RingProducer, channels: usize, overrun: Arc<AtomicU64>) -> Self {
        Self {
            producer: Mutex::new(producer),
            channels,
            overrun,
            scratch: Mutex::new(Vec::with_capacity(8192)),
        }
    }

    /// Downmix `data` to mono f32 and push it onto the ring.
    fn push<T: Copy>(&self, data: &[T], to_f32: impl Fn(T) -> f32) {
        let mut scratch = self.scratch.lock();
        scratch.clear();
        if self.channels <= 1 {
            scratch.extend(data.iter().map(|&sample| to_f32(sample)));
        } else {
            let channels = self.channels;
            for frame in data.chunks_exact(channels) {
                let sum: f32 = frame.iter().map(|&sample| to_f32(sample)).sum();
                scratch.push(sum / channels as f32);
            }
        }
        self.push_mono(&scratch);
    }

    fn push_mono(&self, mono: &[f32]) {
        let mut producer = self.producer.lock();
        let pushed = producer.push_slice(mono);
        let dropped = mono.len().saturating_sub(pushed);
        if dropped > 0 {
            self.overrun.fetch_add(dropped as u64, Ordering::Relaxed);
        }
    }
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
        .map_err(|err| AudioCaptureError::DefaultConfig(err.to_string()))?;
    let sample_rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();

    let ring = HeapRb::<f32>::new(ring_capacity_samples.max(sample_rate as usize));
    let (producer, consumer) = ring.split();

    let stream = build_stream(&device, &config, sample_format, channels, producer, overrun)?;
    stream
        .play()
        .map_err(|err| AudioCaptureError::PlayStream(err.to_string()))?;

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
    let device_hint_owned = device_hint.map(str::to_string);

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
    let ring_capacity = ASSUMED_NATIVE_RATE
        .saturating_mul((max_recording_secs as usize).saturating_add(1))
        .max(ASSUMED_NATIVE_RATE * 2);
    let ProducerStream {
        consumer,
        native_rate,
        stream,
    } = open_producer_stream(device_hint, ring_capacity, overrun)?;

    let max_pcm_samples = (TARGET_SAMPLE_RATE as usize).saturating_mul(max_recording_secs as usize);
    let pcm = consumer::drain_to_pcm(consumer, native_rate, max_pcm_samples, stop_flag)?;
    drop(stream);
    Ok(pcm)
}

/// Startup check that a configured `mic_device` exists; logs the available
/// input devices. Only a named device that cannot be found or enumerated is rejected.
pub fn validate(config: &VoiceConfig) -> Result<(), DeviceValidationError> {
    if !config.enabled {
        return Ok(());
    }

    let host = cpal::default_host();
    let requested = config.mic_device.as_deref();
    let names: Vec<String> = match host.input_devices() {
        Ok(devices) => devices
            .map(|device| device_name(&device).unwrap_or_else(|_| "<unknown>".to_string()))
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
        .and_then(|device| device_name(&device).ok())
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
    if names.iter().any(|name| name == requested) {
        return Ok(());
    }
    Err(DeviceValidationError::NotFound {
        requested: requested.to_string(),
        listing: device_listing(&names),
    })
}

fn device_listing(names: &[String]) -> String {
    if names.is_empty() {
        return "<no input devices reported by cpal>".to_string();
    }
    names
        .iter()
        .map(|name| format!("{name:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn device_name(device: &Device) -> Result<String, CpalError> {
    device.description().map(|description| {
        description
            .driver()
            .map(str::to_string)
            .unwrap_or_else(|| description.name().to_string())
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
                .map_err(|err| AudioCaptureError::DefaultConfig(err.to_string()))?;
            devices
                .find(|device| device_name(device).is_ok_and(|candidate| candidate == name))
                .ok_or_else(|| AudioCaptureError::NoMatchingDevice(name.to_string()))
        }
    }
}

fn build_stream(
    device: &Device,
    config: &StreamConfig,
    format: SampleFormat,
    channels: usize,
    producer: RingProducer,
    overrun: Arc<AtomicU64>,
) -> Result<Stream, AudioCaptureError> {
    let state = CallbackState::new(producer, channels, overrun);
    match format {
        SampleFormat::F32 => build_input_stream(device, config, state, |sample: f32| sample),
        SampleFormat::I16 => build_input_stream(device, config, state, i16_to_f32),
        SampleFormat::U16 => build_input_stream(device, config, state, offset_binary_u16_to_f32),
        _ => Err(AudioCaptureError::UnsupportedFormat),
    }
}

fn i16_to_f32(sample: i16) -> f32 {
    f32::from(sample) / f32::from(i16::MAX)
}

/// u16 PCM is offset-binary: 32768 is silence.
fn offset_binary_u16_to_f32(sample: u16) -> f32 {
    (f32::from(sample) - 32768.0) / 32768.0
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
            |err: CpalError| warn!(target: "assistd::voice::mic", "cpal stream error: {err}"),
            None,
        )
        .map_err(|err| AudioCaptureError::BuildStream(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_ok_when_voice_disabled() {
        let config = VoiceConfig {
            enabled: false,
            mic_device: Some("never-checked".to_string()),
            ..VoiceConfig::default()
        };
        validate(&config).expect("disabled voice should skip mic validation");
    }

    #[test]
    fn validate_ok_when_no_device_name() {
        let config = VoiceConfig {
            enabled: true,
            mic_device: None,
            ..VoiceConfig::default()
        };
        validate(&config).expect("missing mic_device should accept the system default");
    }

    #[test]
    fn validate_errors_with_listing_for_unknown_device() {
        let config = VoiceConfig {
            enabled: true,
            mic_device: Some("assistd-test-definitely-missing-device".to_string()),
            ..VoiceConfig::default()
        };
        let err = validate(&config).expect_err("unknown device must not pass validation");
        assert!(
            matches!(&err, DeviceValidationError::NotFound { requested, .. }
                if requested == "assistd-test-definitely-missing-device"),
            "got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("Set voice.mic_device = null"),
            "error should hint at the null default: {msg}"
        );
    }
}

//! cpal stream construction and the audio-thread callback.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Error as CpalError, ErrorKind, SampleFormat, Stream, StreamConfig};
use parking_lot::Mutex;
use ringbuf::traits::{Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use thiserror::Error;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::warn;

use super::consumer::{self, ChunkSink};
use crate::device::device_name;

/// The transcriber's input rate; the consumer resamples the device rate to it.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Seconds of native audio the ring holds while the consumer catches up.
const RING_SECS: usize = 5;

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
pub(crate) enum DeviceValidationError {
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

/// Handles to a running capture that ends with a `T`. `opened` resolves once
/// the device is recording and errors if it failed to open.
#[derive(Debug)]
pub struct CaptureSession<T> {
    pub stop_flag: Arc<AtomicBool>,
    pub opened: oneshot::Receiver<()>,
    pub handle: JoinHandle<Result<T, AudioCaptureError>>,
}

/// A running input stream and the ring consumer it feeds. `stream` is
/// `!Send` on ALSA, so drop it last, on the thread that built it.
pub struct ProducerStream {
    pub consumer: HeapCons<f32>,
    pub native_rate: u32,
    pub failure: StreamFailure,
    /// Samples the callback dropped on a full ring.
    pub overrun: Arc<AtomicU64>,
    pub stream: Stream,
}

impl fmt::Debug for ProducerStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProducerStream")
            .field("native_rate", &self.native_rate)
            .finish_non_exhaustive()
    }
}

/// Why an input stream stopped delivering audio, shared between cpal's
/// error callback and the consumer draining the stream's ring.
#[derive(Debug, Clone, Default)]
pub struct StreamFailure(Arc<OnceLock<String>>);

impl StreamFailure {
    /// Log `error`, keeping it as the failure when it ends the stream.
    fn observe(&self, error: &CpalError) {
        warn!(target: "assistd::voice::mic", "cpal stream error: {error}");
        if ends_stream(error.kind()) {
            let _ = self.0.set(error.to_string());
        }
    }

    /// [`AudioCaptureError::DeviceError`] once the stream has failed.
    pub fn check(&self) -> Result<(), AudioCaptureError> {
        match self.0.get() {
            Some(reason) => Err(AudioCaptureError::DeviceError(reason.clone())),
            None => Ok(()),
        }
    }
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
/// at the native rate into a ring.
pub fn open_producer_stream(
    device_hint: Option<&str>,
) -> Result<ProducerStream, AudioCaptureError> {
    let host = cpal::default_host();
    let device = select_device(&host, device_hint)?;
    let device_name = device_name(&device);

    let supported = device
        .default_input_config()
        .map_err(|err| AudioCaptureError::DefaultConfig(err.to_string()))?;
    let sample_rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();

    let ring_capacity = sample_rate as usize * RING_SECS;
    let (producer, consumer) = HeapRb::<f32>::new(ring_capacity).split();

    let overrun = Arc::new(AtomicU64::new(0));
    let state = CallbackState::new(producer, channels, Arc::clone(&overrun));
    let failure = StreamFailure::default();
    let stream = build_stream(&device, &config, sample_format, state, failure.clone())?;
    stream
        .play()
        .map_err(|err| AudioCaptureError::PlayStream(err.to_string()))?;

    tracing::info!(
        target: "assistd::voice::mic",
        device = %device_name,
        sample_rate,
        channels,
        format = ?sample_format,
        ring_capacity,
        "audio stream started"
    );

    Ok(ProducerStream {
        consumer,
        native_rate: sample_rate,
        failure,
        overrun,
        stream,
    })
}

/// Spawn a blocking worker that feeds `sink` 16 kHz mono audio from the
/// device until the stop flag is set or the sink ends the capture.
pub fn start<S: ChunkSink>(device_hint: Option<&str>, sink: S) -> CaptureSession<S::Captured> {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let device_hint_owned = device_hint.map(str::to_string);

    let (opened_tx, opened) = oneshot::channel();
    let worker_stop = Arc::clone(&stop_flag);
    let handle = tokio::task::spawn_blocking(move || {
        capture_into(device_hint_owned.as_deref(), &worker_stop, sink, opened_tx)
    });

    CaptureSession {
        stop_flag,
        opened,
        handle,
    }
}

fn capture_into<S: ChunkSink>(
    device_hint: Option<&str>,
    stop_flag: &AtomicBool,
    sink: S,
    opened: oneshot::Sender<()>,
) -> Result<S::Captured, AudioCaptureError> {
    let ProducerStream {
        consumer,
        native_rate,
        failure,
        overrun,
        stream,
    } = open_producer_stream(device_hint)?;
    let _ = opened.send(());

    let captured = consumer::drain(consumer, native_rate, stop_flag, &failure, sink);

    let overrun_samples = overrun.load(Ordering::Relaxed);
    if overrun_samples > 0 {
        warn!(
            target: "assistd::voice::mic",
            overrun_samples,
            "ring buffer overrun during capture; audio was truncated"
        );
    }

    drop(stream);
    captured
}

/// Check that the named input device exists, logging the available ones when
/// it does. `None` (the system default) always passes; a named device that
/// cannot be found or enumerated is rejected.
pub(crate) fn validate(requested: Option<&str>) -> Result<(), DeviceValidationError> {
    let host = cpal::default_host();
    let names: Vec<String> = match host.input_devices() {
        Ok(devices) => devices.map(|device| device_name(&device)).collect(),
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
    if let Some(requested) = requested
        && !names.iter().any(|name| name == requested)
    {
        return Err(DeviceValidationError::NotFound {
            requested: requested.to_string(),
            listing: device_listing(&names),
        });
    }
    log_devices(&host, &names);
    Ok(())
}

fn log_devices(host: &cpal::Host, names: &[String]) {
    let default_name = host
        .default_input_device()
        .map_or_else(|| "<none>".to_string(), |device| device_name(&device));
    tracing::info!(
        target: "assistd::voice::mic",
        default = %default_name,
        available = ?names,
        "cpal input devices"
    );
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
                .find(|device| device_name(device) == name)
                .ok_or_else(|| AudioCaptureError::NoMatchingDevice(name.to_string()))
        }
    }
}

fn build_stream(
    device: &Device,
    config: &StreamConfig,
    format: SampleFormat,
    state: CallbackState,
    failure: StreamFailure,
) -> Result<Stream, AudioCaptureError> {
    match format {
        SampleFormat::F32 => {
            build_input_stream(device, config, state, failure, |sample: f32| sample)
        }
        SampleFormat::I16 => build_input_stream(device, config, state, failure, i16_to_f32),
        SampleFormat::U16 => {
            build_input_stream(device, config, state, failure, offset_binary_u16_to_f32)
        }
        _ => Err(AudioCaptureError::UnsupportedFormat),
    }
}

/// Whether cpal delivers no more audio after reporting an error of `kind`.
fn ends_stream(kind: ErrorKind) -> bool {
    matches!(
        kind,
        ErrorKind::DeviceNotAvailable | ErrorKind::StreamInvalidated
    )
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
    failure: StreamFailure,
    to_f32: impl Fn(T) -> f32 + Send + 'static,
) -> Result<Stream, AudioCaptureError> {
    device
        .build_input_stream(
            *config,
            move |data: &[T], _| state.push(data, &to_f32),
            move |error: CpalError| failure.observe(&error),
            None,
        )
        .map_err(|err| AudioCaptureError::BuildStream(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::consumer::PcmRecording;
    use super::*;

    #[test]
    fn a_vanished_device_fails_the_recording_and_an_xrun_does_not() {
        let released = AtomicBool::new(true);
        let failure = StreamFailure::default();
        let drain = || {
            let (_producer, consumer) = HeapRb::<f32>::new(16).split();
            consumer::drain(
                consumer,
                TARGET_SAMPLE_RATE,
                &released,
                &failure,
                PcmRecording::capped_at(16),
            )
        };

        failure.observe(&CpalError::new(ErrorKind::Xrun));
        assert!(matches!(drain(), Ok(pcm) if pcm.is_empty()));

        failure.observe(&CpalError::new(ErrorKind::DeviceNotAvailable));
        assert!(matches!(drain(), Err(AudioCaptureError::DeviceError(_))));
    }
}

//! Blocking capture worker that streams 20 ms 16 kHz i16 frames to
//! the VAD task until signalled.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use super::consumer::drain_to_frames;
use super::vad::FRAME_SAMPLES;
use crate::mic::capture::{AudioCaptureError, ProducerStream, open_producer_stream};

const LISTEN_RING_SECONDS: usize = 5;
const LISTEN_RING_NATIVE_RATE_ASSUMED: usize = 48_000;

/// Handles to a running continuous capture.
pub struct ListenCaptureSession {
    pub stop_flag: Arc<AtomicBool>,
    pub overrun: Arc<AtomicU64>,
    pub handle: JoinHandle<Result<(), AudioCaptureError>>,
}

/// Spawn a blocking worker streaming 20 ms frames into `frame_tx`
/// until `stop_flag` is set or the receiver is dropped.
pub fn start(
    device_hint: Option<&str>,
    frame_tx: mpsc::Sender<Box<[i16; FRAME_SAMPLES]>>,
) -> ListenCaptureSession {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let overrun = Arc::new(AtomicU64::new(0));
    let device_hint_owned = device_hint.map(str::to_string);

    let worker_stop = Arc::clone(&stop_flag);
    let worker_overrun = Arc::clone(&overrun);
    let handle = tokio::task::spawn_blocking(move || {
        capture_continuous(
            device_hint_owned.as_deref(),
            worker_stop,
            worker_overrun,
            frame_tx,
        )
    });

    ListenCaptureSession {
        stop_flag,
        overrun,
        handle,
    }
}

fn capture_continuous(
    device_hint: Option<&str>,
    stop_flag: Arc<AtomicBool>,
    overrun: Arc<AtomicU64>,
    frame_tx: mpsc::Sender<Box<[i16; FRAME_SAMPLES]>>,
) -> Result<(), AudioCaptureError> {
    let ring_capacity = LISTEN_RING_SECONDS.saturating_mul(LISTEN_RING_NATIVE_RATE_ASSUMED);
    let ProducerStream {
        consumer,
        native_rate,
        stream,
    } = open_producer_stream(device_hint, ring_capacity, overrun.clone())?;

    debug!(
        target: "assistd::voice::listen",
        native_rate,
        "listen capture started"
    );

    let result = drain_to_frames(consumer, native_rate, stop_flag, frame_tx);

    let total_overrun = overrun.load(Ordering::Relaxed);
    if total_overrun > 0 {
        warn!(
            target: "assistd::voice::listen",
            overrun_samples = total_overrun,
            "ring buffer overrun during continuous listening; audio was truncated"
        );
    }

    drop(stream);
    result
}

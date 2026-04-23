//! Blocking worker that opens cpal, streams 16 kHz mono 20 ms i16
//! frames to the VAD task, and closes the device when signalled.
//!
//! This is the continuous-listen counterpart to
//! [`crate::mic::capture::capture_worker`]. They both use the shared
//! [`open_producer_stream`] helper to get a cpal `Stream` + ring
//! consumer; what differs is the drain strategy — PTT accumulates
//! into one `Vec<i16>`, this worker streams frame-sized chunks.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use super::consumer::stream_frames;
use super::vad::FRAME_SAMPLES;
use crate::mic::capture::{AudioCaptureError, ProducerStream, open_producer_stream};

/// Ring capacity in mono native-rate samples. Continuous listening
/// only needs enough headroom for the VAD task to catch up during a
/// scheduling hiccup — ~5 s at 48 kHz is plenty and small enough to
/// avoid ballooning memory.
const LISTEN_RING_SECONDS: usize = 5;
const LISTEN_RING_NATIVE_RATE_ASSUMED: usize = 48_000;

/// Handles for the running capture session. Dropping these (or
/// setting `stop_flag`) brings the stream down.
pub struct ListenCaptureSession {
    pub stop_flag: Arc<AtomicBool>,
    pub overrun: Arc<AtomicU64>,
    pub handle: JoinHandle<Result<(), AudioCaptureError>>,
}

/// Start a continuous capture: opens cpal, begins streaming 20 ms
/// frames into `frame_tx`. The returned `handle` resolves when the
/// worker exits (stop_flag set, or `frame_tx` dropped).
pub fn start(
    device_hint: Option<&str>,
    frame_tx: mpsc::Sender<Box<[i16; FRAME_SAMPLES]>>,
) -> ListenCaptureSession {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let overrun = Arc::new(AtomicU64::new(0));
    let device_hint_owned = device_hint.map(|s| s.to_string());

    let worker_stop = Arc::clone(&stop_flag);
    let worker_overrun = Arc::clone(&overrun);
    let handle = tokio::task::spawn_blocking(move || {
        listen_worker(
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

fn listen_worker(
    device_hint: Option<&str>,
    stop_flag: Arc<AtomicBool>,
    overrun: Arc<AtomicU64>,
    frame_tx: mpsc::Sender<Box<[i16; FRAME_SAMPLES]>>,
) -> Result<(), AudioCaptureError> {
    let ring_cap = LISTEN_RING_SECONDS.saturating_mul(LISTEN_RING_NATIVE_RATE_ASSUMED);
    let ProducerStream {
        consumer,
        native_rate,
        stream,
    } = open_producer_stream(device_hint, ring_cap, overrun.clone())?;

    debug!(
        target: "assistd::voice::listen",
        native_rate,
        "listen capture started"
    );

    let result = stream_frames(consumer, native_rate, stop_flag, frame_tx);

    let total_overrun = overrun.load(Ordering::Relaxed);
    if total_overrun > 0 {
        warn!(
            target: "assistd::voice::listen",
            overrun_samples = total_overrun,
            "ring buffer overrun during continuous listening — audio was truncated"
        );
    }

    // Drop cpal stream on this same thread to avoid `!Send` drop issues.
    drop(stream);
    result
}

//! Streaming consumer for continuous listening.
//!
//! Drains the SPSC ring buffer continuously, resamples from the
//! device's native rate to 16 kHz via `rubato::FastFixedIn`, and
//! emits fixed-size 20 ms i16 frames over a tokio mpsc channel. The
//! VAD task reads from the other end one frame at a time.
//!
//! Runs on a `tokio::task::spawn_blocking` worker because the cpal
//! stream it cooperates with is `!Send` on ALSA.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ringbuf::HeapCons;
use ringbuf::traits::{Consumer, Observer};
use rubato::{FastFixedIn, PolynomialDegree, Resampler};
use tokio::sync::mpsc;
use tracing::debug;

use super::vad::FRAME_SAMPLES;
use crate::mic::capture::{AudioCaptureError, TARGET_SAMPLE_RATE};
use crate::mic::consumer::f32_to_i16;

/// Chunk size (native-rate samples) pulled from the ring before each
/// resampler step. Matches the PTT pipeline — 1024 samples ~= 21 ms
/// at 48 kHz. Tuned to amortize resampler overhead without overshooting
/// frame boundaries badly.
const DRAIN_CHUNK_SIZE: usize = 1024;

/// How long to park when the ring is empty and we haven't been asked
/// to stop. Short enough to keep latency to whisper-ready under
/// ~20 ms of ring drain.
const IDLE_PARK: Duration = Duration::from_millis(10);

/// Pull samples out of the ring, resample to 16 kHz, convert to i16,
/// and emit 20-ms frames on `frame_tx` until `stop_flag` flips. A
/// dropped receiver is treated as a silent stop.
pub fn stream_frames(
    mut consumer: HeapCons<f32>,
    native_rate: u32,
    stop_flag: Arc<AtomicBool>,
    frame_tx: mpsc::Sender<Box<[i16; FRAME_SAMPLES]>>,
) -> Result<(), AudioCaptureError> {
    let needs_resample = native_rate != TARGET_SAMPLE_RATE;

    let mut resampler: Option<FastFixedIn<f32>> = if needs_resample {
        let ratio = TARGET_SAMPLE_RATE as f64 / native_rate as f64;
        Some(
            FastFixedIn::<f32>::new(ratio, 1.0, PolynomialDegree::Linear, DRAIN_CHUNK_SIZE, 1)
                .map_err(|e| AudioCaptureError::BuildStream(format!("rubato init: {e}")))?,
        )
    } else {
        None
    };

    let mut in_buf = vec![0.0f32; DRAIN_CHUNK_SIZE];
    let out_cap = resampler
        .as_ref()
        .map(|r| r.output_frames_max())
        .unwrap_or(DRAIN_CHUNK_SIZE);
    let mut out_buf = vec![vec![0.0f32; out_cap]; 1];

    // Rolling pending buffer — accumulates 16 kHz i16 samples until
    // we have at least FRAME_SAMPLES (320) to emit a 20 ms frame.
    let mut pending: Vec<i16> = Vec::with_capacity(FRAME_SAMPLES * 2);

    let mut frames_emitted: u64 = 0;

    loop {
        let available = consumer.occupied_len();
        if available >= DRAIN_CHUNK_SIZE {
            let got = consumer.pop_slice(&mut in_buf);
            if got < DRAIN_CHUNK_SIZE {
                for s in in_buf.iter_mut().skip(got) {
                    *s = 0.0;
                }
            }

            if let Some(r) = &mut resampler {
                let (_read, written) = r
                    .process_into_buffer(&[&in_buf[..]], &mut out_buf, None)
                    .map_err(|e| AudioCaptureError::DeviceError(format!("rubato process: {e}")))?;
                for &s in &out_buf[0][..written] {
                    pending.push(f32_to_i16(s));
                }
            } else {
                for &s in &in_buf[..] {
                    pending.push(f32_to_i16(s));
                }
            }

            while pending.len() >= FRAME_SAMPLES {
                let mut frame = Box::new([0i16; FRAME_SAMPLES]);
                frame.copy_from_slice(&pending[..FRAME_SAMPLES]);
                pending.drain(..FRAME_SAMPLES);
                match frame_tx.blocking_send(frame) {
                    Ok(()) => frames_emitted += 1,
                    Err(_) => {
                        debug!(
                            target: "assistd::voice::listen",
                            frames_emitted,
                            "frame receiver dropped; stopping capture"
                        );
                        return Ok(());
                    }
                }
            }
            continue;
        }

        if stop_flag.load(Ordering::Relaxed) {
            debug!(
                target: "assistd::voice::listen",
                frames_emitted,
                "stop_flag observed; ending stream"
            );
            return Ok(());
        }

        std::thread::park_timeout(IDLE_PARK);
    }
}

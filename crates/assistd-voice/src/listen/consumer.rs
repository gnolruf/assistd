//! Continuous-listen ring consumer: drains native-rate f32 samples,
//! resamples to 16 kHz, and emits 20 ms i16 frames over a channel.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ringbuf::HeapCons;
use ringbuf::traits::{Consumer, Observer};
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Async, FixedAsync, PolynomialDegree, Resampler};
use tokio::sync::mpsc;
use tracing::debug;

use super::vad::FRAME_SAMPLES;
use crate::mic::capture::{AudioCaptureError, TARGET_SAMPLE_RATE};
use crate::mic::consumer::f32_to_i16;

const DRAIN_CHUNK_SIZE: usize = 1024;
const IDLE_PARK: Duration = Duration::from_millis(10);

/// Emit 20 ms frames on `frame_tx` until `stop_flag` is set. A dropped
/// receiver is a silent stop.
pub fn stream_frames(
    mut consumer: HeapCons<f32>,
    native_rate: u32,
    stop_flag: Arc<AtomicBool>,
    frame_tx: mpsc::Sender<Box<[i16; FRAME_SAMPLES]>>,
) -> Result<(), AudioCaptureError> {
    let needs_resample = native_rate != TARGET_SAMPLE_RATE;

    let mut resampler: Option<Async<f32>> = if needs_resample {
        let ratio = TARGET_SAMPLE_RATE as f64 / native_rate as f64;
        Some(
            Async::<f32>::new_poly(
                ratio,
                1.0,
                PolynomialDegree::Linear,
                DRAIN_CHUNK_SIZE,
                1,
                FixedAsync::Input,
            )
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
    let mut out_buf = vec![0.0f32; out_cap];

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
                let input_adapter =
                    SequentialSlice::new(&in_buf[..], 1, in_buf.len()).map_err(|e| {
                        AudioCaptureError::DeviceError(format!("rubato input adapter: {e}"))
                    })?;
                let out_buf_len = out_buf.len();
                let mut output_adapter = SequentialSlice::new_mut(&mut out_buf[..], 1, out_buf_len)
                    .map_err(|e| {
                        AudioCaptureError::DeviceError(format!("rubato output adapter: {e}"))
                    })?;
                let (_read, written) = r
                    .process_into_buffer(&input_adapter, &mut output_adapter, None)
                    .map_err(|e| AudioCaptureError::DeviceError(format!("rubato process: {e}")))?;
                for &s in &out_buf[..written] {
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

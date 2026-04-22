//! Consumer side of the PTT pipeline.
//!
//! Runs on a `tokio::task::spawn_blocking` worker. Drains raw mono
//! f32 samples out of the SPSC ring at the native device sample rate,
//! resamples to 16 kHz via `rubato::FastFixedIn`, converts to i16,
//! and returns the accumulated PCM when stopped.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ringbuf::HeapCons;
use ringbuf::traits::{Consumer, Observer};
use rubato::{FastFixedIn, PolynomialDegree, Resampler};
use tracing::debug;

use super::capture::{AudioCaptureError, TARGET_SAMPLE_RATE};

/// How many native-rate samples the consumer pulls from the ring
/// before handing a chunk off to the resampler. Smaller = lower
/// drain latency; larger = fewer resampler calls. 1024 samples at 48
/// kHz is ~21 ms, which keeps the ring shallow without excessive
/// resampler overhead.
const DRAIN_CHUNK_SIZE: usize = 1024;

/// How long to park when the ring is empty and we haven't been
/// asked to stop. Much shorter than whisper inference latency; a
/// smaller value would burn CPU.
const IDLE_PARK: Duration = Duration::from_millis(10);

pub fn drain_loop(
    mut consumer: HeapCons<f32>,
    native_rate: u32,
    max_pcm_samples: usize,
    stop_flag: Arc<AtomicBool>,
) -> Result<Vec<i16>, AudioCaptureError> {
    // Short-circuit when native rate already equals the whisper
    // target: skip rubato entirely. Some USB mics and headsets do
    // offer 16 kHz directly.
    let needs_resample = native_rate != TARGET_SAMPLE_RATE;

    let mut pcm: Vec<i16> = Vec::with_capacity(max_pcm_samples.min(16_000 * 10));

    if !needs_resample {
        drain_no_resample(&mut consumer, &mut pcm, max_pcm_samples, &stop_flag);
        return Ok(pcm);
    }

    // rubato's FastFixedIn wants a fixed input chunk size per call.
    // We pad the last partial chunk with zeros on shutdown so the
    // very end of the utterance isn't dropped.
    let ratio = TARGET_SAMPLE_RATE as f64 / native_rate as f64;
    let mut resampler = FastFixedIn::<f32>::new(
        ratio,
        1.0, // fixed ratio — no runtime modulation
        PolynomialDegree::Linear,
        DRAIN_CHUNK_SIZE,
        1, // mono
    )
    .map_err(|e| AudioCaptureError::BuildStream(format!("rubato init: {e}")))?;

    let mut in_buf = vec![0.0f32; DRAIN_CHUNK_SIZE];
    let mut out_buf = vec![vec![0.0f32; resampler.output_frames_max()]; 1];

    loop {
        let available = consumer.occupied_len();
        if available >= DRAIN_CHUNK_SIZE {
            let got = consumer.pop_slice(&mut in_buf);
            if got < DRAIN_CHUNK_SIZE {
                // Partial pop despite `occupied_len >= chunk`: race
                // with producer drop on teardown — pad and finish.
                for s in in_buf.iter_mut().skip(got) {
                    *s = 0.0;
                }
            }
            resample_and_append(
                &mut resampler,
                &in_buf,
                &mut out_buf,
                &mut pcm,
                max_pcm_samples,
            )?;
            if pcm.len() >= max_pcm_samples {
                break;
            }
            continue;
        }

        if stop_flag.load(Ordering::Relaxed) {
            // Drain whatever fractional chunk is left, padded with
            // zeros to meet the fixed input-chunk requirement.
            let got = consumer.pop_slice(&mut in_buf);
            if got > 0 {
                for s in in_buf.iter_mut().skip(got) {
                    *s = 0.0;
                }
                resample_and_append(
                    &mut resampler,
                    &in_buf,
                    &mut out_buf,
                    &mut pcm,
                    max_pcm_samples,
                )?;
            }
            break;
        }

        std::thread::park_timeout(IDLE_PARK);
    }

    debug!(
        target: "assistd::voice::mic",
        pcm_samples = pcm.len(),
        "consumer drained"
    );
    Ok(pcm)
}

fn drain_no_resample(
    consumer: &mut HeapCons<f32>,
    pcm: &mut Vec<i16>,
    max_pcm_samples: usize,
    stop_flag: &AtomicBool,
) {
    let mut scratch = vec![0.0f32; DRAIN_CHUNK_SIZE];
    loop {
        let got = consumer.pop_slice(&mut scratch);
        if got > 0 {
            for &s in &scratch[..got] {
                if pcm.len() >= max_pcm_samples {
                    return;
                }
                pcm.push(f32_to_i16(s));
            }
        } else if stop_flag.load(Ordering::Relaxed) {
            return;
        } else {
            std::thread::park_timeout(IDLE_PARK);
        }
    }
}

fn resample_and_append(
    resampler: &mut FastFixedIn<f32>,
    input: &[f32],
    output: &mut [Vec<f32>],
    pcm: &mut Vec<i16>,
    max_pcm_samples: usize,
) -> Result<(), AudioCaptureError> {
    let input_channels: [&[f32]; 1] = [input];
    let (_read, written) = resampler
        .process_into_buffer(&input_channels, output, None)
        .map_err(|e| AudioCaptureError::DeviceError(format!("rubato process: {e}")))?;

    for &s in &output[0][..written] {
        if pcm.len() >= max_pcm_samples {
            return Ok(());
        }
        pcm.push(f32_to_i16(s));
    }
    Ok(())
}

#[inline]
fn f32_to_i16(s: f32) -> i16 {
    // Clamp to avoid wrap on occasional out-of-range float noise.
    let clamped = s.clamp(-1.0, 1.0);
    (clamped * i16::MAX as f32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_to_i16_saturates_cleanly() {
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(1.0), i16::MAX);
        assert_eq!(f32_to_i16(-1.0), -i16::MAX);
        // Out-of-range input clamps, doesn't wrap.
        assert_eq!(f32_to_i16(2.0), i16::MAX);
        assert_eq!(f32_to_i16(-2.0), -i16::MAX);
    }
}

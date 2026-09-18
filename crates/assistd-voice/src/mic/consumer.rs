//! Push-to-talk ring consumer: drains native-rate f32 samples,
//! resamples to 16 kHz, and accumulates i16 PCM until stopped.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ringbuf::HeapCons;
use ringbuf::traits::{Consumer, Observer};
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Async, FixedAsync, PolynomialDegree, Resampler};
use tracing::debug;

use super::capture::{AudioCaptureError, TARGET_SAMPLE_RATE};

/// Native-rate samples per resampler call; ~21 ms at 48 kHz.
const DRAIN_CHUNK_SIZE: usize = 1024;

const IDLE_PARK: Duration = Duration::from_millis(10);

/// Drain the ring until `stop_flag` is set or `max_pcm_samples` is
/// reached, returning 16 kHz i16 PCM. The final partial chunk is
/// zero-padded so the tail of the utterance is kept.
pub fn drain_loop(
    mut consumer: HeapCons<f32>,
    native_rate: u32,
    max_pcm_samples: usize,
    stop_flag: Arc<AtomicBool>,
) -> Result<Vec<i16>, AudioCaptureError> {
    let needs_resample = native_rate != TARGET_SAMPLE_RATE;

    let mut pcm: Vec<i16> = Vec::with_capacity(max_pcm_samples.min(16_000 * 10));

    if !needs_resample {
        drain_no_resample(&mut consumer, &mut pcm, max_pcm_samples, &stop_flag);
        return Ok(pcm);
    }

    let ratio = TARGET_SAMPLE_RATE as f64 / native_rate as f64;
    let mut resampler = Async::<f32>::new_poly(
        ratio,
        1.0,
        PolynomialDegree::Linear,
        DRAIN_CHUNK_SIZE,
        1,
        FixedAsync::Input,
    )
    .map_err(|e| AudioCaptureError::BuildStream(format!("rubato init: {e}")))?;

    let mut in_buf = vec![0.0f32; DRAIN_CHUNK_SIZE];
    let mut out_buf = vec![0.0f32; resampler.output_frames_max()];

    loop {
        let available = consumer.occupied_len();
        if available >= DRAIN_CHUNK_SIZE {
            let got = consumer.pop_slice(&mut in_buf);
            if got < DRAIN_CHUNK_SIZE {
                // The producer was dropped mid-pop on teardown.
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
    resampler: &mut Async<f32>,
    input: &[f32],
    output: &mut [f32],
    pcm: &mut Vec<i16>,
    max_pcm_samples: usize,
) -> Result<(), AudioCaptureError> {
    let input_adapter = SequentialSlice::new(input, 1, input.len())
        .map_err(|e| AudioCaptureError::DeviceError(format!("rubato input adapter: {e}")))?;
    let output_len = output.len();
    let mut output_adapter = SequentialSlice::new_mut(output, 1, output_len)
        .map_err(|e| AudioCaptureError::DeviceError(format!("rubato output adapter: {e}")))?;
    let (_read, written) = resampler
        .process_into_buffer(&input_adapter, &mut output_adapter, None)
        .map_err(|e| AudioCaptureError::DeviceError(format!("rubato process: {e}")))?;

    for &s in &output[..written] {
        if pcm.len() >= max_pcm_samples {
            return Ok(());
        }
        pcm.push(f32_to_i16(s));
    }
    Ok(())
}

#[inline]
pub(crate) fn f32_to_i16(s: f32) -> i16 {
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
        assert_eq!(f32_to_i16(2.0), i16::MAX);
        assert_eq!(f32_to_i16(-2.0), -i16::MAX);
    }
}

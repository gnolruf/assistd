//! Chunked native-rate to 16 kHz resampling shared by the push-to-talk
//! and continuous-listen consumers.

use ringbuf::HeapCons;
use ringbuf::traits::Consumer;
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Async, FixedAsync, PolynomialDegree, Resampler};

use super::capture::{AudioCaptureError, TARGET_SAMPLE_RATE};

/// Native-rate samples per resampler call; ~21 ms at 48 kHz.
pub(crate) const CHUNK_SIZE: usize = 1024;

/// Pulls fixed-size chunks from a ring and resamples them to 16 kHz.
/// A device already at 16 kHz passes samples through untouched.
pub(crate) struct ChunkResampler {
    resampler: Option<Async<f32>>,
    input: Vec<f32>,
    output: Vec<f32>,
}

impl ChunkResampler {
    pub fn new(native_rate: u32) -> Result<Self, AudioCaptureError> {
        let resampler = if native_rate == TARGET_SAMPLE_RATE {
            None
        } else {
            let ratio = f64::from(TARGET_SAMPLE_RATE) / f64::from(native_rate);
            Some(
                Async::<f32>::new_poly(
                    ratio,
                    1.0,
                    PolynomialDegree::Linear,
                    CHUNK_SIZE,
                    1,
                    FixedAsync::Input,
                )
                .map_err(|e| AudioCaptureError::BuildStream(format!("rubato init: {e}")))?,
            )
        };
        let output_len = resampler
            .as_ref()
            .map_or(CHUNK_SIZE, Resampler::output_frames_max);
        Ok(Self {
            resampler,
            input: vec![0.0; CHUNK_SIZE],
            output: vec![0.0; output_len],
        })
    }

    /// Pull up to one chunk from `consumer` and return it at 16 kHz.
    /// `None` when the ring is empty. A partial chunk is zero-padded
    /// to the resampler's fixed input size, so the tail of an
    /// utterance is kept.
    pub fn pull(
        &mut self,
        consumer: &mut HeapCons<f32>,
    ) -> Result<Option<&[f32]>, AudioCaptureError> {
        let got = consumer.pop_slice(&mut self.input);
        if got == 0 {
            return Ok(None);
        }
        let Some(resampler) = &mut self.resampler else {
            return Ok(Some(&self.input[..got]));
        };
        self.input[got..].fill(0.0);

        let input = SequentialSlice::new(&self.input[..], 1, CHUNK_SIZE)
            .map_err(|e| AudioCaptureError::DeviceError(format!("rubato input adapter: {e}")))?;
        let output_len = self.output.len();
        let mut output = SequentialSlice::new_mut(&mut self.output[..], 1, output_len)
            .map_err(|e| AudioCaptureError::DeviceError(format!("rubato output adapter: {e}")))?;
        let (_, written) = resampler
            .process_into_buffer(&input, &mut output, None)
            .map_err(|e| AudioCaptureError::DeviceError(format!("rubato process: {e}")))?;
        Ok(Some(&self.output[..written]))
    }
}

#[inline]
pub(crate) fn f32_to_i16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16
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

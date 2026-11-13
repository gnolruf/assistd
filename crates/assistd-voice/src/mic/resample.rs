//! Chunked native-rate to 16 kHz resampling shared by the push-to-talk
//! and continuous-listen consumers.

use num_traits::ToPrimitive;
use ringbuf::HeapCons;
use ringbuf::traits::Consumer;
use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::{Fft, FixedSync, Resampler, WindowFunction};

use super::capture::{AudioCaptureError, TARGET_SAMPLE_RATE};

/// Native-rate samples per resampler call; ~21 ms at 48 kHz.
pub(crate) const CHUNK_SIZE: usize = 1024;

/// One FFT block per chunk gives the sharpest low-pass a chunk allows.
const FFT_BLOCKS_PER_CHUNK: usize = 1;

const MONO: usize = 1;

/// Pulls fixed-size chunks from a ring and resamples them to 16 kHz behind
/// an anti-alias filter. A device already at 16 kHz passes samples through
/// untouched.
pub(crate) struct ChunkResampler {
    converter: Option<RateConverter>,
    input: Vec<f32>,
}

impl ChunkResampler {
    pub(crate) fn new(native_rate: u32) -> Result<Self, AudioCaptureError> {
        let converter = if native_rate == TARGET_SAMPLE_RATE {
            None
        } else {
            Some(RateConverter::new(native_rate)?)
        };
        Ok(Self {
            converter,
            input: vec![0.0; CHUNK_SIZE],
        })
    }

    /// Pull up to one chunk from `consumer` at 16 kHz, empty while the filter
    /// is still filling; `None` when the ring is empty. A partial chunk must
    /// be the capture's last: only [`Self::flush`] may follow it.
    pub(crate) fn pull(
        &mut self,
        consumer: &mut HeapCons<f32>,
    ) -> Result<Option<&[f32]>, AudioCaptureError> {
        let popped = consumer.pop_slice(&mut self.input);
        if popped == 0 {
            return Ok(None);
        }
        let Some(converter) = &mut self.converter else {
            return Ok(Some(&self.input[..popped]));
        };
        self.input[popped..].fill(0.0);
        converter.convert(&self.input, popped).map(Some)
    }

    /// Next piece of pulled audio the filter still holds; `None` once all of
    /// it has come out. Call until `None` to keep an utterance's tail.
    pub(crate) fn flush(&mut self) -> Result<Option<&[f32]>, AudioCaptureError> {
        let Some(converter) = self
            .converter
            .as_mut()
            .filter(|converter| converter.samples_owed() > 0)
        else {
            return Ok(None);
        };
        self.input.fill(0.0);
        converter.convert(&self.input, 0).map(Some)
    }
}

/// FFT resampler to 16 kHz whose output lines up with its input: the
/// filter's startup delay is dropped and nothing past the audio's end is kept.
struct RateConverter {
    fft: Fft<f32>,
    native_rate: usize,
    output: Vec<f32>,
    startup_samples_left: usize,
    native_samples_in: usize,
    samples_out: usize,
}

impl RateConverter {
    fn new(native_rate: u32) -> Result<Self, AudioCaptureError> {
        let native_rate = native_rate as usize;
        let fft = Fft::new_custom(
            native_rate,
            TARGET_SAMPLE_RATE as usize,
            CHUNK_SIZE,
            FFT_BLOCKS_PER_CHUNK,
            MONO,
            WindowFunction::BlackmanHarris2,
            FixedSync::Input,
        )
        .map_err(|e| AudioCaptureError::BuildStream(format!("rubato init: {e}")))?;
        Ok(Self {
            output: vec![0.0; fft.output_frames_max()],
            startup_samples_left: fft.output_delay(),
            fft,
            native_rate,
            native_samples_in: 0,
            samples_out: 0,
        })
    }

    /// 16 kHz samples the audio taken so far amounts to, less those returned.
    fn samples_owed(&self) -> usize {
        (self.native_samples_in * TARGET_SAMPLE_RATE as usize).div_ceil(self.native_rate)
            - self.samples_out
    }

    /// Resample one chunk whose first `captured` samples are audio and the
    /// rest zero padding.
    fn convert(&mut self, chunk: &[f32], captured: usize) -> Result<&[f32], AudioCaptureError> {
        self.native_samples_in += captured;
        let input = SequentialSlice::new(chunk, MONO, CHUNK_SIZE)
            .map_err(|e| AudioCaptureError::DeviceError(format!("rubato input adapter: {e}")))?;
        let capacity = self.output.len();
        let mut output = SequentialSlice::new_mut(&mut self.output[..], MONO, capacity)
            .map_err(|e| AudioCaptureError::DeviceError(format!("rubato output adapter: {e}")))?;
        let (_, written) = self
            .fft
            .process_into_buffer(&input, &mut output, None)
            .map_err(|e| AudioCaptureError::DeviceError(format!("rubato process: {e}")))?;

        let skipped = written.min(self.startup_samples_left);
        self.startup_samples_left -= skipped;
        let kept = (written - skipped).min(self.samples_owed());
        self.samples_out += kept;
        Ok(&self.output[skipped..skipped + kept])
    }
}

#[inline]
pub(crate) fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX))
        .to_i16()
        .unwrap_or(0)
}

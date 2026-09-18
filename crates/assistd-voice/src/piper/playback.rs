//! Audio playback via rodio. The device sink is `!Send` on ALSA and
//! must outlive every utterance, so it lives on a dedicated
//! `std::thread` rather than the `spawn_blocking` pool, which may
//! retire idle threads. The `Player` is `Send + Sync`, so callers
//! append samples to it directly.

use std::num::NonZero;
use std::sync::Arc;
use std::thread;

use rodio::buffer::SamplesBuffer;
use rodio::cpal::traits::HostTrait;
use rodio::stream::DeviceSinkBuilder;
use rodio::{ChannelCount, DeviceTrait, Player, SampleRate, cpal};
use tokio::sync::oneshot;

use crate::piper::error::PiperError;
use crate::piper::synth::SynthOutput;

/// Owns the rodio playback path: a device thread and the player
/// queued onto it.
pub struct RodioPlaybackWorker {
    player: Arc<Player>,
    shutdown_tx: Option<std::sync::mpsc::Sender<()>>,
    device_thread: Option<thread::JoinHandle<()>>,
}

impl RodioPlaybackWorker {
    /// Open the named output device (as listed by `aplay -L`), or the
    /// system default for `None`. A name that isn't found falls back
    /// to the default with a warning.
    pub fn start(device_name: Option<&str>) -> Result<Self, PiperError> {
        let host = cpal::default_host();
        match host.output_devices() {
            Ok(devs) => {
                let names: Vec<String> = devs
                    .map(|d| {
                        d.description()
                            .map(|x| x.to_string())
                            .unwrap_or_else(|_| "<no-description>".into())
                    })
                    .collect();
                tracing::info!(
                    target: "assistd::voice::piper",
                    available = ?names,
                    "cpal output devices (set `voice.synthesis.output_device` to one of these)"
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "assistd::voice::piper",
                    error = %e,
                    "could not enumerate cpal output devices"
                );
            }
        }
        let selected = match device_name {
            Some(name) => match host.output_devices() {
                Ok(devs) => {
                    let mut found = None;
                    for d in devs {
                        let dn = d.description().map(|x| x.to_string()).unwrap_or_default();
                        if dn == name {
                            found = Some(d);
                            break;
                        }
                    }
                    match found {
                        Some(d) => {
                            tracing::info!(
                                target: "assistd::voice::piper",
                                device = name,
                                "cpal output device (configured)"
                            );
                            Some(d)
                        }
                        None => {
                            tracing::warn!(
                                target: "assistd::voice::piper",
                                requested = name,
                                "cpal output device not found by name; falling back to default"
                            );
                            host.default_output_device()
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "assistd::voice::piper",
                        error = %e,
                        "could not enumerate cpal output devices; falling back to default"
                    );
                    host.default_output_device()
                }
            },
            None => {
                let dev = host.default_output_device();
                if let Some(d) = &dev {
                    let desc = d
                        .description()
                        .map(|x| x.to_string())
                        .unwrap_or_else(|_| "<no-description>".into());
                    tracing::info!(
                        target: "assistd::voice::piper",
                        device = %desc,
                        "cpal default output device"
                    );
                } else {
                    tracing::warn!(
                        target: "assistd::voice::piper",
                        "cpal reports no default output device"
                    );
                }
                dev
            }
        };

        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<Player, String>>();
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();

        let device_thread = thread::Builder::new()
            .name("piper-rodio".into())
            .spawn(move || {
                let opened = match selected {
                    Some(dev) => DeviceSinkBuilder::from_device(dev)
                        .and_then(|b| b.open_stream())
                        .map_err(|e| format!("open configured device: {e}")),
                    None => DeviceSinkBuilder::open_default_sink()
                        .map_err(|e| format!("open default sink: {e}")),
                };
                let mut device_sink = match opened {
                    Ok(d) => d,
                    Err(msg) => {
                        let _ = init_tx.send(Err(msg));
                        return;
                    }
                };
                device_sink.log_on_drop(false);

                let player = Player::connect_new(device_sink.mixer());
                if init_tx.send(Ok(player)).is_err() {
                    return;
                }

                let _ = shutdown_rx.recv();
                drop(device_sink);
            })
            .map_err(|e| PiperError::Audio(format!("spawn audio thread: {e}")))?;

        let player = init_rx
            .recv()
            .map_err(|_| PiperError::Audio("audio init thread died".into()))?
            .map_err(PiperError::Audio)?;

        Ok(Self {
            player: Arc::new(player),
            shutdown_tx: Some(shutdown_tx),
            device_thread: Some(device_thread),
        })
    }

    /// Queue an utterance. Returns once enqueued; see
    /// [`drain`](Self::drain) to wait for playback.
    pub fn play(&self, output: SynthOutput) -> Result<(), PiperError> {
        let channels: ChannelCount = NonZero::new(1u16).expect("1 is non-zero");
        let sample_rate: SampleRate = NonZero::new(output.sample_rate).ok_or_else(|| {
            PiperError::Audio("voice config reports sample_rate=0; check the .onnx.json".into())
        })?;
        let samples_f32: Vec<f32> = output
            .samples
            .iter()
            .map(|&s| s as f32 / (i16::MAX as f32))
            .collect();
        let buffer = SamplesBuffer::new(channels, sample_rate, samples_f32);
        self.player.append(buffer);
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "playback_enqueued",
            "voice latency stage"
        );
        Ok(())
    }

    /// Wait until the queue has finished playing.
    pub async fn drain(&self) -> Result<(), PiperError> {
        let player = self.player.clone();
        let (tx, rx) = oneshot::channel();
        thread::Builder::new()
            .name("piper-rodio-drain".into())
            .spawn(move || {
                player.sleep_until_end();
                let _ = tx.send(());
            })
            .map_err(|e| PiperError::Audio(format!("spawn drain thread: {e}")))?;
        rx.await.map_err(|_| PiperError::PlaybackClosed)
    }

    /// Drop pending audio. rodio 0.22's `Player::clear` also pauses the
    /// player, so `play` must follow or later appends are silent.
    pub fn clear(&self) {
        self.player.clear();
        self.player.play();
    }

    /// True when the queue has completely drained.
    #[allow(dead_code)]
    pub fn empty(&self) -> bool {
        self.player.empty()
    }
}

/// How long `Drop` waits for the device thread before abandoning the
/// join so a wedged audio device cannot hang daemon shutdown.
const DROP_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

impl Drop for RodioPlaybackWorker {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.device_thread.take() {
            let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
            // The watchdog owns the join; on timeout it is left to
            // finish on its own.
            let spawned = thread::Builder::new()
                .name("piper-rodio-drop-watchdog".into())
                .spawn(move || {
                    let _ = handle.join();
                    let _ = done_tx.send(());
                });
            match spawned {
                Ok(_) => {
                    if done_rx.recv_timeout(DROP_JOIN_TIMEOUT).is_err() {
                        tracing::error!(
                            target: "assistd::voice::piper",
                            timeout_secs = DROP_JOIN_TIMEOUT.as_secs(),
                            "rodio device thread did not exit within budget; abandoning join"
                        );
                    }
                }
                Err(e) => {
                    // The closure owned `handle`, so the thread is
                    // detached and reaped at process exit.
                    tracing::warn!(
                        target: "assistd::voice::piper",
                        error = %e,
                        "could not spawn drop watchdog; device thread detached"
                    );
                }
            }
        }
    }
}

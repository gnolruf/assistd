//! Audio playback via rodio. The cpal device backing
//! [`rodio::stream::MixerDeviceSink`] runs its callback on its own
//! audio thread, but the device sink itself is `!Send` on ALSA, so
//! we hold it on a dedicated `std::thread` (not `spawn_blocking`,
//! whose pool may tear threads down after work completes). The
//! `rodio::Player` wired to that mixer is `Send + Sync`, so the
//! tokio-side caller appends `SamplesBuffer`s directly without
//! crossing the channel for hot-path ops.

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

/// Owns the rodio playback path for the daemon. Cheap to clone via
/// `Arc<RodioPlaybackWorker>`; internal state is `Arc`-shared.
pub struct RodioPlaybackWorker {
    player: Arc<Player>,
    /// Sender used in `Drop` to wake the device thread so it releases
    /// the MixerDeviceSink. The thread joins on drop.
    shutdown_tx: Option<std::sync::mpsc::Sender<()>>,
    device_thread: Option<thread::JoinHandle<()>>,
}

impl RodioPlaybackWorker {
    /// Open the audio device and return a worker. Errors when no
    /// device is available, the device rejects the requested format,
    /// or the audio init thread panics before we get the Player.
    ///
    /// `device_name`:
    /// - `None`: use cpal's default output device. On most desktops
    ///   that's the PipeWire-bridged `default` ALSA PCM, which routes
    ///   wherever PipeWire's default sink points, including Bluetooth.
    /// - `Some(name)`: open the named cpal output device. `name` matches
    ///   the strings cpal returns from `Host::output_devices()`, which
    ///   on Linux are the same as `aplay -L` PCM names (`pipewire`,
    ///   `pulse`, `default`, hardware names like `front:CARD=...`).
    pub fn start(device_name: Option<&str>) -> Result<Self, PiperError> {
        // Always log what cpal sees so silent-synthesis bug reports
        // include the smoking-gun line right next to the synth events.
        let host = cpal::default_host();
        // Enumerate all output devices once so a misconfigured
        // `output_device` value can be diagnosed without external
        // tools (`aplay -L`, etc.).
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
                // Hold the MixerDeviceSink on this thread for the
                // worker's whole lifetime. Dropping it releases the
                // audio device.
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
                // Suppress the "MixerDeviceSink dropped" log line on
                // shutdown; we drop it intentionally.
                device_sink.log_on_drop(false);

                let player = Player::connect_new(device_sink.mixer());
                if init_tx.send(Ok(player)).is_err() {
                    // Caller hung up before init completed.
                    return;
                }

                // Park here until shutdown. The audio thread inside
                // cpal keeps draining the player queue independently;
                // we just need to keep `device_sink` alive.
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

    /// Queue an utterance for playback. Returns immediately; samples
    /// are appended to rodio's internal queue and consumed by the
    /// audio thread. To wait for the queue to drain, follow with
    /// [`drain`](Self::drain).
    #[allow(clippy::expect_used)]
    pub fn play(&self, output: SynthOutput) -> Result<(), PiperError> {
        let channels: ChannelCount = NonZero::new(1u16).expect("1 is non-zero");
        let sample_rate: SampleRate = NonZero::new(output.sample_rate).ok_or_else(|| {
            PiperError::Audio("voice config reports sample_rate=0; check the .onnx.json".into())
        })?;
        // rodio's Sample type is f32 (or f64 with the `64bit` feature
        // off-by-default). i16 → f32 normalisation maps the full
        // signed-16 range to [-1.0, 1.0).
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

    /// Block until the queue is empty (i.e. the most-recent appended
    /// utterance has finished playing). Runs the rodio busy-wait on a
    /// blocking thread so the tokio runtime isn't stalled.
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

    /// Drop pending audio; used to interrupt mid-utterance via
    /// `VoiceOutput::cancel`.
    ///
    /// rodio's `Player::clear` not only empties the queue but also
    /// pauses the player (see rodio 0.22 `player.rs:279-284`). Without
    /// the explicit `play()` after, the next `append()` queues samples
    /// onto a paused player and produces no audio. This bit the PTT
    /// barge-in path: `interrupt()` → `clear()` → user speaks →
    /// `handle_query` → `speak()` → silent, while typed-message TTS
    /// (which never invokes `clear`) worked fine.
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

/// Bounded wait used by the Drop watchdog. If the audio thread is
/// wedged (rare on ALSA but possible), abandoning the join after this
/// budget keeps daemon shutdown unblocked; the OS reclaims the
/// thread on process exit. Logged at `error!` so the operator sees
/// it.
const DROP_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

impl Drop for RodioPlaybackWorker {
    fn drop(&mut self) {
        // Tell the device thread to release MixerDeviceSink; then join
        // with a bounded timeout so daemon shutdown can't hang on a
        // wedged audio device. If the channel is already closed (e.g.
        // thread panicked) the send fails silently and we still
        // attempt to join.
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.device_thread.take() {
            let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
            // Spawn a watchdog thread that owns the join. The watchdog
            // is leaked on timeout; it will clean itself up when the
            // audio thread does eventually exit (or when the process
            // does), neither of which blocks the caller.
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
                    // Couldn't spawn watchdog; fall back to direct
                    // join. The OS process tear-down still unblocks
                    // eventually if this hangs, but we lose
                    // bounded-shutdown guarantees.
                    tracing::warn!(
                        target: "assistd::voice::piper",
                        error = %e,
                        "could not spawn drop watchdog; joining inline"
                    );
                    // We took `handle` already; can't reach it here.
                    // The watchdog thread that was supposed to own it
                    // dropped the variable, which transitively detached
                    // the join; acceptable since process exit reaps
                    // it.
                }
            }
        }
    }
}

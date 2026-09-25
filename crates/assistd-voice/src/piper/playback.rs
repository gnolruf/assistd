//! Audio playback via rodio.

use std::num::NonZero;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use rodio::buffer::SamplesBuffer;
use rodio::cpal::traits::HostTrait;
use rodio::stream::DeviceSinkBuilder;
use rodio::{ChannelCount, DeviceTrait, Player, SampleRate, cpal};

use crate::piper::error::PiperError;
use crate::piper::synth::SynthOutput;

const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// How long `Drop` waits for the device thread before abandoning the join.
const DROP_JOIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Owns the rodio playback path. The `!Send` device sink lives on a dedicated
/// thread for the worker's lifetime; `Player` calls that can block on rodio's
/// queue mutex run on the blocking pool.
pub struct RodioPlaybackWorker {
    player: Arc<Player>,
    shutdown_tx: Option<mpsc::Sender<()>>,
    device_thread: Option<thread::JoinHandle<()>>,
}

impl RodioPlaybackWorker {
    /// Open the named output device (as listed by `aplay -L`), or the
    /// system default for `None`. A name that isn't found falls back
    /// to the default with a warning.
    pub fn start(device_name: Option<&str>) -> Result<Self, PiperError> {
        let host = cpal::default_host();
        let selected = select_output_device(&host, device_name);

        let (init_tx, init_rx) = mpsc::channel::<Result<Player, String>>();
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();
        let device_thread = spawn_device_thread(selected, init_tx, shutdown_rx)?;

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
    pub async fn play(&self, output: SynthOutput) -> Result<(), PiperError> {
        let channels: ChannelCount = NonZero::new(1u16).expect("1 is non-zero");
        let sample_rate: SampleRate = NonZero::new(output.sample_rate).ok_or_else(|| {
            PiperError::Audio("voice config reports sample_rate=0; check the .onnx.json".into())
        })?;
        let samples_f32: Vec<f32> = output
            .samples
            .iter()
            .map(|&sample| sample as f32 / (i16::MAX as f32))
            .collect();
        let buffer = SamplesBuffer::new(channels, sample_rate, samples_f32);
        let player = self.player.clone();
        tokio::task::spawn_blocking(move || player.append(buffer))
            .await
            .map_err(|err| PiperError::Audio(format!("playback append task failed: {err}")))?;
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "playback_enqueued",
            "voice latency stage"
        );
        Ok(())
    }

    /// Wait until the queue has finished playing. Polls because rodio's
    /// `sleep_until_end` holds the queue mutex and would block every later `append`.
    pub async fn drain(&self) {
        while !self.player.empty() {
            tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
        }
    }

    /// Drop pending audio. rodio 0.22's `Player::clear` also pauses the
    /// player, so `play` must follow or later appends are silent.
    pub async fn clear(&self) {
        let player = self.player.clone();
        let _ = tokio::task::spawn_blocking(move || {
            player.clear();
            player.play();
        })
        .await;
    }
}

impl Drop for RodioPlaybackWorker {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(device_thread) = self.device_thread.take() {
            join_with_timeout(device_thread);
        }
    }
}

fn describe(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.to_string())
        .unwrap_or_else(|_| "<no-description>".into())
}

/// Log the available outputs, then pick the named one or the default.
/// A name that isn't found falls back to the default with a warning.
fn select_output_device(host: &cpal::Host, name: Option<&str>) -> Option<cpal::Device> {
    let devices: Vec<(cpal::Device, String)> = match host.output_devices() {
        Ok(devices) => devices
            .map(|device| {
                let description = describe(&device);
                (device, description)
            })
            .collect(),
        Err(err) => {
            tracing::warn!(
                target: "assistd::voice::piper",
                error = %err,
                "could not enumerate cpal output devices; falling back to default"
            );
            return host.default_output_device();
        }
    };
    let names: Vec<&str> = devices
        .iter()
        .map(|(_, description)| description.as_str())
        .collect();
    tracing::info!(
        target: "assistd::voice::piper",
        available = ?names,
        "cpal output devices (set `voice.synthesis.output_device` to one of these)"
    );

    let Some(name) = name else {
        return default_output_device(host);
    };
    match devices
        .into_iter()
        .find(|(_, description)| description == name)
    {
        Some((device, _)) => {
            tracing::info!(
                target: "assistd::voice::piper",
                device = name,
                "cpal output device (configured)"
            );
            Some(device)
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

fn default_output_device(host: &cpal::Host) -> Option<cpal::Device> {
    let device = host.default_output_device();
    match &device {
        Some(device) => tracing::info!(
            target: "assistd::voice::piper",
            device = %describe(device),
            "cpal default output device"
        ),
        None => tracing::warn!(
            target: "assistd::voice::piper",
            "cpal reports no default output device"
        ),
    }
    device
}

fn spawn_device_thread(
    device: Option<cpal::Device>,
    init_tx: mpsc::Sender<Result<Player, String>>,
    shutdown_rx: mpsc::Receiver<()>,
) -> Result<thread::JoinHandle<()>, PiperError> {
    thread::Builder::new()
        .name("piper-rodio".into())
        .spawn(move || run_device_thread(device, init_tx, shutdown_rx))
        .map_err(|err| PiperError::Audio(format!("spawn audio thread: {err}")))
}

/// Report the connected player (or the open error) on `init_tx`, then hold
/// the device sink until `shutdown_rx` fires or hangs up.
fn run_device_thread(
    device: Option<cpal::Device>,
    init_tx: mpsc::Sender<Result<Player, String>>,
    shutdown_rx: mpsc::Receiver<()>,
) {
    let opened = match device {
        Some(device) => DeviceSinkBuilder::from_device(device)
            .and_then(|builder| builder.open_stream())
            .map_err(|err| format!("open configured device: {err}")),
        None => DeviceSinkBuilder::open_default_sink()
            .map_err(|err| format!("open default sink: {err}")),
    };
    let mut device_sink = match opened {
        Ok(device_sink) => device_sink,
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
}

/// Join `device_thread` from a watchdog thread so a wedged audio device cannot
/// hang shutdown; on timeout or spawn failure the thread is left detached.
fn join_with_timeout(device_thread: thread::JoinHandle<()>) {
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let spawned = thread::Builder::new()
        .name("piper-rodio-drop-watchdog".into())
        .spawn(move || {
            let _ = device_thread.join();
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
        Err(err) => tracing::warn!(
            target: "assistd::voice::piper",
            error = %err,
            "could not spawn drop watchdog; device thread detached"
        ),
    }
}

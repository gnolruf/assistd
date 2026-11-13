//! Audio playback via rodio.

use std::fmt;
use std::num::NonZero;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use rodio::buffer::SamplesBuffer;
use rodio::cpal::traits::HostTrait;
use rodio::stream::DeviceSinkBuilder;
use rodio::{ChannelCount, DeviceTrait, Player, cpal};
use tokio::sync::Mutex;
use tokio::time::Instant;

use crate::tts::engine::Utterance;
use crate::tts::error::SpeechError;

const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// How long past the expected end of the queued audio `drain` waits before
/// calling the output stalled.
const DRAIN_STALL_MARGIN: Duration = Duration::from_secs(5);

/// How long `Drop` waits for the device thread before abandoning the join.
const DROP_JOIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Owns the rodio playback path. `Player` calls that can block on rodio's
/// queue mutex run on the blocking pool. A stream that stalls is closed by
/// [`drain`](Self::drain) and reopened by the next [`play`](Self::play).
#[derive(Debug)]
pub struct RodioPlaybackWorker {
    device_name: Option<String>,
    /// `None` between a stall and the `play` that reopens the device.
    stream: Mutex<Option<Arc<OutputStream>>>,
}

impl RodioPlaybackWorker {
    /// Open the named output device (as listed by `aplay -L`), or the
    /// system default for `None`. A name that isn't found falls back
    /// to the default with a warning.
    pub async fn start(device_name: Option<String>) -> Result<Self, SpeechError> {
        let stream = OutputStream::open(device_name.clone()).await?;
        Ok(Self {
            device_name,
            stream: Mutex::new(Some(Arc::new(stream))),
        })
    }

    /// Queue an utterance. Returns once enqueued; see
    /// [`drain`](Self::drain) to wait for playback. Errors when a stalled
    /// stream cannot be reopened.
    pub async fn play(&self, utterance: Utterance) -> Result<(), SpeechError> {
        let stream = self.open_stream().await?;
        let channels: ChannelCount = NonZero::new(1u16).expect("1 is non-zero");
        let audio = utterance.duration();
        let buffer = SamplesBuffer::new(channels, utterance.sample_rate, utterance.samples);
        stream.extend_queue(audio);
        tokio::task::spawn_blocking(move || stream.player.append(buffer))
            .await
            .map_err(|err| SpeechError::Audio(format!("playback append task failed: {err}")))?;
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "playback_enqueued",
            "voice latency stage"
        );
        Ok(())
    }

    /// Wait until the queue has finished playing. Polls because rodio's
    /// `sleep_until_end` holds the queue mutex and would block every later `append`.
    /// Errors, closing the stream, once the queue is five seconds overdue.
    pub async fn drain(&self) -> Result<(), SpeechError> {
        let Some(stream) = self.stream.lock().await.clone() else {
            return Ok(());
        };
        while !stream.player.empty() {
            if stream.is_overdue() {
                self.close_stalled(&stream).await;
                return Err(SpeechError::OutputStalled);
            }
            tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
        }
        Ok(())
    }

    /// Drop pending audio. rodio's `Player::clear` also pauses the
    /// player, so `play` must follow or later appends are silent.
    pub async fn clear(&self) {
        let Some(stream) = self.stream.lock().await.clone() else {
            return;
        };
        let _ = tokio::task::spawn_blocking(move || {
            stream.player.clear();
            stream.player.play();
            stream.empty_queue();
        })
        .await;
    }

    /// The open stream, first reopening the device if a stall closed it.
    async fn open_stream(&self) -> Result<Arc<OutputStream>, SpeechError> {
        let mut open = self.stream.lock().await;
        if let Some(stream) = open.as_ref() {
            return Ok(Arc::clone(stream));
        }
        let stream = Arc::new(OutputStream::open(self.device_name.clone()).await?);
        tracing::info!(target: "assistd::voice::tts", "audio output reopened after a stall");
        Ok(Arc::clone(open.insert(stream)))
    }

    /// Close `stalled`, unless it has been replaced already.
    async fn close_stalled(&self, stalled: &Arc<OutputStream>) {
        let closed = self
            .stream
            .lock()
            .await
            .take_if(|open| Arc::ptr_eq(open, stalled));
        if closed.is_some() {
            tracing::warn!(
                target: "assistd::voice::tts",
                margin_secs = DRAIN_STALL_MARGIN.as_secs(),
                "audio output stopped playing its queue; closed it to reopen on the next sentence"
            );
        }
    }
}

/// One open output stream. The `!Send` device sink lives on a dedicated
/// thread until the stream is dropped.
struct OutputStream {
    player: Player,
    /// When the audio queued so far should have finished playing.
    queued_until: parking_lot::Mutex<Instant>,
    shutdown_tx: Option<mpsc::Sender<()>>,
    device_thread: Option<thread::JoinHandle<()>>,
}

impl fmt::Debug for OutputStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputStream")
            .field("queued_until", &self.queued_until)
            .field("device_thread", &self.device_thread)
            .finish_non_exhaustive()
    }
}

impl OutputStream {
    async fn open(device_name: Option<String>) -> Result<Self, SpeechError> {
        tokio::task::spawn_blocking(move || Self::open_blocking(device_name.as_deref()))
            .await
            .map_err(|err| SpeechError::Audio(format!("audio init task failed: {err}")))?
    }

    fn open_blocking(device_name: Option<&str>) -> Result<Self, SpeechError> {
        let host = cpal::default_host();
        let selected = select_output_device(&host, device_name);

        let (init_tx, init_rx) = mpsc::channel::<Result<Player, String>>();
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();
        let device_thread = spawn_device_thread(selected, init_tx, shutdown_rx)?;

        let player = init_rx
            .recv()
            .map_err(|_| SpeechError::Audio("audio init thread died".into()))?
            .map_err(SpeechError::Audio)?;

        Ok(Self {
            player,
            queued_until: parking_lot::Mutex::new(Instant::now()),
            shutdown_tx: Some(shutdown_tx),
            device_thread: Some(device_thread),
        })
    }

    /// Count `audio` as queued behind whatever is still due to play.
    fn extend_queue(&self, audio: Duration) {
        let mut queued_until = self.queued_until.lock();
        *queued_until = (*queued_until).max(Instant::now()) + audio;
    }

    fn empty_queue(&self) {
        *self.queued_until.lock() = Instant::now();
    }

    fn is_overdue(&self) -> bool {
        Instant::now() > *self.queued_until.lock() + DRAIN_STALL_MARGIN
    }
}

impl Drop for OutputStream {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(device_thread) = self.device_thread.take() {
            join_off_runtime(device_thread);
        }
    }
}

fn device_name(device: &cpal::Device) -> String {
    device.description().map_or_else(
        |_| "<unknown>".into(),
        |description| {
            description
                .driver()
                .map_or_else(|| description.name().to_string(), str::to_string)
        },
    )
}

/// Log the available outputs, then pick the named one or the default.
/// A name that isn't found falls back to the default with a warning.
fn select_output_device(host: &cpal::Host, name: Option<&str>) -> Option<cpal::Device> {
    let devices: Vec<(cpal::Device, String)> = match host.output_devices() {
        Ok(devices) => devices
            .map(|device| {
                let name = device_name(&device);
                (device, name)
            })
            .collect(),
        Err(err) => {
            tracing::warn!(
                target: "assistd::voice::tts",
                error = %err,
                "could not enumerate cpal output devices; falling back to default"
            );
            return host.default_output_device();
        }
    };
    let names: Vec<&str> = devices
        .iter()
        .map(|(_, device_name)| device_name.as_str())
        .collect();
    tracing::info!(
        target: "assistd::voice::tts",
        available = ?names,
        "cpal output devices (set `voice.synthesis.output_device` to one of these)"
    );

    let Some(name) = name else {
        return default_output_device(host);
    };
    match devices
        .into_iter()
        .find(|(_, device_name)| device_name == name)
    {
        Some((device, _)) => {
            tracing::info!(
                target: "assistd::voice::tts",
                device = name,
                "cpal output device (configured)"
            );
            Some(device)
        }
        None => {
            tracing::warn!(
                target: "assistd::voice::tts",
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
            target: "assistd::voice::tts",
            device = %device_name(device),
            "cpal default output device"
        ),
        None => tracing::warn!(
            target: "assistd::voice::tts",
            "cpal reports no default output device"
        ),
    }
    device
}

fn spawn_device_thread(
    device: Option<cpal::Device>,
    init_tx: mpsc::Sender<Result<Player, String>>,
    shutdown_rx: mpsc::Receiver<()>,
) -> Result<thread::JoinHandle<()>, SpeechError> {
    thread::Builder::new()
        .name("tts-rodio".into())
        .spawn(move || run_device_thread(device, &init_tx, &shutdown_rx))
        .map_err(|err| SpeechError::Audio(format!("spawn audio thread: {err}")))
}

/// Report the connected player (or the open error) on `init_tx`, then hold
/// the device sink until `shutdown_rx` fires or hangs up.
fn run_device_thread(
    device: Option<cpal::Device>,
    init_tx: &mpsc::Sender<Result<Player, String>>,
    shutdown_rx: &mpsc::Receiver<()>,
) {
    let opened = match device {
        Some(device) => DeviceSinkBuilder::from_device(device)
            .and_then(DeviceSinkBuilder::open_stream)
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

/// Run [`join_with_timeout`] on the blocking pool when dropped inside a
/// runtime, so no async worker waits on the device; runtime shutdown still
/// waits for it.
fn join_off_runtime(device_thread: thread::JoinHandle<()>) {
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => drop(runtime.spawn_blocking(move || join_with_timeout(device_thread))),
        Err(_) => join_with_timeout(device_thread),
    }
}

/// Join `device_thread` from a watchdog thread so a wedged audio device cannot
/// hang shutdown; on timeout or spawn failure the thread is left detached.
fn join_with_timeout(device_thread: thread::JoinHandle<()>) {
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let spawned = thread::Builder::new()
        .name("tts-rodio-drop-watchdog".into())
        .spawn(move || {
            let _ = device_thread.join();
            let _ = done_tx.send(());
        });
    match spawned {
        Ok(_) => {
            if done_rx.recv_timeout(DROP_JOIN_TIMEOUT).is_err() {
                tracing::error!(
                    target: "assistd::voice::tts",
                    timeout_secs = DROP_JOIN_TIMEOUT.as_secs(),
                    "rodio device thread did not exit within budget; abandoning join"
                );
            }
        }
        Err(err) => tracing::warn!(
            target: "assistd::voice::tts",
            error = %err,
            "could not spawn drop watchdog; device thread detached"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn drain_fails_and_closes_a_stream_that_never_plays_its_queue() {
        let (player, _never_played) = Player::new();
        let playback = RodioPlaybackWorker {
            device_name: None,
            stream: Mutex::new(Some(Arc::new(OutputStream {
                player,
                queued_until: parking_lot::Mutex::new(Instant::now()),
                shutdown_tx: None,
                device_thread: None,
            }))),
        };
        let two_seconds = Utterance {
            sample_rate: NonZero::new(16_000).expect("non-zero"),
            samples: vec![0.0; 32_000],
        };
        let queued_at = Instant::now();
        playback.play(two_seconds).await.expect("enqueue");

        let drained = playback.drain().await;

        assert!(matches!(drained, Err(SpeechError::OutputStalled)));
        assert!(queued_at.elapsed() >= Duration::from_secs(2) + DRAIN_STALL_MARGIN);
        assert!(
            playback.stream.lock().await.is_none(),
            "the next play must reopen the device"
        );
    }
}

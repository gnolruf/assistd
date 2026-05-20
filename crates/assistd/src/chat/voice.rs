//! Chat-TUI voice glue, daemon-window edition.
//!
//! The TUI no longer owns a `MicVoiceInput`; the daemon does. We still
//! spawn the global hotkey listener locally because PTT keystrokes need
//! to arrive at the foreground process. The press/release callbacks
//! dispatch `Request::PttStart` / `Request::PttStop` to the daemon
//! through [`IpcVoiceProxy`]; the daemon's response stream
//! (`VoiceState` → `Transcription` → `Delta`s → `Done`) feeds back into
//! the App's `ChatEvent` channel as `ChatEvent::Wire(_)` so the same
//! reducer that handles query-driven streaming updates the listening
//! indicator and the output pane uniformly.

use std::sync::Arc;

use assistd_core::Config;
use assistd_ipc::{Event, IpcClient};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use super::app::ChatEvent;
use crate::hotkey;
use crate::ipc_voice_proxy::IpcVoiceProxy;

/// Buffered Events between the IPC proxy and the chat reducer. Twelve
/// is plenty: the proxy emits at the daemon's stream cadence (single-
/// digit events per second on a typical reply) and the adapter forwards
/// each one as it arrives.
const EVENT_BRIDGE_CAPACITY: usize = 12;

/// Handles owned by the voice glue. Held by the caller for the
/// lifetime of the TUI session so the hotkey listener doesn't drop
/// its `Arc` references mid-run. Voice itself runs in the daemon;
/// this struct exists only to keep the local hotkey thread alive.
#[allow(dead_code)]
pub struct VoicePipeline {
    pub hotkey_handle: Option<JoinHandle<()>>,
    pub bridge_handle: Option<JoinHandle<()>>,
}

/// Build the chat-TUI voice pipeline.
///
/// - When `voice.enabled = false`, returns a placeholder with no
///   hotkey bound; the daemon's voice IPC remains reachable for
///   anything that wants to drive PTT manually (`assistd ptt-start`).
/// - Otherwise spawns `crate::hotkey::spawn_listener` against an
///   [`IpcVoiceProxy`] whose `start_recording` / `stop_and_transcribe`
///   issue Unix-socket requests to the daemon. The daemon's streaming
///   response flows back through an Event bridge into `chat_tx` so the
///   reducer sees Whisper transitions and the auto-dispatched query
///   response.
pub async fn spawn(
    config: &Config,
    ipc: Arc<IpcClient>,
    chat_tx: mpsc::Sender<ChatEvent>,
    shutdown_rx: watch::Receiver<bool>,
) -> VoicePipeline {
    if !config.voice.enabled {
        info!("voice: disabled in config; PTT hotkey will not bind");
        return VoicePipeline {
            hotkey_handle: None,
            bridge_handle: None,
        };
    }

    info!(
        "voice: routing PTT through daemon IPC (hotkey={:?})",
        config.voice.hotkey
    );

    let (event_tx, event_rx) = mpsc::channel::<Event>(EVENT_BRIDGE_CAPACITY);
    let bridge_handle = Some(tokio::spawn(bridge_events(event_rx, chat_tx)));

    let proxy: Arc<dyn assistd_voice::VoiceInput> =
        Arc::new(IpcVoiceProxy::new(ipc, Some(event_tx)));
    let hotkey_handle = hotkey::spawn_listener(
        &config.presence,
        &config.voice,
        hotkey::Subsystems {
            presence: None, // chat TUI doesn't own a PresenceManager
            voice: proxy,
            listener: None,     // continuous-listen runs in the daemon
            voice_output: None, // VoiceOutputController also lives in the daemon
        },
        shutdown_rx,
    );

    VoicePipeline {
        hotkey_handle,
        bridge_handle,
    }
}

/// Forward each `Event` produced by the proxy onto the chat reducer's
/// channel as `ChatEvent::Wire`. Exits when the proxy drops its sender
/// (process shutdown).
async fn bridge_events(mut event_rx: mpsc::Receiver<Event>, chat_tx: mpsc::Sender<ChatEvent>) {
    while let Some(ev) = event_rx.recv().await {
        if chat_tx.send(ChatEvent::Wire(ev)).await.is_err() {
            break;
        }
    }
}

#[allow(dead_code)] // used only when the `chat` feature compiles voice off
fn _suppress_unused_warn() {
    warn!("voice glue unused");
}

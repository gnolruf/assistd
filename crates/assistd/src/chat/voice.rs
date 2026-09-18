//! Push-to-talk for the chat TUI. The hotkey is grabbed locally because
//! keystrokes must reach the foreground process; the daemon does the
//! rest, and its reply stream feeds the reducer like a typed query's.

use std::sync::Arc;

use assistd_core::Config;
use assistd_ipc::{Event, IpcClient};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::info;

use super::app::{ChatEvent, WireStream};
use crate::hotkey;
use crate::ipc_voice_proxy::IpcVoiceProxy;

const EVENT_BRIDGE_CAPACITY: usize = 12;

/// Tasks the caller holds for the TUI session's lifetime.
pub struct VoicePipeline {
    hotkey_handle: Option<JoinHandle<()>>,
    bridge_handle: Option<JoinHandle<()>>,
}

impl VoicePipeline {
    /// Abort both tasks and wait for them to stop.
    pub async fn shutdown(self) {
        for handle in [self.hotkey_handle, self.bridge_handle]
            .into_iter()
            .flatten()
        {
            handle.abort();
            let _ = handle.await;
        }
    }
}

/// With voice disabled, returns an empty pipeline and binds no hotkey.
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
            presence: None,
            voice: proxy,
            listener: None,
            voice_output: None,
        },
        shutdown_rx,
    );

    VoicePipeline {
        hotkey_handle,
        bridge_handle,
    }
}

/// Tagged [`WireStream::Reply`]: the daemon streams the answer on the
/// PTT connection, so these events own the output pane.
async fn bridge_events(mut event_rx: mpsc::Receiver<Event>, chat_tx: mpsc::Sender<ChatEvent>) {
    while let Some(ev) = event_rx.recv().await {
        let tagged = ChatEvent::Wire {
            stream: WireStream::Reply,
            event: ev,
        };
        if chat_tx.send(tagged).await.is_err() {
            break;
        }
    }
}

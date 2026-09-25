//! Daemon connections: one-shot requests and the two-way query dialog.
//! Each runs as an app task and reports a failure before its terminal
//! event as a [`ChatEvent::WireError`] on its stream.

use assistd_ipc::{ImageAttachment, IpcClient, Request};
use assistd_tools::Attachment;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::{ActiveReply, App, ChatEvent, WireStream};

const QUERY_WRITER_CAPACITY: usize = 8;

impl App {
    /// Send `req` on a fresh connection and feed its events to the reducer
    /// tagged `stream`.
    pub(super) fn spawn_one_shot(&mut self, req: Request, stream: WireStream, label: &'static str) {
        let ipc = self.ipc.clone();
        let chat_tx = self.chat_tx.clone();
        self.tasks.spawn(async move {
            if let Err(message) = pump_one_shot(&ipc, req, stream, label, &chat_tx).await {
                let _ = chat_tx.send(ChatEvent::WireError { stream, message }).await;
            }
        });
    }

    /// Open a query dialog for `text` and make it the active reply, whose
    /// writer carries confirmation answers back to the daemon.
    pub(super) fn spawn_query(&mut self, text: String, attachments: Vec<Attachment>) {
        let (writer_tx, writer_rx) = mpsc::channel::<Request>(QUERY_WRITER_CAPACITY);
        let id = Uuid::new_v4().to_string();
        self.active_reply = Some(ActiveReply {
            id: id.clone(),
            writer: Some(writer_tx),
        });
        let req = Request::query_with_attachments(id, text, wire_attachments(attachments));
        let ipc = self.ipc.clone();
        let chat_tx = self.chat_tx.clone();
        self.tasks.spawn(async move {
            if let Err(message) = drive_query(&ipc, req, writer_rx, &chat_tx).await {
                let _ = chat_tx
                    .send(ChatEvent::WireError {
                        stream: WireStream::Reply,
                        message,
                    })
                    .await;
            }
        });
    }
}

fn wire_attachments(attachments: Vec<Attachment>) -> Vec<ImageAttachment> {
    attachments
        .into_iter()
        .map(|attachment| match attachment {
            Attachment::Image { mime, bytes } => ImageAttachment::from_bytes(mime, &bytes),
        })
        .collect()
}

/// Forward the events answering `req` until a terminal one. `Err` is the
/// wire-error message.
async fn pump_one_shot(
    ipc: &IpcClient,
    req: Request,
    stream: WireStream,
    label: &str,
    chat_tx: &mpsc::Sender<ChatEvent>,
) -> Result<(), String> {
    let mut events = ipc
        .one_shot(req)
        .await
        .map_err(|e| format!("{label} connect: {e}"))?;
    loop {
        match events.next_event().await {
            Ok(Some(event)) => {
                let terminal = event.is_terminal();
                let _ = chat_tx.send(ChatEvent::Wire { stream, event }).await;
                if terminal {
                    return Ok(());
                }
            }
            Ok(None) => return Err(format!("{label}: daemon closed stream mid-flight")),
            Err(e) => return Err(format!("{label} read: {e}")),
        }
    }
}

/// Forward the dialog's events to the reducer and `writer_rx`'s requests
/// to the daemon until a terminal event. The dialog outlives a closed
/// writer channel. `Err` is the wire-error message.
async fn drive_query(
    ipc: &IpcClient,
    req: Request,
    mut writer_rx: mpsc::Receiver<Request>,
    chat_tx: &mpsc::Sender<ChatEvent>,
) -> Result<(), String> {
    let mut conn = ipc
        .open_dialog(req)
        .await
        .map_err(|e| format!("query connect: {e}"))?;
    let mut writer_open = true;
    loop {
        tokio::select! {
            next = conn.next_event() => match next {
                Ok(Some(event)) => {
                    let terminal = event.is_terminal();
                    let _ = chat_tx
                        .send(ChatEvent::Wire {
                            stream: WireStream::Reply,
                            event,
                        })
                        .await;
                    if terminal {
                        return Ok(());
                    }
                }
                Ok(None) => return Err("daemon closed connection mid-stream".into()),
                Err(e) => return Err(format!("read: {e}")),
            },
            outgoing = writer_rx.recv(), if writer_open => match outgoing {
                Some(req) => conn.send(req).await.map_err(|e| format!("write: {e}"))?,
                None => writer_open = false,
            },
        }
    }
}

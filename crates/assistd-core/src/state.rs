use crate::Config;
use anyhow::Result;
use assistd_ipc::{Event, Request};
use assistd_llm::{LlmBackend, LlmEvent};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Shared, long-lived daemon state handed to every request handler.
///
/// Subsystem handles (tools, voice, window manager, memory) will be
/// added here as their respective milestones land. They are expected to
/// be `Arc`-held and cheaply cloneable (actor façades over background
/// tasks) so that multiple concurrent requests can all hold
/// `Arc<AppState>` without contention.
pub struct AppState {
    pub config: Config,
    pub llm: Arc<dyn LlmBackend>,
}

impl AppState {
    pub fn new(config: Config, llm: Arc<dyn LlmBackend>) -> Self {
        Self { config, llm }
    }

    /// Route a single incoming request to the appropriate subsystem.
    ///
    /// Events are streamed back through `tx`. When this function returns
    /// (either `Ok` or `Err`), no further events will be sent. The caller
    /// is responsible for surfacing `Err` to the client as an
    /// [`Event::Error`] if one hasn't been emitted already.
    pub async fn dispatch(self: Arc<Self>, req: Request, tx: mpsc::Sender<Event>) -> Result<()> {
        match req {
            Request::Query { id, text } => self.handle_query(id, text, tx).await,
        }
    }

    async fn handle_query(
        self: Arc<Self>,
        id: String,
        text: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let (llm_tx, mut llm_rx) = mpsc::channel::<LlmEvent>(32);
        let llm = self.llm.clone();
        let generator = tokio::spawn(async move { llm.generate(text, llm_tx).await });

        while let Some(llm_event) = llm_rx.recv().await {
            let wire_event = match llm_event {
                LlmEvent::Delta { text } => Event::Delta {
                    id: id.clone(),
                    text,
                },
                LlmEvent::Done => Event::Done { id: id.clone() },
            };
            if tx.send(wire_event).await.is_err() {
                // Client has disconnected; stop forwarding events.
                break;
            }
        }

        match generator.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("llm backend error: {e}"),
                    })
                    .await;
                Err(e)
            }
            Err(join_err) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("llm backend panicked: {join_err}"),
                    })
                    .await;
                Err(anyhow::anyhow!("llm backend panicked: {join_err}"))
            }
        }
    }
}

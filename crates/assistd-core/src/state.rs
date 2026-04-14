use crate::{Config, PresenceManager};
use anyhow::Result;
use assistd_ipc::{Event, PresenceState, Request};
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
    pub presence: Arc<PresenceManager>,
}

impl AppState {
    pub fn new(config: Config, llm: Arc<dyn LlmBackend>, presence: Arc<PresenceManager>) -> Self {
        Self {
            config,
            llm,
            presence,
        }
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
            Request::SetPresence { id, target } => self.handle_set_presence(id, target, tx).await,
            Request::GetPresence { id } => self.handle_get_presence(id, tx).await,
            Request::Cycle { id } => self.handle_cycle(id, tx).await,
        }
    }

    async fn handle_query(
        self: Arc<Self>,
        id: String,
        text: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        // Auto-wake and take an in-flight guard: a query in any
        // non-Active state blocks here until the daemon is ready to
        // serve. The returned guard keeps the daemon `Active` for the
        // lifetime of the generation — a concurrent `sleep()` will
        // wait until this guard (and the generator that follows) has
        // finished. Failures surface to the client as an Error event
        // so the client doesn't hang.
        let _request_guard = match self.presence.acquire_request_guard().await {
            Ok(g) => g,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id: id.clone(),
                        message: format!("wake failed: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };

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

    async fn handle_set_presence(
        self: Arc<Self>,
        id: String,
        target: PresenceState,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.presence.set_presence(target).await {
            Ok(()) => {
                let _ = tx
                    .send(Event::Presence {
                        id: id.clone(),
                        state: self.presence.state(),
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("set_presence failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    async fn handle_get_presence(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let state = self.presence.state();
        let _ = tx
            .send(Event::Presence {
                id: id.clone(),
                state,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    async fn handle_cycle(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) -> Result<()> {
        match self.presence.cycle().await {
            Ok(new_state) => {
                let _ = tx
                    .send(Event::Presence {
                        id: id.clone(),
                        state: new_state,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("cycle failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }
}

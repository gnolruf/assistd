//! `AppState` and the request dispatcher; each submodule adds the
//! handlers for one family of `Request` variants.

use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;
use tokio::sync::mpsc;
use tokio::task::JoinError;
use tracing::warn;

use assistd_embed::EmbedError;
use assistd_ipc::{Event, Request, Role};
use assistd_llm::{LlmBackend, LlmError};
use assistd_memory::{MemoryError, PersistedRole};
use assistd_tools::ToolRegistry;
use assistd_voice::{
    ContinuousListener, ListenError, VoiceInput, VoiceInputError, VoiceOutputController,
};

use crate::{Config, PresenceError, PresenceManager};

pub(crate) mod branches;
pub(crate) mod capabilities;
pub(crate) mod context;
pub(crate) mod memory_handlers;
pub(crate) mod memory_stack;
pub(crate) mod persistence;
pub(crate) mod presence_handlers;
pub(crate) mod query;
pub(crate) mod runtime;
pub(crate) mod subscribe;
pub(crate) mod subsystems;
pub(crate) mod voice_handlers;
pub(crate) mod wire;

pub use self::branches::history_entries;
pub use self::memory_stack::MemoryStack;
pub use self::runtime::{BusSubscription, ConversationContext, RuntimeState};
pub use self::subsystems::{McpStartupFailure, Subsystems};

/// Why a request handler failed.
#[derive(Debug, Error)]
pub enum DispatchError {
    #[error(transparent)]
    Presence(#[from] PresenceError),

    #[error("invalid attachment: base64 decode failed for {mime}: {source}")]
    InvalidAttachment {
        mime: String,
        #[source]
        source: base64::DecodeError,
    },

    #[error(transparent)]
    Llm(#[from] LlmError),

    #[error("llm backend panicked: {0}")]
    AgentPanicked(#[source] JoinError),

    #[error(transparent)]
    Memory(#[from] MemoryError),

    #[error(transparent)]
    Embed(#[from] EmbedError),

    #[error(transparent)]
    VoiceInput(#[from] VoiceInputError),

    #[error(transparent)]
    Listen(#[from] ListenError),
}

/// Shared, long-lived daemon state handed to every request handler.
pub struct AppState {
    pub config: Config,
    pub subsystems: Subsystems,
    pub memory: MemoryStack,
    pub runtime: RuntimeState,
}

impl AppState {
    /// An `AppState` with no-op memory and embedding backends.
    pub fn new(
        config: Config,
        llm: Arc<dyn LlmBackend>,
        presence: Arc<PresenceManager>,
        tools: Arc<ToolRegistry>,
        voice: Arc<dyn VoiceInput>,
        listener: Arc<dyn ContinuousListener>,
        voice_output: Arc<VoiceOutputController>,
    ) -> Self {
        let subsystems = Subsystems::new(llm, presence, tools, voice, listener, voice_output);
        let memory = MemoryStack::disabled(config.embedding.clone());
        let runtime = RuntimeState::new();
        Self {
            config,
            subsystems,
            memory,
            runtime,
        }
    }

    /// Route one request to its handler, streaming events back on `tx`.
    /// No events are sent after this returns. Every request except
    /// `Subscribe` is bounded by the dispatch envelope timeout.
    pub async fn dispatch(
        self: Arc<Self>,
        req: Request,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        if matches!(req, Request::Subscribe { .. }) {
            return self.dispatch_inner(req, tx).await;
        }
        let envelope = Duration::from_secs(self.config.timeouts.dispatch_envelope_secs);
        let req_id = req.id().to_string();
        let req_kind = req.kind();
        let tx_for_timeout = tx.clone();
        let inner = self.clone().dispatch_inner(req, tx);
        match tokio::time::timeout(envelope, inner).await {
            Ok(result) => result,
            Err(_) => {
                warn!(
                    target: "assistd::state",
                    id = %req_id,
                    kind = req_kind,
                    timeout_secs = self.config.timeouts.dispatch_envelope_secs,
                    "dispatch envelope timeout exceeded; aborting request"
                );
                send_error(
                    &tx_for_timeout,
                    req_id,
                    format!(
                        "request exceeded {}s envelope timeout",
                        self.config.timeouts.dispatch_envelope_secs
                    ),
                )
                .await;
                Ok(())
            }
        }
    }

    async fn dispatch_inner(
        self: Arc<Self>,
        req: Request,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match req {
            Request::Query {
                id,
                text,
                attachments,
            } => return self.handle_query(id, text, attachments, tx).await,
            Request::SetPresence { id, target } => {
                return self.handle_set_presence(id, target, tx).await;
            }
            Request::Cycle { id } => return self.handle_cycle(id, tx).await,
            Request::PttStart { id } => return self.handle_ptt_start(id, tx).await,
            Request::PttStop { id } => return self.handle_ptt_stop(id, tx).await,
            Request::ListenStart { id } => return self.handle_listen_start(id, tx).await,
            Request::ListenStop { id } => return self.handle_listen_stop(id, tx).await,
            Request::ListenToggle { id } => return self.handle_listen_toggle(id, tx).await,
            Request::MemorySave { id, key, value } => {
                return self.handle_memory_save(id, key, value, tx).await;
            }
            Request::MemoryLoad { id, key } => return self.handle_memory_load(id, key, tx).await,
            Request::MemoryList { id, prefix } => {
                return self.handle_memory_list(id, prefix, tx).await;
            }
            Request::MemoryListAll { id, prefix, limit } => {
                return self.handle_memory_list_all(id, prefix, limit, tx).await;
            }
            Request::MemoryDelete { id, key } => {
                return self.handle_memory_delete(id, key, tx).await;
            }
            Request::MemoryForget { id, memory_id } => {
                return self.handle_memory_forget(id, memory_id, tx).await;
            }
            Request::MemorySemanticSearch { id, query, limit } => {
                return self
                    .handle_memory_semantic_search(id, query, limit, tx)
                    .await;
            }
            Request::MemoryReindex { id } => return self.handle_memory_reindex(id, tx).await,
            Request::GetPresence { id } => self.handle_get_presence(id, tx).await,
            Request::GetListenState { id } => self.handle_get_listen_state(id, tx).await,
            Request::VoiceToggle { id } => self.handle_voice_toggle(id, tx).await,
            Request::VoiceSkip { id } => self.handle_voice_skip(id, tx).await,
            Request::InterruptTurn { id } => self.handle_interrupt_turn(id, tx).await,
            Request::GetVoiceState { id } => self.handle_get_voice_state(id, tx).await,
            Request::GetCapabilities { id } => self.handle_get_capabilities(id, tx).await,
            Request::Fork { id, name } => self.handle_fork(id, name, tx).await,
            Request::Branches { id } => self.handle_branches(id, tx).await,
            Request::Switch { id, target } => self.handle_switch(id, target, tx).await,
            Request::Undo { id } => self.handle_undo(id, tx).await,
            Request::ResumeOrNew { id, recency_secs } => {
                self.handle_resume_or_new(id, recency_secs, tx).await
            }
            Request::NewSession { id } => self.handle_new_session(id, tx).await,
            Request::Subscribe { id, filter } => self.handle_subscribe(id, filter, tx).await,
            Request::ConfirmResponse { id, confirm_id, .. } => {
                send_error(
                    &tx,
                    id,
                    format!(
                        "ConfirmResponse(confirm_id={confirm_id}) received with no matching \
                         ConfirmRequest in flight on this connection"
                    ),
                )
                .await;
            }
        }
        Ok(())
    }
}

async fn send_error(tx: &mpsc::Sender<Event>, id: String, message: String) {
    let _ = tx.send(Event::Error { id, message }).await;
}

fn wire_role(role: PersistedRole) -> Role {
    match role {
        PersistedRole::System => Role::System,
        PersistedRole::User => Role::User,
        PersistedRole::Assistant => Role::Assistant,
        PersistedRole::Tool => Role::Tool,
    }
}

#[cfg(test)]
mod tests;

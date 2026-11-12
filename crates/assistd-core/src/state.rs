//! `AppState` and the request dispatcher; each submodule adds the
//! handlers for one family of `Request` variants.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::task::JoinError;
use tracing::warn;

use assistd_embed::EmbedError;
use assistd_ipc::{Event, EventBody, Request, RequestBody, Role, TaskRequest, WakeWordOp};
use assistd_llm::{BackendJudgeModel, LlmBackend, LlmError};
use assistd_memory::{MemoryError, PersistedRole};
use assistd_tools::{JUDGE_TIMEOUT, SafetyJudge, ToolRegistry};
use assistd_voice::{CaptureUnavailable, ListenError, VoiceInputError, VoiceManager};

use crate::planner::PlanError;
use crate::{Config, PresenceError, PresenceManager};

pub(crate) mod branches;
pub(crate) mod capabilities;
pub(crate) mod chat_focus;
pub(crate) mod context;
pub(crate) mod conversation;
pub(crate) mod event_triggers;
pub(crate) mod fire_times;
pub(crate) mod lanes;
pub(crate) mod mcp_handlers;
pub(crate) mod memory_handlers;
pub(crate) mod memory_stack;
pub(crate) mod path_watch;
pub(crate) mod persistence;
pub(crate) mod presence_handlers;
pub(crate) mod query;
pub(crate) mod readiness;
pub(crate) mod relay_trace;
pub(crate) mod run_lane;
pub(crate) mod runtime;
pub(crate) mod seq_ledger;
pub(crate) mod session_unlock;
pub(crate) mod subscribe;
pub(crate) mod subsystems;
pub(crate) mod task_announce;
pub(crate) mod task_confirms;
pub(crate) mod task_dispatch;
pub(crate) mod task_events;
pub(crate) mod task_handlers;
pub(crate) mod task_plans;
pub(crate) mod task_queue;
pub(crate) mod task_runner;
pub(crate) mod task_runs;
pub(crate) mod task_scheduler;
pub(crate) mod transcript_writer;
pub(crate) mod turn_admission;
pub(crate) mod voice_handlers;
pub(crate) mod wakeword_handlers;
pub(crate) mod wire;

pub use self::branches::branch_replay;
pub use self::fire_times::TriggerError;
pub use self::mcp_handlers::{McpAccountError, McpAccounts, McpLogin};
pub use self::memory_stack::MemoryStack;
pub use self::query::TurnOrigin;
pub use self::runtime::{BusSubscription, RuntimeState, SessionContext};
pub use self::subsystems::{McpServerStatus, Subsystems};
pub use self::task_plans::TaskPlanError;
pub use self::task_runner::TaskRunner;
pub use self::wire::AttachmentError;

/// Why a `SetPermissionMode` sent on its own connection is refused.
const PERMISSION_MODE_OFF_QUERY: &str = "SetPermissionMode is honored only on a Query connection";

/// Why a `PlanVerdict` sent as a connection's first request is refused.
const PLAN_VERDICT_OFF_QUERY: &str = "PlanVerdict answers a plan review on its Query connection";

/// Why a request handler failed.
#[derive(Debug, Error)]
pub enum DispatchError {
    #[error(transparent)]
    Presence(#[from] PresenceError),

    #[error("invalid attachment: {0}")]
    InvalidAttachment(#[from] AttachmentError),

    #[error(transparent)]
    InvalidTrigger(#[from] TriggerError),

    #[error("vision not available: model does not support images")]
    VisionUnsupported,

    #[error("auto mode is off (tools.allow_auto_mode), so a task cannot run in it")]
    AutoModeOff,

    #[error(transparent)]
    Llm(#[from] LlmError),

    #[error("llm backend panicked: {0}")]
    AgentPanicked(#[source] JoinError),

    #[error(transparent)]
    Memory(#[from] MemoryError),

    #[error(transparent)]
    Embed(#[from] EmbedError),

    #[error(transparent)]
    VoiceUnavailable(#[from] CaptureUnavailable),

    #[error(transparent)]
    VoiceInput(#[from] VoiceInputError),

    #[error(transparent)]
    Listen(#[from] ListenError),

    #[error(transparent)]
    TaskPlan(#[from] TaskPlanError),

    #[error(transparent)]
    Plan(#[from] PlanError),
}

/// Shared, long-lived daemon state handed to every request handler.
#[derive(Debug)]
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
        voice: Arc<VoiceManager>,
    ) -> Self {
        let subsystems = Subsystems::new(llm, presence, tools, voice);
        let memory = MemoryStack::disabled(config.embedding.clone());
        let runtime = RuntimeState::new();
        Self {
            config,
            subsystems,
            memory,
            runtime,
        }
    }

    /// The judge that answers for the user in Auto mode, when `tools.allow_auto_mode` allows it.
    pub fn safety_judge(&self) -> Option<SafetyJudge> {
        self.config.tools.allow_auto_mode.then(|| {
            let model = BackendJudgeModel::new(Arc::clone(&self.subsystems.llm));
            SafetyJudge::new(Arc::new(model), JUDGE_TIMEOUT)
        })
    }

    /// Route one request from the client running as `peer_pid` to its handler, streaming
    /// events on `tx` and none after returning. Every request except `Subscribe` is bounded
    /// by the dispatch envelope timeout.
    pub async fn dispatch(
        self: Arc<Self>,
        req: Request,
        peer_pid: Option<u32>,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        if Self::outlives_dispatch_envelope(&req) {
            return self.dispatch_inner(req, peer_pid, tx).await;
        }
        let envelope = Duration::from_secs(self.config.timeouts.dispatch_envelope_secs);
        let req_id = req.id.clone();
        let req_kind = req.kind();
        let tx_for_timeout = tx.clone();
        let inner = self.clone().dispatch_inner(req, peer_pid, tx);
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

    /// Requests the envelope timeout leaves alone: a subscription lives as
    /// long as its client, an agent or planning turn is bounded per step instead,
    /// a wake-word session by its own idle timeout, and an MCP login by its sign-in timeout.
    fn outlives_dispatch_envelope(req: &Request) -> bool {
        matches!(
            req.body,
            RequestBody::Subscribe { .. }
                | RequestBody::Query { .. }
                | RequestBody::Task(TaskRequest::Refine { .. })
                | RequestBody::PttStop
                | RequestBody::McpLogin { .. }
                | RequestBody::WakeWord {
                    op: WakeWordOp::TestStart
                }
        )
    }

    async fn dispatch_inner(
        self: Arc<Self>,
        req: Request,
        peer_pid: Option<u32>,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let Request { id, body } = req;
        match body {
            RequestBody::Query {
                text,
                attachments,
                plan,
            } => {
                self.handle_query(id, text, attachments, (TurnOrigin::Typed, plan), tx)
                    .await
            }
            RequestBody::SetPresence { target } => self.handle_set_presence(id, target, tx).await,
            RequestBody::PttStart => self.handle_ptt_start(id, tx).await,
            RequestBody::PttStop => self.handle_ptt_stop(id, tx).await,
            RequestBody::ListenStart => self.handle_listen_start(id, tx).await,
            RequestBody::ListenStop => self.handle_listen_stop(id, tx).await,
            RequestBody::ListenToggle => self.handle_listen_toggle(id, tx).await,
            RequestBody::WakeWord { op } => self.handle_wake_word(id, op, tx).await,
            RequestBody::MemorySave { key, value } => {
                self.handle_memory_save(id, key, value, tx).await
            }
            RequestBody::MemoryLoad { key } => self.handle_memory_load(id, key, tx).await,
            RequestBody::MemoryListAll { prefix, limit } => {
                self.handle_memory_list_all(id, prefix, limit, tx).await
            }
            RequestBody::MemoryDelete { key } => self.handle_memory_delete(id, key, tx).await,
            RequestBody::MemoryForget { memory_id } => {
                self.handle_memory_forget(id, memory_id, tx).await
            }
            RequestBody::MemorySemanticSearch { query, limit } => {
                self.handle_semantic_search(id, query, limit, tx).await
            }
            RequestBody::MemoryReindex => self.handle_memory_reindex(id, tx).await,
            RequestBody::GetPresence => replied(self.handle_get_presence(id, tx)).await,
            RequestBody::GetListenState => replied(self.handle_get_listen_state(id, tx)).await,
            RequestBody::VoiceToggle => replied(self.handle_voice_toggle(id, tx)).await,
            RequestBody::VoiceSkip => replied(self.handle_voice_skip(id, tx)).await,
            RequestBody::InterruptTurn => replied(self.handle_interrupt_turn(id, tx)).await,
            RequestBody::GetVoiceState => replied(self.handle_get_voice_state(id, tx)).await,
            RequestBody::GetCapabilities => replied(self.handle_get_capabilities(id, tx)).await,
            RequestBody::Fork { name } => replied(self.handle_fork(id, name, tx)).await,
            RequestBody::Branches => replied(self.handle_branches(id, tx)).await,
            RequestBody::Switch { target } => replied(self.handle_switch(id, target, tx)).await,
            RequestBody::Undo => replied(self.handle_undo(id, tx)).await,
            RequestBody::ResumeOrNew { recency_secs } => {
                replied(self.handle_resume_or_new(id, recency_secs, tx)).await
            }
            RequestBody::NewSession => replied(self.handle_new_session(id, tx)).await,
            RequestBody::ChatState { focused } => {
                replied(self.handle_chat_state(id, focused, peer_pid, tx)).await
            }
            RequestBody::ChatClosed => replied(self.handle_chat_closed(id, peer_pid, tx)).await,
            RequestBody::GetChatFocus => replied(self.handle_get_chat_focus(id, tx)).await,
            RequestBody::GetReadiness => replied(self.handle_get_readiness(id, tx)).await,
            RequestBody::Subscribe { filter } => {
                replied(self.handle_subscribe(id, filter, tx)).await
            }
            RequestBody::McpLogin { server } => {
                replied(self.handle_mcp_login(id, server, tx)).await
            }
            RequestBody::McpLogout { server } => {
                replied(self.handle_mcp_logout(id, server, tx)).await
            }
            RequestBody::McpStatus => replied(self.handle_mcp_status(id, tx)).await,
            RequestBody::Task(task) => self.dispatch_task_request(id, task, tx).await,
            RequestBody::ConfirmResponse { confirm_id, .. } => {
                replied(send_error(&tx, id, unmatched_confirm_response(&confirm_id))).await
            }
            RequestBody::SetPermissionMode { .. } => {
                replied(send_error(&tx, id, PERMISSION_MODE_OFF_QUERY.into())).await
            }
            RequestBody::PlanVerdict { .. } => {
                replied(send_error(&tx, id, PLAN_VERDICT_OFF_QUERY.into())).await
            }
        }
    }

    async fn dispatch_task_request(
        self: Arc<Self>,
        id: String,
        req: TaskRequest,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        match req {
            TaskRequest::List => self.handle_task_list(id, tx).await,
            TaskRequest::Save { task, spec } => self.handle_task_save(id, task, spec, tx).await,
            TaskRequest::Delete { task } => self.handle_task_delete(id, task, tx).await,
            TaskRequest::SetEnabled { task, enabled } => {
                self.handle_task_set_enabled(id, task, enabled, tx).await
            }
            TaskRequest::Run { task } => self.handle_task_run(id, task, tx).await,
            TaskRequest::Cancel { run } => replied(self.handle_task_cancel(id, run, tx)).await,
            TaskRequest::Queue { op } => replied(self.handle_task_queue(id, op, tx)).await,
            TaskRequest::Confirm {
                run,
                confirm_id,
                allow,
                always,
            } => {
                let approval = task_confirms::approval(allow, always);
                self.handle_task_confirm(id, run, confirm_id, approval, tx)
                    .await
            }
            TaskRequest::Runs { task } => self.handle_task_runs(id, task, tx).await,
            TaskRequest::Refine {
                prompt,
                mode,
                allowed_hosts,
                feedback,
            } => {
                let ask = task_plans::RefineAsk {
                    prompt,
                    mode,
                    allowed_hosts,
                    feedback,
                };
                self.handle_task_refine(id, ask, tx).await
            }
            TaskRequest::CheckTrigger { task, trigger } => {
                self.handle_task_check_trigger(id, task, trigger, tx).await
            }
        }
    }
}

fn unmatched_confirm_response(confirm_id: &str) -> String {
    format!(
        "ConfirmResponse(confirm_id={confirm_id}) received with no matching ConfirmRequest in \
         flight on this connection"
    )
}

async fn replied(reply: impl Future<Output = ()>) -> Result<(), DispatchError> {
    reply.await;
    Ok(())
}

async fn send_error(tx: &mpsc::Sender<Event>, id: String, message: String) {
    let _ = tx.send(Event::new(id, EventBody::Error { message })).await;
}

fn wire_role(role: PersistedRole) -> Role {
    match role {
        PersistedRole::User => Role::User,
        PersistedRole::Assistant => Role::Assistant,
        PersistedRole::Tool => Role::Tool,
    }
}

/// `at` as the RFC 3339 text (UTC, millisecond precision) the wire protocol carries.
fn wire_timestamp(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests;

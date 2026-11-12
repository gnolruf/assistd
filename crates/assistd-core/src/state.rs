//! `AppState` and the request dispatcher, which routes each family of `Request`
//! variants to the service that owns its state.

use std::future::Future;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};
use thiserror::Error;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinError;
use tracing::warn;

use assistd_embed::EmbedError;
use assistd_ipc::{Event, EventBody, Request, RequestBody, Role, TaskRequest, WakeWordOp};
use assistd_llm::{BackendJudgeModel, LlmBackend, LlmError, LlmHealthProbe};
use assistd_memory::{MemoryError, PersistedRole, SessionId};
use assistd_tools::{JUDGE_TIMEOUT, SafetyJudge, ToolRegistry, TurnContext};
use assistd_voice::{CaptureUnavailable, ListenError, VoiceInputError, VoiceManager};

use self::persistence::TranscriptLog;
use self::query::TurnDeps;
use self::task_service::TaskDeps;
use self::turn_admission::TurnAdmission;
use self::voice_handlers::HeardPrompt;
use crate::bus::{EventBus, EventSink};
use crate::planner::{PlanError, Planner};
use crate::presence::PresenceLlmHealthProbe;
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
pub(crate) mod seq_ledger;
pub(crate) mod session;
pub(crate) mod session_unlock;
pub(crate) mod status_service;
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
pub(crate) mod task_service;
pub(crate) mod transcript_writer;
pub(crate) mod turn_admission;
pub(crate) mod turn_driver;
pub(crate) mod voice_handlers;
pub(crate) mod wakeword_handlers;
pub(crate) mod wire;

pub use self::branches::{BranchService, branch_replay};
pub use self::fire_times::TriggerError;
pub use self::mcp_handlers::{McpAccountError, McpAccounts, McpHost, McpLogin, McpService};
pub use self::memory_stack::MemoryStack;
pub use self::query::{Query, TurnOrigin, TurnService};
pub use self::session::SessionContext;
pub use self::status_service::StatusService;
pub use self::subsystems::{McpServerStatus, Subsystems};
pub use self::task_confirms::TaskConfirmError;
pub use self::task_plans::TaskPlanError;
pub use self::task_queue::QueueOpError;
pub use self::task_runner::TaskRunner;
pub use self::task_service::TaskService;
pub use self::voice_handlers::VoiceService;
pub use self::wakeword_handlers::MicError;
pub use self::wire::AttachmentError;

/// Why a request handler failed. Its text is the message of the request's `Error` event.
#[derive(Debug, Error)]
pub enum DispatchError {
    /// Another failure, prefixed with the step of the request it happened in.
    #[error("{context}: {source}")]
    Context {
        context: &'static str,
        source: Box<DispatchError>,
    },

    #[error("request exceeded {0}s envelope timeout")]
    EnvelopeTimeout(u64),

    #[error(
        "ConfirmResponse(confirm_id={0}) received with no matching ConfirmRequest in flight on \
         this connection"
    )]
    UnmatchedConfirmResponse(String),

    #[error("SetPermissionMode is honored only on a Query connection")]
    PermissionModeOffQuery,

    #[error("PlanVerdict answers a plan review on its Query connection")]
    PlanVerdictOffQuery,

    #[error("/fork: name must not be empty")]
    EmptyBranchName,

    #[error("/resume: no branch named {0:?}")]
    UnknownBranch(String),

    #[error("MCP is disabled (mcp.enabled = false) or has no sandbox to run in")]
    McpDisabled,

    #[error(transparent)]
    McpAccount(#[from] McpAccountError),

    #[error(transparent)]
    Mic(#[from] MicError),

    #[error(transparent)]
    QueueOp(#[from] QueueOpError),

    #[error(transparent)]
    TaskConfirm(#[from] TaskConfirmError),

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

impl DispatchError {
    /// This failure with `context` in front of its message.
    pub(crate) fn context(self, context: &'static str) -> Self {
        Self::Context {
            context,
            source: Box::new(self),
        }
    }
}

/// Names the step of a request a failure happened in.
trait FailedStep<T> {
    fn context(self, context: &'static str) -> Result<T, DispatchError>;
}

impl<T, E: Into<DispatchError>> FailedStep<T> for Result<T, E> {
    fn context(self, context: &'static str) -> Result<T, DispatchError> {
        self.map_err(|e| e.into().context(context))
    }
}

/// The client behind a request. The default is no client at all: no process, and
/// nobody to answer a turn's prompts.
#[derive(Debug, Clone, Default)]
pub struct Requester {
    /// The client's process, when the socket reported one.
    pub pid: Option<u32>,
    /// What a turn the request starts runs in, its prompts answered over the connection.
    pub turn: TurnContext,
}

/// Shared, long-lived daemon state: the subsystems and stores, and the services that own
/// each request family's state.
#[derive(Debug)]
pub struct AppState {
    pub config: Arc<Config>,
    pub subsystems: Subsystems,
    pub memory: MemoryStack,
    /// The session and branch the user's conversation is on.
    pub session_ctx: Arc<SessionContext>,
    pub bus: EventBus,
    pub tasks: Arc<TaskService>,
    pub voice: VoiceService,
    pub branches: Arc<BranchService>,
    pub turns: TurnService,
    pub mcp: McpService,
    pub status: StatusService,
    transcript: Arc<TranscriptLog>,
}

impl AppState {
    /// An `AppState` with no-op memory and embedding backends, on a new, unsaved session
    /// with one model lane.
    pub fn new(
        config: Config,
        llm: Arc<dyn LlmBackend>,
        presence: Arc<PresenceManager>,
        tools: Arc<ToolRegistry>,
        voice: Arc<VoiceManager>,
    ) -> Self {
        let subsystems = Subsystems::new(llm, presence, tools, voice);
        let memory = MemoryStack::disabled(config.embedding.clone());
        let session_ctx = Arc::new(SessionContext::new(SessionId::new(), None));
        let one_slot = watch::channel(NonZeroU32::MIN).1;
        Self::from_parts(config, subsystems, memory, session_ctx, one_slot)
    }

    /// Assemble the state over `subsystems` and `memory`, on the session `session_ctx`
    /// points at, with as many model lanes as `slots`, the server's slot count, reports.
    pub fn from_parts(
        config: Config,
        subsystems: Subsystems,
        memory: MemoryStack,
        session_ctx: Arc<SessionContext>,
        slots: watch::Receiver<NonZeroU32>,
    ) -> Self {
        let config = Arc::new(config);
        let bus = EventBus::new();
        let admission = Arc::new(TurnAdmission::new(Arc::clone(&config), slots));
        let embed_wake = memory
            .embedding_cfg
            .enabled
            .then(|| Arc::clone(&memory.embed_wake));
        let transcript = Arc::new(TranscriptLog::new(
            Arc::clone(&memory.transcripts),
            embed_wake,
            Arc::clone(&session_ctx),
        ));
        let planner = Arc::new(planner(&config, &subsystems));
        let tasks = TaskService::new(TaskDeps {
            config: Arc::clone(&config),
            store: Arc::clone(&memory.tasks),
            transcript: Arc::clone(&transcript),
            admission: Arc::clone(&admission),
            session_ctx: Arc::clone(&session_ctx),
            bus: bus.clone(),
            llm: Arc::clone(&subsystems.llm),
            presence: Arc::clone(&subsystems.presence),
            vision_revalidator: subsystems.vision_revalidator.clone(),
            tools: Arc::clone(&subsystems.tools),
            planner: Arc::clone(&planner),
            slot_stash: subsystems.slot_stash.clone(),
            voice: Arc::clone(&subsystems.voice),
        });
        let voice = VoiceService::new(
            Arc::clone(&config),
            Arc::clone(&subsystems.voice),
            Arc::clone(&subsystems.presence),
            Arc::clone(&admission),
        );
        let branches = Arc::new(BranchService {
            session_ctx: Arc::clone(&session_ctx),
            transcripts: Arc::clone(&memory.transcripts),
            transcript: Arc::clone(&transcript),
            admission: Arc::clone(&admission),
            llm: Arc::clone(&subsystems.llm),
            presence: Arc::clone(&subsystems.presence),
            slot_stash: subsystems.slot_stash.clone(),
            bus: bus.clone(),
        });
        let turns = TurnService::new(TurnDeps {
            config: Arc::clone(&config),
            llm: Arc::clone(&subsystems.llm),
            presence: Arc::clone(&subsystems.presence),
            tools: Arc::clone(&subsystems.tools),
            planner,
            vision_revalidator: subsystems.vision_revalidator.clone(),
            voice: Arc::clone(&subsystems.voice),
            window_manager: Arc::clone(&subsystems.window_manager),
            embedder: Arc::clone(&memory.embedder),
            semantic: Arc::clone(&memory.semantic),
            embedding_cfg: memory.embedding_cfg.clone(),
            session_ctx: Arc::clone(&session_ctx),
            admission: Arc::clone(&admission),
            transcript: Arc::clone(&transcript),
            branches: Arc::clone(&branches),
            bus: bus.clone(),
        });
        let mcp = McpService {
            host: McpHost::new(
                Arc::clone(&subsystems.tools),
                Arc::clone(&subsystems.mcp_servers),
                bus.clone(),
            ),
            accounts: subsystems.mcp_accounts.clone(),
        };
        let status = StatusService {
            config: Arc::clone(&config),
            presence: Arc::clone(&subsystems.presence),
            voice: Arc::clone(&subsystems.voice),
            vision_revalidator: subsystems.vision_revalidator.clone(),
            mcp_servers: Arc::clone(&subsystems.mcp_servers),
            tools_disabled: subsystems.tools_disabled.clone(),
            web_search: Arc::clone(&subsystems.web_search),
            embedder: Arc::clone(&memory.embedder),
            admission: Arc::clone(&admission),
            tasks: Arc::clone(&tasks),
            bus: bus.clone(),
        };
        Self {
            config,
            subsystems,
            memory,
            session_ctx,
            bus,
            tasks,
            voice,
            branches,
            turns,
            mcp,
            status,
            transcript,
        }
    }

    /// Where the events `req` produces for `client` go: also onto the bus, unless the
    /// client is subscribing to it.
    pub fn events_of(&self, req: &Request, client: mpsc::Sender<Event>) -> EventSink {
        match req.body {
            RequestBody::Subscribe { .. } => EventSink::unpublished(client),
            _ => EventSink::to_client(self.bus.clone(), client),
        }
    }

    /// The judge that answers for the user in Auto mode, when `tools.allow_auto_mode` allows it.
    pub fn safety_judge(&self) -> Option<SafetyJudge> {
        safety_judge(&self.config, &self.subsystems.llm)
    }

    /// Wait for every queued transcript write and every background
    /// persistence task; none may be started afterwards.
    pub async fn finish_persistence(&self) {
        self.transcript.finish().await;
    }

    /// Route one request from `requester` to its handler, streaming events on `tx` and none
    /// after returning; every event but a subscriber's is published on the bus as it is
    /// produced. A failure is sent as the request's one `Error` event and returned. Requests
    /// other than the long-lived ones are bounded by the dispatch envelope timeout.
    pub async fn dispatch(
        self: Arc<Self>,
        req: Request,
        requester: Requester,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let tx = self.events_of(&req, tx);
        let id = req.id.clone();
        let result = self
            .dispatch_within_envelope(req, requester, tx.clone())
            .await;
        if let Err(e) = &result {
            let message = e.to_string();
            let _ = tx.send(Event::new(id, EventBody::Error { message })).await;
        }
        result
    }

    async fn dispatch_within_envelope(
        self: Arc<Self>,
        req: Request,
        requester: Requester,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        if Self::outlives_dispatch_envelope(&req) {
            return self.dispatch_inner(req, requester, tx).await;
        }
        let timeout_secs = self.config.timeouts.dispatch_envelope_secs;
        let req_id = req.id.clone();
        let req_kind = req.kind();
        let inner = self.dispatch_inner(req, requester, tx);
        tokio::time::timeout(Duration::from_secs(timeout_secs), inner)
            .await
            .unwrap_or_else(|_| {
                warn!(
                    target: "assistd::state",
                    id = %req_id,
                    kind = req_kind,
                    timeout_secs,
                    "dispatch envelope timeout exceeded; aborting request"
                );
                Err(DispatchError::EnvelopeTimeout(timeout_secs))
            })
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

    /// Transcribe the push-to-talk press being released and run what it heard as a turn.
    async fn handle_ptt_stop(&self, id: String, tx: EventSink) -> Result<(), DispatchError> {
        match self.voice.handle_ptt_stop(id, &tx).await? {
            Some(HeardPrompt { query, cancel }) => self.turns.run_query(query, tx, cancel).await,
            None => Ok(()),
        }
    }

    async fn dispatch_inner(
        self: Arc<Self>,
        req: Request,
        requester: Requester,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let Request { id, body } = req;
        let peer_pid = requester.pid;
        match body {
            RequestBody::Query {
                text,
                attachments,
                plan,
            } => {
                let query = Query::typed(id, text, attachments, plan, requester.turn);
                self.turns.handle_query(query, tx).await
            }
            RequestBody::SetPresence { target } => {
                self.status.handle_set_presence(id, target, tx).await
            }
            RequestBody::PttStart => self.voice.handle_ptt_start(id, tx).await,
            RequestBody::PttStop => self.handle_ptt_stop(id, tx).await,
            RequestBody::ListenStart => self.voice.handle_listen_start(id, tx).await,
            RequestBody::ListenStop => self.voice.handle_listen_stop(id, tx).await,
            RequestBody::ListenToggle => self.voice.handle_listen_toggle(id, tx).await,
            RequestBody::WakeWord { op } => self.voice.handle_wake_word(id, op, tx).await,
            RequestBody::MemorySave { key, value } => {
                self.memory.handle_memory_save(id, key, value, tx).await
            }
            RequestBody::MemoryLoad { key } => self.memory.handle_memory_load(id, key, tx).await,
            RequestBody::MemoryListAll { prefix, limit } => {
                self.memory
                    .handle_memory_list_all(id, prefix, limit, tx)
                    .await
            }
            RequestBody::MemoryDelete { key } => {
                self.memory.handle_memory_delete(id, key, tx).await
            }
            RequestBody::MemoryForget { memory_id } => {
                self.memory.handle_memory_forget(id, memory_id, tx).await
            }
            RequestBody::MemorySemanticSearch { query, limit } => {
                self.memory
                    .handle_semantic_search(id, query, limit, tx)
                    .await
            }
            RequestBody::MemoryReindex => self.memory.handle_memory_reindex(id, tx).await,
            RequestBody::GetPresence => replied(self.status.handle_get_presence(id, tx)).await,
            RequestBody::GetListenState => {
                replied(self.voice.handle_get_listen_state(id, tx)).await
            }
            RequestBody::VoiceToggle => replied(self.voice.handle_voice_toggle(id, tx)).await,
            RequestBody::VoiceSkip => replied(self.voice.handle_voice_skip(id, tx)).await,
            RequestBody::InterruptTurn => replied(self.voice.handle_interrupt_turn(id, tx)).await,
            RequestBody::GetVoiceState => replied(self.voice.handle_get_voice_state(id, tx)).await,
            RequestBody::GetCapabilities => {
                replied(self.status.handle_get_capabilities(id, tx)).await
            }
            RequestBody::Fork { name } => self.branches.handle_fork(id, name, tx).await,
            RequestBody::Branches => self.branches.handle_branches(id, tx).await,
            RequestBody::Switch { target } => self.branches.handle_switch(id, target, tx).await,
            RequestBody::Undo => self.branches.handle_undo(id, tx).await,
            RequestBody::ResumeOrNew { recency_secs } => {
                self.branches
                    .handle_resume_or_new(id, recency_secs, tx)
                    .await
            }
            RequestBody::NewSession => self.branches.handle_new_session(id, tx).await,
            RequestBody::ChatState { focused } => {
                replied(self.turns.handle_chat_state(id, focused, peer_pid, tx)).await
            }
            RequestBody::ChatClosed => {
                replied(self.turns.handle_chat_closed(id, peer_pid, tx)).await
            }
            RequestBody::GetChatFocus => replied(self.turns.handle_get_chat_focus(id, tx)).await,
            RequestBody::GetReadiness => replied(self.status.handle_get_readiness(id, tx)).await,
            RequestBody::Subscribe { filter } => {
                replied(self.status.handle_subscribe(id, filter, tx)).await
            }
            RequestBody::McpLogin { server } => self.mcp.handle_mcp_login(id, server, tx).await,
            RequestBody::McpLogout { server } => self.mcp.handle_mcp_logout(id, server, tx).await,
            RequestBody::McpStatus => replied(self.mcp.handle_mcp_status(id, tx)).await,
            RequestBody::Task(task) => self.tasks.dispatch(id, task, tx).await,
            RequestBody::ConfirmResponse { confirm_id, .. } => {
                Err(DispatchError::UnmatchedConfirmResponse(confirm_id))
            }
            RequestBody::SetPermissionMode { .. } => Err(DispatchError::PermissionModeOffQuery),
            RequestBody::PlanVerdict { .. } => Err(DispatchError::PlanVerdictOffQuery),
        }
    }
}

/// The planner every planning turn and task refine goes through.
fn planner(config: &Config, subsystems: &Subsystems) -> Planner {
    let health: Arc<dyn LlmHealthProbe> = Arc::new(PresenceLlmHealthProbe::new(Arc::clone(
        &subsystems.presence,
    )));
    Planner::new(
        subsystems.planning.clone(),
        Arc::clone(&subsystems.tools),
        Some(health),
        Duration::from_secs(config.timeouts.tool_call_secs),
        config.tools.allow_auto_mode,
    )
}

/// The judge that answers for the user in Auto mode, when `tools.allow_auto_mode` allows it.
fn safety_judge(config: &Config, llm: &Arc<dyn LlmBackend>) -> Option<SafetyJudge> {
    config.tools.allow_auto_mode.then(|| {
        let model = BackendJudgeModel::new(Arc::clone(llm));
        SafetyJudge::new(Arc::new(model), JUDGE_TIMEOUT)
    })
}

async fn replied(reply: impl Future<Output = ()>) -> Result<(), DispatchError> {
    reply.await;
    Ok(())
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

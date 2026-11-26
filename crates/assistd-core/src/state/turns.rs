//! `TurnService`: runs each user turn from its `Query` through admission, the agent loop,
//! persistence and the spoken reply.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::{JoinError, JoinHandle};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{Instrument, debug, warn};

use assistd_config::{Config, EmbeddingConfig};
use assistd_embed::EmbedderHandle;
use assistd_ipc::{Attachment, EventBody, ImageAttachment, PlanMode};
use assistd_llm::{LlmBackend, LlmError, LlmEvent, SlotAssignment, SlotHolder};
use assistd_memory::{PersistedMessage, RecallStore, SessionId};
use assistd_tools::{ConfirmRouter, HostReads, ToolCatalog, TurnContext, VoiceGate};
use assistd_voice::VoiceManager;
use assistd_wm::WindowManager;

use super::branches::BranchService;
use super::chat_focus::ChatFocusSlot;
use super::context::{combine_context_blocks, voice_approvals_note};
use super::host_reads::BranchHostReads;
use super::session_lapse::{SessionLapse, TurnInFlight};
use super::wire::decode_wire_attachments;
use super::{DispatchError, FailedStep, default_permission_mode};
use crate::admission::interaction::InteractionTurn;
use crate::admission::{MainLaneTurn, TurnAdmission};
use crate::agent::{Agent, AgentDefinition, AgentHost, AgentTask};
use crate::bus::{EventBus, EventSink};
use crate::documents::system_prompt;
use crate::judge::{BackendJudgeModel, safety_judge};
use crate::planner::Planner;
use crate::presence::{PresenceManager, RequestGuard};
use crate::recovery::{Component, spawn_supervised};
use crate::transcript::{PersistTarget, SessionContext, TranscriptLog};
use crate::turn_driver::{SpeechPipeline, TurnDriver, run_speech_worker};
use crate::vision::VisionRevalidator;
use crate::voice_gate::VoiceConfirmationGate;

/// One prompt to run as a user turn.
#[derive(Debug)]
pub struct Query {
    /// Tags every event of the turn.
    pub id: String,
    pub text: String,
    pub attachments: Vec<ImageAttachment>,
    pub origin: TurnOrigin,
    pub plan: PlanMode,
}

impl Query {
    /// A prompt typed by the client that answers `turn`'s prompts.
    pub(super) fn typed(
        id: String,
        text: String,
        attachments: Vec<ImageAttachment>,
        plan: PlanMode,
        turn: TurnContext,
    ) -> Self {
        Self {
            id,
            text,
            attachments,
            origin: TurnOrigin::Typed(turn),
            plan,
        }
    }
}

/// How a turn's prompt reached the daemon.
#[derive(Debug)]
pub enum TurnOrigin {
    /// Text from a client: the chat or another tool. The client answers
    /// the turn's prompts through this context.
    Typed(TurnContext),
    /// A push-to-talk or continuous-listening transcription. The turn is headless: no client
    /// shows its prompts, so they are asked out loud.
    Voice,
}

impl TurnOrigin {
    pub(super) fn is_voice(&self) -> bool {
        matches!(self, Self::Voice)
    }

    /// The context turn `id` runs in, a headless turn running in the default permission
    /// mode with its prompts going to the voice gate.
    fn into_turn_context(self, id: &str, deps: &TurnDeps) -> TurnContext {
        match self {
            Self::Typed(turn) => turn,
            Self::Voice => {
                let mode = default_permission_mode(&deps.config.tools);
                let judge_model = BackendJudgeModel::new(Arc::clone(&deps.llm));
                let judge = safety_judge(&deps.config, judge_model);
                let gate = Arc::clone(&deps.voice_gate) as Arc<dyn VoiceGate>;
                let router = ConfirmRouter::for_voice_turn(id.to_string(), mode, judge, gate);
                TurnContext::answered_by(router)
            }
        }
    }
}

/// A turn whose reply is complete but may still be being spoken, and what it holds until
/// the speech ends.
struct ReplySpeech<'turn> {
    worker: JoinHandle<()>,
    _daemon_awake: RequestGuard,
    _interaction: InteractionTurn,
    _session_in_use: TurnInFlight<'turn>,
}

impl ReplySpeech<'_> {
    async fn finished(self) {
        let _ = self.worker.await;
    }
}

/// What the turn service borrows from the rest of the daemon.
#[derive(Debug)]
pub(super) struct TurnDeps {
    pub(super) config: Arc<Config>,
    pub(super) llm: Arc<dyn LlmBackend>,
    pub(super) presence: Arc<PresenceManager>,
    pub(super) tools: Arc<ToolCatalog>,
    pub(super) agents: AgentHost,
    pub(super) planner: Arc<Planner>,
    pub(super) vision_revalidator: Option<Arc<VisionRevalidator>>,
    pub(super) voice: Arc<VoiceManager>,
    pub(super) voice_gate: Arc<VoiceConfirmationGate>,
    pub(super) window_manager: Arc<dyn WindowManager>,
    pub(super) embedder: Arc<EmbedderHandle>,
    pub(super) recall: Arc<dyn RecallStore>,
    pub(super) embedding_cfg: EmbeddingConfig,
    pub(super) session_ctx: Arc<SessionContext>,
    pub(super) admission: Arc<TurnAdmission>,
    pub(super) transcript: Arc<TranscriptLog>,
    pub(super) branches: Arc<BranchService>,
    pub(super) bus: EventBus,
}

/// Runs user turns and handles the chat's focus reports,
/// which shape how a turn replies.
#[derive(Debug)]
pub struct TurnService {
    pub(super) deps: TurnDeps,
    /// The chat's last focus report.
    pub(super) chat_focus: ChatFocusSlot,
    host_reads: BranchHostReads,
    session_lapse: SessionLapse,
}

impl TurnService {
    pub(super) fn new(deps: TurnDeps) -> Self {
        let lapse_secs = deps.config.voice.session_hold_secs.get();
        Self {
            session_lapse: SessionLapse::new(Duration::from_secs(lapse_secs.into())),
            deps,
            chat_focus: ChatFocusSlot::default(),
            host_reads: BranchHostReads::default(),
        }
    }

    /// Handle a single user query: wake the daemon if needed, run the agent loop under
    /// its plan mode, stream events through `tx`, and persist the turn. `Done` is sent
    /// when the reply is complete; returns once it has been spoken.
    pub async fn handle_query(&self, query: Query, tx: EventSink) -> Result<(), DispatchError> {
        let cancel = self.deps.admission.turn_cancellation();
        self.run_query(query, tx, cancel).await
    }

    /// [`Self::handle_query`] for a turn whose cancellation token was
    /// taken earlier; the prompt is dropped if `cancel` has already fired.
    pub(super) async fn run_query(
        &self,
        query: Query,
        tx: EventSink,
        cancel: CancellationToken,
    ) -> Result<(), DispatchError> {
        let id = query.id.clone();
        let speech = self.reply_to(query, &tx, cancel).await?;
        if speech.is_none() {
            debug!(
                target: "assistd::state",
                id = %id,
                "turn interrupted or abandoned before reaching the model; dropping the prompt"
            );
        }
        tx.reply(id, EventBody::Done).await;
        if let Some(speech) = speech {
            speech.finished().await;
        }
        Ok(())
    }

    /// Run `query`'s turn until its reply is complete; `None` when the turn was
    /// interrupted or its client left before it reached the model, its prompt dropped.
    async fn reply_to(
        &self,
        query: Query,
        tx: &EventSink,
        cancel: CancellationToken,
    ) -> Result<Option<ReplySpeech<'_>>, DispatchError> {
        let Query {
            id,
            text,
            attachments: wire_attachments,
            origin,
            plan,
        } = query;
        let _cancel_on_return = cancel.clone().drop_guard();
        let attachments = decode_wire_attachments(wire_attachments)?;
        let Some((daemon_awake, main_lane)) = self.admit_turn(&id, &text, tx, &cancel).await?
        else {
            return Ok(None);
        };
        let MainLaneTurn {
            lane: agent_guard,
            user_turn,
            turn: interaction,
        } = main_lane;
        if !self.revalidate_vision(tx, &cancel).await {
            return Ok(None);
        }
        self.require_vision_for(&attachments)?;
        if origin.is_voice() {
            self.leave_lapsed_session(&id, tx).await;
        }
        let session_in_use = self.session_lapse.turn();

        let (current_session, persist) = self.open_persistence_turn(&text).await;
        self.refresh_chat_focus(&id);
        self.assemble_transient_context(&text, &origin).await;

        let title_user_text = text.clone();
        let (llm_tx, llm_rx) = mpsc::channel::<LlmEvent>(32);
        let turn = origin
            .into_turn_context(&id, &self.deps)
            .reading_into(self.active_branch_host_reads().await);
        let agent_task =
            self.spawn_agent_task(turn, (text, attachments), plan, llm_tx, cancel.clone());

        let (speech, speech_rx) = SpeechPipeline::new(&self.deps.config.voice.synthesis);
        let voice_output = self.deps.voice.speech();
        voice_output.skip().await;
        let start_epoch = voice_output.current_epoch();
        let reply_speech = ReplySpeech {
            worker: self.spawn_speech_worker(id.clone(), start_epoch, speech_rx),
            _daemon_awake: daemon_awake,
            _interaction: interaction,
            _session_in_use: session_in_use,
        };

        let transcript = Arc::clone(&self.deps.transcript);
        let finished = TurnDriver::new(id.clone(), transcript, persist, tx.clone())
            .speaking(speech)
            .drive(llm_rx, &cancel)
            .await
            .is_some();

        let agent_result = agent_task.await;
        drop((agent_guard, user_turn));

        if let Err(e) = agent_outcome(agent_result) {
            reply_speech.finished().await;
            return Err(e);
        }
        if finished {
            self.deps
                .branches
                .spawn_session_title_broadcast(id, current_session, &title_user_text);
        }
        Ok(Some(reply_speech))
    }

    /// Begin a new session for a spoken prompt when the one before it has lapsed and no chat
    /// is open to show it.
    async fn leave_lapsed_session(&self, id: &str, tx: &EventSink) {
        if self.session_lapse.lapsed() && !self.chat_focus.open() {
            self.deps.branches.begin_session_for_turn(id, tx).await;
        }
    }

    fn require_vision_for(&self, attachments: &[Attachment]) -> Result<(), DispatchError> {
        let supported = self
            .deps
            .vision_revalidator
            .as_ref()
            .is_some_and(|revalidator| revalidator.gate().supported());
        if attachments.is_empty() || supported {
            Ok(())
        } else {
            Err(DispatchError::VisionUnsupported)
        }
    }

    /// The main lane and the daemon awake for query `id`'s turn; `None` when `cancel`
    /// fired, the client left or the turn was dropped from the queue first.
    async fn admit_turn(
        &self,
        id: &str,
        text: &str,
        tx: &EventSink,
        cancel: &CancellationToken,
    ) -> Result<Option<(RequestGuard, MainLaneTurn)>, DispatchError> {
        let admission = async {
            let Some(main_lane) = self.deps.admission.take_main_lane(id, text, tx).await else {
                return Ok(None);
            };
            let request_guard = self.acquire_request_guard(id, tx).await?;
            Ok(Some((request_guard, main_lane)))
        };
        tokio::select! {
            biased;
            () = cancel.cancelled() => Ok(None),
            () = tx.closed() => Ok(None),
            admitted = admission => admitted,
        }
    }

    /// Re-probe vision support if stale; `false` when `cancel` fired or the client left first.
    async fn revalidate_vision(&self, tx: &EventSink, cancel: &CancellationToken) -> bool {
        let Some(revalidator) = &self.deps.vision_revalidator else {
            return !cancel.is_cancelled();
        };
        tokio::select! {
            biased;
            () = cancel.cancelled() => false,
            () = tx.closed() => false,
            () = revalidator.revalidate_if_stale(&self.deps.presence) => true,
        }
    }

    async fn acquire_request_guard(
        &self,
        id: &str,
        tx: &EventSink,
    ) -> Result<RequestGuard, DispatchError> {
        self.deps
            .presence
            .acquire_request_guard_with_progress(id.to_string(), tx.clone())
            .await
            .context("wake failed")
    }

    /// Save the session if needed and queue the write opening the turn with
    /// the user's prompt; the target is `None` when the session could not be saved.
    async fn open_persistence_turn(&self, text: &str) -> (SessionId, Option<PersistTarget>) {
        match self.deps.transcript.saved_session().await {
            Ok((session, branch)) => {
                let target = self
                    .deps
                    .transcript
                    .open_user_turn(branch, PersistedMessage::user(text))
                    .await;
                (session, Some(target))
            }
            Err(e) => {
                warn!(
                    target: "assistd::memory",
                    error = %e,
                    "saving the session failed; turn will not be persisted"
                );
                let (session, _branch) = self.deps.session_ctx.current().await;
                (session, None)
            }
        }
    }

    async fn active_branch_host_reads(&self) -> HostReads {
        let (_session, branch) = self.deps.session_ctx.current().await;
        self.host_reads
            .of(branch, self.deps.transcript.store().as_ref())
            .await
    }

    async fn assemble_transient_context(&self, text: &str, origin: &TurnOrigin) {
        let semantic = if self.recalls_past_context() {
            match self.build_semantic_context(text).await {
                Ok(block) => block,
                Err(e) => {
                    debug!(
                        target: "assistd::embed",
                        error = %e,
                        "semantic context injection failed; continuing without it",
                    );
                    None
                }
            }
        } else {
            None
        };
        let window = self.build_window_context().await;
        let mode = default_permission_mode(&self.deps.config.tools);
        let approvals = voice_approvals_note(origin, mode);
        let delivery = self.spoken_reply_note();
        if let Some(block) = combine_context_blocks([semantic, window, approvals, delivery])
            && let Err(e) = self.deps.llm.set_transient_context(block).await
        {
            debug!(
                target: "assistd::context",
                error = %e,
                "set_transient_context failed; continuing without it",
            );
        }
    }

    fn recalls_past_context(&self) -> bool {
        let embedding = &self.deps.embedding_cfg;
        self.deps.config.memory.enabled && embedding.enabled && embedding.auto_inject
    }

    /// Run the agent of `turn` on the prompt under `plan`, starting it planning when asked to.
    fn spawn_agent_task(
        &self,
        turn: TurnContext,
        (text, attachments): (String, Vec<Attachment>),
        plan: PlanMode,
        llm_tx: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> AbortOnDropHandle<Result<(), LlmError>> {
        let agent = self.query_agent(&turn, plan);
        if plan == PlanMode::Planning
            && let Some(router) = turn.router()
        {
            router.set_planning(true);
        }
        AbortOnDropHandle::new(tokio::spawn(
            async move {
                let task = AgentTask { text, attachments };
                let outcome = agent.run(task, &turn, llm_tx, cancel).await;
                outcome.end.into_result()
            }
            .in_current_span(),
        ))
    }

    /// The agent of `turn` under `plan`: one that can move through plan mode unless it is off,
    /// and for a headless turn one whose plan is put to the user out loud.
    fn query_agent(&self, turn: &TurnContext, plan: PlanMode) -> Agent {
        let main = AgentDefinition::new(
            system_prompt(&self.deps.config),
            self.deps.tools.snapshot(),
            SlotAssignment::Shared(SlotHolder::Main),
        );
        if turn.asks_aloud() {
            return self.deps.planner.headless_turn_agent(turn, main);
        }
        if !plan.is_off() {
            return self.deps.planner.user_turn_agent(turn, main);
        }
        self.deps.agents.main(main)
    }

    fn spawn_speech_worker(
        &self,
        id: String,
        start_epoch: u64,
        speech_rx: mpsc::Receiver<String>,
    ) -> JoinHandle<()> {
        spawn_supervised(
            "speech_worker",
            Component::Voice,
            run_speech_worker(
                self.deps.voice.speech().clone(),
                self.deps.bus.sender().clone(),
                id,
                start_epoch,
                speech_rx,
            )
            .in_current_span(),
        )
    }
}

fn agent_outcome(
    agent_result: Result<Result<(), LlmError>, JoinError>,
) -> Result<(), DispatchError> {
    match agent_result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(DispatchError::from(e).context("llm backend error")),
        Err(join_err) => Err(DispatchError::AgentPanicked(join_err)),
    }
}

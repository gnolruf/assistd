//! `handle_query`: per-turn agent loop driver.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, mpsc};
use tokio::task::{JoinError, JoinHandle};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{Instrument, debug, warn};

use assistd_config::{Config, EmbeddingConfig};
use assistd_embed::EmbedderHandle;
use assistd_ipc::{Event, EventBody, ImageAttachment, PlanMode};
use assistd_llm::{LlmBackend, LlmError, LlmEvent, LlmHealthProbe};
use assistd_memory::{PersistedMessage, SemanticStore, SessionId};
use assistd_tools::{Attachment, ConfirmRouter, ToolCatalog, TurnContext};
use assistd_voice::{
    SpeakDecision, SpeakingGuard, VoiceManager, VoiceOutput, VoiceOutputController,
};
use assistd_wm::WindowManager;

use super::branches::BranchService;
use super::chat_focus::ChatFocusSlot;
use super::context::{combine_context_blocks, voice_approvals_note};
use super::persistence::TranscriptLog;
use super::session::SessionContext;
use super::transcript_writer::PersistTarget;
use super::turn_admission::{MainLaneTurn, TurnAdmission};
use super::turn_driver::{SpeechPipeline, TurnDriver};
use super::wire::decode_wire_attachments;
use super::{DispatchError, FailedStep};
use crate::Agent;
use crate::bus::{EventBus, EventSink};
use crate::planner::Planner;
use crate::presence::{PresenceLlmHealthProbe, PresenceManager, RequestGuard};
use crate::recovery::{Component, spawn_supervised};
use crate::vision::VisionRevalidator;

const SENTENCE_PREVIEW_CHARS: usize = 60;

/// One prompt to run as a turn of the user's conversation.
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
    /// Text from a client: the chat, `assistd query`, or another tool. The client answers
    /// the turn's prompts through this context.
    Typed(TurnContext),
    /// A push-to-talk or continuous-listening transcription, which nobody can answer
    /// prompts for.
    Voice,
}

impl TurnOrigin {
    pub(super) fn is_voice(&self) -> bool {
        matches!(self, Self::Voice)
    }

    /// The context turn `id` runs in.
    fn into_turn_context(self, id: &str) -> TurnContext {
        match self {
            Self::Typed(turn) => turn,
            Self::Voice => TurnContext::answered_by(ConfirmRouter::for_voice_turn(id.to_string())),
        }
    }
}

/// What the turn service borrows from the rest of the daemon.
#[derive(Debug)]
pub(super) struct TurnDeps {
    pub(super) config: Arc<Config>,
    pub(super) llm: Arc<dyn LlmBackend>,
    pub(super) presence: Arc<PresenceManager>,
    pub(super) tools: Arc<ToolCatalog>,
    pub(super) planner: Arc<Planner>,
    pub(super) vision_revalidator: Option<Arc<VisionRevalidator>>,
    pub(super) voice: Arc<VoiceManager>,
    pub(super) window_manager: Arc<dyn WindowManager>,
    pub(super) embedder: Arc<EmbedderHandle>,
    pub(super) semantic: Arc<dyn SemanticStore>,
    pub(super) embedding_cfg: EmbeddingConfig,
    pub(super) session_ctx: Arc<SessionContext>,
    pub(super) admission: Arc<TurnAdmission>,
    pub(super) transcript: Arc<TranscriptLog>,
    pub(super) branches: Arc<BranchService>,
    pub(super) bus: EventBus,
}

/// Runs the turns of the user's conversation and handles the chat's focus reports,
/// which shape how a turn replies.
#[derive(Debug)]
pub struct TurnService {
    pub(super) deps: TurnDeps,
    /// The chat's last focus report.
    pub(super) chat_focus: ChatFocusSlot,
}

impl TurnService {
    pub(super) fn new(deps: TurnDeps) -> Self {
        Self {
            deps,
            chat_focus: ChatFocusSlot::default(),
        }
    }

    /// Handle a single user query: wake the daemon if needed, run the agent loop under
    /// its plan mode, stream events through `tx`, and persist the turn.
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
        let Query {
            id,
            text,
            attachments: wire_attachments,
            origin,
            plan,
        } = query;
        let _cancel_on_return = cancel.clone().drop_guard();
        let attachments = decode_wire_attachments(wire_attachments)?;
        let Some((_request_guard, main_lane)) = self.admit_turn(&id, &text, &tx, &cancel).await?
        else {
            return finish_interrupted_before_start(&tx, id).await;
        };
        let MainLaneTurn {
            lane: agent_guard,
            user_turn,
            turn: _conversation_turn,
        } = main_lane;
        if !self.revalidate_vision(&cancel).await {
            return finish_interrupted_before_start(&tx, id).await;
        }
        self.require_vision_for(&attachments)?;

        let (current_session, persist) = self.open_persistence_turn(&text).await;
        let chat_focused = self.refresh_chat_focus(&id);
        self.assemble_transient_context(&text, &origin, chat_focused)
            .await;

        let title_user_text = text.clone();
        let (llm_tx, llm_rx) = mpsc::channel::<LlmEvent>(32);
        let turn = origin.into_turn_context(&id);
        let agent_task =
            self.spawn_agent_task(turn, (text, attachments), plan, llm_tx, cancel.clone());

        let (speech, speech_rx) = SpeechPipeline::new(&self.deps.config.voice.synthesis);
        let voice_output = self.deps.voice.speech();
        voice_output.skip().await;
        let start_epoch = voice_output.current_epoch();
        let speech_handle = self.spawn_speech_worker(id.clone(), start_epoch, speech_rx);

        let transcript = Arc::clone(&self.deps.transcript);
        let done_emitted = TurnDriver::new(id.clone(), transcript, persist, tx.clone())
            .speaking(speech)
            .drive(llm_rx, &cancel)
            .await
            .is_some();

        let agent_result = agent_task.await;
        drop((agent_guard, user_turn));

        if done_emitted && matches!(&agent_result, Ok(Ok(()))) {
            self.deps.branches.spawn_session_title_broadcast(
                id.clone(),
                current_session,
                &title_user_text,
            );
        }

        self.finalize_turn(id, agent_result, speech_handle, &tx, done_emitted)
            .await
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
    /// fired or the turn was dropped from the queue first.
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
            admitted = admission => admitted,
        }
    }

    /// Re-probe vision support if stale; `false` when `cancel` fired first.
    async fn revalidate_vision(&self, cancel: &CancellationToken) -> bool {
        let Some(revalidator) = &self.deps.vision_revalidator else {
            return !cancel.is_cancelled();
        };
        tokio::select! {
            biased;
            () = cancel.cancelled() => false,
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

    async fn assemble_transient_context(
        &self,
        text: &str,
        origin: &TurnOrigin,
        chat_focused: bool,
    ) {
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
        let approvals = voice_approvals_note(origin);
        let brief = self.brief_reply_note(origin, chat_focused);
        if let Some(block) = combine_context_blocks([semantic, window, approvals, brief])
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
                agent
                    .run_turn(&turn, text, attachments, llm_tx, cancel)
                    .await
            }
            .in_current_span(),
        ))
    }

    /// The agent of `turn` under `plan`: one that can move through plan mode unless it is off.
    fn query_agent(&self, turn: &TurnContext, plan: PlanMode) -> Agent {
        let llm = self.deps.llm.clone();
        let tools = self.deps.tools.snapshot();
        if !plan.is_off() {
            return self.deps.planner.chat_agent(turn, llm, &tools);
        }
        let health: Option<Arc<dyn LlmHealthProbe>> = Some(Arc::new(PresenceLlmHealthProbe::new(
            self.deps.presence.clone(),
        )));
        Agent::new(
            llm,
            tools,
            health,
            Duration::from_secs(self.deps.config.timeouts.tool_call_secs),
        )
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

    /// Wait for speech and report the agent's outcome. Sends `Done` if the
    /// stream did not, as a cancelled turn ends without one.
    async fn finalize_turn(
        &self,
        id: String,
        agent_result: Result<Result<(), LlmError>, JoinError>,
        speech_handle: JoinHandle<()>,
        tx: &EventSink,
        done_emitted: bool,
    ) -> Result<(), DispatchError> {
        let _ = speech_handle.await;

        match agent_result {
            Ok(Ok(())) => {
                if !done_emitted {
                    let _ = tx.send(Event::new(id, EventBody::Done)).await;
                }
                Ok(())
            }
            Ok(Err(e)) => Err(DispatchError::from(e).context("llm backend error")),
            Err(join_err) => Err(DispatchError::AgentPanicked(join_err)),
        }
    }
}

pub(super) async fn finish_interrupted_before_start(
    tx: &EventSink,
    id: String,
) -> Result<(), DispatchError> {
    debug!(
        target: "assistd::state",
        id = %id,
        "turn interrupted before reaching the model; dropping the prompt"
    );
    let _ = tx.send(Event::new(id, EventBody::Done)).await;
    Ok(())
}

pub(super) async fn run_speech_worker(
    voice_output: Arc<VoiceOutputController>,
    events_bus: broadcast::Sender<Event>,
    id: String,
    start_epoch: u64,
    mut speech_rx: mpsc::Receiver<String>,
) {
    let mut speaking: Option<SpeakingGuard> = None;
    while let Some(sentence) = speech_rx.recv().await {
        match voice_output.should_speak(start_epoch) {
            SpeakDecision::Speak => {
                let Ok(output) = voice_output.output() else {
                    continue;
                };
                if speaking.is_none() {
                    let _ = events_bus.send(Event::new(
                        id.clone(),
                        EventBody::SpeakingState { speaking: true },
                    ));
                    speaking = Some(voice_output.begin_speaking());
                }
                voice_output.remember_spoken(&sentence);
                speak_sentence(&voice_output, output.as_ref(), sentence, start_epoch).await;
            }
            SpeakDecision::DropSilent | SpeakDecision::DropForSkip => {}
        }
    }
    if let Ok(output) = voice_output.output()
        && let Err(e) = output.wait_idle().await
    {
        debug!(
            target: "assistd::voice",
            error = %e,
            "voice_output.wait_idle failed (non-fatal)"
        );
    }
    if speaking.take().is_some() {
        let _ = events_bus.send(Event::new(id, EventBody::SpeakingState { speaking: false }));
    }
}

/// Speak one sentence, then cancel playback if a skip landed meanwhile,
/// since `speak()` may append audio after the skip cleared the queue.
async fn speak_sentence(
    voice_output: &VoiceOutputController,
    output: &dyn VoiceOutput,
    sentence: String,
    start_epoch: u64,
) {
    let preview: String = sentence.chars().take(SENTENCE_PREVIEW_CHARS).collect();
    if let Err(e) = output.speak(sentence).await {
        warn!(
            target: "assistd::voice",
            error = %e,
            sentence_preview = %preview,
            "voice_output.speak failed; sentence dropped"
        );
    }
    if matches!(
        voice_output.should_speak(start_epoch),
        SpeakDecision::DropForSkip
    ) {
        output.cancel().await;
    }
}

//! Handlers for `TaskRun` and `TaskCancel`, and a saved task run as an
//! unattended turn in a conversation and transcript session of its own, its prompts approved
//! by the task's mode and hosts or paused for the user.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{Instrument, info, info_span, warn};

use assistd_ipc::{ConfirmSource, Event, EventBody, TaskId, TaskRunId, TaskTrigger};
use assistd_llm::{ConversationKey, LlmBackend, LlmError, LlmEvent, LlmHealthProbe};
use assistd_memory::{
    MemoryError, PersistedMessage, Result as MemoryResult, SessionId, TaskRecord, TaskRunEnd,
};
use assistd_tools::{ConfirmRouter, TurnContext};

use super::branches::delete_answer_files;
use super::event_triggers::Occurrence;
use super::lanes::LaneLease;
use super::run_lane::{LaneHold, RunLane, YieldingBackend};
use super::task_announce::TaskReply;
use super::task_confirms::TaskRunParking;
use super::task_queue::QueuedRun;
use super::task_runner::RegisteredRun;
use super::task_service::TaskService;
use super::transcript_writer::PersistTarget;
use super::turn_driver::TurnDriver;
use super::{DispatchError, FailedStep, safety_judge};
use crate::Agent;
use crate::bus::EventSink;
use crate::presence::{PresenceLlmHealthProbe, RequestGuard};

const EVENT_BUFFER: usize = 32;

/// Follows the main system prompt in every task run's conversation.
const TASK_PREAMBLE: &str = "You are running a saved task on your own. No user is present: \
     nobody will answer a question, so ask none. A command needing confirmation waits until \
     the user answers it later. Do the work with your tools, then make your final reply the \
     task's result, complete in itself.";

/// Why a run the daemon stopped mid-turn failed.
pub(super) const DAEMON_STOPPED: &str = "the daemon stopped before the run finished";

/// One run of a saved task: the request that queued it, the task, and the run's id.
struct TaskRunTurn {
    id: String,
    task: TaskRecord,
    run: TaskRunId,
}

impl TaskService {
    pub(super) async fn handle_task_run(
        &self,
        id: String,
        task: TaskId,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        let (record, run) = self.queue_task_run(task).await.context("task run failed")?;
        self.deps.admission.task_queue.push(run, record, id.clone());
        let _ = tx
            .send(Event::new(
                id.clone(),
                EventBody::TaskRunQueued { task, run },
            ))
            .await;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    pub(super) async fn handle_task_cancel(
        &self,
        id: String,
        run: TaskRunId,
        tx: EventSink,
    ) -> Result<(), DispatchError> {
        self.cancel_task_run(run).await?;
        let _ = tx.send(Event::new(id, EventBody::Done)).await;
        Ok(())
    }

    async fn queue_task_run(&self, task: TaskId) -> MemoryResult<(TaskRecord, TaskRunId)> {
        let record = self.deps.store.get(task).await?;
        let run = self.deps.store.queue_run(task, TaskTrigger::Manual).await?;
        Ok((record, run))
    }

    /// Start `queued` on the lane `lease` holds, in the background.
    pub(super) fn spawn_task_run(self: &Arc<Self>, queued: QueuedRun, lease: LaneLease) {
        let span = info_span!("task_run", task = %queued.task.id, run = %queued.run);
        let run = queued.run;
        let turn = TaskRunTurn {
            id: queued.request_id,
            task: queued.task,
            run,
        };
        let tasks = Arc::clone(self);
        self.runner.spawn(run, move |registered| {
            tasks.run_task(turn, lease, registered).instrument(span)
        });
    }

    /// Drive the run to its end, publishing its events as `TaskRun`, then give up
    /// its place in the queue and its lane.
    async fn run_task(
        self: Arc<Self>,
        turn: TaskRunTurn,
        lease: LaneLease,
        registered: RegisteredRun,
    ) {
        let events = EventSink::of_task_run(self.deps.bus.clone(), turn.run);
        let reply = self
            .drive_task_run(&turn, lease, &registered, &events)
            .await;
        self.deps.admission.task_queue.finish(turn.run);
        if let Some(reply) = reply {
            let announce = self.announce_task_reply(TaskReply {
                id: turn.id,
                run: turn.run,
                task_name: turn.task.spec.name,
                reply,
            });
            tokio::select! {
                () = registered.cancel.cancelled() => {}
                () = announce => {}
            }
        }
    }

    /// Run the turn, record how it ended, then end its event stream with `Error` for a
    /// failed run and `Done` for a cancelled one, whose turn sent none. Returns the final
    /// reply of a successful run whose task announces it.
    async fn drive_task_run(
        self: &Arc<Self>,
        turn: &TaskRunTurn,
        lease: LaneLease,
        registered: &RegisteredRun,
        tx: &EventSink,
    ) -> Option<String> {
        let outcome = self
            .task_run_turn(turn, lease, &registered.cancel, tx)
            .await;
        let end = run_end(outcome, registered);
        let id = turn.id.clone();
        let (terminal, reply) = match &end {
            TaskRunEnd::Succeeded { reply } => {
                (None, turn.task.spec.announce.then(|| reply.clone()))
            }
            TaskRunEnd::Failed { reason } => (
                Some(Event::new(
                    id,
                    EventBody::Error {
                        message: reason.clone(),
                    },
                )),
                None,
            ),
            TaskRunEnd::Cancelled => (Some(Event::new(id, EventBody::Done)), None),
        };
        self.record_run_end(turn, end).await;
        if let Some(terminal) = terminal {
            let _ = tx.send(terminal).await;
        }
        reply
    }

    /// Record how the run ended, then fire the triggers waiting on its task finishing
    /// unless it was cancelled.
    async fn record_run_end(&self, turn: &TaskRunTurn, end: TaskRunEnd) {
        info!(target: "assistd::tasks", end = ?end, "task run ended");
        let succeeded = match &end {
            TaskRunEnd::Succeeded { .. } => Some(true),
            TaskRunEnd::Failed { .. } => Some(false),
            TaskRunEnd::Cancelled => None,
        };
        if let Err(e) = self.deps.store.finish_run(turn.run, end).await {
            warn!(target: "assistd::tasks", error = %e, "recording how the task run ended failed");
            return;
        }
        self.prune_task_runs().await;
        if let Some(succeeded) = succeeded {
            let task = turn.task.id;
            self.fire_event_triggers(Occurrence::TaskFinished { task, succeeded })
                .await;
        }
    }

    /// Delete each task's finished runs beyond those it keeps, `tasks.kept_runs` unless it
    /// names its own, with their sessions, sparing
    /// the session the user is in; skipped while a user turn or switch holds it.
    pub(super) async fn prune_task_runs(&self) {
        let Ok(_user_turn) = self.deps.admission.user_turn_lock.clone().try_lock_owned() else {
            return;
        };
        let (session, _branch) = self.deps.session_ctx.current().await;
        match self
            .deps
            .store
            .prune_runs(self.deps.config.tasks.kept_runs.get(), session)
            .await
        {
            Ok(pruned) => {
                if pruned.runs > 0 {
                    info!(target: "assistd::tasks", runs = pruned.runs, "pruned old task runs");
                }
                delete_answer_files(pruned.released_answer_files).await;
            }
            Err(e) => warn!(target: "assistd::tasks", error = %e, "pruning old task runs failed"),
        }
    }

    /// The run's final reply, `None` when it was cancelled or ended without one.
    async fn task_run_turn(
        self: &Arc<Self>,
        turn: &TaskRunTurn,
        lease: LaneLease,
        cancel: &CancellationToken,
        tx: &EventSink,
    ) -> Result<Option<String>, DispatchError> {
        let Some(awake) = self.wake_for_task_run(&turn.id, tx, cancel).await? else {
            return Ok(None);
        };
        let system_prompt = task_system_prompt(&self.deps.config.chat.system_prompt);
        let key = ConversationKey::TaskRun(turn.run);
        let conversation = self
            .deps
            .llm
            .isolated_conversation(system_prompt, lease.slot(key))?;
        let lane = Arc::new(RunLane {
            tasks: Arc::clone(self),
            task: turn.task.id,
            run: turn.run,
            request_id: turn.id.clone(),
            held: LaneHold::new(lease, awake),
            backend: Arc::clone(&conversation),
            events: tx.clone(),
            cancel: cancel.clone(),
        });
        let persist = self.open_task_run_session(turn).await;
        let outcome = self
            .unattended_turn(turn, conversation, Arc::clone(&lane), persist, cancel, tx)
            .await;
        self.end_run_conversation(key, lane.on_main_lane()).await;
        outcome
    }

    /// Drop the ended run's slot state and, when it held the main lane, restore the main
    /// conversation's while the daemon is still idle.
    async fn end_run_conversation(&self, run: ConversationKey, on_main_lane: bool) {
        if let Some(stash) = &self.deps.slot_stash {
            stash.end(run).await;
            if on_main_lane {
                stash.switch_to(ConversationKey::Main).await;
            }
        }
    }

    /// Wake the model as a query does and re-probe it if it reloaded; `None` when
    /// `cancel` fired first.
    pub(super) async fn wake_for_task_run(
        &self,
        id: &str,
        tx: &EventSink,
        cancel: &CancellationToken,
    ) -> Result<Option<RequestGuard>, DispatchError> {
        let awake = async {
            let request_guard = self
                .deps
                .presence
                .acquire_request_guard_with_progress(id.to_string(), tx.clone())
                .await?;
            if let Some(revalidator) = &self.deps.vision_revalidator {
                revalidator.revalidate_if_stale(&self.deps.presence).await;
            }
            Ok(request_guard)
        };
        tokio::select! {
            biased;
            () = cancel.cancelled() => Ok(None),
            awake = awake => awake.map(Some),
        }
    }

    /// Save a new session for the run, mark the run as running in it, and queue the
    /// write opening its turn with the task's prompt; `None` when the session could not
    /// be saved, so the run is not persisted.
    async fn open_task_run_session(&self, turn: &TaskRunTurn) -> Option<PersistTarget> {
        let session = SessionId::new();
        let opened = async {
            let branch = self
                .deps
                .transcript
                .store()
                .begin_session_with_main_branch(session)
                .await?;
            self.deps.store.start_run(turn.run, session).await?;
            Ok::<_, MemoryError>(branch)
        };
        match opened.await {
            Ok(branch) => {
                let prompt = PersistedMessage::user(turn.task.spec.prompt.as_str());
                Some(
                    self.deps
                        .transcript
                        .open_task_run_turn(branch, prompt)
                        .await,
                )
            }
            Err(e) => {
                warn!(
                    target: "assistd::tasks",
                    error = %e,
                    "saving the task run's session failed; the run will not be persisted"
                );
                None
            }
        }
    }

    /// Run the task's prompt in `conversation` on `lane`, persisting to `persist` and sending
    /// its wire events to `tx`, with nothing spoken; returns the final reply once the turn
    /// emitted `Done`.
    async fn unattended_turn(
        &self,
        turn: &TaskRunTurn,
        conversation: Arc<dyn LlmBackend>,
        lane: Arc<RunLane>,
        persist: Option<PersistTarget>,
        cancel: &CancellationToken,
        tx: &EventSink,
    ) -> Result<Option<String>, DispatchError> {
        let (llm_tx, llm_rx) = mpsc::channel(EVENT_BUFFER);
        let agent_task = self.spawn_task_agent(turn, conversation, lane, llm_tx, cancel.clone());
        let transcript = Arc::clone(&self.deps.transcript);
        let reply = TurnDriver::new(turn.id.clone(), transcript, persist, tx.clone())
            .drive(llm_rx, cancel)
            .await;
        match agent_task.await {
            Ok(Ok(())) => Ok(reply),
            Ok(Err(e)) => Err(e.into()),
            Err(join_err) => Err(DispatchError::AgentPanicked(join_err)),
        }
    }

    /// The main agent's tools over `conversation`, each model step giving way to the user's
    /// conversation on `lane`, with every confirmation labelled as the run's and approved by
    /// the task's mode or hosts, or parked until answered.
    fn spawn_task_agent(
        &self,
        turn: &TaskRunTurn,
        conversation: Arc<dyn LlmBackend>,
        lane: Arc<RunLane>,
        llm_tx: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> AbortOnDropHandle<Result<(), LlmError>> {
        let health: Option<Arc<dyn LlmHealthProbe>> = Some(Arc::new(PresenceLlmHealthProbe::new(
            self.deps.presence.clone(),
        )));
        let backend = Arc::new(YieldingBackend {
            inner: conversation,
            lane: Arc::clone(&lane),
        });
        let agent = Agent::new(
            backend,
            self.deps.tools.snapshot(),
            health,
            Duration::from_secs(self.deps.config.timeouts.tool_call_secs),
        );
        let source = ConfirmSource::Task {
            task: turn.task.id,
            run: turn.run,
        };
        let parking = TaskRunParking {
            lane: Arc::clone(&lane),
        };
        let spec = &turn.task.spec;
        let router = ConfirmRouter::for_task_run(
            turn.id.clone(),
            source,
            spec.mode,
            safety_judge(&self.deps.config, &self.deps.llm),
            spec.allowed_hosts.clone(),
            Arc::new(parking),
        );
        let prompt = turn.task.spec.prompt.as_str().to_string();
        let context = TurnContext::answered_by(router).on_lane(lane);
        let agent_turn = async move {
            agent
                .run_turn(&context, prompt, Vec::new(), llm_tx, cancel)
                .await
        };
        AbortOnDropHandle::new(tokio::spawn(agent_turn.in_current_span()))
    }
}

/// The main system prompt followed by [`TASK_PREAMBLE`].
pub(super) fn task_system_prompt(main: &str) -> String {
    if main.trim().is_empty() {
        TASK_PREAMBLE.to_string()
    } else {
        format!("{main}\n\n{TASK_PREAMBLE}")
    }
}

/// How a run whose turn ended with `outcome` finished. A run cut short by shutdown
/// without being cancelled on request failed.
fn run_end(
    outcome: Result<Option<String>, DispatchError>,
    registered: &RegisteredRun,
) -> TaskRunEnd {
    match outcome {
        Ok(Some(reply)) => TaskRunEnd::Succeeded { reply },
        _ if registered.cancel_requested() => TaskRunEnd::Cancelled,
        _ if registered.cancel.is_cancelled() => TaskRunEnd::Failed {
            reason: DAEMON_STOPPED.into(),
        },
        Ok(None) => TaskRunEnd::Failed {
            reason: "the run ended without a reply".into(),
        },
        Err(e) => TaskRunEnd::Failed {
            reason: e.to_string(),
        },
    }
}

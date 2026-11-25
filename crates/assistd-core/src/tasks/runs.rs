//! Handlers for `TaskRun` and `TaskCancel`, and a saved task run as an
//! unattended turn on a backend and transcript session of its own, its prompts approved
//! by the task's mode and hosts or paused for the user.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{Instrument, info, info_span, warn};

use assistd_config::Config;
use assistd_ipc::{ConfirmSource, EventBody, TaskEvent, TaskId, TaskRunId, TaskTrigger};
use assistd_llm::{LlmBackend, LlmError, LlmEvent, SlotHolder};
use assistd_memory::{
    MemoryError, PersistedMessage, Result as MemoryResult, SessionId, TaskRecord, TaskRunEnd,
};
use assistd_tools::{ConfirmRouter, HostReads, TurnContext};

use super::announce::TaskReply;
use super::confirms::TaskRunParking;
use super::event_triggers::Occurrence;
use super::run_lane::{LaneHold, RunLane, YieldingBackend};
use super::runner::RegisteredRun;
use super::{FailedStep, TaskError, TaskService};
use crate::admission::lanes::LaneLease;
use crate::admission::queue::QueuedRun;
use crate::bus::EventSink;
use crate::documents::system_prompt;
use crate::judge::{BackendJudgeModel, safety_judge};
use crate::presence::RequestGuard;
use crate::relay::delete_answer_files;
use crate::transcript::PersistTarget;
use crate::turn_driver::TurnDriver;

const EVENT_BUFFER: usize = 32;

/// Follows the main system prompt in every task run's history.
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
    ) -> Result<(), TaskError> {
        let (record, run) = self.queue_task_run(task).await.context("task run failed")?;
        self.deps.admission.task_queue.push(run, record, id.clone());
        tx.reply(id, EventBody::TaskRunQueued { task, run }).await;
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

    /// Drive the run to its end, then give up its place in the queue and its lane.
    async fn run_task(
        self: Arc<Self>,
        turn: TaskRunTurn,
        lease: LaneLease,
        registered: RegisteredRun,
    ) {
        let reply = self.drive_task_run(&turn, lease, &registered).await;
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

    /// Run the turn and record how it ended. Returns the final reply of a successful run
    /// whose task announces it.
    async fn drive_task_run(
        self: &Arc<Self>,
        turn: &TaskRunTurn,
        lease: LaneLease,
        registered: &RegisteredRun,
    ) -> Option<String> {
        let outcome = self.task_run_turn(turn, lease, &registered.cancel).await;
        let end = run_end(outcome, registered);
        let reply = match &end {
            TaskRunEnd::Succeeded { reply } if turn.task.spec.announce => Some(reply.clone()),
            _ => None,
        };
        self.record_run_end(turn, end).await;
        reply
    }

    /// Have the task's path watches drop the run's own changes, record how the run ended,
    /// then fire the triggers waiting on its task finishing unless it was cancelled.
    async fn record_run_end(&self, turn: &TaskRunTurn, end: TaskRunEnd) {
        info!(target: "assistd::tasks", end = ?end, "task run ended");
        self.events.note_run_ended(turn.task.id);
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
    ) -> Result<Option<String>, TaskError> {
        let Some(awake) = self.wake_for_task_run(None, cancel).await? else {
            return Ok(None);
        };
        let system_prompt = task_system_prompt(&self.deps.config);
        let holder = SlotHolder::TaskRun(turn.run);
        let backend = self
            .deps
            .llm
            .isolated_backend(system_prompt, lease.slot(holder))?;
        let lane = Arc::new(RunLane {
            tasks: Arc::clone(self),
            task: turn.task.id,
            run: turn.run,
            held: LaneHold::new(lease, awake),
            backend: Arc::clone(&backend),
            cancel: cancel.clone(),
        });
        let persist = self.open_task_run_session(turn).await;
        let host_reads = self.host_reads_at_start(turn.run).await;
        let outcome = self
            .unattended_turn(
                turn,
                backend,
                Arc::clone(&lane),
                persist,
                host_reads,
                cancel,
            )
            .await;
        self.end_run_backend(holder, lane.on_main_lane()).await;
        outcome
    }

    /// Drop the ended run's slot state and, when it held the main lane, restore the main
    /// backend's while the daemon is still idle.
    async fn end_run_backend(&self, run: SlotHolder, on_main_lane: bool) {
        if let Some(stash) = &self.deps.slot_stash {
            stash.end(run).await;
            if on_main_lane {
                stash.switch_to(SlotHolder::Main).await;
            }
        }
    }

    /// Wake the model as a query does and re-probe it if it reloaded, reporting a slow
    /// load to `progress`, a request id and its sink, when given; `None` when `cancel`
    /// fired first.
    pub(super) async fn wake_for_task_run(
        &self,
        progress: Option<(&str, &EventSink)>,
        cancel: &CancellationToken,
    ) -> Result<Option<RequestGuard>, TaskError> {
        let awake = async {
            let presence = &self.deps.presence;
            let request_guard = match progress {
                Some((id, tx)) => {
                    presence
                        .acquire_request_guard_with_progress(id.to_string(), tx.clone())
                        .await?
                }
                None => presence.acquire_request_guard().await?,
            };
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

    /// What run `run`'s history counts as having read before its first step: this machine
    /// when a changed path started it, since whoever wrote there chose what the run reads,
    /// and when what started it cannot be told.
    async fn host_reads_at_start(&self, run: TaskRunId) -> HostReads {
        let started_by_path = match self.deps.store.run_trigger(run).await {
            Ok(trigger) => matches!(
                trigger,
                TaskTrigger::Event {
                    event: TaskEvent::PathChanged { .. },
                    ..
                }
            ),
            Err(e) => {
                warn!(
                    target: "assistd::tasks",
                    error = %e,
                    "reading what started the run failed; treating it as started by a changed path"
                );
                true
            }
        };
        HostReads::known(started_by_path)
    }

    /// Run the task's prompt over `backend` on `lane`, persisting to `persist`, with
    /// nothing sent to a client or spoken and its history starting from `host_reads`;
    /// returns the final reply once the turn finished.
    async fn unattended_turn(
        &self,
        turn: &TaskRunTurn,
        backend: Arc<dyn LlmBackend>,
        lane: Arc<RunLane>,
        persist: Option<PersistTarget>,
        host_reads: HostReads,
        cancel: &CancellationToken,
    ) -> Result<Option<String>, TaskError> {
        let (llm_tx, llm_rx) = mpsc::channel(EVENT_BUFFER);
        let agent_task =
            self.spawn_task_agent(turn, backend, lane, host_reads, llm_tx, cancel.clone());
        let transcript = Arc::clone(&self.deps.transcript);
        let unheard = EventSink::discarding();
        let reply = TurnDriver::new(turn.id.clone(), transcript, persist, unheard)
            .drive(llm_rx, cancel)
            .await;
        match agent_task.await {
            Ok(Ok(())) => Ok(reply),
            Ok(Err(e)) => Err(e.into()),
            Err(join_err) => Err(TaskError::AgentPanicked(join_err)),
        }
    }

    /// The main agent's tools over `backend`, its history starting from `host_reads`, each
    /// model step giving way to the user's interaction on `lane`, every confirmation labelled
    /// as the run's and approved by the task's mode or hosts, or parked until answered.
    fn spawn_task_agent(
        &self,
        turn: &TaskRunTurn,
        backend: Arc<dyn LlmBackend>,
        lane: Arc<RunLane>,
        host_reads: HostReads,
        llm_tx: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> AbortOnDropHandle<Result<(), LlmError>> {
        let backend = Arc::new(YieldingBackend {
            inner: backend,
            lane: Arc::clone(&lane),
        });
        let agent = self.deps.agents.agent(backend, self.deps.tools.snapshot());
        let source = ConfirmSource::Task {
            task: turn.task.id,
            run: turn.run,
        };
        let parking = TaskRunParking {
            lane: Arc::clone(&lane),
        };
        let spec = &turn.task.spec;
        let prompt = spec.prompt.as_str().to_string();
        let judge_model = BackendJudgeModel::for_prompt(Arc::clone(&self.deps.llm), prompt.clone());
        let router = ConfirmRouter::for_task_run(
            turn.id.clone(),
            source,
            spec.mode,
            safety_judge(&self.deps.config, judge_model),
            spec.allowed_hosts.clone(),
            Arc::new(parking),
        );
        let context = TurnContext::answered_by(router)
            .on_lane(lane)
            .reading_into(host_reads);
        let agent_turn = async move {
            agent
                .run_turn(&context, prompt, Vec::new(), llm_tx, cancel)
                .await
        };
        AbortOnDropHandle::new(tokio::spawn(agent_turn.in_current_span()))
    }
}

/// The main system prompt followed by [`TASK_PREAMBLE`].
pub(super) fn task_system_prompt(config: &Config) -> String {
    let main = system_prompt(config);
    if main.trim().is_empty() {
        TASK_PREAMBLE.to_string()
    } else {
        format!("{main}\n\n{TASK_PREAMBLE}")
    }
}

/// How a run whose turn ended with `outcome` finished. A run cut short by shutdown
/// without being cancelled on request failed.
fn run_end(outcome: Result<Option<String>, TaskError>, registered: &RegisteredRun) -> TaskRunEnd {
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

//! Handlers for `TaskRun` and `TaskCancel`: a saved task run as an unattended
//! turn in a conversation and transcript session of its own.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedMutexGuard, mpsc};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{Instrument, info, info_span, warn};

use assistd_ipc::{ConfirmSource, Event, TaskId, TaskRunId, TaskTrigger};
use assistd_llm::{LlmBackend, LlmError, LlmEvent, LlmHealthProbe};
use assistd_memory::{
    MemoryError, PersistedMessage, Result as MemoryResult, SessionId, TaskRecord, TaskRunEnd,
};
use assistd_tools::{CONFIRM_ROUTER, ConfirmRouter};
use assistd_voice::SentenceBuffer;

use super::persistence::PersistTarget;
use super::query::{TurnScope, TurnTranslator};
use super::task_runner::RegisteredRun;
use super::{AppState, DispatchError, send_error};
use crate::Agent;
use crate::presence::{PresenceLlmHealthProbe, RequestGuard};

const EVENT_BUFFER: usize = 32;

/// Follows the main system prompt in every task run's conversation.
const TASK_PREAMBLE: &str = "You are running a saved task on your own. No user is present: \
     nobody will answer a question or confirm a command, so ask neither. Do the work with \
     your tools, then make your final reply the task's result, complete in itself.";

const DAEMON_STOPPED: &str = "the daemon stopped before the run finished";

/// One run of a saved task: the request that queued it, the task, and the run's id.
struct TaskRunTurn {
    id: String,
    task: TaskRecord,
    run: TaskRunId,
}

impl AppState {
    pub(super) async fn handle_task_run(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let (record, run) = match self.queue_task_run(task).await {
            Ok(queued) => queued,
            Err(e) => {
                send_error(&tx, id, format!("task run failed: {e}")).await;
                return Err(e.into());
            }
        };
        let turn = TaskRunTurn {
            id: id.clone(),
            task: record,
            run,
        };
        let span = info_span!("task_run", %task, %run);
        let state = Arc::clone(&self);
        self.runtime.task_runner.spawn(run, move |registered| {
            state.run_task(turn, registered).instrument(span)
        });
        let _ = tx
            .send(Event::TaskRunQueued {
                id: id.clone(),
                task,
                run,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    pub(super) async fn handle_task_cancel(
        self: Arc<Self>,
        id: String,
        run: TaskRunId,
        tx: mpsc::Sender<Event>,
    ) {
        if self.runtime.task_runner.cancel(run) {
            let _ = tx.send(Event::Done { id }).await;
        } else {
            send_error(&tx, id, format!("no task run {run} is queued or running")).await;
        }
    }

    async fn queue_task_run(&self, task: TaskId) -> MemoryResult<(TaskRecord, TaskRunId)> {
        let record = self.memory.tasks.get(task).await?;
        let run = self
            .memory
            .tasks
            .queue_run(task, TaskTrigger::Manual)
            .await?;
        Ok((record, run))
    }

    /// Drive the run to its end while publishing its events as `TaskRun`.
    async fn run_task(self: Arc<Self>, turn: TaskRunTurn, registered: RegisteredRun) {
        let (tx, rx) = mpsc::channel(EVENT_BUFFER);
        tokio::join!(
            self.drive_task_run(&turn, &registered, tx),
            self.publish_task_run_events(TurnScope::TaskRun(turn.run), rx)
        );
    }

    async fn publish_task_run_events(&self, scope: TurnScope, mut rx: mpsc::Receiver<Event>) {
        while let Some(event) = rx.recv().await {
            self.runtime.publish(&scope.wrap(event));
        }
    }

    /// Run the turn, record how it ended, then end its event stream with `Error` for a
    /// failed run and `Done` for a cancelled one, whose turn sent none.
    async fn drive_task_run(
        &self,
        turn: &TaskRunTurn,
        registered: &RegisteredRun,
        tx: mpsc::Sender<Event>,
    ) {
        let outcome = self.task_run_turn(turn, &registered.cancel, &tx).await;
        let end = run_end(outcome, registered);
        let id = turn.id.clone();
        let terminal = match &end {
            TaskRunEnd::Succeeded { .. } => None,
            TaskRunEnd::Failed { reason } => Some(Event::Error {
                id,
                message: reason.clone(),
            }),
            TaskRunEnd::Cancelled => Some(Event::Done { id }),
        };
        info!(target: "assistd::tasks", end = ?end, "task run ended");
        if let Err(e) = self.memory.tasks.finish_run(turn.run, end).await {
            warn!(target: "assistd::tasks", error = %e, "recording how the task run ended failed");
        }
        if let Some(terminal) = terminal {
            let _ = tx.send(terminal).await;
        }
    }

    /// The run's final reply, `None` when it was cancelled or ended without one.
    async fn task_run_turn(
        &self,
        turn: &TaskRunTurn,
        cancel: &CancellationToken,
        tx: &mpsc::Sender<Event>,
    ) -> Result<Option<String>, DispatchError> {
        let Some(_admitted) = self.admit_task_run(&turn.id, tx, cancel).await? else {
            return Ok(None);
        };
        let system_prompt = task_system_prompt(&self.config.chat.system_prompt);
        let backend = self.subsystems.llm.isolated_conversation(system_prompt)?;
        let persist = self.open_task_run_session(turn).await;
        let unattended = self.unattended_turn(turn, backend, persist, cancel, tx);
        match &self.subsystems.main_slot {
            Some(main_slot) => main_slot.preserve_across(unattended).await,
            None => unattended.await,
        }
    }

    /// Take the agent turn lock and wake the model as a query does; `None` when `cancel`
    /// fired first.
    async fn admit_task_run(
        &self,
        id: &str,
        tx: &mpsc::Sender<Event>,
        cancel: &CancellationToken,
    ) -> Result<Option<(RequestGuard, OwnedMutexGuard<()>)>, DispatchError> {
        let admission = async {
            let agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
            let request_guard = self
                .subsystems
                .presence
                .acquire_request_guard_with_progress(id.to_string(), tx.clone())
                .await?;
            Ok((request_guard, agent_guard))
        };
        tokio::select! {
            biased;
            () = cancel.cancelled() => Ok(None),
            admitted = admission => admitted.map(Some),
        }
    }

    /// Save a new session for the run, mark the run as running in it, and queue the
    /// write opening its turn with the task's prompt; `None` when the session could not
    /// be saved, so the run is not persisted.
    async fn open_task_run_session(&self, turn: &TaskRunTurn) -> Option<PersistTarget> {
        let session = SessionId::new();
        let opened = async {
            let branch = self
                .memory
                .transcripts
                .begin_session_with_main_branch(session, std::process::id())
                .await?;
            self.memory.tasks.start_run(turn.run, session).await?;
            Ok::<_, MemoryError>(branch)
        };
        match opened.await {
            Ok(branch) => {
                let prompt = PersistedMessage::user(turn.task.spec.prompt.as_str());
                Some(self.open_turn_fire_and_forget(branch, prompt))
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

    /// Run the task's prompt on `backend`, persisting to `persist` and sending its wire
    /// events to `tx`; returns the final reply once the turn emitted `Done`.
    async fn unattended_turn(
        &self,
        turn: &TaskRunTurn,
        backend: Arc<dyn LlmBackend>,
        persist: Option<PersistTarget>,
        cancel: &CancellationToken,
        tx: &mpsc::Sender<Event>,
    ) -> Result<Option<String>, DispatchError> {
        let (llm_tx, llm_rx) = mpsc::channel(EVENT_BUFFER);
        let agent_task = self.spawn_task_agent(turn, backend, llm_tx, cancel.clone());
        let mut translator =
            TurnTranslator::new(turn.id.clone(), TurnScope::TaskRun(turn.run), persist);
        self.forward_task_run_events(&mut translator, llm_rx, tx)
            .await;
        match agent_task.await {
            Ok(Ok(())) => Ok(translator.take_reply()),
            Ok(Err(e)) => Err(e.into()),
            Err(join_err) => Err(DispatchError::AgentPanicked(join_err)),
        }
    }

    /// The main agent's tools over `backend`, with every confirmation denied and labelled
    /// as the run's.
    fn spawn_task_agent(
        &self,
        turn: &TaskRunTurn,
        backend: Arc<dyn LlmBackend>,
        llm_tx: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> AbortOnDropHandle<Result<(), LlmError>> {
        let health: Option<Arc<dyn LlmHealthProbe>> = Some(Arc::new(PresenceLlmHealthProbe::new(
            self.subsystems.presence.clone(),
        )));
        let agent = Agent::new(
            backend,
            self.subsystems.tools.snapshot(),
            health,
            Duration::from_secs(self.config.timeouts.tool_call_secs),
        );
        let source = ConfirmSource::Task {
            task: turn.task.id,
            run: turn.run,
        };
        let router = ConfirmRouter::unattended(turn.id.clone(), source);
        let prompt = turn.task.spec.prompt.as_str().to_string();
        AbortOnDropHandle::new(tokio::spawn(
            CONFIRM_ROUTER
                .scope(router, async move {
                    agent.run_turn(prompt, Vec::new(), llm_tx, cancel).await
                })
                .in_current_span(),
        ))
    }

    /// Translate and persist the run's agent events until the agent finishes; nothing is
    /// spoken.
    async fn forward_task_run_events(
        &self,
        translator: &mut TurnTranslator,
        mut llm_rx: mpsc::Receiver<LlmEvent>,
        tx: &mpsc::Sender<Event>,
    ) {
        let max_sentence_chars = self.config.voice.synthesis.max_sentence_chars.get() as usize;
        let mut unspoken = SentenceBuffer::new(max_sentence_chars);
        while let Some(event) = llm_rx.recv().await {
            if let Some((wire, _)) = translator.translate(self, event, &mut unspoken) {
                let _ = tx.send(wire).await;
            }
        }
        translator.persist_turn_stats(self);
    }
}

/// The main system prompt followed by [`TASK_PREAMBLE`].
fn task_system_prompt(main: &str) -> String {
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

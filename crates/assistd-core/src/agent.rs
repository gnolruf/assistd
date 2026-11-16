//! Per-turn agent loop: step the LLM, dispatch the tool calls it
//! requests, feed the results back, repeat until it answers.

use std::any::Any;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::FutureExt;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, instrument, warn};

use assistd_ipc::{Attachment, RelayEvent, StatusKind};
use assistd_llm::{
    HealthWaitError, LlmError, LlmEvent, LlmHealthProbe, StepOutcome, ToolCall, ToolResultPayload,
    TurnBackend,
};
use assistd_tools::{RelayObserver, ToolOutput, ToolRegistry, TurnContext};
use assistd_utils::text::panic_message;

use crate::recovery::{Component, StatusSeverity};
use crate::recovery_event;

/// How long a turn waits for an LLM restart: the supervisor's worst-case
/// backoff (63s) plus spawn and probe headroom.
const REPLAY_WAIT_BUDGET: Duration = Duration::from_secs(75);

/// Consecutive identical tool calls after which the model is treated as
/// stuck.
const DUPLICATE_CALL_LIMIT: usize = 3;

/// Ceiling on tool-calling steps per turn that keep making different calls.
const MAX_TOOL_STEPS: u32 = 200;

/// Consecutive retries of a step cut off at the output-token limit before
/// the turn fails.
const TRUNCATION_RETRY_LIMIT: u32 = 2;

/// Ends the reply kept for a turn cut off before the model finished.
const INTERRUPTED_MARKER: &str = "[interrupted]";

/// Events one model step may have waiting for the turn.
const STEP_EVENT_BUFFER: usize = 32;

/// Tells the model why its last response vanished and how to avoid a repeat.
const TRUNCATION_NOTE: &str = "Your previous response reached the output-token limit before \
     it finished, so it was discarded and none of its tool calls ran. Keep your reasoning \
     brief and your tool calls small: split a long script into several shorter calls.";

/// Long-lived agent dependencies, reused across turns. With a `health` probe the loop
/// replays a step once after a llama-server crash; without one, a crash ends the turn.
/// A tool still running after `tool_deadline` is reported to the model as failed.
#[derive(Debug)]
pub struct Agent {
    backend: Arc<dyn TurnBackend>,
    tools: Arc<ToolRegistry>,
    /// Offered in place of `tools` while the turn plans.
    planning_tools: Option<Arc<ToolRegistry>>,
    health: Option<Arc<dyn LlmHealthProbe>>,
    tool_deadline: Duration,
}

impl Agent {
    /// Build an agent over `backend` and `tools`; see [`Agent`] for how
    /// `health` and `tool_deadline` shape a turn.
    pub fn new(
        backend: Arc<dyn TurnBackend>,
        tools: Arc<ToolRegistry>,
        health: Option<Arc<dyn LlmHealthProbe>>,
        tool_deadline: Duration,
    ) -> Self {
        Self {
            backend,
            tools,
            planning_tools: None,
            health,
            tool_deadline,
        }
    }

    /// Offer `planning` instead of the acting tools whenever the turn plans; the choice is
    /// made before each step.
    pub fn with_planning_tools(mut self, planning: Arc<ToolRegistry>) -> Self {
        self.planning_tools = Some(planning);
        self
    }

    /// Run one turn in `context`, streaming events on `tx` until the model answers or `tx`
    /// closes or `cancel` fires. Turns must not run concurrently on one agent. Errors when the
    /// backend fails; an incomplete restart surfaces as [`LlmError::ServerRestarting`].
    #[instrument(skip_all, name = "agent_turn")]
    pub async fn run_turn(
        &self,
        context: &TurnContext,
        user_text: String,
        user_attachments: Vec<Attachment>,
        tx: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> Result<(), LlmError> {
        self.backend.push_user(user_text, user_attachments).await?;
        let tools = Arc::clone(self.tools_for(context));
        let mut turn = Turn {
            context: context
                .clone()
                .observed_by(Arc::new(TurnRelayObserver(tx.clone()))),
            tx,
            cancel,
            schemas: tools.openai_schemas(),
            tools,
            tools_withdrawn: None,
            streak: CallStreak::default(),
            iteration: 0,
            truncated_steps: 0,
        };

        loop {
            if turn.stop_requested() {
                debug!(
                    target: "assistd::agent",
                    iteration = turn.iteration,
                    cancelled = turn.cancel.is_cancelled(),
                    "stopping between iterations (client gone or explicit cancel)"
                );
                return self.record_interruption(&turn, "").await;
            }

            let outcome = match self.step_with_replay(&turn).await? {
                StepEnd::Outcome(outcome) => outcome,
                StepEnd::Interrupted { streamed } => {
                    return self.record_interruption(&turn, &streamed).await;
                }
            };

            match outcome {
                StepOutcome::Final => {
                    let _ = turn.tx.send(LlmEvent::Done).await;
                    return Ok(());
                }
                StepOutcome::Truncated => self.retry_truncated_step(&mut turn).await?,
                StepOutcome::ToolCalls(calls) if turn.tools_withdrawn.is_some() => {
                    return self.refuse_withdrawn_calls(&turn, &calls).await;
                }
                StepOutcome::ToolCalls(calls) => {
                    turn.truncated_steps = 0;
                    let (results, stuck) = self.dispatch_tool_calls(&mut turn, calls).await;
                    self.backend.push_tool_results(results).await?;
                    turn.offer(self.tools_for(&turn.context));
                    turn.iteration += 1;
                    if let Some(why) = ToolBudgetExhausted::check(stuck, turn.iteration) {
                        self.withdraw_tools(&mut turn, why).await;
                    }
                }
            }
        }
    }

    /// The planning tools while `context`'s turn plans, else the acting ones.
    fn tools_for(&self, context: &TurnContext) -> &Arc<ToolRegistry> {
        match &self.planning_tools {
            Some(planning) if context.planning() => planning,
            _ => &self.tools,
        }
    }

    async fn refuse_withdrawn_calls(
        &self,
        turn: &Turn,
        calls: &[ToolCall],
    ) -> Result<(), LlmError> {
        warn!(
            target: "assistd::agent",
            iteration = turn.iteration,
            "model requested tools after they were withdrawn; ending turn"
        );
        let _ = turn
            .tx
            .send(LlmEvent::ToolCallsRequested {
                calls: calls.to_vec(),
            })
            .await;
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            let result = cancelled_tool_result(call, "ended; tools are unavailable");
            results.push(send_tool_result(turn, call, result).await);
        }
        self.backend.push_tool_results(results).await?;
        let _ = turn.tx.send(LlmEvent::Done).await;
        Ok(())
    }

    /// Tell the client and the model that the last response was cut off so
    /// the next step retries it; errors once [`TRUNCATION_RETRY_LIMIT`]
    /// consecutive retries are spent.
    async fn retry_truncated_step(&self, turn: &mut Turn) -> Result<(), LlmError> {
        turn.truncated_steps += 1;
        if turn.truncated_steps > TRUNCATION_RETRY_LIMIT {
            return Err(LlmError::OutputLimit(turn.truncated_steps));
        }
        recovery_event!(
            StatusSeverity::Warning,
            Component::Llm,
            "output_truncated",
            iteration = turn.iteration,
            retry = turn.truncated_steps,
            "response hit the output-token limit; retrying the step"
        );
        let _ = turn.tx.send(LlmEvent::StepDiscarded).await;
        let _ = turn
            .tx
            .send(status_event(
                StatusSeverity::Warning,
                Component::Llm,
                StatusKind::OutputTruncated,
                format!(
                    "LLM response hit the output-token limit; retrying ({}/{TRUNCATION_RETRY_LIMIT})",
                    turn.truncated_steps
                ),
            ))
            .await;
        if let Err(e) = self
            .backend
            .set_transient_note(turn.truncation_note())
            .await
        {
            warn!(
                target: "assistd::agent",
                error = %e,
                "set_transient_note failed; retrying without the note"
            );
        }
        Ok(())
    }

    /// Keep what the cut-off step streamed as the turn's reply, marked as
    /// interrupted, in the history and for whoever records the turn.
    async fn record_interruption(&self, turn: &Turn, streamed: &str) -> Result<(), LlmError> {
        let reply = interrupted_reply(streamed);
        self.backend.push_interrupted_reply(reply.clone()).await?;
        let _ = turn.tx.send(LlmEvent::Interrupted { reply }).await;
        Ok(())
    }

    /// One LLM step. A crash is replayed once after the server is ready
    /// again; any other failure ends the turn with an error and no `Done`.
    async fn step_with_replay(&self, turn: &Turn) -> Result<StepEnd, LlmError> {
        let mut restart_attempted = false;
        loop {
            let replay_probe = self.health.as_ref().filter(|_| !restart_attempted);
            let _ = turn.tx.send(LlmEvent::StepStarted).await;
            match (self.relay_step(turn).await, replay_probe) {
                (Err(LlmError::ServerRestarting(reason)), Some(probe)) => {
                    restart_attempted = true;
                    match await_restart(probe, turn, &reason).await {
                        Replay::Ready => continue,
                        Replay::Cancelled => {
                            return Ok(StepEnd::Interrupted {
                                streamed: String::new(),
                            });
                        }
                        Replay::Abandoned(final_msg) => {
                            return Err(LlmError::ServerRestarting(final_msg));
                        }
                    }
                }
                (step_end, _) => return step_end,
            }
        }
    }

    /// Run one model step, passing its events on to the turn. The step is
    /// dropped once the turn is cancelled or its consumer leaves.
    async fn relay_step(&self, turn: &Turn) -> Result<StepEnd, LlmError> {
        let (step_tx, mut step_rx) = mpsc::channel(STEP_EVENT_BUFFER);
        let mut step = pin!(self.backend.step(turn.schemas.clone(), step_tx));
        let mut streamed = String::new();
        loop {
            tokio::select! {
                biased;
                () = turn.stopped() => {
                    debug!(
                        target: "assistd::agent",
                        iteration = turn.iteration,
                        cancelled = turn.cancel.is_cancelled(),
                        "stopping during LLM step (client gone or explicit cancel)"
                    );
                    return Ok(StepEnd::Interrupted { streamed });
                }
                Some(event) = step_rx.recv() => turn.pass_on(event, &mut streamed).await,
                result = &mut step => {
                    while let Ok(event) = step_rx.try_recv() {
                        turn.pass_on(event, &mut streamed).await;
                    }
                    return result.map(StepEnd::Outcome);
                }
            }
        }
    }

    /// Announce the step's calls and dispatch each in order, answering any left unrun after
    /// the client leaves or cancellation with a cancelled payload so the batch stays
    /// complete. Returns `true` when a call hit [`DUPLICATE_CALL_LIMIT`].
    async fn dispatch_tool_calls(
        &self,
        turn: &mut Turn,
        calls: Vec<ToolCall>,
    ) -> (Vec<ToolResultPayload>, bool) {
        let mut results = Vec::with_capacity(calls.len());
        let mut stuck = false;
        let _ = turn
            .tx
            .send(LlmEvent::ToolCallsRequested {
                calls: calls.clone(),
            })
            .await;
        let mut pending = calls.into_iter();
        for call in pending.by_ref() {
            stuck |= turn.streak.record(&call) >= DUPLICATE_CALL_LIMIT;
            if turn.stop_requested() {
                debug!(
                    target: "assistd::agent",
                    iteration = turn.iteration,
                    tool = %call.name,
                    cancelled = turn.cancel.is_cancelled(),
                    "stopping mid-call (client gone or explicit cancel) without dispatch"
                );
                results.push(skip_tool_call(turn, &call).await);
                break;
            }

            let (payload, cancelled_mid_dispatch) = self.run_tool_call(turn, &call).await;
            results.push(payload);
            if cancelled_mid_dispatch {
                break;
            }
        }
        for call in pending {
            results.push(skip_tool_call(turn, &call).await);
        }
        (results, stuck)
    }

    /// Announce and run one call, or abandon it when the turn is cancelled
    /// meanwhile. The flag is `true` when cancellation fired.
    async fn run_tool_call(&self, turn: &Turn, call: &ToolCall) -> (ToolResultPayload, bool) {
        let _ = turn
            .tx
            .send(LlmEvent::ToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
            .await;

        let dispatched = tokio::select! {
            biased;
            () = turn.cancel.cancelled() => None,
            r = dispatch_tool_call(turn, call, self.tool_deadline) => Some(r),
        };
        let result = dispatched.unwrap_or_else(|| {
            warn!(
                target: "assistd::agent",
                iteration = turn.iteration,
                tool = %call.name,
                "cancellation fired during tool dispatch; abandoning tool"
            );
            cancelled_tool_result(call, "cancelled during dispatch")
        });
        let cancelled_mid_dispatch = turn.cancel.is_cancelled();
        (
            send_tool_result(turn, call, result).await,
            cancelled_mid_dispatch,
        )
    }

    async fn withdraw_tools(&self, turn: &mut Turn, why: ToolBudgetExhausted) {
        recovery_event!(
            StatusSeverity::Warning,
            Component::Agent,
            "tools_withdrawn",
            iteration = turn.iteration,
            reason = %why.reason(),
            "withdrawing tools; asking model to answer from what it has"
        );
        let _ = turn
            .tx
            .send(status_event(
                StatusSeverity::Warning,
                Component::Agent,
                StatusKind::ToolsWithdrawn,
                format!(
                    "Agent stopped using tools ({}); answering with what it has",
                    why.reason()
                ),
            ))
            .await;
        if let Err(e) = self.backend.set_transient_note(why.model_note()).await {
            warn!(
                target: "assistd::agent",
                error = %e,
                "set_transient_note failed; answering without the note"
            );
        }
        turn.tools_withdrawn = Some(why);
    }
}

struct Turn {
    /// The caller's context, with the turn's own events observing its relays.
    context: TurnContext,
    tx: mpsc::Sender<LlmEvent>,
    cancel: CancellationToken,
    tools: Arc<ToolRegistry>,
    schemas: Vec<Value>,
    tools_withdrawn: Option<ToolBudgetExhausted>,
    streak: CallStreak,
    iteration: u32,
    truncated_steps: u32,
}

impl Turn {
    fn stop_requested(&self) -> bool {
        self.tx.is_closed() || self.cancel.is_cancelled()
    }

    /// Resolves once the turn is cancelled or its consumer has left.
    async fn stopped(&self) {
        tokio::select! {
            () = self.cancel.cancelled() => {}
            () = self.tx.closed() => {}
        }
    }

    /// Send a step's `event` to the turn's consumer, keeping `streamed` equal
    /// to the text the step has shown so far.
    async fn pass_on(&self, event: LlmEvent, streamed: &mut String) {
        match &event {
            LlmEvent::Delta { text } => streamed.push_str(text),
            LlmEvent::StepDiscarded => streamed.clear(),
            _ => {}
        }
        let _ = self.tx.send(event).await;
    }

    /// Offer `tools` from the next step on.
    fn offer(&mut self, tools: &Arc<ToolRegistry>) {
        if !Arc::ptr_eq(&self.tools, tools) {
            self.tools = Arc::clone(tools);
            self.schemas = tools.openai_schemas();
        }
    }

    /// [`TRUNCATION_NOTE`], restating the withdrawal the retry would otherwise lose.
    fn truncation_note(&self) -> String {
        match self.tools_withdrawn {
            Some(why) => format!("{TRUNCATION_NOTE} {}", why.model_note()),
            None => TRUNCATION_NOTE.to_string(),
        }
    }
}

/// Shows a turn the relays its tool calls run, as [`LlmEvent::Relay`].
#[derive(Debug)]
struct TurnRelayObserver(mpsc::Sender<LlmEvent>);

#[async_trait]
impl RelayObserver for TurnRelayObserver {
    async fn report(&self, event: RelayEvent) {
        let _ = self.0.send(LlmEvent::Relay(event)).await;
    }
}

/// How a model step that did not fail ended.
enum StepEnd {
    Outcome(StepOutcome),
    /// The turn stopped mid-step, after the step had streamed this text.
    Interrupted {
        streamed: String,
    },
}

enum Replay {
    Ready,
    Cancelled,
    Abandoned(String),
}

/// Why the loop stopped accepting tool calls for the rest of the turn. The schemas
/// stay in the request after withdrawal; without them the server stops parsing
/// tool-call markup and would stream it as text.
#[derive(Debug, Clone, Copy)]
enum ToolBudgetExhausted {
    Repeating,
    StepCeiling,
}

impl ToolBudgetExhausted {
    fn check(stuck: bool, iteration: u32) -> Option<Self> {
        if stuck {
            Some(Self::Repeating)
        } else if iteration >= MAX_TOOL_STEPS {
            Some(Self::StepCeiling)
        } else {
            None
        }
    }

    fn reason(self) -> String {
        match self {
            Self::Repeating => {
                format!("the same tool call was repeated {DUPLICATE_CALL_LIMIT} times in a row")
            }
            Self::StepCeiling => {
                format!("the turn reached its ceiling of {MAX_TOOL_STEPS} tool steps")
            }
        }
    }

    fn model_note(self) -> String {
        format!(
            "Tool access for this turn has been withdrawn because {}. Do not request \
             any more tool calls. Answer the user now using what you have already \
             learned, and say plainly which parts you could not finish.",
            self.reason()
        )
    }
}

/// How many times in a row the model has issued the same tool call.
#[derive(Default)]
struct CallStreak {
    last: Option<(String, Value)>,
    count: usize,
}

impl CallStreak {
    fn record(&mut self, call: &ToolCall) -> usize {
        let same = self
            .last
            .as_ref()
            .is_some_and(|(name, args)| *name == call.name && *args == call.arguments);
        if same {
            self.count += 1;
        } else {
            self.last = Some((call.name.clone(), call.arguments.clone()));
            self.count = 1;
        }
        self.count
    }
}

/// `streamed` ending in [`INTERRUPTED_MARKER`], or the marker alone when
/// nothing was streamed.
fn interrupted_reply(streamed: &str) -> String {
    match streamed.trim_end() {
        "" => INTERRUPTED_MARKER.to_string(),
        shown => format!("{shown}\n\n{INTERRUPTED_MARKER}"),
    }
}

/// Tell the client the server crashed, wait for the supervisor to bring
/// it back, and report whether the step can be replayed.
async fn await_restart(probe: &Arc<dyn LlmHealthProbe>, turn: &Turn, reason: &str) -> Replay {
    announce_crash(turn, reason).await;

    let wait_result = tokio::select! {
        biased;
        () = turn.cancel.cancelled() => {
            debug!(
                target: "assistd::agent",
                iteration = turn.iteration,
                "cancellation fired while waiting for LLM restart"
            );
            return Replay::Cancelled;
        }
        res = probe.wait_for_ready(REPLAY_WAIT_BUDGET) => res,
    };

    match wait_result {
        Ok(()) => {
            announce_replay(turn).await;
            Replay::Ready
        }
        Err(wait_err) => Replay::Abandoned(abandon_replay(turn, wait_err).await),
    }
}

async fn announce_crash(turn: &Turn, reason: &str) {
    recovery_event!(
        StatusSeverity::Warning,
        Component::Llm,
        "crash_detected",
        iteration = turn.iteration,
        reason = %reason,
        "llama-server died mid-step; waiting for supervisor restart and replaying"
    );
    let _ = turn.tx.send(LlmEvent::StepDiscarded).await;
    let _ = turn
        .tx
        .send(status_event(
            StatusSeverity::Warning,
            Component::Llm,
            StatusKind::Restarting,
            "LLM crashed: restarting and replaying your query".to_string(),
        ))
        .await;
}

async fn announce_replay(turn: &Turn) {
    recovery_event!(
        StatusSeverity::Info,
        Component::Llm,
        "replay_ready",
        iteration = turn.iteration,
        "supervisor reported Ready; replaying user query"
    );
    let _ = turn
        .tx
        .send(status_event(
            StatusSeverity::Info,
            Component::Llm,
            StatusKind::Replaying,
            "LLM restored: replaying your query".to_string(),
        ))
        .await;
}

/// Report the failed restart to the client and return the turn's final
/// error message.
async fn abandon_replay(turn: &Turn, wait_err: HealthWaitError) -> String {
    recovery_event!(
        StatusSeverity::Error,
        Component::Llm,
        "replay_abandoned",
        iteration = turn.iteration,
        wait_error = %wait_err,
        "supervisor did not return to Ready; abandoning replay"
    );
    let final_msg = match wait_err {
        HealthWaitError::Timeout => "LLM did not recover before timeout",
        HealthWaitError::Degraded => "LLM server keeps failing; it is retried every few minutes",
        HealthWaitError::NoService => "LLM service is not currently attached",
    };
    let _ = turn
        .tx
        .send(status_event(
            StatusSeverity::Error,
            Component::Llm,
            StatusKind::Degraded,
            final_msg.to_string(),
        ))
        .await;
    final_msg.to_string()
}

fn status_event(
    severity: StatusSeverity,
    component: Component,
    event: StatusKind,
    message: String,
) -> LlmEvent {
    LlmEvent::Status {
        severity,
        component,
        event,
        message,
    }
}

/// Answer a call that will never run, reporting the result like any other
/// so the client and the transcript see the batch completed.
async fn skip_tool_call(turn: &Turn, call: &ToolCall) -> ToolResultPayload {
    send_tool_result(
        turn,
        call,
        cancelled_tool_result(call, "cancelled before dispatch"),
    )
    .await
}

/// Report `result` to the client and return what the model is shown.
async fn send_tool_result(turn: &Turn, call: &ToolCall, result: ToolOutput) -> ToolResultPayload {
    let payload = ToolResultPayload {
        call_id: call.id.clone(),
        name: call.name.clone(),
        content: result.output.clone(),
        attachments: result.attachments.clone(),
    };
    let _ = turn
        .tx
        .send(LlmEvent::ToolResult {
            id: call.id.clone(),
            name: call.name.clone(),
            result: Box::new(result.report()),
        })
        .await;
    payload
}

fn cancelled_tool_result(call: &ToolCall, reason: &str) -> ToolOutput {
    error_tool_result(
        &format!("[error] {}: agent turn {reason}.", call.name),
        Duration::ZERO,
    )
}

fn error_tool_result(message: &str, duration: Duration) -> ToolOutput {
    let duration_ms = duration.as_millis();
    ToolOutput::failed(format!("{message}\n[exit:-1 | {duration_ms}ms]"), duration)
}

fn panicked_tool_result(
    call: &ToolCall,
    iteration: u32,
    duration: Duration,
    panic: &(dyn Any + Send),
) -> ToolOutput {
    let message = panic_message(panic);
    warn!(
        target: "assistd::agent",
        iteration,
        tool = %call.name,
        duration_ms = duration.as_millis(),
        panic = %message,
        "tool invocation panicked"
    );
    error_tool_result(
        &format!(
            "[error] {}: the tool crashed ({message}). Try: a different command or source.",
            call.name
        ),
        duration,
    )
}

/// `fut`'s output, or `None` once it has run `deadline` without one; the clock stops while
/// `context`'s turn has a prompt parked.
async fn within_deadline<F: Future>(
    context: &TurnContext,
    deadline: Duration,
    fut: F,
) -> Option<F::Output> {
    let Some(mut parked) = context.parked() else {
        return tokio::time::timeout(deadline, fut).await.ok();
    };
    let mut fut = pin!(fut);
    let mut left = deadline;
    loop {
        let paused = *parked.borrow_and_update();
        let started = Instant::now();
        let clock = async {
            if paused {
                std::future::pending::<()>().await;
            } else {
                tokio::time::sleep(left).await;
            }
        };
        tokio::select! {
            output = &mut fut => return Some(output),
            () = clock => return None,
            changed = parked.changed() => {
                if !paused {
                    left = left.saturating_sub(started.elapsed());
                }
                if changed.is_err() {
                    return tokio::time::timeout(left, fut).await.ok();
                }
            }
        }
    }
}

async fn dispatch_tool_call(turn: &Turn, call: &ToolCall, deadline: Duration) -> ToolOutput {
    let start = Instant::now();
    let (tools, iteration) = (turn.tools.as_ref(), turn.iteration);

    let Some(tool) = tools.get(&call.name) else {
        return unknown_tool_result(tools, call, iteration, start.elapsed());
    };

    let invocation =
        AssertUnwindSafe(tool.invoke(&turn.context, call.arguments.clone())).catch_unwind();
    let finished = if tool.bounded_by_tool_deadline() {
        within_deadline(&turn.context, deadline, invocation).await
    } else {
        Some(invocation.await)
    };
    if tool.reads_host(&call.arguments) {
        turn.context.record_host_read();
    }
    let Some(result) = finished else {
        let duration = start.elapsed();
        warn!(
            target: "assistd::agent",
            iteration,
            tool = %call.name,
            duration_ms = duration.as_millis(),
            "tool call exceeded its deadline; abandoning it"
        );
        return error_tool_result(
            &format!(
                "[error] {}: no result after {}s; call abandoned. Try: a faster or narrower command.",
                call.name,
                deadline.as_secs()
            ),
            duration,
        );
    };
    let duration = start.elapsed();

    match result {
        Ok(Ok(output)) => {
            log_completed_call(call, &output, iteration, duration);
            output
        }
        Err(panic) => panicked_tool_result(call, iteration, duration, panic.as_ref()),
        Ok(Err(e)) => {
            warn!(
                target: "assistd::agent",
                iteration,
                tool = %call.name,
                duration_ms = duration.as_millis(),
                error = %e,
                "tool invocation errored"
            );
            error_tool_result(
                &format!(
                    "[error] {}: tool invocation failed. Check: {e}. Try: a different command.",
                    call.name
                ),
                duration,
            )
        }
    }
}

fn unknown_tool_result(
    tools: &ToolRegistry,
    call: &ToolCall,
    iteration: u32,
    duration: Duration,
) -> ToolOutput {
    warn!(
        target: "assistd::agent",
        iteration,
        tool = %call.name,
        duration_ms = duration.as_millis(),
        "unknown tool"
    );
    let available = tools.names().collect::<Vec<_>>().join(", ");
    error_tool_result(
        &format!(
            "[error] agent: unknown tool '{}'. Available: {available}.",
            call.name
        ),
        duration,
    )
}

fn log_completed_call(call: &ToolCall, output: &ToolOutput, iteration: u32, duration: Duration) {
    let command = call
        .arguments
        .get("command")
        .and_then(Value::as_str)
        .unwrap_or("");
    info!(
        target: "assistd::agent",
        iteration,
        tool = %call.name,
        command = %command,
        exit_code = output.exit_code,
        output_size = output.output.len(),
        duration_ms = duration.as_millis(),
        "tool call complete"
    );
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use parking_lot::Mutex as StdMutex;
    use tokio::sync::Notify;

    use assistd_llm::{HealthSnapshot, LlmResult};
    use assistd_tools::commands::EchoCommand;
    use assistd_tools::{
        CommandRegistry, OutputCap, PresentSpec, RunTool, Tool, ToolError, ToolOutput, TurnContext,
    };

    use super::*;

    /// Scripted mock backend: returns queued step outcomes in order, then
    /// `Final`. Records what the loop pushed back.
    #[derive(Debug)]
    struct MockBackend {
        outcomes: StdMutex<Vec<StepOutcome>>,
        pushed_users: StdMutex<Vec<String>>,
        pushed_results: StdMutex<Vec<Vec<ToolResultPayload>>>,
        /// Every entry into `step`, including cancelled ones.
        step_calls: AtomicUsize,
        slow_step: StdMutex<Option<Duration>>,
        /// Tool schemas offered on each completed `step`.
        step_tool_counts: StdMutex<Vec<usize>>,
        transient_notes: StdMutex<Vec<String>>,
    }

    impl MockBackend {
        fn with(outcomes: Vec<StepOutcome>) -> Arc<Self> {
            Arc::new(Self {
                outcomes: StdMutex::new(outcomes),
                pushed_users: StdMutex::new(Vec::new()),
                pushed_results: StdMutex::new(Vec::new()),
                step_calls: AtomicUsize::new(0),
                slow_step: StdMutex::new(None),
                step_tool_counts: StdMutex::new(Vec::new()),
                transient_notes: StdMutex::new(Vec::new()),
            })
        }

        fn slow_step_ms(self: Arc<Self>, ms: u64) -> Arc<Self> {
            *self.slow_step.lock() = Some(Duration::from_millis(ms));
            self
        }
    }

    #[async_trait]
    impl TurnBackend for MockBackend {
        async fn push_user(&self, text: String, _attachments: Vec<Attachment>) -> LlmResult<()> {
            self.pushed_users.lock().push(text);
            Ok(())
        }

        async fn push_tool_results(&self, results: Vec<ToolResultPayload>) -> LlmResult<()> {
            self.pushed_results.lock().push(results);
            Ok(())
        }

        async fn step(
            &self,
            tools: Vec<Value>,
            tx: mpsc::Sender<LlmEvent>,
        ) -> LlmResult<StepOutcome> {
            self.step_calls.fetch_add(1, Ordering::SeqCst);
            let delay = *self.slow_step.lock();
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            self.step_tool_counts.lock().push(tools.len());
            let outcome = {
                let mut queue = self.outcomes.lock();
                if queue.is_empty() {
                    StepOutcome::Final
                } else {
                    queue.remove(0)
                }
            };
            if matches!(outcome, StepOutcome::Final) {
                let _ = tx.send(LlmEvent::Delta { text: "ok".into() }).await;
            }
            Ok(outcome)
        }

        async fn push_interrupted_reply(&self, _reply: String) -> LlmResult<()> {
            Ok(())
        }

        async fn set_transient_note(&self, text: String) -> LlmResult<()> {
            self.transient_notes.lock().push(text);
            Ok(())
        }
    }

    const TOOL_DEADLINE: Duration = Duration::from_secs(300);

    /// A `run` tool whose invocation never completes.
    #[derive(Debug)]
    struct HangingTool {
        entered: Arc<Notify>,
    }

    #[async_trait]
    impl Tool for HangingTool {
        fn name(&self) -> &'static str {
            "run"
        }
        fn description(&self) -> Cow<'_, str> {
            "never returns".into()
        }
        fn parameters_schema(&self) -> Value {
            serde_json::json!({"type":"object"})
        }
        async fn invoke(&self, _turn: &TurnContext, _args: Value) -> Result<ToolOutput, ToolError> {
            self.entered.notify_one();
            std::future::pending::<()>().await;
            unreachable!("hanging tool must never resolve")
        }
    }

    /// A `run` tool that answers after `delay` and opts out of the tool deadline.
    #[derive(Debug)]
    struct SelfBoundedTool {
        delay: Duration,
    }

    #[async_trait]
    impl Tool for SelfBoundedTool {
        fn name(&self) -> &'static str {
            "run"
        }
        fn description(&self) -> Cow<'_, str> {
            "slow but bounded".into()
        }
        fn parameters_schema(&self) -> Value {
            serde_json::json!({"type":"object"})
        }
        fn bounded_by_tool_deadline(&self) -> bool {
            false
        }
        async fn invoke(&self, _turn: &TurnContext, _args: Value) -> Result<ToolOutput, ToolError> {
            tokio::time::sleep(self.delay).await;
            Ok(ToolOutput::text("finished", self.delay))
        }
    }

    /// A `run` tool whose invocation panics.
    #[derive(Debug)]
    struct PanickingTool;

    #[async_trait]
    impl Tool for PanickingTool {
        fn name(&self) -> &'static str {
            "run"
        }
        fn description(&self) -> Cow<'_, str> {
            "panics".into()
        }
        fn parameters_schema(&self) -> Value {
            serde_json::json!({"type":"object"})
        }
        async fn invoke(&self, _turn: &TurnContext, _args: Value) -> Result<ToolOutput, ToolError> {
            panic!("not implemented")
        }
    }

    fn call(id: &str, command: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "run".into(),
            arguments: serde_json::json!({ "command": command }),
        }
    }

    fn tools_with_echo() -> Arc<ToolRegistry> {
        let mut commands = CommandRegistry::new();
        commands.register(EchoCommand);
        let output_cap = Arc::new(OutputCap::new(PresentSpec {
            overflow_dir: std::env::temp_dir()
                .join(format!("assistd-agent-test-{}", std::process::id())),
            ..PresentSpec::default()
        }));
        let mut tools = ToolRegistry::with_output_cap(Arc::clone(&output_cap));
        tools.register(RunTool::new(Arc::new(commands), output_cap));
        Arc::new(tools)
    }

    async fn collect(rx: &mut mpsc::Receiver<LlmEvent>) -> Vec<LlmEvent> {
        let mut out = Vec::new();
        while let Some(ev) = rx.recv().await {
            out.push(ev);
        }
        out
    }

    /// Run one uncancelled turn, draining events concurrently so a long
    /// turn never blocks on channel capacity.
    async fn run_turn(agent: Agent, user_text: &str) -> (Result<(), LlmError>, Vec<LlmEvent>) {
        let (tx, mut rx) = mpsc::channel(16);
        let context = TurnContext::default();
        let turn = agent.run_turn(
            &context,
            user_text.into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
        );
        tokio::join!(turn, collect(&mut rx))
    }

    fn status_kinds(events: &[LlmEvent]) -> Vec<StatusKind> {
        events
            .iter()
            .filter_map(|e| match e {
                LlmEvent::Status { event, .. } => Some(*event),
                _ => None,
            })
            .collect()
    }

    /// A failed turn streams no text for its failure and never sends `Done`.
    fn assert_failed_without_done(events: &[LlmEvent]) {
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, LlmEvent::Done | LlmEvent::Delta { .. })),
            "{events:?}"
        );
    }

    fn ok_then_done() -> [LlmEvent; 2] {
        [LlmEvent::Delta { text: "ok".into() }, LlmEvent::Done]
    }

    #[tokio::test]
    async fn tool_call_result_is_fed_back_before_final_answer() {
        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![call("c-1", "echo hello")]),
            StepOutcome::Final,
        ]);
        let (res, events) = run_turn(
            Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
            "say hello",
        )
        .await;
        res.unwrap();

        let [
            first_step,
            requested,
            tool_call,
            tool_result,
            second_step,
            rest @ ..,
        ] = events.as_slice()
        else {
            panic!("too few events: {events:?}");
        };
        assert_eq!(
            [first_step, second_step],
            [&LlmEvent::StepStarted, &LlmEvent::StepStarted]
        );
        assert_eq!(
            *requested,
            LlmEvent::ToolCallsRequested {
                calls: vec![call("c-1", "echo hello")]
            }
        );
        assert_eq!(
            *tool_call,
            LlmEvent::ToolCall {
                id: "c-1".into(),
                name: "run".into(),
                arguments: serde_json::json!({ "command": "echo hello" }),
            }
        );
        assert!(
            matches!(tool_result, LlmEvent::ToolResult { id, name, .. } if id == "c-1" && name == "run"),
            "{tool_result:?}"
        );
        assert_eq!(rest, ok_then_done());

        let pushed = backend.pushed_results.lock();
        let [step] = pushed.as_slice() else {
            panic!("expected one push: {pushed:?}");
        };
        let [payload] = step.as_slice() else {
            panic!("expected one result: {step:?}");
        };
        assert_eq!(payload.call_id, "c-1");
        assert!(payload.content.contains("hello"), "{:?}", payload.content);
    }

    #[tokio::test]
    async fn repeated_identical_calls_withdraw_tools_then_answer() {
        let outcomes: Vec<StepOutcome> = (0..DUPLICATE_CALL_LIMIT)
            .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), "echo same")]))
            .collect();
        let backend = MockBackend::with(outcomes);
        let (res, events) = run_turn(
            Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
            "loop it",
        )
        .await;
        res.unwrap();

        assert_eq!(status_kinds(&events), [StatusKind::ToolsWithdrawn]);
        assert_eq!(events.last(), Some(&LlmEvent::Done));
        assert_eq!(backend.pushed_results.lock().len(), DUPLICATE_CALL_LIMIT);
        assert_eq!(
            *backend.step_tool_counts.lock(),
            [1, 1, 1, 1],
            "the answer step must still offer the schema so a stray call is parsed"
        );
        assert_eq!(
            *backend.transient_notes.lock(),
            [ToolBudgetExhausted::Repeating.model_note()]
        );
    }

    #[tokio::test]
    async fn step_ceiling_withdraws_tools_then_answer() {
        let outcomes: Vec<StepOutcome> = (0..MAX_TOOL_STEPS + 10)
            .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), &format!("echo {i}"))]))
            .collect();
        let backend = MockBackend::with(outcomes);
        let (res, events) = run_turn(
            Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
            "go",
        )
        .await;
        res.unwrap();

        assert_eq!(status_kinds(&events), [StatusKind::ToolsWithdrawn]);
        assert_eq!(events.last(), Some(&LlmEvent::Done));
        let counts = backend.step_tool_counts.lock();
        assert_eq!(counts.len(), MAX_TOOL_STEPS as usize + 1);
        assert_eq!(counts.last(), Some(&1));
        assert_eq!(
            *backend.transient_notes.lock(),
            [ToolBudgetExhausted::StepCeiling.model_note()]
        );
    }

    #[tokio::test]
    async fn tool_request_after_withdrawal_ends_turn_with_synthetic_results() {
        let outcomes: Vec<StepOutcome> = (0..=DUPLICATE_CALL_LIMIT)
            .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), "echo same")]))
            .collect();
        let backend = MockBackend::with(outcomes);
        let (res, events) = run_turn(
            Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
            "loop it",
        )
        .await;
        res.unwrap();

        assert_eq!(events.last(), Some(&LlmEvent::Done));
        let pushed = backend.pushed_results.lock();
        assert_eq!(pushed.len(), DUPLICATE_CALL_LIMIT + 1);
        let [synthetic] = pushed[DUPLICATE_CALL_LIMIT].as_slice() else {
            panic!("expected one synthetic result: {pushed:?}");
        };
        assert_eq!(synthetic.call_id, format!("c-{DUPLICATE_CALL_LIMIT}"));
        assert_eq!(
            synthetic.content,
            "[error] run: agent turn ended; tools are unavailable.\n[exit:-1 | 0ms]"
        );
        assert_eq!(
            backend.step_calls.load(Ordering::SeqCst),
            DUPLICATE_CALL_LIMIT + 1
        );
    }

    #[tokio::test]
    async fn truncated_step_is_retried_with_a_note() {
        let backend = MockBackend::with(vec![StepOutcome::Truncated, StepOutcome::Final]);
        let (res, events) = run_turn(
            Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
            "go",
        )
        .await;
        res.unwrap();

        assert_eq!(status_kinds(&events), [StatusKind::OutputTruncated]);
        assert!(events.ends_with(&ok_then_done()), "{events:?}");
        assert_eq!(*backend.transient_notes.lock(), [TRUNCATION_NOTE]);
        assert!(backend.pushed_results.lock().is_empty());
    }

    #[tokio::test]
    async fn truncations_past_the_retry_limit_fail_the_turn() {
        let outcomes = (0..=TRUNCATION_RETRY_LIMIT)
            .map(|_| StepOutcome::Truncated)
            .collect();
        let backend = MockBackend::with(outcomes);
        let (res, events) = run_turn(
            Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
            "go",
        )
        .await;

        assert!(
            matches!(res, Err(LlmError::OutputLimit(n)) if n == TRUNCATION_RETRY_LIMIT + 1),
            "{res:?}"
        );
        assert_eq!(
            status_kinds(&events),
            [StatusKind::OutputTruncated; TRUNCATION_RETRY_LIMIT as usize]
        );
        assert_failed_without_done(&events);
    }

    #[tokio::test]
    async fn truncation_after_withdrawal_restates_the_withdrawal() {
        let mut outcomes: Vec<StepOutcome> = (0..DUPLICATE_CALL_LIMIT)
            .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), "echo same")]))
            .collect();
        outcomes.push(StepOutcome::Truncated);
        let backend = MockBackend::with(outcomes);
        let (res, _events) = run_turn(
            Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
            "loop it",
        )
        .await;
        res.unwrap();

        let withdrawal = ToolBudgetExhausted::Repeating.model_note();
        assert_eq!(
            *backend.transient_notes.lock(),
            [
                withdrawal.clone(),
                format!("{TRUNCATION_NOTE} {withdrawal}")
            ]
        );
    }

    #[tokio::test]
    async fn unknown_tool_passes_error_to_next_step() {
        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![ToolCall {
                id: "c-1".into(),
                name: "nonexistent".into(),
                arguments: serde_json::json!({}),
            }]),
            StepOutcome::Final,
        ]);
        let (res, _) = run_turn(
            Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
            "go",
        )
        .await;
        res.unwrap();
        let pushed = backend.pushed_results.lock();
        assert_eq!(pushed.len(), 1);
        let payload = &pushed[0][0];
        assert!(
            payload.content.starts_with(
                "[error] agent: unknown tool 'nonexistent'. Available: run.\n[exit:-1 |"
            ),
            "{:?}",
            payload.content
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cancellation_during_slow_step_preempts_loop() {
        let backend = MockBackend::with(vec![StepOutcome::Final]).slow_step_ms(2_000);
        let (tx, _rx) = mpsc::channel::<LlmEvent>(16);
        let token = CancellationToken::new();
        let kicker = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            kicker.cancel();
        });
        Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE)
            .run_turn(&TurnContext::default(), "go".into(), Vec::new(), tx, token)
            .await
            .unwrap();
        assert_eq!(backend.step_calls.load(Ordering::SeqCst), 1);
        assert!(
            backend.step_tool_counts.lock().is_empty(),
            "slow step ran to completion instead of being preempted"
        );
    }

    #[tokio::test]
    async fn cancellation_mid_batch_answers_every_remaining_call() {
        let entered = Arc::new(Notify::new());
        let mut tools = ToolRegistry::new();
        tools.register(HangingTool {
            entered: entered.clone(),
        });
        let backend = MockBackend::with(vec![StepOutcome::ToolCalls(vec![
            call("c-1", "hang"),
            call("c-2", "echo b"),
            call("c-3", "echo c"),
        ])]);
        let (tx, mut rx) = mpsc::channel::<LlmEvent>(16);
        let token = CancellationToken::new();
        let kicker = token.clone();
        tokio::spawn(async move {
            entered.notified().await;
            kicker.cancel();
        });

        let agent = Agent::new(backend.clone(), Arc::new(tools), None, TOOL_DEADLINE);
        let context = TurnContext::default();
        let turn = agent.run_turn(&context, "go".into(), Vec::new(), tx, token);
        tokio::time::timeout(Duration::from_secs(5), turn)
            .await
            .expect("cancellation did not preempt hung tool")
            .unwrap();
        assert_eq!(backend.step_calls.load(Ordering::SeqCst), 1);

        let events = collect(&mut rx).await;
        let answered: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                LlmEvent::ToolResult { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(answered, ["c-1", "c-2", "c-3"], "{events:?}");

        let pushed = backend.pushed_results.lock();
        let [step] = pushed.as_slice() else {
            panic!("batch was not answered exactly once: {pushed:?}");
        };
        let contents: Vec<(&str, &str)> = step
            .iter()
            .map(|payload| (payload.call_id.as_str(), payload.content.as_str()))
            .collect();
        assert_eq!(
            contents,
            [
                (
                    "c-1",
                    "[error] run: agent turn cancelled during dispatch.\n[exit:-1 | 0ms]"
                ),
                (
                    "c-2",
                    "[error] run: agent turn cancelled before dispatch.\n[exit:-1 | 0ms]"
                ),
                (
                    "c-3",
                    "[error] run: agent turn cancelled before dispatch.\n[exit:-1 | 0ms]"
                ),
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn hung_tool_past_deadline_becomes_error_result_and_turn_continues() {
        let mut tools = ToolRegistry::new();
        tools.register(HangingTool {
            entered: Arc::new(Notify::new()),
        });
        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![call("c-1", "hang")]),
            StepOutcome::Final,
        ]);
        let agent = Agent::new(
            backend.clone(),
            Arc::new(tools),
            None,
            Duration::from_secs(2),
        );
        let (res, _) = tokio::time::timeout(Duration::from_secs(5), run_turn(agent, "go"))
            .await
            .expect("tool deadline did not fire");
        res.unwrap();

        assert_eq!(
            backend.step_calls.load(Ordering::SeqCst),
            2,
            "turn did not continue after the abandoned call"
        );
        let pushed = backend.pushed_results.lock();
        assert_eq!(pushed.len(), 1);
        assert_eq!(pushed[0][0].call_id, "c-1");
        assert!(
            pushed[0][0]
                .content
                .starts_with("[error] run: no result after 2s; call abandoned."),
            "{:?}",
            pushed[0][0].content
        );
    }

    #[tokio::test]
    async fn panicking_tool_becomes_error_result_and_turn_continues() {
        let mut tools = ToolRegistry::new();
        tools.register(PanickingTool);
        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![call("c-1", "web fetch https://example.com")]),
            StepOutcome::Final,
        ]);
        let agent = Agent::new(
            backend.clone(),
            Arc::new(tools),
            None,
            Duration::from_secs(2),
        );

        run_turn(agent, "go").await.0.unwrap();

        assert_eq!(backend.step_calls.load(Ordering::SeqCst), 2);
        let pushed = backend.pushed_results.lock();
        assert!(
            pushed[0][0]
                .content
                .starts_with("[error] run: the tool crashed (not implemented)."),
            "{:?}",
            pushed[0][0].content
        );
    }

    #[tokio::test(start_paused = true)]
    async fn self_bounded_tool_outlives_the_tool_deadline() {
        let mut tools = ToolRegistry::new();
        tools.register(SelfBoundedTool {
            delay: Duration::from_secs(60),
        });
        let backend = MockBackend::with(vec![
            StepOutcome::ToolCalls(vec![call("c-1", "relay")]),
            StepOutcome::Final,
        ]);
        let agent = Agent::new(
            backend.clone(),
            Arc::new(tools),
            None,
            Duration::from_secs(2),
        );

        run_turn(agent, "go").await.0.unwrap();

        let pushed = backend.pushed_results.lock();
        assert_eq!(pushed[0][0].content, "finished");
    }

    #[derive(Debug)]
    struct MockProbe {
        wait_result: Result<(), HealthWaitError>,
        wait_calls: AtomicUsize,
    }

    impl MockProbe {
        fn new(wait_result: Result<(), HealthWaitError>) -> Arc<Self> {
            Arc::new(Self {
                wait_result,
                wait_calls: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl LlmHealthProbe for MockProbe {
        async fn snapshot(&self) -> Option<HealthSnapshot> {
            None
        }

        async fn wait_for_ready(&self, _budget: Duration) -> Result<(), HealthWaitError> {
            self.wait_calls.fetch_add(1, Ordering::SeqCst);
            self.wait_result
        }
    }

    /// Fails each `step` with the next queued error, then answers `Final`.
    #[derive(Debug)]
    struct ErrorInjectingBackend {
        errors: StdMutex<Vec<LlmError>>,
        step_calls: AtomicUsize,
    }

    impl ErrorInjectingBackend {
        fn new(errors: Vec<LlmError>) -> Arc<Self> {
            Arc::new(Self {
                errors: StdMutex::new(errors),
                step_calls: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl TurnBackend for ErrorInjectingBackend {
        async fn push_user(&self, _text: String, _attachments: Vec<Attachment>) -> LlmResult<()> {
            Ok(())
        }

        async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> LlmResult<()> {
            Ok(())
        }

        async fn step(
            &self,
            _tools: Vec<Value>,
            tx: mpsc::Sender<LlmEvent>,
        ) -> LlmResult<StepOutcome> {
            self.step_calls.fetch_add(1, Ordering::SeqCst);
            let next_err = {
                let mut errors = self.errors.lock();
                (!errors.is_empty()).then(|| errors.remove(0))
            };
            if let Some(e) = next_err {
                return Err(e);
            }
            let _ = tx.send(LlmEvent::Delta { text: "ok".into() }).await;
            Ok(StepOutcome::Final)
        }

        async fn push_interrupted_reply(&self, _reply: String) -> LlmResult<()> {
            Ok(())
        }

        async fn set_transient_note(&self, _text: String) -> LlmResult<()> {
            Ok(())
        }
    }

    fn restart_error(err: &LlmError) -> Option<&str> {
        match err {
            LlmError::ServerRestarting(reason) => Some(reason),
            _ => None,
        }
    }

    #[tokio::test]
    async fn replay_once_on_server_restarting() {
        let backend = ErrorInjectingBackend::new(vec![LlmError::ServerRestarting("boom".into())]);
        let probe = MockProbe::new(Ok(()));
        let (res, events) = run_turn(
            Agent::new(
                backend.clone(),
                tools_with_echo(),
                Some(probe.clone()),
                TOOL_DEADLINE,
            ),
            "hello",
        )
        .await;
        res.expect("replay should succeed");

        assert_eq!(
            status_kinds(&events),
            [StatusKind::Restarting, StatusKind::Replaying]
        );
        assert!(events.ends_with(&ok_then_done()), "{events:?}");
        assert_eq!(probe.wait_calls.load(Ordering::SeqCst), 1);
        assert_eq!(backend.step_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn replay_does_not_loop_on_repeated_server_restarting() {
        let backend = ErrorInjectingBackend::new(vec![
            LlmError::ServerRestarting("first".into()),
            LlmError::ServerRestarting("second".into()),
        ]);
        let probe = MockProbe::new(Ok(()));
        let (res, events) = run_turn(
            Agent::new(
                backend.clone(),
                tools_with_echo(),
                Some(probe),
                TOOL_DEADLINE,
            ),
            "hello",
        )
        .await;
        let err = res.expect_err("second ServerRestarting must be terminal");
        assert_eq!(restart_error(&err), Some("second"), "{err}");

        assert_eq!(
            status_kinds(&events),
            [StatusKind::Restarting, StatusKind::Replaying]
        );
        assert_failed_without_done(&events);
        assert_eq!(backend.step_calls.load(Ordering::SeqCst), 2);
    }
}

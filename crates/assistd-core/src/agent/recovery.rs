//! Recovering a model step: one replay after a llama-server crash, and retries of a
//! response cut off at the output-token limit.

use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, warn};

use assistd_ipc::StatusKind;
use assistd_llm::{HealthWaitError, LlmError, LlmEvent, LlmHealthProbe};

use super::Agent;
use super::run::{StepEnd, Turn, status_event};
use crate::recovery::{Component, StatusSeverity};
use crate::recovery_event;

/// How long a turn waits for an LLM restart: the supervisor's worst-case
/// backoff (63s) plus spawn and probe headroom.
const REPLAY_WAIT_BUDGET: Duration = Duration::from_secs(75);

/// Consecutive retries of a step cut off at the output-token limit before
/// the turn fails.
pub(super) const TRUNCATION_RETRY_LIMIT: u32 = 2;

/// Tells the model why its last response vanished and how to avoid a repeat.
pub(super) const TRUNCATION_NOTE: &str = "Your previous response reached the output-token limit before \
     it finished, so it was discarded and none of its tool calls ran. Keep your reasoning \
     brief and your tool calls small: split a long script into several shorter calls.";

impl Agent {
    /// One LLM step. A crash is replayed once after the server is ready
    /// again; any other failure ends the turn with an error and no `Done`.
    pub(super) async fn step_with_replay(&self, turn: &Turn) -> Result<StepEnd, LlmError> {
        let mut restart_attempted = false;
        loop {
            let replay_probe = self.host.health.as_ref().filter(|_| !restart_attempted);
            turn.emit(LlmEvent::StepStarted).await;
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

    /// Tell the client and the model that the last response was cut off so
    /// the next step retries it; errors once [`TRUNCATION_RETRY_LIMIT`]
    /// consecutive retries are spent.
    pub(super) async fn retry_truncated_step(&self, turn: &mut Turn) -> Result<(), LlmError> {
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
        turn.emit(LlmEvent::StepDiscarded).await;
        turn.emit(status_event(
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
}

enum Replay {
    Ready,
    Cancelled,
    Abandoned(String),
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
    turn.emit(LlmEvent::StepDiscarded).await;
    turn.emit(status_event(
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
    turn.emit(status_event(
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
    turn.emit(status_event(
        StatusSeverity::Error,
        Component::Llm,
        StatusKind::Degraded,
        final_msg.to_string(),
    ))
    .await;
    final_msg.to_string()
}

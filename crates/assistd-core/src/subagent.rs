//! Sub-agent runs: the agent loop over a throwaway backend on the shared
//! llama-server, saving the model's final answer and returning its contract as a
//! [`RelayReport`].

use std::fs::{DirBuilder, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use assistd_ipc::{Access, RelayEvent, StatusKind};
use assistd_llm::{LlmBackend, LlmError, LlmEvent, SlotAssignment, TurnBackend};
use assistd_tools::policy::on_blocking_pool;
use assistd_tools::{ToolRegistry, TurnContext};
use assistd_utils::time::saturating_millis;

use crate::agent::AgentSpec;

mod report;

use report::{RelayContract, saved_answer};

pub use report::RelayReport;

const EVENT_BUFFER: usize = 32;

const ANSWER_DIR_MODE: u32 = 0o700;
const ANSWER_FILE_MODE: u32 = 0o600;

/// Asks for the return contract once the run has answered.
const CONTRACT_INSTRUCTION: &str = "Your answer above is saved in full and handed back. \
     Now sum it up for the agent that gave you the task, as JSON: `status` is done when the \
     task is finished, partial when only part of it is, and blocked when you could not make \
     progress; `summary` states your conclusions in a few lines; `evidence` lists the \
     file:line references or URLs they rest on; `caveats` lists what you did not verify or \
     finish.";

/// Why a sub-agent run ended without an answer.
#[derive(Debug, Error)]
pub enum SubAgentError {
    /// The run was cancelled before the model answered.
    #[error("sub-agent cancelled before it answered")]
    Cancelled,
    /// The run failed.
    #[error("sub-agent failed: {0}")]
    Llm(#[from] LlmError),
    /// The answer could not be saved.
    #[error("cannot save the sub-agent's answer to {}: {source}", path.display())]
    Save {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// How a sub-agent run ended, and what its tool calls touched either way.
#[derive(Debug)]
pub struct SubAgentRun {
    pub report: Result<RelayReport, SubAgentError>,
    /// Paths are as the run's tools saw them.
    pub accesses: Vec<Access>,
}

/// What a sub-agent run streamed: its last step's text, `None` when it never
/// finished, how many tool-calling steps it took, and what those touched.
struct FinalAnswer {
    text: Option<String>,
    tool_steps: u32,
    accesses: Vec<Access>,
}

/// Runs sub-agents in backends of their own beside `parent`'s, on the
/// same server and model.
#[derive(Debug)]
pub struct SubAgentRunner {
    parent: Arc<dyn LlmBackend>,
    agents: AgentSpec,
}

impl SubAgentRunner {
    /// A runner beside `parent` whose runs are driven by agents built from `agents`.
    pub fn new(parent: Arc<dyn LlmBackend>, agents: AgentSpec) -> Self {
        Self { parent, agents }
    }

    /// Run `task` on a backend with an empty history, `system_prompt` and `tools` until the model
    /// answers, save that answer whole to the new file `answer_file`, then ask for its
    /// contract. Only the report reaches the model; `turn`, the calling turn, answers the
    /// run's prompts and is shown its tool calls, and the backend is dropped when the
    /// run ends or `cancel` fires. What its tool calls touched comes back however it ends.
    pub async fn run(
        &self,
        system_prompt: String,
        task: String,
        tools: Arc<ToolRegistry>,
        answer_file: PathBuf,
        cancel: CancellationToken,
        turn: &TurnContext,
    ) -> SubAgentRun {
        let isolated = self
            .parent
            .isolated_backend(system_prompt, SlotAssignment::Unassigned);
        let backend = match isolated {
            Ok(backend) => backend,
            Err(error) => {
                return SubAgentRun {
                    report: Err(error.into()),
                    accesses: Vec::new(),
                };
            }
        };
        let agent = self
            .agents
            .agent(Arc::clone(&backend) as Arc<dyn TurnBackend>, tools);
        let (tx, rx) = mpsc::channel(EVENT_BUFFER);
        let (outcome, finished) = tokio::join!(
            agent.run_turn(turn, task, Vec::new(), tx, cancel.clone()),
            final_answer(rx, turn)
        );
        let FinalAnswer {
            text,
            tool_steps,
            accesses,
        } = finished;
        let report = match outcome {
            Ok(()) => {
                let answered = Answered { text, tool_steps };
                report_of(answered, backend.as_ref(), answer_file, &cancel, turn).await
            }
            Err(error) => Err(error.into()),
        };
        SubAgentRun { report, accesses }
    }
}

/// What a run that ended without an error streamed: its last step's text, `None` when
/// it never finished, and how many tool-calling steps it took.
struct Answered {
    text: Option<String>,
    tool_steps: u32,
}

/// Save the run's answer whole to the new file `answer_file`, ask `backend` for its
/// contract, and tell `turn` the run answered.
async fn report_of(
    answered: Answered,
    backend: &dyn LlmBackend,
    answer_file: PathBuf,
    cancel: &CancellationToken,
    turn: &TurnContext,
) -> Result<RelayReport, SubAgentError> {
    let answer = answered.text.ok_or(SubAgentError::Cancelled)?;
    let saved = saved_answer(&answer);
    let answer_file = save_answer(saved.contents, answer_file).await?;
    let contract = contract_of(backend, cancel).await?;
    turn.report_relay(RelayEvent::Answered {
        answer_chars: u32::try_from(saved.chars).unwrap_or(u32::MAX),
        tool_steps: answered.tool_steps,
        answer_file: answer_file.clone(),
    })
    .await;
    Ok(RelayReport::new(contract, answer_file))
}

/// Write `contents` to the new owner-only file `path`, creating its directory,
/// and return `path`.
async fn save_answer(contents: String, path: PathBuf) -> Result<PathBuf, SubAgentError> {
    on_blocking_pool(move || match write_new_file(&path, contents.as_bytes()) {
        Ok(()) => Ok(path),
        Err(source) => Err(SubAgentError::Save { path, source }),
    })
    .await
}

fn write_new_file(path: &Path, contents: &[u8]) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        DirBuilder::new()
            .recursive(true)
            .mode(ANSWER_DIR_MODE)
            .create(dir)?;
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(ANSWER_FILE_MODE)
        .open(path)?
        .write_all(contents)
}

/// Ask `backend` for its contract, held to the contract schema by grammar. A reply
/// that cannot be read becomes a contract saying so, since the answer is already saved.
async fn contract_of(
    backend: &dyn LlmBackend,
    cancel: &CancellationToken,
) -> Result<RelayContract, SubAgentError> {
    let reply = tokio::select! {
        biased;
        () = cancel.cancelled() => return Err(SubAgentError::Cancelled),
        reply = backend.complete_with_schema(
            CONTRACT_INSTRUCTION.into(),
            RelayContract::schema(),
        ) => reply?,
    };
    Ok(RelayContract::parse(&reply).unwrap_or_else(|error| {
        warn!(target: "assistd::relay", %error, "sub-agent's contract is unreadable");
        RelayContract::unreadable()
    }))
}

/// Collect the run's answer, the text of its last step, while showing `turn`
/// what the run streams.
async fn final_answer(mut rx: mpsc::Receiver<LlmEvent>, turn: &TurnContext) -> FinalAnswer {
    let mut text = String::new();
    let mut done = false;
    let mut tool_steps = 0;
    let mut accesses = Vec::new();
    let mut compacting_since = None;
    while let Some(mut event) = rx.recv().await {
        match &mut event {
            LlmEvent::StepStarted | LlmEvent::StepDiscarded => text.clear(),
            LlmEvent::Delta { text: delta } => text.push_str(delta),
            LlmEvent::ToolCallsRequested { .. } => tool_steps += 1,
            LlmEvent::ToolResult {
                accesses: touched, ..
            } => accesses.append(touched),
            LlmEvent::Done => done = true,
            _ => {}
        }
        if let Some(shown) = relay_event_of(event, &mut compacting_since) {
            turn.report_relay(shown).await;
        }
    }
    FinalAnswer {
        text: done.then_some(text),
        tool_steps,
        accesses,
    }
}

/// What the user is shown of `event`, timing a compaction from the
/// announcement that starts it to the one that ends it.
fn relay_event_of(event: LlmEvent, compacting_since: &mut Option<Instant>) -> Option<RelayEvent> {
    match event {
        LlmEvent::Delta { text } => Some(RelayEvent::Delta { text }),
        LlmEvent::ReasoningDelta { text } => Some(RelayEvent::ReasoningDelta { text }),
        LlmEvent::ContextSize { total, .. } => Some(RelayEvent::ContextSize { total }),
        LlmEvent::ToolCall {
            name, arguments, ..
        } => Some(RelayEvent::ToolCall {
            name,
            args: arguments,
        }),
        LlmEvent::ToolResult { name, result, .. } => Some(RelayEvent::ToolResult {
            name,
            result: result.without_attachments(),
        }),
        LlmEvent::Status {
            event: StatusKind::CompactingHistory,
            ..
        } => {
            *compacting_since = Some(Instant::now());
            Some(RelayEvent::Compacting)
        }
        LlmEvent::Status {
            event: StatusKind::HistoryCompacted,
            ..
        } => compacting_since.take().map(|since| RelayEvent::Compacted {
            duration_ms: saturating_millis(since.elapsed()),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests;

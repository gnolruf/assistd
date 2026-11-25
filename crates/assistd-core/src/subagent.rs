//! Sub-agent runs: the agent loop over a throwaway backend on the shared
//! llama-server, saving the model's final answer and returning it as a
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

use assistd_ipc::{Access, RelayEvent, StatusKind};
use assistd_llm::{LlmBackend, LlmError, LlmEvent, SlotAssignment, TurnBackend};
use assistd_tools::policy::on_blocking_pool;
use assistd_tools::{ToolRegistry, TurnContext};
use assistd_utils::time::saturating_millis;

use crate::agent::AgentSpec;

mod report;

use report::{RelayStatus, saved_answer};

pub use report::RelayReport;

const EVENT_BUFFER: usize = 32;

const ANSWER_DIR_MODE: u32 = 0o700;
const ANSWER_FILE_MODE: u32 = 0o600;

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
/// finished, whether its tools were withdrawn, how many tool-calling steps it took,
/// and what those touched.
struct FinalAnswer {
    text: Option<String>,
    status: RelayStatus,
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
    /// answers, and save that answer whole to the new file `answer_file`. Only the report
    /// reaches the model; `turn`, the calling turn, answers the run's prompts and is shown
    /// its tool calls, and the backend is dropped when the run ends or `cancel` fires.
    /// What its tool calls touched comes back however it ends.
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
        let agent = self.agents.agent(backend as Arc<dyn TurnBackend>, tools);
        let (tx, rx) = mpsc::channel(EVENT_BUFFER);
        let (outcome, finished) = tokio::join!(
            agent.run_turn(turn, task, Vec::new(), tx, cancel),
            final_answer(rx, turn)
        );
        let FinalAnswer {
            text,
            status,
            tool_steps,
            accesses,
        } = finished;
        let report = match outcome {
            Ok(()) => {
                let answered = Answered {
                    text,
                    status,
                    tool_steps,
                };
                report_of(answered, answer_file, turn).await
            }
            Err(error) => Err(error.into()),
        };
        SubAgentRun { report, accesses }
    }
}

/// What a run that ended without an error streamed: its last step's text, `None` when
/// it never finished, whether its tools were withdrawn, and how many tool-calling
/// steps it took.
struct Answered {
    text: Option<String>,
    status: RelayStatus,
    tool_steps: u32,
}

/// Save the run's answer whole to the new file `answer_file` and tell `turn` the run
/// answered.
async fn report_of(
    answered: Answered,
    answer_file: PathBuf,
    turn: &TurnContext,
) -> Result<RelayReport, SubAgentError> {
    let answer = answered.text.ok_or(SubAgentError::Cancelled)?;
    let saved = saved_answer(&answer);
    let answer_file = save_answer(saved.contents, answer_file).await?;
    turn.report_relay(RelayEvent::Answered {
        answer_chars: u32::try_from(saved.chars).unwrap_or(u32::MAX),
        tool_steps: answered.tool_steps,
        answer_file: answer_file.clone(),
    })
    .await;
    Ok(RelayReport::new(answered.status, answer, answer_file))
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

/// Collect the run's answer, the text of its last step, while showing `turn`
/// what the run streams.
async fn final_answer(mut rx: mpsc::Receiver<LlmEvent>, turn: &TurnContext) -> FinalAnswer {
    let mut text = String::new();
    let mut done = false;
    let mut status = RelayStatus::Done;
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
            LlmEvent::Status {
                event: StatusKind::ToolsWithdrawn,
                ..
            } => status = RelayStatus::Partial,
            LlmEvent::Done => done = true,
            _ => {}
        }
        if let Some(shown) = relay_event_of(event, &mut compacting_since) {
            turn.report_relay(shown).await;
        }
    }
    FinalAnswer {
        text: done.then_some(text),
        status,
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

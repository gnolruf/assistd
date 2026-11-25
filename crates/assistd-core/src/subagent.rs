//! Sub-agent runs: the agent loop over a throwaway backend on the shared
//! llama-server, saving what the model wrote and returning its last message as a
//! [`RelayReport`].

use std::fs::{DirBuilder, OpenOptions};
use std::io::{self, Write};
use std::mem;
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

use report::{RelayStatus, Shortfall, Written, saved_answer};

pub use report::RelayReport;

const EVENT_BUFFER: usize = 32;

const ANSWER_DIR_MODE: u32 = 0o700;
const ANSWER_FILE_MODE: u32 = 0o600;

/// Why a sub-agent run ended with nothing to report.
#[derive(Debug, Error)]
pub enum SubAgentError {
    /// The run was cancelled before the model answered.
    #[error("sub-agent cancelled before it answered")]
    Cancelled,
    /// The run failed before any step streamed text.
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

/// The text a run streams, kept step by step.
#[derive(Default)]
struct StepTexts {
    /// The text of each earlier step that streamed any, oldest first.
    earlier: Vec<String>,
    current: String,
    /// The longest attempt at the current step that was cut off at the output-token limit.
    longest_cut_off: String,
    /// The current step asked for tools, so its text outlasts the start of the next step.
    called_tools: bool,
}

impl StepTexts {
    fn observe(&mut self, event: &LlmEvent) {
        match event {
            LlmEvent::StepStarted => {
                let text = mem::take(&mut self.current);
                if mem::take(&mut self.called_tools) && !text.is_empty() {
                    self.earlier.push(text);
                }
            }
            LlmEvent::StepDiscarded => {
                let attempt = mem::take(&mut self.current);
                if attempt.len() > self.longest_cut_off.len() {
                    self.longest_cut_off = attempt;
                }
            }
            LlmEvent::Delta { text } => self.current.push_str(text),
            LlmEvent::ToolCallsRequested { .. } => {
                self.called_tools = true;
                self.longest_cut_off.clear();
            }
            _ => {}
        }
    }

    /// What a run that answered wrote: its last step's text is the message.
    fn answered(self) -> Written {
        Written {
            notes: self.earlier,
            message: self.current,
        }
    }

    /// What a run that stopped early wrote: the longest attempt at the step it stopped
    /// in, or else the text of the last step that streamed any. `None` when no step did.
    fn stopped_early(mut self) -> Option<Written> {
        let attempt = if self.current.len() >= self.longest_cut_off.len() {
            self.current
        } else {
            self.longest_cut_off
        };
        let message = if attempt.is_empty() {
            self.earlier.pop()?
        } else {
            attempt
        };
        Some(Written {
            notes: self.earlier,
            message,
        })
    }
}

/// What a sub-agent run streamed: its text, whether it finished, whether its tools
/// were withdrawn, how many tool-calling steps it took, and what those touched.
struct Streamed {
    texts: StepTexts,
    done: bool,
    tools_withdrawn: bool,
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
    /// answers, and save what it wrote to the new file `answer_file`. A run that fails after
    /// streaming text reports that text as partial; one that streamed none is an error.
    /// `turn`, the calling turn, answers the run's prompts and is shown its tool calls, and
    /// the backend is dropped when the run ends or `cancel` fires.
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
        let (outcome, streamed) = tokio::join!(
            agent.run_turn(turn, task, Vec::new(), tx, cancel),
            streamed_by_run(rx, turn)
        );
        let Streamed {
            texts,
            done,
            tools_withdrawn,
            tool_steps,
            accesses,
        } = streamed;
        let ended = match outcome {
            Ok(()) if !done => Err(SubAgentError::Cancelled),
            Ok(()) if tools_withdrawn => Ok((
                RelayStatus::Partial(Shortfall::ToolsWithdrawn),
                texts.answered(),
            )),
            Ok(()) => Ok((RelayStatus::Done, texts.answered())),
            Err(error) => match texts.stopped_early() {
                Some(written) => {
                    warn!(target: "assistd::relay", %error, "sub-agent stopped early; reporting what it wrote");
                    Ok((RelayStatus::Partial(Shortfall::of(&error)), written))
                }
                None => Err(error.into()),
            },
        };
        let report = match ended {
            Ok((status, written)) => {
                report_of(status, written, tool_steps, answer_file, turn).await
            }
            Err(error) => Err(error),
        };
        SubAgentRun { report, accesses }
    }
}

/// Save what the run wrote to the new file `answer_file` and tell `turn` the run
/// answered.
async fn report_of(
    status: RelayStatus,
    written: Written,
    tool_steps: u32,
    answer_file: PathBuf,
    turn: &TurnContext,
) -> Result<RelayReport, SubAgentError> {
    let saved = saved_answer(&written);
    let answer_file = save_answer(saved.contents, answer_file).await?;
    turn.report_relay(RelayEvent::Answered {
        answer_chars: u32::try_from(saved.chars).unwrap_or(u32::MAX),
        tool_steps,
        answer_file: answer_file.clone(),
    })
    .await;
    Ok(RelayReport::new(status, written.message, answer_file))
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

/// Collect what the run streams while showing it to `turn`.
async fn streamed_by_run(mut rx: mpsc::Receiver<LlmEvent>, turn: &TurnContext) -> Streamed {
    let mut streamed = Streamed {
        texts: StepTexts::default(),
        done: false,
        tools_withdrawn: false,
        tool_steps: 0,
        accesses: Vec::new(),
    };
    let mut compacting_since = None;
    while let Some(mut event) = rx.recv().await {
        streamed.texts.observe(&event);
        match &mut event {
            LlmEvent::ToolCallsRequested { .. } => streamed.tool_steps += 1,
            LlmEvent::ToolResult {
                accesses: touched, ..
            } => streamed.accesses.append(touched),
            LlmEvent::Status {
                event: StatusKind::ToolsWithdrawn,
                ..
            } => streamed.tools_withdrawn = true,
            LlmEvent::Done => streamed.done = true,
            _ => {}
        }
        if let Some(shown) = relay_event_of(event, &mut compacting_since) {
            turn.report_relay(shown).await;
        }
    }
    streamed
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

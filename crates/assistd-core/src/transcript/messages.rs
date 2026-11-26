//! The messages an agent's history is stored as, drafted from what its run streams.

use std::mem;
use std::time::Instant;

use assistd_ipc::ToolReport;
use assistd_llm::ToolCall;
use assistd_memory::{PersistedMessage, PersistedToolCall, Reasoning, ToolOutcome};
use assistd_utils::time::saturating_millis;

/// What the model step in flight has streamed, until it becomes a message.
#[derive(Default)]
pub(crate) struct StepDraft {
    text: String,
    reasoning: Option<ReasoningTrace>,
}

impl StepDraft {
    /// The reply text streamed so far.
    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn push_text(&mut self, delta: &str) {
        self.text.push_str(delta);
    }

    pub(crate) fn push_reasoning(&mut self, delta: &str) {
        self.reasoning
            .get_or_insert_with(ReasoningTrace::new)
            .push(delta);
    }

    /// Drop what the step streamed: it runs again or ends the run without it.
    pub(crate) fn discard(&mut self) {
        *self = Self::default();
    }

    /// The step as the message that requested `calls`, its text kept unless blank.
    pub(crate) fn requesting(&mut self, calls: &[ToolCall]) -> PersistedMessage {
        let narration = mem::take(&mut self.text);
        let content = if narration.trim().is_empty() {
            String::new()
        } else {
            narration
        };
        let calls = calls
            .iter()
            .map(|call| PersistedToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
            .collect();
        PersistedMessage::assistant_tool_calls(content, calls, self.take_reasoning())
    }

    /// The step as the reply that ended the run; `None` when it streamed no text.
    pub(crate) fn reply(&mut self) -> Option<PersistedMessage> {
        let reasoning = self.take_reasoning();
        let text = mem::take(&mut self.text);
        (!text.is_empty()).then(|| PersistedMessage::assistant_text(text, reasoning))
    }

    /// `reply`, which the history keeps in place of the step that was cut off.
    pub(crate) fn interrupted(&mut self, reply: String) -> PersistedMessage {
        self.text.clear();
        PersistedMessage::assistant_text(reply, self.take_reasoning())
    }

    fn take_reasoning(&mut self) -> Option<Reasoning> {
        self.reasoning.take().map(ReasoningTrace::into_reasoning)
    }
}

/// Reasoning streamed so far for the current step.
struct ReasoningTrace {
    text: String,
    started_at: Instant,
    last_delta_at: Instant,
}

impl ReasoningTrace {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            text: String::new(),
            started_at: now,
            last_delta_at: now,
        }
    }

    fn push(&mut self, delta: &str) {
        self.text.push_str(delta);
        self.last_delta_at = Instant::now();
    }

    fn into_reasoning(self) -> Reasoning {
        let elapsed = self.last_delta_at.duration_since(self.started_at);
        Reasoning {
            text: self.text,
            duration_ms: saturating_millis(elapsed),
        }
    }
}

/// The message `result` of tool call `call_id` to tool `name` is stored as.
pub(crate) fn tool_result_message(
    call_id: &str,
    name: &str,
    result: &ToolReport,
    read_host: bool,
) -> PersistedMessage {
    let outcome = ToolOutcome {
        exit_code: result.exit_code.into(),
        duration_ms: result.duration_ms,
    };
    PersistedMessage::tool_result(result.output.clone(), call_id, name, outcome, read_host)
}

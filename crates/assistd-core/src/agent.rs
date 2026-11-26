//! The agent: what one is, the loop that runs its turn, and how one is started on the
//! server the daemon's agents share.

use std::sync::Arc;
use std::time::Duration;

use assistd_ipc::Attachment;
use assistd_llm::{LlmBackend, LlmHealthProbe, LlmResult, TurnBackend};

mod definition;
mod dispatch;
mod outcome;
mod recovery;
mod run;

pub use definition::{AgentDefinition, AgentLimits, McpAccess, Permissions};
pub use outcome::{AgentEnd, AgentOutcome, Written};

/// Where the daemon's agents run: the backend whose server they share, the probe a step
/// waits on to be replayed once after a llama-server crash (without one a crash ends the
/// turn), and how long a tool call may run before the model is told it failed.
#[derive(Debug, Clone)]
pub struct AgentHost {
    server: Arc<dyn LlmBackend>,
    health: Option<Arc<dyn LlmHealthProbe>>,
    tool_deadline: Duration,
}

impl AgentHost {
    pub fn new(
        server: Arc<dyn LlmBackend>,
        health: Option<Arc<dyn LlmHealthProbe>>,
        tool_deadline: Duration,
    ) -> Self {
        Self {
            server,
            health,
            tool_deadline,
        }
    }

    /// The agent over the server backend's own history, which was opened with
    /// `definition`'s prompt and slot.
    pub fn main(&self, definition: AgentDefinition) -> Agent {
        self.agent_over(Arc::clone(&self.server), definition)
    }

    /// An agent over an empty history of its own on the same server and model, opened
    /// with `definition`'s prompt and slot. Errors when the server backend opens none.
    pub fn child(&self, definition: AgentDefinition) -> LlmResult<Agent> {
        let history = self
            .server
            .isolated_backend(definition.system_prompt.clone(), definition.slot)?;
        Ok(self.agent_over(history, definition))
    }

    fn agent_over(&self, history: Arc<dyn LlmBackend>, definition: AgentDefinition) -> Agent {
        Agent {
            host: self.clone(),
            definition,
            backend: Arc::clone(&history) as Arc<dyn TurnBackend>,
            history,
        }
    }
}

/// One agent: its definition and the history its turns run over.
#[derive(Debug, Clone)]
pub struct Agent {
    host: AgentHost,
    definition: AgentDefinition,
    history: Arc<dyn LlmBackend>,
    /// What the loop drives: `history`, or a wrapper of it.
    backend: Arc<dyn TurnBackend>,
}

impl Agent {
    pub fn history(&self) -> &Arc<dyn LlmBackend> {
        &self.history
    }

    /// Have the loop drive `backend`, a wrapper of this agent's history, in its place.
    pub fn driven_through(mut self, backend: Arc<dyn TurnBackend>) -> Self {
        self.backend = backend;
        self
    }
}

/// What an agent's turn is asked: the text, and any images sent with it.
#[derive(Debug)]
pub struct AgentTask {
    pub text: String,
    pub attachments: Vec<Attachment>,
}

impl From<String> for AgentTask {
    fn from(text: String) -> Self {
        Self {
            text,
            attachments: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests;

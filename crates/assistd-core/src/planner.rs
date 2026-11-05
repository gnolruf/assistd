//! Plan mode: a turn over a conversation that may run only what asks for nothing, ending in a
//! [`Plan`] of steps and the approvals each expects to need.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::{Map, Value};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use assistd_ipc::Plan;
use assistd_llm::{
    LlmBackend, LlmError, LlmEvent, LlmHealthProbe, LlmResult, StepOutcome, ToolResultPayload,
};
use assistd_tools::policy::on_blocking_pool;
use assistd_tools::{
    Attachment, CONFIRM_ROUTER, ConfirmRouter, GrantCheck, SearchPath, Tool, ToolCatalog,
    ToolError, ToolOutput, ToolRegistry,
};

use crate::agent::Agent;

mod draft;

use draft::PlanDraft;

/// Rendered at the tail of the conversation on every step of a planning turn.
const PLAN_PREAMBLE: &str = "Plan mode: do not do the work yet. Find out what it needs \
     using only commands that change nothing, then reply with a plan: the steps you would \
     take and what each would need approved. Anything that would ask for confirmation is \
     refused while planning; when a command is refused, describe that step in the plan \
     instead of looking for another way to run it.";

/// Appended to a planning tool's result when something in it was refused for asking.
const REFUSED_NOTE: &str = "\n[plan mode: this needed approval, so it was refused and did not \
     run. Describe the step, and the approval it needs, in your plan instead.]";

/// Asks for the plan once the planning turn has answered.
const PLAN_INSTRUCTION: &str = "Now give your plan as JSON. `summary` says in a few lines \
     what the work will do. `steps` lists the steps in order, each with a `description` and \
     the `actions` it needs approved to run without asking. An action's `kind` is `program` \
     to run a program that is not read-only (`target` is its name, and `subcommand` its \
     subcommand for a program such as git that is approved per subcommand), `web_host` to \
     reach a web host (`target` is the host name), `mcp_tool` to call an MCP tool (`target` \
     is its full name), `app` to launch an app (`target` is its desktop entry id), or \
     `write` to change files (`target` is the absolute directory they are in). A step that \
     needs no approval has no actions.";

/// What a planning turn is asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanAsk {
    /// Plan this piece of work.
    Work(String),
    /// Revise the conversation's last plan with this feedback.
    Revise(String),
}

/// Why a planning turn ended without a plan.
#[derive(Debug, Error)]
pub enum PlanError {
    #[error("planning was cancelled before the plan was made")]
    Cancelled,
    #[error("planning failed: {0}")]
    Llm(#[from] LlmError),
    #[error("the model's plan could not be read: {0}")]
    Unreadable(#[source] serde_json::Error),
}

/// What a planning turn may use: tools that ask for nothing, and the checks its anticipated
/// actions face.
#[derive(Debug, Clone)]
pub struct PlanningTools {
    pub registry: Arc<ToolRegistry>,
    pub checks: GrantCheck,
}

impl Default for PlanningTools {
    /// No tools, and checks that find no program.
    fn default() -> Self {
        Self {
            registry: Arc::new(ToolRegistry::new()),
            checks: GrantCheck::new(SearchPath { dirs: Vec::new() }),
        }
    }
}

/// Runs planning turns. Every ask for confirmation in one is refused without prompting, so it
/// runs only what needs no approval.
#[derive(Debug)]
pub struct Planner {
    tools: PlanningTools,
    /// The tools ordinary turns are offered, which an anticipated MCP tool must be among.
    catalog: Arc<ToolCatalog>,
    health: Option<Arc<dyn LlmHealthProbe>>,
    tool_deadline: Duration,
}

impl Planner {
    /// A planner over `tools`; `health` and `tool_deadline` act as they do for [`Agent`].
    pub fn new(
        tools: PlanningTools,
        catalog: Arc<ToolCatalog>,
        health: Option<Arc<dyn LlmHealthProbe>>,
        tool_deadline: Duration,
    ) -> Self {
        Self {
            tools,
            catalog,
            health,
            tool_deadline,
        }
    }

    /// Run a planning turn on `ask` in `conversation`, streaming it on `tx`, then ask for the
    /// plan, each action checked against the machine. The plan-mode preamble rides at the
    /// conversation's tail, so its system prompt is unchanged; `request_id` labels refusals.
    pub async fn plan(
        &self,
        conversation: Arc<dyn LlmBackend>,
        ask: PlanAsk,
        request_id: String,
        tx: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> Result<Plan, PlanError> {
        let conversation: Arc<dyn LlmBackend> = Arc::new(PlanModeConversation::new(conversation));
        let agent = Agent::new(
            Arc::clone(&conversation),
            Arc::clone(&self.tools.registry),
            self.health.clone(),
            self.tool_deadline,
        );
        let router = ConfirmRouter::for_planning(request_id);
        let turn = agent.run_turn(ask.into_text(), Vec::new(), tx, cancel.clone());
        CONFIRM_ROUTER.scope(router, turn).await?;
        if cancel.is_cancelled() {
            return Err(PlanError::Cancelled);
        }
        let draft = draft_of(conversation.as_ref(), &cancel).await?;
        let checks = self.tools.checks.clone();
        let tools = self.catalog.snapshot();
        Ok(on_blocking_pool(move || draft.into_plan(&checks, &tools)).await)
    }
}

impl PlanAsk {
    fn into_text(self) -> String {
        match self {
            Self::Work(text) => text,
            Self::Revise(feedback) => format!("Revise your plan with this feedback: {feedback}"),
        }
    }
}

/// Ask `conversation` for its plan, held to the draft schema by grammar.
async fn draft_of(
    conversation: &dyn LlmBackend,
    cancel: &CancellationToken,
) -> Result<PlanDraft, PlanError> {
    let reply = tokio::select! {
        biased;
        () = cancel.cancelled() => return Err(PlanError::Cancelled),
        reply = conversation.complete_with_schema(PLAN_INSTRUCTION.into(), PlanDraft::schema()) => {
            reply?
        }
    };
    serde_json::from_str(&reply).map_err(PlanError::Unreadable)
}

/// A conversation whose every step ends with [`PLAN_PREAMBLE`], ahead of any one-shot note
/// the agent loop adds.
#[derive(Debug)]
struct PlanModeConversation {
    inner: Arc<dyn LlmBackend>,
    note: Mutex<Option<String>>,
}

impl PlanModeConversation {
    fn new(inner: Arc<dyn LlmBackend>) -> Self {
        Self {
            inner,
            note: Mutex::new(None),
        }
    }
}

#[async_trait]
impl LlmBackend for PlanModeConversation {
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> LlmResult<()> {
        self.inner.generate(prompt, tx).await
    }

    async fn push_user(&self, text: String, attachments: Vec<Attachment>) -> LlmResult<()> {
        self.inner.push_user(text, attachments).await
    }

    async fn push_tool_results(&self, results: Vec<ToolResultPayload>) -> LlmResult<()> {
        self.inner.push_tool_results(results).await
    }

    async fn step(&self, tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> LlmResult<StepOutcome> {
        let note = match self.note.lock().take() {
            Some(extra) => format!("{PLAN_PREAMBLE}\n\n{extra}"),
            None => PLAN_PREAMBLE.to_string(),
        };
        self.inner.set_transient_note(note).await?;
        self.inner.step(tools, tx).await
    }

    async fn set_transient_context(&self, text: String) -> LlmResult<()> {
        self.inner.set_transient_context(text).await
    }

    async fn set_transient_note(&self, text: String) -> LlmResult<()> {
        *self.note.lock() = Some(text);
        Ok(())
    }

    async fn complete_with_schema(
        &self,
        instruction: String,
        schema: Map<String, Value>,
    ) -> LlmResult<String> {
        self.inner.complete_with_schema(instruction, schema).await
    }
}

/// A tool offered while planning, whose result tells the model when something it tried was
/// refused for needing approval.
#[derive(Debug)]
pub(crate) struct PlanModeTool<T> {
    inner: T,
}

impl<T: Tool> PlanModeTool<T> {
    pub(crate) fn new(inner: T) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<T: Tool> Tool for PlanModeTool<T> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> Cow<'_, str> {
        self.inner.description()
    }

    fn parameters_schema(&self) -> Value {
        self.inner.parameters_schema()
    }

    fn bounded_by_tool_deadline(&self) -> bool {
        self.inner.bounded_by_tool_deadline()
    }

    async fn invoke(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let before = refused_while_planning();
        let mut output = self.inner.invoke(args).await?;
        if refused_while_planning() > before {
            output.output.push_str(REFUSED_NOTE);
        }
        Ok(output)
    }
}

/// How many asks the planning router in scope has refused; 0 outside a planning turn.
fn refused_while_planning() -> usize {
    CONFIRM_ROUTER
        .try_with(|router| router.refused_while_planning())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;

//! Plan mode: a turn over a conversation that may run only what asks for nothing, ending in a
//! plan the user accepts or revises, or in a task prompt rewritten with a [`Plan`] of its steps.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use assistd_ipc::{HostName, ModeRecommendation, PermissionMode, Plan, TaskPrompt, TaskSpecError};
use assistd_llm::{
    LlmBackend, LlmError, LlmEvent, LlmHealthProbe, LlmResult, StepOutcome, ToolResultPayload,
};
use assistd_tools::policy::on_blocking_pool;
use assistd_tools::{
    ActionCheck, Attachment, ConfirmRouter, SearchPath, Tool, ToolCatalog, ToolError, ToolOutput,
    ToolRegistry, TurnContext,
};

use crate::agent::Agent;

mod draft;
mod recommend;
mod switch;

use draft::RefinementDraft;
use recommend::recommended_mode;
use switch::{EnterPlanModeTool, ExitPlanModeTool};

/// Opens the note rendered at the tail of the conversation on every step of a planning turn.
const PLAN_MODE_RULES: &str = "Plan mode: do not do the work yet. Find out what it needs \
     using only commands that change nothing. Anything that would ask for confirmation is \
     refused while planning; when a command is refused, note that the step needs approval \
     instead of looking for another way to run it.";

/// Ends the note of a refine's planning turn, whose plan is asked for afterwards.
const FINDINGS_ENDING: &str = "Once you know enough, reply with only a short list of what \
     you found: the paths, commands and hosts the work would use, and anything missing or \
     refused. Do not write the plan or the steps yet; you will be asked for them next, in a \
     fixed format.";

/// Ends the note of a chat turn while it plans.
const CHAT_PLAN_ENDING: &str = "If something about the work is unclear, ask the user and \
     stop. Once you know enough, call exit_plan_mode with the whole plan; do not write it as \
     a reply. The user accepts it there, and you then carry it out, or tells you what to \
     change.";

/// Appended to a planning tool's result when something in it was refused for asking.
const REFUSED_NOTE: &str = "\n[plan mode: this needed approval, so it was refused and did not \
     run. Note the step, and the approval it needs, for your plan instead.]";

/// Asks for the task prompt rewritten, and its plan, once the planning turn has answered.
const REFINE_INSTRUCTION: &str = "This work will run unattended, with nobody to answer \
     questions. Now give, as JSON, `prompt`: the work rewritten as a self-contained brief for \
     that run. Say what it is for; list the steps, naming the exact paths, commands and hosts \
     you found; say what done looks like and where the result goes; and say what to do when \
     something fails or there is nothing to do. Keep everything the user asked for and add \
     no work they did not ask for. Then give the plan for that brief.";

/// How the plan is laid out, after the refine instruction.
const PLAN_LAYOUT: &str = "`summary` says in a few lines \
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
    /// Revise `shown`, what the user was given from the conversation's last answer, with
    /// `feedback`.
    Revise { shown: String, feedback: String },
}

/// A task prompt rewritten for unattended runs, the plan of the rewrite, and the mode its
/// runs are best given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refinement {
    pub prompt: TaskPrompt,
    pub plan: Plan,
    pub recommended: ModeRecommendation,
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
    #[error("the model's prompt was refused: {0}")]
    Prompt(#[from] TaskSpecError),
}

/// What a planning turn may use: tools that ask for nothing, and the checks its anticipated
/// actions face.
#[derive(Debug, Clone)]
pub struct PlanningTools {
    pub registry: Arc<ToolRegistry>,
    pub checks: ActionCheck,
}

impl Default for PlanningTools {
    /// No tools, and checks that find no program and no approval.
    fn default() -> Self {
        Self {
            registry: Arc::new(ToolRegistry::new()),
            checks: ActionCheck::new(SearchPath { dirs: Vec::new() }, None),
        }
    }
}

/// What the planned work will run under, which decides the actions it will ask about.
#[derive(Debug, Clone, Default)]
pub struct WorkPermissions {
    pub mode: PermissionMode,
    /// Hosts it fetches from unasked, on top of the approved hosts.
    pub allowed_hosts: Vec<HostName>,
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
    /// Auto mode may be recommended.
    auto_offered: bool,
}

impl Planner {
    /// A planner over `tools`; `health` and `tool_deadline` act as they do for [`Agent`], and
    /// a refine recommends Auto only when `auto_offered`.
    pub fn new(
        tools: PlanningTools,
        catalog: Arc<ToolCatalog>,
        health: Option<Arc<dyn LlmHealthProbe>>,
        tool_deadline: Duration,
        auto_offered: bool,
    ) -> Self {
        Self {
            tools,
            catalog,
            health,
            tool_deadline,
            auto_offered,
        }
    }

    /// An agent for the chat turn `turn` over `conversation`. It acts with `acting` and a tool
    /// that asks the user to plan first; while the turn plans, it is offered the planning tools
    /// and one that hands the plan to the user, with the plan-mode note at the conversation's
    /// tail.
    pub fn chat_agent(
        &self,
        turn: &TurnContext,
        conversation: Arc<dyn LlmBackend>,
        acting: &ToolRegistry,
    ) -> Agent {
        let mut acting = acting.clone();
        acting.register(EnterPlanModeTool);
        let mut planning = ToolRegistry::clone(&self.tools.registry);
        planning.register(ExitPlanModeTool);
        Agent::new(
            PlanModeConversation::new(conversation, CHAT_PLAN_ENDING, turn.clone()),
            Arc::new(acting),
            self.health.clone(),
            self.tool_deadline,
        )
        .with_planning_tools(Arc::new(planning))
    }

    /// Run a planning turn on `ask` in `conversation`, streaming it on `tx`, then ask for the
    /// task prompt rewritten as a brief for an unattended run, with the plan of that brief,
    /// each action checked against the machine and `permissions`, and the mode its runs are
    /// best given.
    pub async fn refine(
        &self,
        conversation: Arc<dyn LlmBackend>,
        ask: PlanAsk,
        permissions: WorkPermissions,
        request_id: String,
        tx: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> Result<Refinement, PlanError> {
        let conversation = self
            .explore(conversation, ask, request_id, tx, &cancel)
            .await?;
        let instruction = format!("{REFINE_INSTRUCTION} {PLAN_LAYOUT}");
        let draft: RefinementDraft = draft_of(
            conversation.as_ref(),
            instruction,
            RefinementDraft::schema(),
            &cancel,
        )
        .await?;
        let (prompt, draft) = draft.into_parts();
        let prompt = TaskPrompt::try_from(prompt)?;
        let checks = self.tools.checks.clone();
        let tools = self.catalog.snapshot();
        let auto_offered = self.auto_offered;
        let (plan, recommended) = on_blocking_pool(move || {
            let plan = draft.into_plan(&checks, &tools, &permissions);
            let hosts = &permissions.allowed_hosts;
            let recommended = recommended_mode(&plan, &checks, hosts, auto_offered);
            (plan, recommended)
        })
        .await;
        Ok(Refinement {
            prompt,
            plan,
            recommended,
        })
    }

    /// Run the planning turn on `ask`, returning the conversation in plan mode.
    async fn explore(
        &self,
        conversation: Arc<dyn LlmBackend>,
        ask: PlanAsk,
        request_id: String,
        tx: mpsc::Sender<LlmEvent>,
        cancel: &CancellationToken,
    ) -> Result<Arc<dyn LlmBackend>, PlanError> {
        let turn = TurnContext::answered_by(ConfirmRouter::for_planning(request_id));
        let conversation: Arc<dyn LlmBackend> =
            PlanModeConversation::new(conversation, FINDINGS_ENDING, turn.clone());
        let agent = Agent::new(
            Arc::clone(&conversation),
            Arc::clone(&self.tools.registry),
            self.health.clone(),
            self.tool_deadline,
        );
        agent
            .run_turn(&turn, ask.into_text(), Vec::new(), tx, cancel.clone())
            .await?;
        if cancel.is_cancelled() {
            return Err(PlanError::Cancelled);
        }
        Ok(conversation)
    }
}

impl PlanAsk {
    fn into_text(self) -> String {
        match self {
            Self::Work(text) => text,
            Self::Revise { shown, feedback } => {
                format!("You gave me this:\n\n{shown}\n\nRevise it with this feedback: {feedback}")
            }
        }
    }
}

/// Ask `conversation` for its answer to `instruction`, held to `schema` by grammar.
async fn draft_of<T: DeserializeOwned>(
    conversation: &dyn LlmBackend,
    instruction: String,
    schema: Map<String, Value>,
    cancel: &CancellationToken,
) -> Result<T, PlanError> {
    let reply = tokio::select! {
        biased;
        () = cancel.cancelled() => return Err(PlanError::Cancelled),
        reply = conversation.complete_with_schema(instruction, schema) => reply?,
    };
    serde_json::from_str(&reply).map_err(PlanError::Unreadable)
}

/// A conversation whose every step, while its turn plans, ends with the plan-mode note,
/// ahead of any one-shot note the agent loop adds.
#[derive(Debug)]
struct PlanModeConversation {
    inner: Arc<dyn LlmBackend>,
    turn: TurnContext,
    /// [`PLAN_MODE_RULES`] and how the turn should end.
    preamble: String,
    note: Mutex<Option<String>>,
}

impl PlanModeConversation {
    /// `inner` with the plan-mode rules and then `ending` at its tail while `turn` plans.
    fn new(inner: Arc<dyn LlmBackend>, ending: &str, turn: TurnContext) -> Arc<Self> {
        Arc::new(Self {
            inner,
            turn,
            preamble: format!("{PLAN_MODE_RULES} {ending}"),
            note: Mutex::new(None),
        })
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
        let extra = self.note.lock().take();
        let note = match (self.turn.planning(), extra) {
            (true, Some(extra)) => Some(format!("{}\n\n{extra}", self.preamble)),
            (true, None) => Some(self.preamble.clone()),
            (false, extra) => extra,
        };
        if let Some(note) = note {
            self.inner.set_transient_note(note).await?;
        }
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

    async fn invoke(&self, turn: &TurnContext, args: Value) -> Result<ToolOutput, ToolError> {
        let before = turn.refused_while_planning();
        let mut output = self.inner.invoke(turn, args).await?;
        if turn.refused_while_planning() > before {
            output.output.push_str(REFUSED_NOTE);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests;

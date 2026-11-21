//! Plan mode for a headless turn: the model enters it unasked, and its plan is put to the user
//! out loud as one question whose yes grants the plan's actions to the turn.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use assistd_ipc::{ActionOutlook, AnticipatedAction, PermissionMode, PlannedAction};
use assistd_tools::policy::on_blocking_pool;
use assistd_tools::{
    ActionCheck, SpokenAnswer, Tool, ToolCatalog, ToolError, ToolOutput, TurnContext,
};

use super::draft::{ACTIONS_MAX_ITEMS, DraftAction, anticipated};
use super::switch::{ENTER_PLAN_MODE, EXIT_PLAN_MODE, UNREVIEWED, revise_reply, router_of};
use super::{ACTION_KINDS, WorkPermissions};

const SUMMARY_MAX_CHARS: usize = 240;

const ENTER_DESCRIPTION: &str = "Switch to plan mode before work that will need the user's \
     approval more than once: web pages to read on sites not yet approved, by you or by a \
     sub-agent, several programs to run, apps to open or MCP tools to call, or files to \
     change outside the writable directories. In plan mode you look \
     around with commands that change nothing, then call exit_plan_mode with what the work \
     needs; the user is asked once, out loud, and what they approve runs without asking for \
     the rest of this turn. Do not use it for questions, lookups, conversation, or work that \
     needs one approval or none; just do those.";

const EXIT_DESCRIPTION: &str = "Ask the user, out loud, to approve your plan and the actions \
     it needs. Call it once you know enough to do the work. The result says whether the user \
     agreed, in which case you leave plan mode and carry the plan out, refused, or said what \
     to change.";

const SUMMARY_PARAMETER: &str = "What you will do, as one short plain sentence spoken to the \
     user, such as \"I'll update the changelog and publish the crate.\" No Markdown, paths or \
     commands.";

const ACTIONS_PARAMETER: &str = "Everything the work needs approved, each once. Leave out \
     reading files and edits inside the writable directories, which need no approval, and \
     name each web host exactly. For web research, run `web search` first and list the host \
     of each site you plan to read.";

const ENTERED: &str = "You are in plan mode now. Do not do the work yet; find out what it \
     needs, then call exit_plan_mode with a one-sentence summary and the actions that need \
     approval.";

const NOTHING_TO_APPROVE: &str = "Nothing in the plan needs approval, and plan mode is over: \
     every tool is back. Carry the plan out now.";

const AGREED: &str = "The user agreed, and plan mode is over: every tool is back. Carry the \
     plan out now. The actions you listed run without asking for the rest of this turn; \
     anything else that needs approval is still asked out loud. A sub-agent you hand web \
     research to is told which sites were approved.";

const REFUSED: &str = "The user did not agree to the plan. You are still in plan mode: do not \
     do the work or look for another way to do it. Say in one sentence that it was not done, \
     and stop.";

/// Starts a headless turn planning, with nobody asked first: planning runs nothing that asks.
#[derive(Debug)]
pub(super) struct EnterSpokenPlanTool;

/// Puts a headless turn's plan to the user out loud. A yes leaves plan mode with the plan's
/// actions granted to the turn, other words revise the plan, and a no keeps the turn planning.
#[derive(Debug)]
pub(super) struct ProposeSpokenPlanTool {
    pub(super) checks: ActionCheck,
    /// The tools an anticipated MCP tool must be among.
    pub(super) catalog: Arc<ToolCatalog>,
}

/// A plan as the model proposes it: what it will do and what that needs approved.
#[derive(Debug, Deserialize)]
struct Proposal {
    summary: String,
    actions: Vec<DraftAction>,
}

#[async_trait]
impl Tool for EnterSpokenPlanTool {
    fn name(&self) -> &str {
        ENTER_PLAN_MODE
    }

    fn description(&self) -> Cow<'_, str> {
        ENTER_DESCRIPTION.into()
    }

    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    fn reads_host(&self, _args: &Value) -> bool {
        false
    }

    async fn invoke(&self, turn: &TurnContext, _args: Value) -> Result<ToolOutput, ToolError> {
        let start = Instant::now();
        router_of(turn)?.set_planning(true);
        Ok(ToolOutput::text(ENTERED, start.elapsed()))
    }
}

impl ProposeSpokenPlanTool {
    /// `proposal`'s actions, each once, checked against this machine and what a headless turn
    /// runs unasked.
    async fn anticipated(&self, actions: Vec<DraftAction>) -> Vec<AnticipatedAction> {
        let checks = self.checks.clone();
        let tools = self.catalog.snapshot();
        let permissions = WorkPermissions {
            mode: PermissionMode::AcceptEdits,
            allowed_hosts: Vec::new(),
        };
        on_blocking_pool(move || anticipated(actions, &checks, &tools, &permissions)).await
    }
}

#[async_trait]
impl Tool for ProposeSpokenPlanTool {
    fn name(&self) -> &str {
        EXIT_PLAN_MODE
    }

    fn description(&self) -> Cow<'_, str> {
        EXIT_DESCRIPTION.into()
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "maxLength": SUMMARY_MAX_CHARS,
                    "description": SUMMARY_PARAMETER
                },
                "actions": {
                    "type": "array",
                    "items": DraftAction::schema(),
                    "maxItems": ACTIONS_MAX_ITEMS,
                    "description": format!("{ACTIONS_PARAMETER} {ACTION_KINDS}")
                }
            },
            "required": ["summary", "actions"]
        })
    }

    fn bounded_by_tool_deadline(&self) -> bool {
        false
    }

    fn reads_host(&self, _args: &Value) -> bool {
        false
    }

    async fn invoke(&self, turn: &TurnContext, args: Value) -> Result<ToolOutput, ToolError> {
        let start = Instant::now();
        let proposal = Proposal::read(args)?;
        let router = router_of(turn)?;
        let anticipated = self.anticipated(proposal.actions).await;
        if let Some(problems) = problems_reply(&anticipated) {
            return Ok(ToolOutput::text(problems, start.elapsed()));
        }
        let asks = asked_about(anticipated);
        if asks.is_empty() {
            router.set_planning(false);
            return Ok(ToolOutput::text(NOTHING_TO_APPROVE, start.elapsed()));
        }
        let reply = match router.propose_grant(&proposal.summary, asks).await {
            Some(SpokenAnswer::Confirm) => {
                router.set_planning(false);
                AGREED.to_string()
            }
            Some(SpokenAnswer::Message(said)) => revise_reply(&said),
            Some(SpokenAnswer::Deny | SpokenAnswer::Unanswered) => REFUSED.to_string(),
            None => UNREVIEWED.to_string(),
        };
        Ok(ToolOutput::text(reply, start.elapsed()))
    }
}

impl Proposal {
    /// The proposal in `args`, its summary a sentence that can be spoken ahead of another.
    fn read(args: Value) -> Result<Self, ToolError> {
        let mut proposal: Self = serde_json::from_value(args).map_err(|error| {
            ToolError::InvalidArgs(format!("the plan could not be read: {error}"))
        })?;
        let summary = proposal.summary.trim();
        if summary.is_empty() || summary.chars().count() > SUMMARY_MAX_CHARS {
            return Err(ToolError::InvalidArgs(format!(
                "`summary` must be one sentence of at most {SUMMARY_MAX_CHARS} characters"
            )));
        }
        if proposal.actions.len() > ACTIONS_MAX_ITEMS {
            return Err(ToolError::InvalidArgs(format!(
                "`actions` holds at most {ACTIONS_MAX_ITEMS} actions"
            )));
        }
        proposal.summary = if summary.ends_with(['.', '!', '?']) {
            summary.to_string()
        } else {
            format!("{summary}.")
        };
        Ok(proposal)
    }
}

/// What to tell the model when some of `anticipated` cannot work here; `None` when all can.
fn problems_reply(anticipated: &[AnticipatedAction]) -> Option<String> {
    let problems: Vec<&str> = anticipated
        .iter()
        .filter_map(|action| action.problem.as_deref())
        .collect();
    (!problems.is_empty()).then(|| {
        format!(
            "The user was not asked, because some actions cannot work on this machine:\n- {}\n\
             You are still in plan mode: fix or drop them, then call exit_plan_mode again.",
            problems.join("\n- ")
        )
    })
}

/// The actions of `anticipated` that would stop the turn to ask.
fn asked_about(anticipated: Vec<AnticipatedAction>) -> Vec<PlannedAction> {
    anticipated
        .into_iter()
        .filter(|action| action.outlook != ActionOutlook::Unasked)
        .map(|action| action.action)
        .collect()
}

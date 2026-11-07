//! The tools a chat turn moves through plan mode with: one asks the user to plan first, the
//! other hands the finished plan to the user to accept or revise.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde_json::{Value, json};

use assistd_ipc::{ConfirmSource, PermissionMode, PlanVerdict};
use assistd_tools::{
    Approval, CONFIRM_ROUTER, ConfirmRouter, ConfirmationRequest, Risk, Tool, ToolError,
    ToolOutput, Unattended,
};

const ENTER_PLAN_MODE: &str = "enter_plan_mode";

const EXIT_PLAN_MODE: &str = "exit_plan_mode";

const ENTER_DESCRIPTION: &str = "Ask the user to switch to plan mode before doing the work. \
     In plan mode you look around with commands that change nothing, then hand the user a \
     plan, and carry it out once they accept it. Use it only when agreeing on the approach \
     first is worth the user's time: work that changes several files or programs, takes many \
     steps, is hard to undo, or could reasonably be done in quite different ways. Do not use \
     it for questions, lookups, conversation, or work with a few obvious steps; just do those.";

const EXIT_DESCRIPTION: &str = "Hand your plan to the user to review. Call it once you know \
     enough to do the work, with the whole plan. The result says whether the user accepted \
     it, in which case you leave plan mode and carry it out, or what to change.";

const PLAN_PARAMETER: &str = "The plan in Markdown: a line on what the work will do, then \
     numbered steps naming the exact paths, commands and hosts each uses, marking the steps \
     that change files, run programs that change things or reach new hosts.";

/// Why the confirmation prompt asks, shown above the model's reason.
const ENTER_PROMPT: &str = "The assistant wants to plan this first: it will look around with \
     commands that change nothing, then show you a plan to accept or revise.";

const ENTER_ACCEPTED: &str = "The user agreed: you are in plan mode now. Do not do the work \
     yet; find out what it needs, then call exit_plan_mode with the plan.";

const ENTER_DECLINED: &str = "The user wants no plan for this. Do the work now.";

const KEEP_PLANNING: &str = "The user did not accept the plan and will say what to change. \
     You are still in plan mode: reply with one short line asking what they would like \
     changed, and stop.";

const UNREVIEWED: &str = "Nobody can review a plan here, so you are still in plan mode. \
     Reply with the plan instead.";

/// Asks the user to switch the turn to plan mode.
#[derive(Debug)]
pub(crate) struct EnterPlanModeTool;

/// Hands the plan to the user, leaving plan mode under the mode they pick if they accept it.
#[derive(Debug)]
pub(crate) struct ExitPlanModeTool;

#[async_trait]
impl Tool for EnterPlanModeTool {
    fn name(&self) -> &str {
        ENTER_PLAN_MODE
    }

    fn description(&self) -> Cow<'_, str> {
        ENTER_DESCRIPTION.into()
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this work is worth planning first, in one sentence \
                                    for the user."
                }
            },
            "required": ["reason"]
        })
    }

    fn bounded_by_tool_deadline(&self) -> bool {
        false
    }

    async fn invoke(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let start = Instant::now();
        let reason = string_arg(&args, "reason")?;
        let router = router_in_scope()?;
        let reply = match router.ask(enter_request(reason)).await {
            Approval::Deny => ENTER_DECLINED,
            Approval::Once | Approval::Always => {
                router.enter_plan_mode().await;
                ENTER_ACCEPTED
            }
        };
        Ok(ToolOutput::text(reply, start.elapsed()))
    }
}

#[async_trait]
impl Tool for ExitPlanModeTool {
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
                "plan": { "type": "string", "description": PLAN_PARAMETER }
            },
            "required": ["plan"]
        })
    }

    fn bounded_by_tool_deadline(&self) -> bool {
        false
    }

    async fn invoke(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let start = Instant::now();
        let plan = string_arg(&args, "plan")?;
        let router = router_in_scope()?;
        let reply = match router.review_plan(plan).await {
            Some(PlanVerdict::Accept { mode }) => {
                router.set_mode(mode);
                router.set_planning(false);
                accepted_reply(mode)
            }
            Some(PlanVerdict::KeepPlanning {
                feedback: Some(feedback),
            }) => revise_reply(&feedback),
            Some(PlanVerdict::KeepPlanning { feedback: None }) => KEEP_PLANNING.to_string(),
            None => UNREVIEWED.to_string(),
        };
        Ok(ToolOutput::text(reply, start.elapsed()))
    }
}

/// The non-empty string argument `name`.
fn string_arg(args: &Value, name: &str) -> Result<String, ToolError> {
    args.get(name)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ToolError::InvalidArgs(format!("`{name}` (non-empty string) is required")))
}

fn router_in_scope() -> Result<Arc<ConfirmRouter>, ToolError> {
    CONFIRM_ROUTER
        .try_with(Arc::clone)
        .map_err(|_| ToolError::InvalidArgs("plan mode is unavailable on this turn".into()))
}

/// The prompt asking the user to plan first, which every mode asks.
fn enter_request(reason: String) -> ConfirmationRequest {
    ConfirmationRequest {
        tool: ENTER_PLAN_MODE.to_string(),
        script: reason,
        matched_pattern: ENTER_PROMPT.to_string(),
        always_allow: Vec::new(),
        risk: Risk::Routine,
        unattended: Unattended::Never,
        source: ConfirmSource::Main,
    }
}

fn accepted_reply(mode: PermissionMode) -> String {
    let asks = match mode {
        PermissionMode::Manual => "Anything that changes something still asks the user first.",
        PermissionMode::AcceptEdits => {
            "File edits inside the edit scope run without asking; anything else that changes \
             something still asks."
        }
        PermissionMode::Auto => {
            "File edits run without asking, and the safety judge answers for the user where it \
             can; anything it doubts still asks."
        }
    };
    format!(
        "The user accepted the plan, and plan mode is over: every tool is back. Carry the plan \
         out now, step by step. {asks}"
    )
}

fn revise_reply(feedback: &str) -> String {
    format!(
        "The user did not accept the plan, and said:\n\n{feedback}\n\nYou are still in plan \
         mode. Revise the plan by this, finding out more if you need to, then call \
         exit_plan_mode again with the whole revised plan."
    )
}

//! The plan as the model writes it, held to a schema by grammar, and the [`Plan`] it becomes
//! once its actions are read and checked.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use assistd_ipc::{AnticipatedAction, Plan, PlanId, PlanStep, PlannedAction};
use assistd_tools::{ActionCheck, ToolRegistry};
use assistd_utils::path::expand_tilde_from_env;

use super::WorkPermissions;

const SUMMARY_MAX_CHARS: usize = 800;
const STEPS_MAX_ITEMS: usize = 16;
const DESCRIPTION_MAX_CHARS: usize = 400;
const ACTIONS_MAX_ITEMS: usize = 8;
const TARGET_MAX_CHARS: usize = 300;
const SUBCOMMAND_MAX_CHARS: usize = 64;

/// The model's plan, before its actions are checked.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PlanDraft {
    summary: String,
    steps: Vec<DraftStep>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DraftStep {
    description: String,
    actions: Vec<DraftAction>,
}

/// An anticipated action as the model names it: a kind and the one thing it applies to.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DraftAction {
    kind: DraftKind,
    target: String,
    subcommand: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DraftKind {
    Program,
    WebHost,
    McpTool,
    App,
    Write,
}

impl PlanDraft {
    /// The JSON schema the plan request is held to by grammar.
    pub(super) fn schema() -> Map<String, Value> {
        let text = |max_chars: usize| json!({"type": "string", "maxLength": max_chars});
        let action = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["program", "web_host", "mcp_tool", "app", "write"]
                },
                "target": text(TARGET_MAX_CHARS),
                "subcommand": text(SUBCOMMAND_MAX_CHARS)
            },
            "required": ["kind", "target"]
        });
        let step = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "description": text(DESCRIPTION_MAX_CHARS),
                "actions": {"type": "array", "items": action, "maxItems": ACTIONS_MAX_ITEMS}
            },
            "required": ["description", "actions"]
        });
        Map::from_iter([
            ("type".into(), json!("object")),
            ("additionalProperties".into(), json!(false)),
            (
                "properties".into(),
                json!({
                    "summary": text(SUMMARY_MAX_CHARS),
                    "steps": {"type": "array", "items": step, "maxItems": STEPS_MAX_ITEMS}
                }),
            ),
            ("required".into(), json!(["summary", "steps"])),
        ])
    }

    /// The plan under a fresh id, each step's repeated actions merged and every action checked
    /// by `checks` against `tools` and the work's `permissions`. Blocks on the filesystem.
    pub(super) fn into_plan(
        self,
        checks: &ActionCheck,
        tools: &ToolRegistry,
        permissions: &WorkPermissions,
    ) -> Plan {
        let steps = self
            .steps
            .into_iter()
            .map(|step| PlanStep {
                description: step.description.trim().to_string(),
                actions: anticipated(step.actions, checks, tools, permissions),
            })
            .collect();
        Plan {
            id: PlanId(uuid::Uuid::new_v4().to_string()),
            summary: self.summary.trim().to_string(),
            steps,
        }
    }
}

impl DraftAction {
    fn into_action(self) -> PlannedAction {
        let target = self.target.trim().to_string();
        match self.kind {
            DraftKind::Program => PlannedAction::Program {
                program: target,
                subcommand: self
                    .subcommand
                    .map(|subcommand| subcommand.trim().to_string())
                    .filter(|subcommand| !subcommand.is_empty()),
            },
            DraftKind::WebHost => PlannedAction::WebHost {
                host: target.to_ascii_lowercase(),
            },
            DraftKind::McpTool => PlannedAction::McpTool { name: target },
            DraftKind::App => PlannedAction::App {
                desktop_entry: target,
            },
            DraftKind::Write => PlannedAction::Write {
                under: expand_tilde_from_env(&target),
            },
        }
    }
}

/// `actions`, each once, with what `checks` finds wrong with it and whether the work's
/// `permissions` let it run unasked.
fn anticipated(
    actions: Vec<DraftAction>,
    checks: &ActionCheck,
    tools: &ToolRegistry,
    permissions: &WorkPermissions,
) -> Vec<AnticipatedAction> {
    let mut distinct: Vec<PlannedAction> = Vec::new();
    for action in actions.into_iter().map(DraftAction::into_action) {
        if !distinct.contains(&action) {
            distinct.push(action);
        }
    }
    distinct
        .into_iter()
        .map(|action| AnticipatedAction {
            problem: checks.problem(&action, tools),
            outlook: checks.outlook(&action, permissions.mode, &permissions.allowed_hosts),
            action,
        })
        .collect()
}

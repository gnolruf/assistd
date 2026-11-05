use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio::sync::mpsc;

use assistd_ipc::Event;
use assistd_tools::commands::BashPolicyCfg;
use assistd_tools::{
    ActionCheck, ApprovalGate, Approvals, CONFIRM_ROUTER, ConfirmRouter, ConfirmationGate,
    IpcConfirmationGate, OutputCap, PresentSpec, SandboxInfo, ToolOutput, ToolRegistry,
};

use super::REFUSED_NOTE;
use crate::{PlanningCommandDeps, PlanningToolDeps, planning_commands, planning_tools};

fn planning_registry() -> Arc<ToolRegistry> {
    let gate: Arc<dyn ConfirmationGate> = Arc::new(IpcConfirmationGate);
    let sandbox = SandboxInfo::none();
    let commands = planning_commands(PlanningCommandDeps {
        bash_cfg: Arc::new(BashPolicyCfg::default()),
        approved_hosts: ApprovalGate::new(Arc::clone(&gate), Arc::new(Approvals::unsaved())),
        sandbox: Arc::clone(&sandbox),
        confirmation_gate: gate,
        search: None,
    });
    planning_tools(PlanningToolDeps {
        commands,
        output_cap: Arc::new(OutputCap::new(PresentSpec::default())),
        scratch_dir: None,
        reminisce: None,
        checks: ActionCheck::new(sandbox.search_path(), None),
    })
    .registry
}

async fn run(tools: &ToolRegistry, line: &str) -> ToolOutput {
    tools
        .get("run")
        .expect("planning offers run")
        .invoke(json!({ "command": line }))
        .await
        .expect("run answers")
}

/// Whether `line` puts a confirmation prompt on a client's wire; the prompt is denied.
async fn gate_prompts(tools: &ToolRegistry, line: &str) -> bool {
    let (wire, mut prompts) = mpsc::channel(8);
    let router = ConfirmRouter::new("r".into(), wire, Duration::from_millis(1), None);
    CONFIRM_ROUTER.scope(router, run(tools, line)).await;
    matches!(prompts.try_recv(), Ok(Event::ConfirmRequest { .. }))
}

#[tokio::test]
async fn planning_refuses_everything_the_gate_would_prompt_for() {
    let dir = tempfile::tempdir().unwrap();
    let kept = dir.path().join("kept");
    std::fs::write(&kept, "notes").unwrap();
    let made = dir.path().join("made");
    let (kept_arg, made_arg) = (kept.display(), made.display());
    let lines = [
        format!("cat {kept_arg}"),
        format!("ls {}", dir.path().display()),
        format!("bash 'wc -c {kept_arg}'"),
        format!("bash 'rm {kept_arg}'"),
        format!("bash 'touch {made_arg}'"),
        "bash 'curl -s https://example.invalid/'".to_string(),
        "web fetch https://example.invalid/".to_string(),
        format!("write {made_arg} hi"),
        "wm focus 1".to_string(),
    ];
    let tools = planning_registry();
    let mut prompted_lines = 0;
    for line in &lines {
        let prompted = gate_prompts(&tools, line).await;
        let router = ConfirmRouter::for_planning("p".into());
        let output = CONFIRM_ROUTER
            .scope(Arc::clone(&router), run(&tools, line))
            .await;
        assert_eq!(router.refused_while_planning() > 0, prompted, "{line}");
        assert_eq!(
            output.output.contains(REFUSED_NOTE),
            prompted,
            "{line}: {}",
            output.output
        );
        prompted_lines += usize::from(prompted);
    }
    assert!(prompted_lines >= 3, "only {prompted_lines} lines asked");
    assert!(kept.exists());
    assert!(!made.exists());
}

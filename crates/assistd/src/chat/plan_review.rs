//! A plan under review: its steps, with each anticipated action a checkbox the
//! user ticks to approve it.

use std::fmt::Write as _;

use assistd_ipc::{AnticipatedAction, Plan, TaskGrant};

/// A plan and which of its actions are checked; actions that failed the daemon's
/// checks start unchecked.
pub(super) struct PlanReview {
    plan: Plan,
    /// One per action, in step order.
    checked: Vec<bool>,
    /// Index into the actions in step order.
    selected: usize,
}

impl PlanReview {
    pub(super) fn new(plan: Plan) -> Self {
        let checked = plan
            .steps
            .iter()
            .flat_map(|step| &step.actions)
            .map(|action| action.problem.is_none())
            .collect();
        Self {
            plan,
            checked,
            selected: 0,
        }
    }

    pub(super) fn plan(&self) -> &Plan {
        &self.plan
    }

    pub(super) fn selected(&self) -> usize {
        self.selected
    }

    pub(super) fn is_checked(&self, action: usize) -> bool {
        self.checked.get(action).copied().unwrap_or(false)
    }

    pub(super) fn action_count(&self) -> usize {
        self.checked.len()
    }

    pub(super) fn select_previous(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub(super) fn select_next(&mut self) {
        self.selected = (self.selected + 1).min(self.checked.len().saturating_sub(1));
    }

    pub(super) fn toggle_selected(&mut self) {
        if let Some(checked) = self.checked.get_mut(self.selected) {
            *checked = !*checked;
        }
    }

    /// The grants of the checked actions, in step order.
    pub(super) fn accepted(&self) -> Vec<TaskGrant> {
        self.actions()
            .zip(&self.checked)
            .filter(|(_, checked)| **checked)
            .map(|(action, _)| action.grant.clone())
            .collect()
    }

    fn actions(&self) -> impl Iterator<Item = &AnticipatedAction> {
        self.plan.steps.iter().flat_map(|step| &step.actions)
    }
}

/// What `grant` lets a run do, in a few words.
pub(super) fn grant_label(grant: &TaskGrant) -> String {
    match grant {
        TaskGrant::Program {
            program,
            subcommand,
        } => {
            let mut label = format!("run {program}");
            if let Some(subcommand) = subcommand {
                let _ = write!(label, " {subcommand}");
            }
            label
        }
        TaskGrant::WebHost { host } => format!("reach {host}"),
        TaskGrant::McpTool { name } => format!("call {name}"),
        TaskGrant::App { desktop_entry } => format!("launch {desktop_entry}"),
        TaskGrant::Write { under } => format!("write under {}", under.display()),
    }
}

#[cfg(test)]
mod tests;

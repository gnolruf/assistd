//! A plan under review: its steps and the actions each expects to need approved, scrolled
//! through before the plan is accepted.

use std::fmt::Write as _;

use assistd_ipc::{ActionOutlook, Plan, PlannedAction};

/// A plan and how far down it is scrolled.
pub(super) struct PlanReview {
    plan: Plan,
    /// First line shown; each frame clamps it.
    scroll: usize,
}

impl PlanReview {
    pub(super) fn new(plan: Plan) -> Self {
        Self { plan, scroll: 0 }
    }

    pub(super) fn plan(&self) -> &Plan {
        &self.plan
    }

    pub(super) fn scroll(&self) -> usize {
        self.scroll
    }

    pub(super) fn scroll_up(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_sub(lines);
    }

    pub(super) fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_add(lines);
    }

    /// Hold the scroll within `last`, the last line that can start the view.
    pub(super) fn clamp_scroll(&mut self, last: usize) {
        self.scroll = self.scroll.min(last);
    }
}

/// What `action` does, in a few words.
pub(super) fn action_label(action: &PlannedAction) -> String {
    match action {
        PlannedAction::Program {
            program,
            subcommand,
        } => {
            let mut label = format!("run {program}");
            if let Some(subcommand) = subcommand {
                let _ = write!(label, " {subcommand}");
            }
            label
        }
        PlannedAction::WebHost { host } => format!("reach {host}"),
        PlannedAction::McpTool { name } => format!("call {name}"),
        PlannedAction::App { desktop_entry } => format!("launch {desktop_entry}"),
        PlannedAction::Write { under } => format!("write under {}", under.display()),
    }
}

/// Whether the work will stop to ask about an action, in a few words.
pub(super) fn outlook_label(outlook: ActionOutlook) -> &'static str {
    match outlook {
        ActionOutlook::Unasked => "runs unasked",
        ActionOutlook::JudgeDecides => "the safety judge decides",
        ActionOutlook::Asks => "asks you",
    }
}

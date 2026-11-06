//! The permission mode a plan's runs are best given: the least permissive one that leaves the
//! fewest of its actions to ask about.

use assistd_ipc::{
    ActionOutlook, HostName, ModeRecommendation, PermissionMode, Plan, PlannedAction,
};
use assistd_tools::ActionCheck;

/// Among Manual, Accept edits and, when `auto_offered`, Auto, the first under which the fewest
/// of `plan`'s actions would ask, with web fetches from `allowed_hosts` unasked. Blocks on the
/// filesystem.
pub(super) fn recommended_mode(
    plan: &Plan,
    checks: &ActionCheck,
    allowed_hosts: &[HostName],
    auto_offered: bool,
) -> ModeRecommendation {
    let mut actions: Vec<&PlannedAction> = Vec::new();
    for anticipated in plan.steps.iter().flat_map(|step| &step.actions) {
        if !actions.contains(&&anticipated.action) {
            actions.push(&anticipated.action);
        }
    }
    let asking = |mode: PermissionMode| -> Vec<PlannedAction> {
        actions
            .iter()
            .filter(|action| checks.outlook(action, mode, allowed_hosts) == ActionOutlook::Asks)
            .map(|action| (*action).clone())
            .collect()
    };
    let manual_asks = asking(PermissionMode::Manual);
    let more_permissive = [PermissionMode::AcceptEdits, PermissionMode::Auto]
        .into_iter()
        .filter(|mode| auto_offered || *mode != PermissionMode::Auto);
    let mut best = (PermissionMode::Manual, manual_asks.clone());
    for mode in more_permissive {
        let asks = asking(mode);
        if asks.len() < best.1.len() {
            best = (mode, asks);
        }
    }
    let (mode, still_asks) = best;
    let covers = manual_asks
        .into_iter()
        .filter(|action| !still_asks.contains(action))
        .collect();
    ModeRecommendation {
        mode,
        covers,
        still_asks,
    }
}

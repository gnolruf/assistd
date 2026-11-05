use assistd_ipc::{AnticipatedAction, Plan, PlanId, PlanStep, TaskGrant};

use super::PlanReview;

fn web(host: &str) -> TaskGrant {
    TaskGrant::WebHost { host: host.into() }
}

#[test]
fn flagged_actions_are_approved_only_once_ticked() {
    let plan = Plan {
        id: PlanId("p".into()),
        summary: "fetch".into(),
        steps: vec![PlanStep {
            description: "fetch both".into(),
            actions: vec![
                AnticipatedAction {
                    grant: web("ok.example"),
                    problem: None,
                },
                AnticipatedAction {
                    grant: web("bad.example"),
                    problem: Some("does not resolve".into()),
                },
            ],
        }],
    };
    let mut review = PlanReview::new(plan);
    assert_eq!(review.accepted(), vec![web("ok.example")]);
    review.select_next();
    review.toggle_selected();
    assert_eq!(
        review.accepted(),
        vec![web("ok.example"), web("bad.example")]
    );
}

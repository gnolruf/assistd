use std::time::Duration;

use futures_util::FutureExt;

use super::*;

fn lanes(slots: u32) -> (Arc<ModelLanes>, watch::Sender<NonZeroU32>) {
    let (tx, rx) = watch::channel(NonZeroU32::new(slots).unwrap());
    (ModelLanes::new(Arc::new(Mutex::new(())), rx), tx)
}

#[tokio::test]
async fn task_runs_never_outnumber_the_lanes_beside_the_main_one() {
    let (lanes, _slots) = lanes(3);
    let _user_turn = lanes.main.clone().lock_owned().await;

    let first = lanes
        .admit(TaskRunId(1))
        .now_or_never()
        .expect("a free lane");
    let second = lanes
        .admit(TaskRunId(2))
        .now_or_never()
        .expect("a free lane");
    assert_ne!(
        first.slot(ConversationKey::Main),
        second.slot(ConversationKey::Main)
    );
    assert!(!first.on_main_lane() && !second.on_main_lane());
    let third = tokio::spawn({
        let lanes = Arc::clone(&lanes);
        async move { lanes.admit(TaskRunId(3)).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!third.is_finished(), "a third run took a lane");
    assert_eq!(
        lanes.holders(),
        [
            LaneHolder::Main,
            LaneHolder::Task { run: TaskRunId(1) },
            LaneHolder::Task { run: TaskRunId(2) },
        ]
    );

    drop(first);
    let third = tokio::time::timeout(Duration::from_secs(1), third)
        .await
        .expect("the freed lane went unused")
        .unwrap();
    assert_eq!(third.slot(ConversationKey::Main), SlotAssignment::Pinned(1));
}

#[tokio::test]
async fn with_one_lane_a_task_run_waits_out_the_user_turn_and_holds_off_the_next() {
    let (lanes, _slots) = lanes(1);
    let user_turn = lanes.main.clone().lock_owned().await;
    let run = tokio::spawn({
        let lanes = Arc::clone(&lanes);
        async move { lanes.admit(TaskRunId(1)).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!run.is_finished(), "a run started during a user turn");

    drop(user_turn);
    let lease = tokio::time::timeout(Duration::from_secs(1), run)
        .await
        .expect("the run never took the free lane")
        .unwrap();
    let key = ConversationKey::TaskRun(TaskRunId(1));
    assert_eq!(lease.slot(key), SlotAssignment::Shared(key));
    assert_eq!(lanes.holders(), [LaneHolder::Task { run: TaskRunId(1) }]);
    assert!(lanes.main.try_lock().is_err(), "a user turn could start");

    drop(lease);
    assert!(lanes.main.try_lock().is_ok());
    assert_eq!(lanes.holders(), [LaneHolder::Main]);
}

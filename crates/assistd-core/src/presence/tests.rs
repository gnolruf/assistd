use super::*;

#[tokio::test]
async fn sleep_from_sleeping_is_noop() {
    let m = PresenceManager::stub(PresenceState::Sleeping);
    assert!(m.sleep().await.is_ok());
    assert!(m.sleep().await.is_ok());
    assert_eq!(m.state(), PresenceState::Sleeping);
}

#[tokio::test]
async fn drowse_from_sleeping_errors() {
    let m = PresenceManager::stub(PresenceState::Sleeping);
    let err = m
        .drowse()
        .await
        .expect_err("drowse from Sleeping must error");
    assert!(err.to_string().contains("Sleeping"));
    assert_eq!(m.state(), PresenceState::Sleeping);
}

#[tokio::test]
async fn ensure_active_is_noop_when_active() {
    let m = PresenceManager::stub(PresenceState::Active);
    assert!(m.ensure_active().await.is_ok());
    assert_eq!(m.state(), PresenceState::Active);
}

#[tokio::test]
async fn wake_from_active_is_noop() {
    let m = PresenceManager::stub(PresenceState::Active);
    assert!(m.wake().await.is_ok());
    assert_eq!(m.state(), PresenceState::Active);
}

#[tokio::test]
async fn drowse_from_drowsy_is_noop() {
    let m = PresenceManager::stub(PresenceState::Drowsy);
    assert!(m.drowse().await.is_ok());
    assert_eq!(m.state(), PresenceState::Drowsy);
}

#[tokio::test]
async fn sleep_from_active_broadcasts_sleeping() {
    let m = PresenceManager::stub(PresenceState::Active);
    let mut rx = m.subscribe();
    m.sleep().await.unwrap();
    assert_eq!(*rx.borrow_and_update(), PresenceState::Sleeping);
}

#[tokio::test]
async fn subscribe_sees_test_set_state() {
    let m = PresenceManager::stub(PresenceState::Sleeping);
    let mut rx = m.subscribe();
    m.set_state_for_test(PresenceState::Drowsy);
    rx.changed().await.unwrap();
    assert_eq!(*rx.borrow_and_update(), PresenceState::Drowsy);
}

#[tokio::test]
async fn cycle_from_sleeping_goes_to_active_logically() {
    let start = PresenceState::Sleeping;
    assert_eq!(start.next(), PresenceState::Active);
}

fn sleep_cfg(drowsy: u64, sleep: u64) -> crate::SleepConfig {
    let mut cfg = crate::Config::default().sleep;
    cfg.idle_to_drowsy_mins = drowsy;
    cfg.idle_to_sleep_mins = sleep;
    cfg
}

#[tokio::test]
async fn ensure_active_resets_activity_timer() {
    let m = PresenceManager::stub(PresenceState::Active);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let before = m.idle_duration();
    assert!(before >= Duration::from_millis(40));
    m.ensure_active().await.unwrap();
    let after = m.idle_duration();
    assert!(after < before);
    assert!(after < Duration::from_millis(20));
}

#[tokio::test]
async fn set_presence_resets_activity_timer() {
    let m = PresenceManager::stub(PresenceState::Active);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(m.idle_duration() >= Duration::from_millis(40));
    m.set_presence(PresenceState::Active).await.unwrap();
    assert!(m.idle_duration() < Duration::from_millis(20));
}

#[tokio::test]
async fn cycle_resets_activity_timer() {
    let m = PresenceManager::stub(PresenceState::Drowsy);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(m.idle_duration() >= Duration::from_millis(40));
    m.cycle().await.unwrap();
    assert!(m.idle_duration() < Duration::from_millis(20));
}

#[tokio::test]
async fn wake_from_active_does_not_reset_activity_timer() {
    let m = PresenceManager::stub(PresenceState::Active);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let before = m.idle_duration();
    m.wake().await.unwrap();
    let after = m.idle_duration();
    assert!(after >= before);
}

#[test]
fn time_until_next_transition_active_counts_down_to_drowsy() {
    let m = PresenceManager::stub(PresenceState::Active);
    let cfg = sleep_cfg(30, 120);
    let d = m.time_until_next_transition(&cfg).unwrap();
    assert!(d <= Duration::from_secs(30 * 60));
    assert!(d >= Duration::from_secs(30 * 60).saturating_sub(Duration::from_secs(5)));
}

#[test]
fn time_until_next_transition_sleeping_returns_none() {
    let m = PresenceManager::stub(PresenceState::Sleeping);
    assert!(m.time_until_next_transition(&sleep_cfg(30, 120)).is_none());
}

#[test]
fn time_until_next_transition_active_with_drowsy_disabled_uses_sleep() {
    let m = PresenceManager::stub(PresenceState::Active);
    let d = m.time_until_next_transition(&sleep_cfg(0, 120)).unwrap();
    assert!(d <= Duration::from_secs(120 * 60));
}

#[test]
fn time_until_next_transition_active_with_both_disabled_returns_none() {
    let m = PresenceManager::stub(PresenceState::Active);
    assert!(m.time_until_next_transition(&sleep_cfg(0, 0)).is_none());
}

#[test]
fn time_until_next_transition_drowsy_with_sleep_disabled_returns_none() {
    let m = PresenceManager::stub(PresenceState::Drowsy);
    assert!(m.time_until_next_transition(&sleep_cfg(30, 0)).is_none());
}

#[test]
fn time_until_next_transition_drowsy_counts_down_to_sleep() {
    let m = PresenceManager::stub(PresenceState::Drowsy);
    let d = m.time_until_next_transition(&sleep_cfg(30, 120)).unwrap();
    assert!(d <= Duration::from_secs(120 * 60));
}

#[tokio::test]
async fn acquire_request_guard_fast_path_when_active() {
    let m = PresenceManager::stub(PresenceState::Active);
    let g = tokio::time::timeout(Duration::from_millis(100), m.acquire_request_guard())
        .await
        .expect("acquire did not complete in time")
        .expect("acquire returned Err");
    drop(g);
    assert_eq!(m.state(), PresenceState::Active);
}

#[tokio::test]
async fn sleep_defers_for_inflight_request() {
    let m = PresenceManager::stub(PresenceState::Active);
    let guard = m.acquire_request_guard().await.unwrap();

    let m2 = Arc::clone(&m);
    let sleep_task = tokio::spawn(async move { m2.sleep().await });

    // Sleep must block while the request guard is alive.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !sleep_task.is_finished(),
        "sleep completed while request guard held"
    );

    // Drop guard; sleep should now proceed.
    drop(guard);
    let res = tokio::time::timeout(Duration::from_secs(2), sleep_task)
        .await
        .expect("sleep did not complete after guard dropped")
        .expect("sleep task panicked");
    res.expect("sleep returned Err");
    assert_eq!(m.state(), PresenceState::Sleeping);
}

#[tokio::test]
async fn drowse_defers_for_inflight_request() {
    let m = PresenceManager::stub(PresenceState::Active);
    let guard = m.acquire_request_guard().await.unwrap();

    let m2 = Arc::clone(&m);
    let drowse_task = tokio::spawn(async move { m2.drowse().await });

    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !drowse_task.is_finished(),
        "drowse must block while request guard is held"
    );

    drop(guard);
    let _ = tokio::time::timeout(Duration::from_secs(2), drowse_task)
        .await
        .expect("drowse did not unblock after guard dropped");
}

#[tokio::test]
async fn stream_guard_increments_and_decrements_count() {
    let m = PresenceManager::stub(PresenceState::Active);
    let mut rx = m.subscribe_llm_streams();
    assert_eq!(*rx.borrow_and_update(), 0);
    let g1 = m.acquire_stream_guard();
    assert_eq!(*m.subscribe_llm_streams().borrow(), 1);
    let g2 = m.acquire_stream_guard();
    assert_eq!(*m.subscribe_llm_streams().borrow(), 2);
    drop(g1);
    assert_eq!(*m.subscribe_llm_streams().borrow(), 1);
    drop(g2);
    assert_eq!(*m.subscribe_llm_streams().borrow(), 0);
}

#[tokio::test]
async fn wait_until_llm_idle_returns_true_immediately_when_zero() {
    let m = PresenceManager::stub(PresenceState::Active);
    let ok = tokio::time::timeout(
        Duration::from_millis(20),
        m.wait_until_llm_idle(Duration::from_secs(5)),
    )
    .await
    .expect("wait did not complete fast");
    assert!(ok);
}

#[tokio::test]
async fn wait_until_llm_idle_times_out_when_busy() {
    let m = PresenceManager::stub(PresenceState::Active);
    let _g = m.acquire_stream_guard();
    let ok = m.wait_until_llm_idle(Duration::from_millis(30)).await;
    assert!(!ok, "wait should have timed out while a guard is held");
}

#[tokio::test]
async fn wait_until_llm_idle_returns_true_after_guard_dropped() {
    let m = PresenceManager::stub(PresenceState::Active);
    let g = m.acquire_stream_guard();
    let m2 = Arc::clone(&m);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        drop(g);
        // Keep the Arc alive for the spawn so the guard drop observes the watch.
        let _ = m2;
    });
    let ok = m.wait_until_llm_idle(Duration::from_millis(500)).await;
    assert!(ok, "wait should have resolved once the guard dropped");
}

#[tokio::test]
async fn stream_guard_does_not_block_sleep() {
    let m = PresenceManager::stub(PresenceState::Active);
    let _stream = m.acquire_stream_guard();
    let m2 = Arc::clone(&m);
    let sleep_task = tokio::spawn(async move { m2.sleep().await });
    let res = tokio::time::timeout(Duration::from_secs(1), sleep_task)
        .await
        .expect("sleep was blocked by an LLM stream guard");
    res.expect("sleep task panicked")
        .expect("sleep returned Err");
    assert_eq!(m.state(), PresenceState::Sleeping);
}

#[tokio::test]
async fn wake_marker_cleared_on_error_path() {
    let m = PresenceManager::stub(PresenceState::Drowsy);
    assert!(m.wake_in_progress().is_none());
    let err = m.wake().await;
    assert!(err.is_err(), "wake must fail against dummy control");
    assert!(
        m.wake_in_progress().is_none(),
        "wake_in_progress must be cleared after wake returns, even on error"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rapid_sleep_guard_loop_no_crash() {
    let m = PresenceManager::stub(PresenceState::Active);

    let mut readers = Vec::new();
    for _ in 0..4 {
        let m = Arc::clone(&m);
        readers.push(tokio::spawn(async move {
            for _ in 0..50 {
                match m.acquire_request_guard().await {
                    Ok(g) => {
                        tokio::task::yield_now().await;
                        drop(g);
                    }
                    Err(_) => tokio::task::yield_now().await,
                }
            }
        }));
    }

    let m2 = Arc::clone(&m);
    let writer = tokio::spawn(async move {
        for _ in 0..100 {
            m2.sleep().await.expect("sleep errored");
            m2.set_state_for_test(PresenceState::Active);
            tokio::task::yield_now().await;
        }
    });

    tokio::time::timeout(Duration::from_secs(30), async move {
        writer.await.expect("writer panicked");
        for r in readers {
            r.await.expect("reader panicked");
        }
    })
    .await
    .expect("rapid toggle workload deadlocked");
}

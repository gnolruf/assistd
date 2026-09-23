use super::*;

#[tokio::test]
async fn transition_to_current_state_is_a_silent_noop() {
    for state in [
        PresenceState::Active,
        PresenceState::Drowsy,
        PresenceState::Sleeping,
    ] {
        let m = PresenceManager::stub(state);
        let rx = m.subscribe();
        m.set_presence(state)
            .await
            .unwrap_or_else(|e| panic!("{state:?}: {e:#}"));
        assert_eq!(m.state(), state);
        assert!(!rx.has_changed().unwrap(), "{state:?}: broadcast a no-op");
    }
}

#[tokio::test]
async fn drowse_from_sleeping_errors() {
    let m = PresenceManager::stub(PresenceState::Sleeping);
    let err = m
        .drowse()
        .await
        .expect_err("drowse from Sleeping must error");
    assert!(matches!(err, PresenceError::DrowseFromSleeping), "{err:?}");
    assert_eq!(m.state(), PresenceState::Sleeping);
}

#[tokio::test]
async fn sleep_from_active_broadcasts_sleeping() {
    let m = PresenceManager::stub(PresenceState::Active);
    let mut rx = m.subscribe();
    m.sleep().await.unwrap();
    assert_eq!(*rx.borrow_and_update(), PresenceState::Sleeping);
    assert_eq!(m.state(), PresenceState::Sleeping);
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
    assert_eq!(m.state(), PresenceState::Active);
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
    assert_eq!(m.cycle().await.unwrap(), PresenceState::Sleeping);
    assert_eq!(m.state(), PresenceState::Sleeping);
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

#[tokio::test]
async fn acquire_request_guard_fast_path_when_active() {
    let m = PresenceManager::stub(PresenceState::Active);
    tokio::time::timeout(
        Duration::from_millis(100),
        m.acquire_request_guard_inner(None),
    )
    .await
    .expect("acquire did not complete in time")
    .expect("acquire returned Err");
    assert_eq!(m.state(), PresenceState::Active);
}

#[tokio::test]
async fn sleep_defers_for_inflight_request() {
    let m = PresenceManager::stub(PresenceState::Active);
    let guard = m.acquire_request_guard_inner(None).await.unwrap();

    let m2 = Arc::clone(&m);
    let sleep_task = tokio::spawn(async move { m2.sleep().await });

    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !sleep_task.is_finished(),
        "sleep completed while request guard held"
    );

    drop(guard);
    tokio::time::timeout(Duration::from_secs(2), sleep_task)
        .await
        .expect("sleep did not complete after guard dropped")
        .expect("sleep task panicked")
        .expect("sleep returned Err");
    assert_eq!(m.state(), PresenceState::Sleeping);
}

#[tokio::test]
async fn drowse_defers_for_inflight_request() {
    let m = PresenceManager::stub(PresenceState::Active);
    let guard = m.acquire_request_guard_inner(None).await.unwrap();

    let m2 = Arc::clone(&m);
    let drowse_task = tokio::spawn(async move { m2.drowse().await });

    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !drowse_task.is_finished(),
        "drowse must block while request guard is held"
    );

    drop(guard);
    tokio::time::timeout(Duration::from_secs(2), drowse_task)
        .await
        .expect("drowse did not unblock after guard dropped")
        .expect("drowse task panicked")
        .expect_err("stub has no llama-server to unload");
    assert_eq!(m.state(), PresenceState::Active);
}

#[tokio::test]
async fn stream_guard_increments_and_decrements_count() {
    let m = PresenceManager::stub(PresenceState::Active);
    let rx = m.stream_count_tx.subscribe();
    assert_eq!(*rx.borrow(), 0);
    let g1 = m.acquire_stream_guard();
    assert_eq!(*rx.borrow(), 1);
    let g2 = m.acquire_stream_guard();
    assert_eq!(*rx.borrow(), 2);
    drop(g1);
    assert_eq!(*rx.borrow(), 1);
    drop(g2);
    assert_eq!(*rx.borrow(), 0);
}

#[tokio::test(start_paused = true)]
async fn wait_until_llm_idle_returns_true_immediately_when_zero() {
    let m = PresenceManager::stub(PresenceState::Active);
    let idle = tokio::time::timeout(
        Duration::from_millis(20),
        m.wait_until_llm_idle(Duration::from_secs(5)),
    )
    .await
    .expect("wait did not complete fast");
    assert!(idle);
}

#[tokio::test(start_paused = true)]
async fn wait_until_llm_idle_times_out_when_busy() {
    let m = PresenceManager::stub(PresenceState::Active);
    let _g = m.acquire_stream_guard();
    assert!(!m.wait_until_llm_idle(Duration::from_millis(30)).await);
}

#[tokio::test(start_paused = true)]
async fn wait_until_llm_idle_returns_true_after_guard_dropped() {
    let m = PresenceManager::stub(PresenceState::Active);
    let g = m.acquire_stream_guard();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        drop(g);
    });
    assert!(m.wait_until_llm_idle(Duration::from_millis(500)).await);
}

#[tokio::test]
async fn stream_guard_does_not_block_sleep() {
    let m = PresenceManager::stub(PresenceState::Active);
    let _stream = m.acquire_stream_guard();
    let m2 = Arc::clone(&m);
    let sleep_task = tokio::spawn(async move { m2.sleep().await });
    tokio::time::timeout(Duration::from_secs(1), sleep_task)
        .await
        .expect("sleep was blocked by an LLM stream guard")
        .expect("sleep task panicked")
        .expect("sleep returned Err");
    assert_eq!(m.state(), PresenceState::Sleeping);
}

#[tokio::test]
async fn wake_marker_cleared_on_error_path() {
    let m = PresenceManager::stub(PresenceState::Drowsy);
    assert!(m.wake_in_progress().is_none());
    m.wake()
        .await
        .expect_err("wake must fail against dummy control");
    assert!(
        m.wake_in_progress().is_none(),
        "wake_in_progress must be cleared after wake returns, even on error"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rapid_sleep_and_guard_churn_does_not_deadlock() {
    let m = PresenceManager::stub(PresenceState::Active);

    let mut readers = Vec::new();
    for _ in 0..4 {
        let m = Arc::clone(&m);
        readers.push(tokio::spawn(async move {
            for _ in 0..50 {
                if let Ok(g) = m.acquire_request_guard_inner(None).await {
                    tokio::task::yield_now().await;
                    drop(g);
                } else {
                    tokio::task::yield_now().await;
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

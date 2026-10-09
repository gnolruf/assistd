use super::*;

#[tokio::test]
async fn transition_to_current_state_is_a_silent_noop() {
    for state in [
        PresenceTarget::Active,
        PresenceTarget::Drowsy,
        PresenceTarget::Sleeping,
    ] {
        let m = PresenceManager::stub(state);
        let rx = m.subscribe();
        m.set_presence(state)
            .await
            .unwrap_or_else(|e| panic!("{state:?}: {e:#}"));
        assert_eq!(m.state(), state.into());
        assert!(!rx.has_changed().unwrap(), "{state:?}: broadcast a no-op");
    }
}

#[tokio::test]
async fn ensure_active_resets_activity_timer() {
    let m = PresenceManager::stub(PresenceTarget::Active);
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
async fn dropped_guard_waiter_leaves_its_wake_running() {
    let m = PresenceManager::stub(PresenceTarget::Drowsy);
    let transition = m.transition.lock().await;
    let waiter = timeout(
        Duration::from_millis(50),
        m.acquire_request_guard_inner(None),
    )
    .await;
    assert!(waiter.is_err(), "acquire finished while a transition ran");
    assert_eq!(Arc::strong_count(&m), 2, "the wake died with its waiter");

    drop(transition);
    timeout(Duration::from_secs(2), async {
        while Arc::strong_count(&m) > 1 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the detached wake never finished");
}

#[tokio::test]
async fn sleep_defers_for_inflight_request() {
    let m = PresenceManager::stub(PresenceTarget::Active);
    let guard = m.acquire_request_guard_inner(None).await.unwrap();

    let m2 = Arc::clone(&m);
    let sleep_task = tokio::spawn(async move { m2.sleep().await });

    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !sleep_task.is_finished(),
        "sleep completed while request guard held"
    );

    drop(guard);
    timeout(Duration::from_secs(2), sleep_task)
        .await
        .expect("sleep did not complete after guard dropped")
        .expect("sleep task panicked")
        .expect("sleep returned Err");
    assert_eq!(m.state(), PresenceState::Sleeping);
}

#[tokio::test(start_paused = true)]
async fn wait_until_llm_idle_returns_true_after_guard_dropped() {
    let m = PresenceManager::stub(PresenceTarget::Active);
    let g = m.acquire_stream_guard();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        drop(g);
    });
    assert!(m.wait_until_llm_idle(Duration::from_millis(500)).await);
}

#[tokio::test]
async fn stream_guard_does_not_block_sleep() {
    let m = PresenceManager::stub(PresenceTarget::Active);
    let _stream = m.acquire_stream_guard();
    let m2 = Arc::clone(&m);
    let sleep_task = tokio::spawn(async move { m2.sleep().await });
    timeout(Duration::from_secs(1), sleep_task)
        .await
        .expect("sleep was blocked by an LLM stream guard")
        .expect("sleep task panicked")
        .expect("sleep returned Err");
    assert_eq!(m.state(), PresenceState::Sleeping);
}

#[tokio::test]
async fn failed_wake_settles_back_to_the_prior_state() {
    let m = PresenceManager::stub(PresenceTarget::Sleeping);
    m.wake()
        .await
        .expect_err("stub llama-server binary does not exist");
    assert_eq!(m.state(), PresenceState::Sleeping);
    assert!(m.wake_in_progress().is_none());
    assert_eq!(m.state().next(), PresenceTarget::Active);
}

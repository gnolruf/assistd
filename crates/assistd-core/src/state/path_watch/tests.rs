use super::*;

const WINDOW: Duration = Duration::from_secs(2);

#[tokio::test(start_paused = true)]
async fn a_burst_of_changes_inside_the_debounce_window_fires_once() {
    let (tx, mut changes) = mpsc::channel(CHANGE_BUFFER);
    let _open = tx.clone();
    let burst = tokio::spawn(async move {
        for _ in 0..4 {
            tx.send(()).await.unwrap();
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    });

    assert!(next_burst(&mut changes, WINDOW).await.is_some());
    burst.await.unwrap();
    let another = tokio::time::timeout(WINDOW * 5, next_burst(&mut changes, WINDOW)).await;
    assert!(another.is_err(), "the burst fired more than once");
}

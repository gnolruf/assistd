use std::future::Future;
use std::thread;

struct PanicHandler<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> Drop for PanicHandler<F> {
    fn drop(&mut self) {
        if thread::panicking()
            && let Some(handler) = self.0.take()
        {
            handler();
        }
    }
}

pub(crate) async fn with_panic_handler<Fut, OnPanic>(future: Fut, on_panic: OnPanic) -> Fut::Output
where
    Fut: Future,
    OnPanic: FnOnce(),
{
    let _handler = PanicHandler(Some(on_panic));
    future.await
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::with_panic_handler;

    #[tokio::test]
    async fn panic_handler_runs_after_an_await_and_preserves_the_panic() {
        let unavailable = Arc::new(AtomicBool::new(false));
        let on_panic = Arc::clone(&unavailable);
        let task = tokio::spawn(with_panic_handler(
            async {
                tokio::task::yield_now().await;
                panic!("startup failed");
            },
            move || on_panic.store(true, Ordering::SeqCst),
        ));

        let error = task
            .await
            .expect_err("startup panic must remain observable");
        assert!(error.is_panic());
        assert!(unavailable.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn successful_startup_does_not_run_the_panic_handler() {
        let result = with_panic_handler(
            async {
                tokio::task::yield_now().await;
                7
            },
            || panic!("unexpected panic"),
        )
        .await;

        assert_eq!(result, 7);
    }
}

//! Supervised task spawning and the daemon panic hook. Recovery events
//! log under `target = "assistd::recovery"`.

use std::any::Any;
use std::future::Future;
use std::time::Duration;

use tokio::task::{JoinHandle, JoinSet};
use tracing::{debug, error, info, warn};

pub use assistd_ipc::{Component, StatusSeverity};

/// Log a recovery event at the level matching `severity`, under
/// `target = "assistd::recovery"`, with `severity`, `component`, and
/// `event` fields ahead of the caller's own.
#[macro_export]
macro_rules! recovery_event {
    ($severity:expr, $component:expr, $event:literal $(, $($field:tt)*)?) => {{
        let __component_str: &'static str = $crate::recovery::Component::as_str($component);
        let __event_str: &'static str = $event;
        match $severity {
            $crate::recovery::StatusSeverity::Info => ::tracing::info!(
                target: "assistd::recovery",
                severity = "info",
                component = __component_str,
                event = __event_str,
                $($($field)*)?
            ),
            $crate::recovery::StatusSeverity::Warning => ::tracing::warn!(
                target: "assistd::recovery",
                severity = "warning",
                component = __component_str,
                event = __event_str,
                $($($field)*)?
            ),
            $crate::recovery::StatusSeverity::Error => ::tracing::error!(
                target: "assistd::recovery",
                severity = "error",
                component = __component_str,
                event = __event_str,
                $($($field)*)?
            ),
        }
    }};
}

/// `tokio::spawn` a detached future and emit a recovery event if it
/// panics, instead of losing the panic in a never-joined `JoinHandle`.
pub fn spawn_supervised<F>(name: &'static str, component: Component, future: F) -> JoinHandle<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let inner = tokio::spawn(future);
    tokio::spawn(async move {
        match inner.await {
            Ok(()) => {}
            Err(join_err) if join_err.is_panic() => {
                let payload = join_err.into_panic();
                let msg = panic_message(&payload);
                recovery_event!(
                    StatusSeverity::Error,
                    component,
                    "task_panic",
                    task = name,
                    panic = %msg,
                    "supervised task panicked"
                );
            }
            Err(join_err) if join_err.is_cancelled() => {
                debug!(
                    target: "assistd::recovery",
                    component = component.as_str(),
                    task = name,
                    "supervised task cancelled"
                );
            }
            Err(_) => {}
        }
    })
}

/// Replace the global panic hook with one that logs a recovery event
/// with the panic's location before chaining to the previous hook.
pub fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info.location().map_or_else(
            || "<unknown>".to_string(),
            |loc| format!("{}:{}", loc.file(), loc.line()),
        );
        let payload_msg = panic_message(info.payload());

        recovery_event!(
            StatusSeverity::Error,
            Component::Daemon,
            "panic",
            location = %location,
            message = %payload_msg,
            "panic"
        );

        previous(info);
    }));
}

/// Wait up to `grace` for every task in `tasks` to finish, then abort
/// the rest. Panics are logged as `"{what} task panicked"`.
pub async fn drain_join_set(tasks: &mut JoinSet<()>, grace: Duration, what: &str) {
    let in_flight = tasks.len();
    if in_flight == 0 {
        return;
    }
    info!(
        grace_secs = grace.as_secs(),
        in_flight, "draining in-flight {what} tasks"
    );
    let drained = tokio::time::timeout(grace, async {
        while let Some(res) = tasks.join_next().await {
            if let Err(e) = res
                && e.is_panic()
            {
                error!("{what} task panicked: {e}");
            }
        }
    })
    .await;
    if drained.is_err() {
        warn!(
            remaining = tasks.len(),
            "shutdown grace expired; aborting remaining {what} tasks"
        );
        tasks.shutdown().await;
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawn_supervised_contains_inner_panic() {
        let handle = spawn_supervised("panicker", Component::Llm, async {
            panic!("boom");
        });
        handle
            .await
            .expect("sentinel must not panic on inner panic");
    }

    #[test]
    fn panic_message_extracts_string_payloads() {
        let cases: [(Box<dyn Any + Send>, &str); 3] = [
            (Box::new("static panic text"), "static panic text"),
            (Box::new("owned panic text".to_string()), "owned panic text"),
            (Box::new(42u32), "<non-string panic payload>"),
        ];
        for (payload, expected) in cases {
            assert_eq!(panic_message(&*payload), expected);
        }
    }
}

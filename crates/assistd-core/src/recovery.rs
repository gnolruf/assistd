//! Supervised task spawning and the daemon panic hook. Recovery events
//! log under `target = "assistd::recovery"`.

use std::any::Any;
use std::future::Future;
use std::sync::Weak;
use std::time::Duration;

use parking_lot::Mutex;

use tokio::task::{JoinHandle, JoinSet};

use crate::PresenceManager;

pub use assistd_ipc::{Component, StatusSeverity};

/// Emit a structured recovery event: the `tracing` macro for the
/// severity, under `target = "assistd::recovery"`, with `severity`,
/// `component`, and `event` fields ahead of the caller's own.
///
/// ```ignore
/// recovery_event!(
///     StatusSeverity::Warning,
///     Component::Llm,
///     "crash_detected",
///     pid = old_pid,
///     "llama-server died mid-response, restarting"
/// );
/// ```
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
                ::tracing::debug!(
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

/// Replace the global panic hook with one that logs a recovery event and
/// best-effort SIGTERMs the llama-server process group before chaining
/// to the previous hook. `presence` is `Weak` so the hook never keeps the
/// manager alive past shutdown.
pub fn install_panic_hook(presence: Weak<PresenceManager>) {
    static PRESENCE: Mutex<Option<Weak<PresenceManager>>> = Mutex::new(None);
    *PRESENCE.lock() = Some(presence);

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "<unknown>".to_string());
        let payload_msg = panic_message(info.payload());

        recovery_event!(
            StatusSeverity::Error,
            Component::Daemon,
            "panic",
            location = %location,
            message = %payload_msg,
            "daemon panic; killing llama-server before propagating"
        );

        let pid_opt = PRESENCE
            .lock()
            .as_ref()
            .and_then(|w| w.upgrade())
            .and_then(|p| p.llama_pid_blocking());
        if let Some(pid) = pid_opt
            && let Some(pgid) = rustix::process::Pid::from_raw(pid as i32)
        {
            let _ = rustix::process::kill_process_group(pgid, rustix::process::Signal::TERM);
            recovery_event!(
                StatusSeverity::Warning,
                Component::Llm,
                "panic_kill",
                pid = pid,
                "sent SIGTERM to llama-server process group from panic hook"
            );
        }

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
    tracing::info!(
        grace_secs = grace.as_secs(),
        in_flight,
        "draining in-flight {what} tasks"
    );
    let drained = tokio::time::timeout(grace, async {
        while let Some(res) = tasks.join_next().await {
            if let Err(e) = res
                && e.is_panic()
            {
                tracing::error!("{what} task panicked: {e}");
            }
        }
    })
    .await;
    if drained.is_err() {
        tracing::warn!(
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

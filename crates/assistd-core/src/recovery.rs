//! Structured recovery vocabulary, supervised task spawning, and the
//! daemon panic hook.
//!
//! Three responsibilities:
//!
//! 1. **Vocabulary**: [`Component`] and [`RecoverySeverity`] give every
//!    recovery event a canonical `severity`/`component` field pair.
//!    Filterable with `RUST_LOG=assistd::recovery=info`.
//! 2. **Panic isolation**: [`spawn_supervised`] wraps a `tokio::spawn`
//!    so panics in detached tasks emit a recovery event instead of
//!    silently disappearing into a never-joined `JoinHandle`.
//! 3. **Daemon panic hook**: [`install_panic_hook`] replaces the global
//!    panic hook so that any panic also tries to SIGTERM the running
//!    llama-server process group before propagating, keeping a child
//!    from being orphaned when the daemon goes down via panic.

use std::any::Any;
use std::future::Future;
use std::sync::Weak;

use parking_lot::Mutex;

use tokio::task::JoinHandle;

use crate::PresenceManager;

/// Severity of a recovery event. Maps 1:1 to a `tracing` log level and
/// to the `severity` string field on the wire (`Event::Status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoverySeverity {
    /// Routine recovery progress (e.g. "replay started"). `info` level.
    Info,
    /// A recoverable failure was observed (e.g. "llama-server crashed,
    /// restarting"). `warn` level.
    Warning,
    /// The recovery itself failed and the operation will not complete
    /// (e.g. "supervisor degraded; replay aborted"). `error` level.
    Error,
}

impl RecoverySeverity {
    /// Returns the canonical lowercase wire string for this severity level.
    pub fn as_str(self) -> &'static str {
        match self {
            RecoverySeverity::Info => "info",
            RecoverySeverity::Warning => "warning",
            RecoverySeverity::Error => "error",
        }
    }
}

/// Canonical component identifier carried as a structured field on every
/// recovery event. New subsystems get a new variant rather than a free-form
/// string so log filters and dashboards can rely on a fixed vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    /// llama-server lifecycle, restarts, in-flight crash detection.
    Llm,
    /// MCP transport / supervisor.
    Mcp,
    /// Voice input/output (whisper, piper, mic, listen).
    Voice,
    /// SQLite-backed memory + conversation persistence.
    Memory,
    /// Window-manager backend (i3/sway/hyprland).
    Wm,
    /// Embedding service + worker task.
    Embed,
    /// Global hotkey listener.
    Hotkey,
    /// Top-level daemon orchestration (signal handler, panics with no
    /// more specific subsystem attribution).
    Daemon,
    /// Idle-monitor task that auto-drowses/sleeps.
    IdleMonitor,
    /// GPU-monitor task that auto-sleeps when a foreground GPU consumer
    /// (game, ML training) shows up.
    GpuMonitor,
    /// Continuous-listen utterance dispatcher.
    ListenDispatcher,
}

impl Component {
    /// Returns the canonical lowercase wire string for this component identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Component::Llm => "llm",
            Component::Mcp => "mcp",
            Component::Voice => "voice",
            Component::Memory => "memory",
            Component::Wm => "wm",
            Component::Embed => "embed",
            Component::Hotkey => "hotkey",
            Component::Daemon => "daemon",
            Component::IdleMonitor => "idle_monitor",
            Component::GpuMonitor => "gpu_monitor",
            Component::ListenDispatcher => "listen_dispatcher",
        }
    }
}

/// Emit a structured recovery event at the given severity level.
///
/// Wraps the corresponding `tracing` macro so every recovery event is
/// emitted under `target = "assistd::recovery"` with `severity` and
/// `component` fields. Additional structured fields (pid, attempt,
/// ran_for_secs, etc.) are passed via the trailing tt-munch.
///
/// # Example
///
/// ```ignore
/// recovery_event!(
///     RecoverySeverity::Warning,
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
            $crate::recovery::RecoverySeverity::Info => ::tracing::info!(
                target: "assistd::recovery",
                severity = "info",
                component = __component_str,
                event = __event_str,
                $($($field)*)?
            ),
            $crate::recovery::RecoverySeverity::Warning => ::tracing::warn!(
                target: "assistd::recovery",
                severity = "warning",
                component = __component_str,
                event = __event_str,
                $($($field)*)?
            ),
            $crate::recovery::RecoverySeverity::Error => ::tracing::error!(
                target: "assistd::recovery",
                severity = "error",
                component = __component_str,
                event = __event_str,
                $($($field)*)?
            ),
        }
    }};
}

/// `tokio::spawn` a future and emit a recovery event if it panics.
///
/// Detached tokio tasks normally swallow panics into their never-joined
/// `JoinHandle`. This wrapper joins the handle from a sentinel task so
/// the panic surfaces as a structured `target = "assistd::recovery"`
/// log line attributed to the named component.
///
/// `name` is a short identifier (e.g. `"signal_handler"`) included as
/// the `task` field on the panic event.
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
                    RecoverySeverity::Error,
                    component,
                    "task_panic",
                    task = name,
                    panic = %msg,
                    "supervised task panicked"
                );
            }
            Err(join_err) if join_err.is_cancelled() => {
                // Cancellation is normal during shutdown; do not
                // promote to a recovery event.
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

/// Replace the global panic hook with one that logs structured recovery
/// fields and best-effort SIGTERMs the running llama-server before
/// chaining to the previous hook.
///
/// `presence` is a `Weak` so the hook does not keep the manager alive
/// past daemon shutdown. Pass `Arc::downgrade(&presence_arc)`.
///
/// Idempotent: installing twice replaces the previous chain, so tests can
/// safely re-install in setup.
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
            RecoverySeverity::Error,
            Component::Daemon,
            "panic",
            location = %location,
            message = %payload_msg,
            "daemon panic; killing llama-server before propagating"
        );

        // Best-effort: try to SIGTERM the llama-server process group so
        // the child doesn't outlive the daemon. We hold the lock only
        // long enough to grab a strong ref + read the pid.
        let pid_opt = PRESENCE
            .lock()
            .as_ref()
            .and_then(|w| w.upgrade())
            .and_then(|p| p.llama_pid_blocking());
        if let Some(pid) = pid_opt {
            // The child was started in its own session via `setsid`, so its
            // pgid equals its pid. `kill_process_group` is the safe rustix
            // wrapper around `killpg(2)`. We ignore the result: the child
            // may have already exited between read and kill, and there is
            // nothing actionable to do from a panic hook regardless.
            if let Some(pgid) = rustix::process::Pid::from_raw(pid as i32) {
                let _ = rustix::process::kill_process_group(pgid, rustix::process::Signal::TERM);
                recovery_event!(
                    RecoverySeverity::Warning,
                    Component::Llm,
                    "panic_kill",
                    pid = pid,
                    "sent SIGTERM to llama-server process group from panic hook"
                );
            }
        }

        previous(info);
    }));
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

    #[test]
    fn component_as_str_is_stable() {
        assert_eq!(Component::Llm.as_str(), "llm");
        assert_eq!(Component::IdleMonitor.as_str(), "idle_monitor");
        assert_eq!(Component::ListenDispatcher.as_str(), "listen_dispatcher");
    }

    #[test]
    fn severity_as_str_matches_tracing_levels() {
        assert_eq!(RecoverySeverity::Info.as_str(), "info");
        assert_eq!(RecoverySeverity::Warning.as_str(), "warning");
        assert_eq!(RecoverySeverity::Error.as_str(), "error");
    }

    #[tokio::test]
    async fn spawn_supervised_completes_normally_for_clean_task() {
        let handle = spawn_supervised("ok_task", Component::Daemon, async {
            tokio::task::yield_now().await;
        });
        // The sentinel handle resolves once the inner task finishes.
        // We don't assert any panic logging here; the absence of a
        // panic event is the whole point.
        handle.await.expect("sentinel join");
    }

    #[tokio::test]
    async fn spawn_supervised_logs_on_panic() {
        // We don't have a captured-subscriber harness wired here; the
        // assertion is that the sentinel itself does not panic and
        // resolves cleanly even when the inner task panicked. Visual
        // verification via test-log output covers the message shape.
        let handle = spawn_supervised("panicker", Component::Llm, async {
            panic!("boom");
        });
        handle
            .await
            .expect("sentinel must not panic on inner panic");
    }

    #[test]
    fn panic_message_extracts_static_str() {
        let payload: Box<dyn Any + Send> = Box::new("static panic text");
        assert_eq!(panic_message(&*payload), "static panic text");
    }

    #[test]
    fn panic_message_extracts_owned_string() {
        let payload: Box<dyn Any + Send> = Box::new("owned panic text".to_string());
        assert_eq!(panic_message(&*payload), "owned panic text");
    }

    #[test]
    fn panic_message_falls_back_for_non_string_payload() {
        let payload: Box<dyn Any + Send> = Box::new(42u32);
        assert_eq!(panic_message(&*payload), "<non-string panic payload>");
    }
}

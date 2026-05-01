//! i3 backend for [`crate::WindowManager`].
//!
//! Speaks the i3 IPC protocol via `tokio-i3ipc`. Two Unix-socket
//! connections are opened at startup: one held under a `Mutex` for
//! command sends, one consumed by [`tokio_i3ipc::I3::listen`] for the
//! event stream. A background task subscribes to `Window` events and
//! keeps a snapshot of the focused window's class up to date so
//! [`I3Backend::focused_window`] reads from memory rather than round-
//! tripping over the socket.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use tokio::sync::{Mutex, RwLock, watch};
use tokio::task::JoinHandle;
use tokio_i3ipc::{
    I3,
    event::{Event, Subscribe, WindowChange},
    reply,
};

use crate::{WindowId, WindowManager, WorkspaceId};

/// Cached focus state. Seeded from `get_tree()` at startup, then updated
/// by the event task on every `Window::Focus` event. The trait surface
/// only reads `focused_window`, so we don't track workspace focus.
#[derive(Default, Clone)]
struct Snapshot {
    focused_window: Option<WindowId>,
}

/// `WindowManager` impl wrapping a single i3 IPC command socket.
/// Held inside `Arc<dyn WindowManager>` by the daemon's `AppState`.
pub struct I3Backend {
    cmd: Arc<Mutex<I3>>,
    snapshot: Arc<RwLock<Snapshot>>,
}

/// Returned by [`I3Backend::start`] alongside the backend itself. The
/// daemon awaits [`I3Handle::shutdown`] in its graceful-shutdown block
/// so the event task drains before the process exits — matching how
/// other long-lived subsystems (`embedder_task_handle`, `hotkey_handle`,
/// etc.) are awaited in `daemon.rs`.
pub struct I3Handle {
    pub backend: Arc<I3Backend>,
    event_task: JoinHandle<()>,
}

impl I3Handle {
    pub async fn shutdown(self) {
        // The daemon flips `shutdown_tx` first; the event task selects
        // on `shutdown.changed()` and exits. Awaiting here just drains.
        let _ = self.event_task.await;
    }
}

impl I3Backend {
    /// Connect to the i3 IPC sockets, seed the focused-window snapshot,
    /// and spawn the background event task.
    ///
    /// Returns `Err` when the i3 socket isn't reachable (i3 not running,
    /// `I3SOCK` unset, non-Linux dev box, …). The daemon catches that
    /// and substitutes `NoWindowManager` so the rest of startup proceeds.
    pub async fn start(mut shutdown: watch::Receiver<bool>) -> Result<I3Handle> {
        // Two sockets: `listen()` consumes its connection by value, so a
        // single socket can't multiplex command + event traffic.
        let mut cmd = I3::connect()
            .await
            .context("connect to i3 IPC (cmd socket)")?;
        let mut events_conn = I3::connect()
            .await
            .context("connect to i3 IPC (events socket)")?;
        events_conn
            .subscribe([Subscribe::Window])
            .await
            .context("subscribe to i3 window events")?;

        // Seed snapshot before wrapping cmd in the Mutex — reuses the
        // command socket since `get_tree()` needs `&mut`.
        let initial = match seed_snapshot(&mut cmd).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("i3 seed_snapshot failed: {e:#}; starting with empty focus state");
                Snapshot::default()
            }
        };
        let snapshot = Arc::new(RwLock::new(initial));

        let snap_for_task = snapshot.clone();
        let event_task = tokio::spawn(async move {
            let mut stream = events_conn.listen();
            loop {
                tokio::select! {
                    _ = shutdown.changed() => {
                        if *shutdown.borrow() { break; }
                    }
                    evt = stream.next() => {
                        match evt {
                            Some(Ok(Event::Window(w))) if matches!(w.change, WindowChange::Focus) => {
                                let class = w
                                    .container
                                    .window_properties
                                    .as_ref()
                                    .and_then(|p| p.class.clone());
                                snap_for_task.write().await.focused_window = class;
                            }
                            Some(Ok(_)) => {}
                            Some(Err(e)) => {
                                tracing::warn!("i3 event stream error: {e}");
                                break;
                            }
                            None => break, // socket closed
                        }
                    }
                }
            }
            tracing::info!("i3 event task exited");
        });

        let backend = Arc::new(Self {
            cmd: Arc::new(Mutex::new(cmd)),
            snapshot,
        });
        Ok(I3Handle {
            backend,
            event_task,
        })
    }

    async fn run(&self, payload: &str) -> Result<()> {
        let results = self
            .cmd
            .lock()
            .await
            .run_command(payload)
            .await
            .with_context(|| format!("i3 RUN_COMMAND failed: {payload}"))?;
        for r in results {
            if !r.success {
                anyhow::bail!(
                    "i3 rejected command `{payload}`: {}",
                    r.error.unwrap_or_default()
                );
            }
        }
        Ok(())
    }
}

#[async_trait]
impl WindowManager for I3Backend {
    async fn focus(&self, window: &WindowId) -> Result<()> {
        let cmd = format!(r#"[class="{}"] focus"#, escape_for_criteria(window));
        self.run(&cmd).await
    }

    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> Result<()> {
        let target = format_workspace_target(workspace);
        let cmd = format!(
            r#"[class="{}"] move container to {}"#,
            escape_for_criteria(window),
            target,
        );
        self.run(&cmd).await
    }

    async fn focused_window(&self) -> Result<Option<WindowId>> {
        Ok(self.snapshot.read().await.focused_window.clone())
    }
}

/// Escape `\` and `"` inside a value that lands between `[class="..."]`
/// or `workspace "..."`. Backslashes are escaped first so we don't
/// double-escape the slashes inserted in front of quotes.
fn escape_for_criteria(s: &str) -> String {
    s.replace('\\', r"\\").replace('"', r#"\""#)
}

/// Numeric workspace IDs become `workspace number N` (robust to renames);
/// non-numeric become `workspace "<escaped name>"`.
fn format_workspace_target(ws: &WorkspaceId) -> String {
    if let Ok(n) = ws.parse::<u32>() {
        format!("workspace number {n}")
    } else {
        format!(r#"workspace "{}""#, escape_for_criteria(ws))
    }
}

async fn seed_snapshot(cmd: &mut I3) -> Result<Snapshot> {
    let tree = cmd.get_tree().await.context("i3 GET_TREE")?;
    Ok(Snapshot {
        focused_window: walk_focused(&tree)
            .and_then(|n| n.window_properties.as_ref())
            .and_then(|p| p.class.clone()),
    })
}

fn walk_focused(node: &reply::Node) -> Option<&reply::Node> {
    if node.focused {
        return Some(node);
    }
    for child in node.nodes.iter().chain(node.floating_nodes.iter()) {
        if let Some(n) = walk_focused(child) {
            return Some(n);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_for_criteria_handles_quotes_and_backslashes() {
        assert_eq!(escape_for_criteria("Firefox"), "Firefox");
        assert_eq!(escape_for_criteria(r#"a"b"#), r#"a\"b"#);
        assert_eq!(escape_for_criteria(r"a\b"), r"a\\b");
        // Order matters: backslash inserted by quote-escape must NOT be
        // re-escaped. `a"b` → `a\"b` (single inserted slash, single quote).
        assert_eq!(escape_for_criteria(r#"a"b\c"#), r#"a\"b\\c"#);
    }

    #[test]
    fn format_workspace_target_numeric() {
        assert_eq!(format_workspace_target(&"3".into()), "workspace number 3");
        assert_eq!(format_workspace_target(&"10".into()), "workspace number 10");
    }

    #[test]
    fn format_workspace_target_named() {
        assert_eq!(
            format_workspace_target(&"scratch".into()),
            r#"workspace "scratch""#
        );
    }

    #[test]
    fn format_workspace_target_named_with_quote_is_escaped() {
        assert_eq!(
            format_workspace_target(&r#"weird"name"#.into()),
            r#"workspace "weird\"name""#
        );
    }

    #[test]
    fn format_workspace_target_u32_parseable_is_numeric() {
        // Any string that parses as `u32` becomes `workspace number N`,
        // including zero-padded variants — `"03"` → `workspace number 3`.
        // i3 normalizes the number, so the semantics match.
        assert_eq!(format_workspace_target(&"03".into()), "workspace number 3");
    }
}

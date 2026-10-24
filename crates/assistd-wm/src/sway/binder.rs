//! Hotkeys bound through Sway IPC: added at connect, re-added after a
//! config reload or reconnect, removed on shutdown.

use futures_util::StreamExt;
use swayipc_async::{Connection, Event, EventStream, EventType, WorkspaceChange};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use assistd_utils::backoff::backoff_delay;

use crate::hotkeys::{BindingCommands, Hotkey, HotkeyEvent};
use crate::{WM_IPC_TIMEOUT, WmError, WmResult};

const EVENT_QUEUE: usize = 16;

/// Hotkeys Sway holds for the daemon, reported as [`HotkeyEvent`]s.
#[derive(Debug)]
pub struct SwayHotkeys {
    events: mpsc::Receiver<HotkeyEvent>,
    task: JoinHandle<()>,
}

impl SwayHotkeys {
    /// Spawn the task that binds every hotkey and reports them,
    /// connecting with backoff until Sway answers. A binding Sway rejects
    /// is logged and skipped.
    pub fn spawn(hotkeys: Vec<Hotkey>, shutdown: watch::Receiver<bool>) -> Self {
        let (tx, rx) = mpsc::channel(EVENT_QUEUE);
        let binder = Binder {
            commands: BindingCommands::new(hotkeys),
            tx,
            shutdown,
        };
        Self {
            events: rx,
            task: tokio::spawn(binder.run()),
        }
    }

    /// The next hotkey press or release; `None` once the task has exited.
    /// Cancel-safe.
    pub async fn next_event(&mut self) -> Option<HotkeyEvent> {
        self.events.recv().await
    }

    /// Await the task, which unbinds every hotkey once the shutdown watch
    /// flips. Flip it first or this blocks.
    pub async fn shutdown(self) {
        let _ = self.task.await;
    }
}

/// Why [`Binder::serve`] stopped.
enum Stop {
    Shutdown,
    Disconnected,
}

struct Binder {
    commands: BindingCommands,
    tx: mpsc::Sender<HotkeyEvent>,
    shutdown: watch::Receiver<bool>,
}

impl Binder {
    async fn run(mut self) {
        while let Some((mut conn, mut events)) = self.connect_with_backoff().await {
            if let Stop::Shutdown = self.serve(&mut conn, &mut events).await {
                self.unbind_all(&mut conn).await;
                return;
            }
            tracing::warn!(target: "assistd::hotkey", "sway hotkey connection lost; reconnecting");
        }
    }

    async fn serve(&mut self, conn: &mut Connection, events: &mut EventStream) -> Stop {
        if self.bind_all(conn).await.is_err() {
            return Stop::Disconnected;
        }
        loop {
            tokio::select! {
                changed = self.shutdown.changed() => {
                    if changed.is_err() || *self.shutdown.borrow() {
                        return Stop::Shutdown;
                    }
                }
                next = events.next() => match next {
                    Some(Ok(event)) => {
                        if self.on_event(conn, event).await.is_err() {
                            return Stop::Disconnected;
                        }
                    }
                    Some(Err(e)) => {
                        tracing::warn!(target: "assistd::hotkey", "sway event stream error: {e}");
                        return Stop::Disconnected;
                    }
                    None => return Stop::Disconnected,
                },
            }
        }
    }

    async fn on_event(&self, conn: &mut Connection, event: Event) -> WmResult<()> {
        match event {
            Event::Binding(binding) => {
                if let Some(hotkey) = self.commands.event_for(&binding.binding.command) {
                    let _ = self.tx.send(hotkey).await;
                }
                Ok(())
            }
            Event::Workspace(workspace) if matches!(workspace.change, WorkspaceChange::Reload) => {
                tracing::info!(target: "assistd::hotkey", "sway config reloaded; rebinding hotkeys");
                self.bind_all(conn).await
            }
            _ => Ok(()),
        }
    }

    /// Connect, retrying with backoff; `None` when shutdown arrives first.
    async fn connect_with_backoff(&mut self) -> Option<(Connection, EventStream)> {
        let mut attempt: u32 = 0;
        loop {
            match connect().await {
                Ok(connected) => return Some(connected),
                Err(e) => tracing::warn!(
                    target: "assistd::hotkey",
                    "sway hotkey connection failed (attempt {}): {e}",
                    attempt + 1
                ),
            }
            tokio::select! {
                changed = self.shutdown.changed() => {
                    if changed.is_err() || *self.shutdown.borrow() {
                        return None;
                    }
                }
                () = tokio::time::sleep(backoff_delay(attempt)) => {}
            }
            attempt = attempt.saturating_add(1);
        }
    }

    async fn bind_all(&self, conn: &mut Connection) -> WmResult<()> {
        for command in self.commands.bind_commands() {
            if let Err(rejected) = run_command(conn, &command).await? {
                tracing::warn!(target: "assistd::hotkey", "sway rejected `{command}`: {rejected}");
            }
        }
        Ok(())
    }

    async fn unbind_all(&self, conn: &mut Connection) {
        for command in self.commands.unbind_commands() {
            match run_command(conn, &command).await {
                Ok(Ok(())) => {}
                Ok(Err(rejected)) => tracing::debug!(
                    target: "assistd::hotkey",
                    "sway rejected `{command}`: {rejected}"
                ),
                Err(e) => {
                    tracing::warn!(target: "assistd::hotkey", "failed to unbind hotkeys: {e}");
                    return;
                }
            }
        }
    }
}

async fn connect() -> WmResult<(Connection, EventStream)> {
    let conn = Connection::new()
        .await
        .map_err(|e| WmError::ipc("connect to sway IPC (hotkey cmd socket)", e))?;
    let events = Connection::new()
        .await
        .map_err(|e| WmError::ipc("connect to sway IPC (hotkey events socket)", e))?
        .subscribe([EventType::Binding, EventType::Workspace])
        .await
        .map_err(|e| WmError::ipc("subscribe to sway binding+workspace events", e))?;
    Ok((conn, events))
}

/// The inner error is Sway's message when it rejects the command.
async fn run_command(conn: &mut Connection, command: &str) -> WmResult<Result<(), String>> {
    let outcomes = tokio::time::timeout(WM_IPC_TIMEOUT, conn.run_command(command))
        .await
        .map_err(|_| WmError::Timeout(WM_IPC_TIMEOUT))?
        .map_err(|e| WmError::ipc("sway RUN_COMMAND", e))?;
    Ok(outcomes
        .into_iter()
        .collect::<Result<(), _>>()
        .map_err(|rejected| rejected.to_string()))
}

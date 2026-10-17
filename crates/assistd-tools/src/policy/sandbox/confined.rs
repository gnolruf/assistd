//! A long-lived thread, barred by Landlock from abstract Unix sockets made
//! outside it, that spawns sandboxed commands sharing the host's network.

use std::io;
use std::sync::mpsc;
use std::thread;

use landlock::{CompatLevel, Compatible, Ruleset, RulesetAttr, RulesetError, Scope};
use tokio::process::{Child, Command as ProcCommand};
use tokio::runtime::Handle;
use tokio::sync::oneshot;

type SpawnRequest = (ProcCommand, Handle, oneshot::Sender<io::Result<Child>>);

/// Why the confined spawning thread could not start.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ConfinementError {
    #[error(
        "the kernel cannot keep sandboxed commands off abstract sockets such as the X \
         server's (needs Landlock scoping, Linux 6.12+): {0}"
    )]
    Landlock(#[from] RulesetError),
    #[error("cannot start the confined spawning thread: {0}")]
    Thread(#[source] io::Error),
}

/// Spawns commands from one confined thread, which children inherit the
/// restriction from. The thread outlives every child it spawns, since
/// `--die-with-parent` kills a child when its spawning thread exits; it
/// stops once this is dropped.
#[derive(Debug)]
pub(crate) struct ConfinedSpawner(mpsc::Sender<SpawnRequest>);

impl ConfinedSpawner {
    /// # Errors
    /// [`ConfinementError`] when the kernel lacks Landlock scoping or the
    /// thread cannot start.
    pub(super) fn start() -> Result<Self, ConfinementError> {
        let (requests, inbox) = mpsc::channel::<SpawnRequest>();
        let (confined_tx, confined) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("assistd-confined-spawn".into())
            .spawn(move || {
                let restricted = forbid_abstract_sockets();
                let is_restricted = restricted.is_ok();
                let _ = confined_tx.send(restricted);
                if is_restricted {
                    spawn_each(inbox);
                }
            })
            .map_err(ConfinementError::Thread)?;
        confined
            .recv()
            .map_err(|_| ConfinementError::Thread(io::Error::other("thread exited")))??;
        Ok(Self(requests))
    }

    pub(crate) async fn spawn(&self, cmd: ProcCommand) -> io::Result<Child> {
        let stopped = || io::Error::other("the confined spawning thread stopped");
        let (reply, child) = oneshot::channel();
        self.0
            .send((cmd, Handle::current(), reply))
            .map_err(|_| stopped())?;
        child.await.map_err(|_| stopped())?
    }
}

fn spawn_each(inbox: mpsc::Receiver<SpawnRequest>) {
    for (mut cmd, runtime, reply) in inbox {
        let _runtime = runtime.enter();
        let _ = reply.send(cmd.spawn());
    }
}

fn forbid_abstract_sockets() -> Result<(), RulesetError> {
    Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .scope(Scope::AbstractUnixSocket)?
        .create()?
        .restrict_self()?;
    Ok(())
}

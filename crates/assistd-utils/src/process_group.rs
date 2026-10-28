//! Owning handle to the process group a spawned child leads, and the
//! parent-death signal that terminates the child if the daemon dies first.

use std::io;
use std::process::ExitStatus;

use rustix::io::Errno;
use rustix::process::{Pid, Signal, kill_process_group, test_kill_process_group};
use tokio::process::{Child, Command};

/// The process group a child spawned with `process_group(0)` leads. The id
/// is fixed at spawn and outlives the leader's own exit. Dropping it
/// SIGKILLs every process still in the group.
#[derive(Debug)]
pub struct ProcessGroup {
    id: Pid,
    emptied: bool,
}

impl ProcessGroup {
    /// The group `child` leads, or `None` once the child has been reaped.
    pub fn led_by(child: &Child) -> Option<Self> {
        child
            .id()
            .and_then(|pid| i32::try_from(pid).ok())
            .and_then(Pid::from_raw)
            .map(|id| Self { id, emptied: false })
    }

    /// The process group id, equal to the leader's pid.
    pub fn id(&self) -> Pid {
        self.id
    }

    /// Wait for `child`, the group's leader, to exit and be reaped. If no
    /// member outlived it, the group is marked empty and never signalled
    /// again, since the kernel may hand its id to an unrelated process.
    pub async fn reap_leader(&mut self, child: &mut Child) -> io::Result<ExitStatus> {
        let status = child.wait().await?;
        self.emptied |= test_kill_process_group(self.id) == Err(Errno::SRCH);
        Ok(status)
    }

    /// Send `signal` to every process in the group, ignoring failures; a
    /// no-op once [`Self::reap_leader`] found the group empty.
    pub fn signal(&self, signal: Signal) {
        if !self.emptied {
            let _ = kill_process_group(self.id, signal);
        }
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        self.signal(Signal::KILL);
    }
}

/// Have the kernel SIGTERM the child `cmd` spawns when the daemon dies, even
/// by SIGKILL; a child whose parent died before the signal was armed exits
/// at once. Only the direct child gets the signal, not the rest of its
/// group. It fires when the spawning *thread* exits, so never spawn from
/// `spawn_blocking`: an idle blocking-pool thread would kill it.
#[cfg(target_os = "linux")]
#[allow(
    unsafe_code,
    reason = "std exposes no safe way to run code between fork and exec"
)]
pub fn set_parent_death_signal(cmd: &mut Command) {
    let parent = rustix::process::getpid();
    // SAFETY: the closure runs in the child between fork() and exec(). It
    // captures only a `Pid` and issues the prctl(PR_SET_PDEATHSIG) and
    // getppid() syscalls, both async-signal-safe; it does not allocate.
    unsafe {
        cmd.pre_exec(move || {
            rustix::process::set_parent_process_death_signal(Some(Signal::TERM))?;
            if rustix::process::getppid() != Some(parent) {
                return Err(Errno::SRCH.into());
            }
            Ok(())
        });
    }
}

/// No-op off Linux, where PDEATHSIG does not exist.
#[cfg(not(target_os = "linux"))]
pub fn set_parent_death_signal(_cmd: &mut Command) {}

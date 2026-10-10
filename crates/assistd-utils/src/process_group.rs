//! Owning handle to the process group a spawned child leads, and the
//! parent-death signal that reaps it if the daemon dies first.

use rustix::process::{Pid, Signal, kill_process_group};
use tokio::process::{Child, Command};

/// The process group a child spawned with `process_group(0)` leads. The id
/// is fixed at spawn and outlives the leader's own exit. Dropping it
/// SIGKILLs every process still in the group.
#[derive(Debug)]
pub struct ProcessGroup(Pid);

impl ProcessGroup {
    /// The group `child` leads, or `None` once the child has been reaped.
    pub fn led_by(child: &Child) -> Option<Self> {
        child
            .id()
            .and_then(|pid| i32::try_from(pid).ok())
            .and_then(Pid::from_raw)
            .map(Self)
    }

    /// The process group id, equal to the leader's pid.
    pub fn id(&self) -> Pid {
        self.0
    }

    /// Send `signal` to every process in the group, ignoring failures.
    pub fn signal(&self, signal: Signal) {
        let _ = kill_process_group(self.0, signal);
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        self.signal(Signal::KILL);
    }
}

/// Have the kernel SIGTERM the child `cmd` spawns when the daemon dies, even
/// by SIGKILL. `pre_exec` is the only way to set PDEATHSIG on a spawned child.
#[cfg(target_os = "linux")]
#[allow(
    unsafe_code,
    reason = "std exposes no safe way to run code between fork and exec"
)]
pub fn set_parent_death_signal(cmd: &mut Command) {
    // SAFETY: the closure runs in the child between fork() and exec(). It
    // captures nothing and only issues the prctl(PR_SET_PDEATHSIG) syscall,
    // which is async-signal-safe.
    unsafe {
        cmd.pre_exec(|| {
            rustix::process::set_parent_process_death_signal(Some(Signal::TERM)).map_err(Into::into)
        });
    }
}

/// No-op off Linux, where PDEATHSIG does not exist.
#[cfg(not(target_os = "linux"))]
pub fn set_parent_death_signal(_cmd: &mut Command) {}

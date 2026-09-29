//! Owning handle to the process group a spawned child leads.

use rustix::process::{Pid, Signal, kill_process_group};
use tokio::process::Child;

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

//! A sandbox value for tests, which can start no command.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

use super::probe::host_path;
use super::scope::CommandScope;
use super::{Profile, Protected, SandboxInfo, SandboxPrograms, SharedDirs};

/// A host-profile bwrap sandbox over `shared` whose programs do not exist, so a command
/// built from it fails to start on every machine.
pub fn inert_sandbox(shared: SharedDirs) -> Arc<SandboxInfo> {
    Arc::new(SandboxInfo {
        programs: SandboxPrograms {
            bwrap: PathBuf::from("/nonexistent/bwrap"),
            prlimit: PathBuf::from("/nonexistent/prlimit"),
            choom: PathBuf::from("/nonexistent/choom"),
            scope: CommandScope {
                systemd_run: PathBuf::from("/nonexistent/systemd-run"),
                runtime_dir: OsString::from("/nonexistent/runtime"),
            },
        },
        extra_args: Vec::new(),
        protected: Protected::default(),
        shared,
        watched: Arc::default(),
        host_path: host_path(&std::env::var_os("PATH").unwrap_or_default()),
        unit_dirs: Vec::new(),
        confined: None,
        profile: Profile::Host,
    })
}

//! The transient systemd user scope each sandboxed command runs in, which caps
//! the memory and tasks of all its processes together.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use assistd_utils::program::{ProgramError, find_program};

/// The variable `systemd-run` finds the user's systemd manager through.
const RUNTIME_DIR: &str = "XDG_RUNTIME_DIR";

/// The variables a scope's command inherits from `systemd-run`, which the
/// sandbox unsets.
pub(super) const SCOPE_ENV: [&str; 2] = [RUNTIME_DIR, "INVOCATION_ID"];

/// Run the command in a new scope of the user's manager, removed once it
/// exits even if it failed, with its arguments passed through unexpanded.
const SCOPE_FLAGS: [&str; 5] = [
    "--user",
    "--scope",
    "--quiet",
    "--collect",
    "--expand-environment=no",
];

/// The caps on each scope, as `systemd-run` flags: the memory all of its
/// processes hold together, with no swap past it, and its tasks.
const SCOPE_LIMITS: [&str; 3] = [
    "--property=MemoryMax=4G",
    "--property=MemorySwapMax=0",
    "--property=TasksMax=512",
];

/// Why sandboxed commands cannot run in a transient systemd scope.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScopeError {
    #[error("cannot run `systemd-run`: {0}")]
    Missing(ProgramError),
    #[error("{RUNTIME_DIR} is not set, so `systemd-run` cannot reach the user's systemd manager")]
    NoRuntimeDir,
    #[error("`systemd-run` cannot start a user scope: {0}")]
    Refused(String),
}

/// How to start a command in a memory- and task-capped scope of the user's
/// systemd manager.
#[derive(Debug, Clone)]
pub(crate) struct CommandScope {
    pub(super) systemd_run: PathBuf,
    pub(super) runtime_dir: OsString,
}

impl CommandScope {
    /// Find `systemd-run` on `path_env` and start one capped scope with it,
    /// so a manager that refuses scopes or their caps is caught at startup.
    /// Blocks until that scope's command exits.
    pub(super) fn probe(path_env: &OsStr) -> Result<Self, ScopeError> {
        let systemd_run =
            find_program(Path::new("systemd-run"), path_env).map_err(ScopeError::Missing)?;
        let runtime_dir = std::env::var_os(RUNTIME_DIR).ok_or(ScopeError::NoRuntimeDir)?;
        let scope = Self {
            systemd_run,
            runtime_dir,
        };
        let output = scope
            .command(Path::new("true"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .output()
            .map_err(|error| ScopeError::Refused(error.to_string()))?;
        if output.status.success() {
            Ok(scope)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(ScopeError::Refused(stderr.trim().to_owned()))
        }
    }

    /// The command that runs `program` in a new capped scope, from an
    /// environment holding only what `systemd-run` needs to reach the
    /// manager. `systemd-run` execs `program`, which keeps its process ID.
    pub(super) fn command(&self, program: &Path) -> Command {
        let mut cmd = Command::new(&self.systemd_run);
        cmd.env_clear()
            .env(RUNTIME_DIR, &self.runtime_dir)
            .args(SCOPE_FLAGS)
            .args(SCOPE_LIMITS)
            .arg(program);
        cmd
    }
}

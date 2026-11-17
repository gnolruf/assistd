//! Resolving the requested sandbox at startup: the programs it needs, a scope the user's
//! systemd manager accepts, and the host's program and unit directories.

use std::ffi::OsStr;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_utils::program::{ProgramError, find_program};
use tracing::{info, warn};

use super::confined::ConfinedSpawner;
use super::scope::{CommandScope, ScopeError};
use super::{
    Profile, Protected, SANDBOX_PATH, SandboxInfo, SandboxPrograms, SandboxRequest, SharedDirs,
    ToolSandbox, ToolsDisabled,
};

/// Why [`probe_sandbox`] could not satisfy the requested sandbox.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error(
        "tools.bash.sandbox = \"bwrap\" but `bwrap` cannot be run: {0}. \
         Install bubblewrap, or set sandbox to \"auto\" to start with tools disabled."
    )]
    BwrapNotFound(ProgramError),
    #[error(
        "tools.bash.sandbox = \"bwrap\" but `{program}` cannot be run from the sandbox's \
         PATH: {reason}. Install util-linux, or set sandbox to \"auto\" to start with tools \
         disabled."
    )]
    UtilLinuxNotFound {
        program: &'static str,
        reason: ProgramError,
    },
    #[error(
        "tools.bash.sandbox = \"bwrap\" but sandboxed commands cannot run in a capped systemd \
         scope: {0}. Run assistd in a systemd user session, or set sandbox to \"auto\" to start \
         with tools disabled."
    )]
    ScopeUnavailable(ScopeError),
}

/// A program the bwrap sandbox needs and cannot run.
#[derive(Debug)]
enum MissingProgram {
    Bwrap(ProgramError),
    UtilLinux {
        program: &'static str,
        reason: ProgramError,
    },
    Scope(ScopeError),
}

impl From<MissingProgram> for SandboxError {
    fn from(missing: MissingProgram) -> Self {
        match missing {
            MissingProgram::Bwrap(reason) => Self::BwrapNotFound(reason),
            MissingProgram::UtilLinux { program, reason } => {
                Self::UtilLinuxNotFound { program, reason }
            }
            MissingProgram::Scope(error) => Self::ScopeUnavailable(error),
        }
    }
}

impl From<MissingProgram> for ToolsDisabled {
    fn from(missing: MissingProgram) -> Self {
        match missing {
            MissingProgram::Bwrap(reason) => Self::BwrapMissing(reason),
            MissingProgram::UtilLinux { program, reason } => {
                Self::UtilLinuxMissing { program, reason }
            }
            MissingProgram::Scope(error) => Self::ScopeUnavailable(error),
        }
    }
}

/// Resolve `request` against the environment, once per process; `Auto` without `bwrap`, like
/// `None`, disables tools. Fails with [`SandboxError::BwrapNotFound`] when `request` is `Bwrap`
/// and no `bwrap` is on `PATH`.
pub fn probe_sandbox(
    request: SandboxRequest,
    extra_args: Vec<String>,
    protected: Protected,
    shared: SharedDirs,
) -> Result<ToolSandbox, SandboxError> {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    probe_sandbox_with_path(request, extra_args, protected, shared, &path_env)
}

fn probe_sandbox_with_path(
    request: SandboxRequest,
    extra_args: Vec<String>,
    protected: Protected,
    shared: SharedDirs,
    path_env: &OsStr,
) -> Result<ToolSandbox, SandboxError> {
    let disabled = match (request, find_sandbox_programs(path_env)) {
        (SandboxRequest::Bwrap, Err(missing)) => return Err(missing.into()),
        (SandboxRequest::None, _) => ToolsDisabled::ByConfig,
        (SandboxRequest::Auto, Err(missing)) => missing.into(),
        (SandboxRequest::Auto | SandboxRequest::Bwrap, Ok(programs)) => {
            match confined_spawner(&programs.bwrap, &extra_args) {
                Ok(confined) => {
                    announce_bwrap(&programs.bwrap);
                    return Ok(ToolSandbox::Bwrap(Arc::new(SandboxInfo {
                        programs,
                        extra_args,
                        protected,
                        shared,
                        host_path: host_path(path_env),
                        unit_dirs: user_unit_dirs(path_env),
                        confined,
                        profile: Profile::Host,
                    })));
                }
                Err(disabled) => disabled,
            }
        }
    };
    warn!(target: "assistd::policy", "{disabled}");
    Ok(ToolSandbox::Disabled(disabled))
}

/// `bwrap` and `systemd-run` on `path_env`, with a scope started to check
/// the manager accepts one, and `prlimit` and `choom` on [`SANDBOX_PATH`],
/// where the sandbox runs them.
fn find_sandbox_programs(path_env: &OsStr) -> Result<SandboxPrograms, MissingProgram> {
    let bwrap = find_program(Path::new("bwrap"), path_env).map_err(MissingProgram::Bwrap)?;
    let prlimit = find_util_linux("prlimit")?;
    let choom = find_util_linux("choom")?;
    let scope = CommandScope::probe(path_env).map_err(MissingProgram::Scope)?;
    Ok(SandboxPrograms {
        bwrap,
        prlimit,
        choom,
        scope,
    })
}

fn find_util_linux(program: &'static str) -> Result<PathBuf, MissingProgram> {
    find_program(Path::new(program), OsStr::new(SANDBOX_PATH))
        .map_err(|reason| MissingProgram::UtilLinux { program, reason })
}

/// The spawner that keeps commands off abstract sockets, needed only when
/// `extra_args` shares the network namespace they belong to.
fn confined_spawner(
    bwrap: &Path,
    extra_args: &[String],
) -> Result<Option<ConfinedSpawner>, ToolsDisabled> {
    if !shares_network(extra_args) {
        return Ok(None);
    }
    let spawner = ConfinedSpawner::start().map_err(|error| {
        warn!(target: "assistd::policy", %error, "cannot confine networked sandboxed commands");
        ToolsDisabled::AbstractSocketsReachable
    })?;
    if is_setuid(bwrap) {
        warn!(
            target: "assistd::policy",
            path = %bwrap.display(),
            "bwrap is setuid, but --share-net runs it with no_new_privs (Landlock requires \
             it), so it gets no root privileges: commands fail unless unprivileged user \
             namespaces are enabled"
        );
    }
    Ok(Some(spawner))
}

fn announce_bwrap(bwrap: &Path) {
    info!(
        target: "assistd::policy",
        path = %bwrap.display(),
        "bash sandbox: bubblewrap enabled"
    );
}

/// The absolute directories `systemd-analyze` on `path_env` says user units
/// load from; empty when it cannot run, leaving the spec's paths guarded.
fn user_unit_dirs(path_env: &OsStr) -> Vec<PathBuf> {
    let output = find_program(Path::new("systemd-analyze"), path_env)
        .ok()
        .and_then(|program| {
            std::process::Command::new(program)
                .args(["--user", "unit-paths"])
                .stdin(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .output()
                .ok()
        })
        .filter(|output| output.status.success());
    let Some(output) = output else {
        info!(
            target: "assistd::policy",
            "systemd-analyze cannot list the user unit path; guarding the XDG defaults only"
        );
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute())
        .collect()
}

/// The absolute directories of `path_env`, or [`SANDBOX_PATH`]'s when it
/// has none. Relative entries resolve against whatever directory a command
/// happens to be in, so they are dropped.
pub(super) fn host_path(path_env: &OsStr) -> Vec<PathBuf> {
    let dirs: Vec<PathBuf> = std::env::split_paths(path_env)
        .filter(|dir| dir.is_absolute())
        .collect();
    if dirs.is_empty() {
        std::env::split_paths(SANDBOX_PATH).collect()
    } else {
        dirs
    }
}

/// Whether `extra_args` share the host's network namespace with commands.
fn shares_network(extra_args: &[String]) -> bool {
    extra_args.iter().any(|arg| arg == "--share-net")
}

#[cfg(unix)]
fn is_setuid(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|md| md.permissions().mode() & 0o4000 != 0)
}

#[cfg(not(unix))]
fn is_setuid(_path: &Path) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_path_keeps_only_absolute_entries() {
        assert_eq!(
            host_path(OsStr::new("/usr/bin::.:bin:/bin")),
            [PathBuf::from("/usr/bin"), PathBuf::from("/bin")]
        );
        assert_eq!(
            host_path(OsStr::new(".:")),
            std::env::split_paths(SANDBOX_PATH).collect::<Vec<_>>()
        );
    }

    #[test]
    fn probe_sandbox_disables_tools_when_disabled_or_bwrap_is_absent() {
        for (request, expected) in [
            (SandboxRequest::None, ToolsDisabled::ByConfig),
            (
                SandboxRequest::Auto,
                ToolsDisabled::BwrapMissing(ProgramError::NotOnPath),
            ),
        ] {
            let probed = probe_sandbox_with_path(
                request,
                Vec::new(),
                Protected::default(),
                SharedDirs::default(),
                OsStr::new(""),
            )
            .unwrap_or_else(|e| panic!("{request:?}: {e}"));
            assert!(
                matches!(&probed, ToolSandbox::Disabled(reason) if *reason == expected),
                "{request:?}: {probed:?}"
            );
        }
    }

    #[test]
    fn probe_sandbox_disables_tools_when_none_even_with_bwrap_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bwrap = dir.path().join("bwrap");
        std::fs::write(&bwrap, "").expect("write");
        std::fs::set_permissions(&bwrap, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let probed = probe_sandbox_with_path(
            SandboxRequest::None,
            Vec::new(),
            Protected::default(),
            SharedDirs::default(),
            dir.path().as_os_str(),
        )
        .expect("none never fails");
        assert!(matches!(
            probed,
            ToolSandbox::Disabled(ToolsDisabled::ByConfig)
        ));
    }

    #[test]
    fn probe_sandbox_bwrap_missing_fails_startup() {
        probe_sandbox_with_path(
            SandboxRequest::Bwrap,
            Vec::new(),
            Protected::default(),
            SharedDirs::default(),
            OsStr::new(""),
        )
        .expect_err("bwrap is required but absent from PATH");
    }
}

//! The bubblewrap sandbox subprocess-spawning commands run in.

use std::ffi::OsStr;
use std::io;
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::Arc;

use assistd_utils::program::ProgramError;
use tokio::process::Command as ProcCommand;

use self::command::{Layout, sandboxed_command};
use self::home::SandboxHome;
use self::mounts::Mounts;
use self::scope::CommandScope;
use super::allowlist::SearchPath;
use super::on_blocking_pool;
use super::path_class::{GuardRoots, PathGuards};

mod command;
mod confined;
mod home;
#[cfg(any(test, feature = "test-support"))]
mod inert;
mod mounts;
mod probe;
mod scope;
mod scratch_cap;
mod seccomp;

pub(crate) use confined::ConfinedSpawner;
pub(crate) use home::is_within_visible_home_entry;
#[cfg(any(test, feature = "test-support"))]
pub use inert::inert_sandbox;
pub use probe::{SandboxError, probe_sandbox};
pub use scope::ScopeError;
pub(crate) use scratch_cap::Guarded;
pub use scratch_cap::{SCRATCH_MARKER, ScratchCap};

/// The `PATH` sandboxed commands run with, and the fallback when the
/// daemon has no usable one.
const SANDBOX_PATH: &str = "/usr/local/bin:/usr/bin:/bin";

/// How sandboxing was requested for subprocess-spawning commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxRequest {
    /// Use bwrap if found on `PATH`; without it the model gets no tools.
    Auto,
    /// Require bwrap; fail startup if missing.
    Bwrap,
    /// Never sandbox, so the model gets no tools.
    None,
}

/// The sandbox the model's tools run in, as resolved by [`probe_sandbox`].
#[derive(Debug)]
pub enum ToolSandbox {
    Bwrap(Arc<SandboxInfo>),
    /// No sandbox is available, so no tool may be offered to the model.
    Disabled(ToolsDisabled),
}

/// Why the model gets no tools.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolsDisabled {
    #[error("tools are disabled: tools.bash.sandbox = \"none\" leaves them unsandboxed")]
    ByConfig,
    #[error("tools are disabled: cannot run bubblewrap (`bwrap`): {0}; install it to enable them")]
    BwrapMissing(ProgramError),
    #[error(
        "tools are disabled: cannot run `{program}` from the sandbox's PATH to cap sandboxed \
         commands: {reason}; install util-linux to enable them"
    )]
    UtilLinuxMissing {
        program: &'static str,
        reason: ProgramError,
    },
    #[error("tools are disabled: sandboxed commands cannot run in a capped systemd scope: {0}")]
    ScopeUnavailable(ScopeError),
    #[error(
        "tools are disabled: tools.bash.bwrap_extra_args shares the network, and the kernel \
         cannot keep sandboxed commands off abstract sockets such as the X server's (needs \
         Landlock scoping, Linux 6.12+); drop --share-net to enable them"
    )]
    AbstractSocketsReachable,
}

/// The programs a bwrap sandbox starts each command through.
#[derive(Debug, Clone)]
pub struct SandboxPrograms {
    pub bwrap: PathBuf,
    /// Found on the sandbox's `PATH`, since it runs inside the sandbox.
    prlimit: PathBuf,
    /// Found on the sandbox's `PATH`, since it runs inside the sandbox.
    choom: PathBuf,
    scope: CommandScope,
}

/// Resolved sandbox configuration shared by every subprocess-spawning
/// command.
#[derive(Debug)]
pub struct SandboxInfo {
    pub programs: SandboxPrograms,
    /// Extra args appended to the bwrap invocation before `--`, verbatim
    /// unless the command's [`HomeAccess`] is read-only.
    pub extra_args: Vec<String>,
    pub protected: Protected,
    pub shared: SharedDirs,
    /// Absolute directories host shells look programs up in.
    host_path: Vec<PathBuf>,
    /// Directories systemd loads user units from, as `systemd-analyze`
    /// reported them at startup.
    unit_dirs: Vec<PathBuf>,
    /// Set when `extra_args` shares the network namespace, which abstract
    /// Unix sockets such as the X server's belong to.
    confined: Option<ConfinedSpawner>,
    profile: Profile,
}

/// What of the host a bwrap sandbox mounts.
#[derive(Debug, Clone)]
enum Profile {
    /// The host's root read-only, with `$HOME`'s visible entries as each
    /// command's [`HomeAccess`] has them, and no `AF_UNIX` sockets.
    Host,
    Quarantine(QuarantineProfile),
}

/// What a host-profile command may change under `$HOME`; a quarantine's
/// home is its own scratch directory, writable either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HomeAccess {
    /// Nothing, there or in what `bwrap_extra_args` mounts, whose device
    /// binds it goes without: only `/tmp` and the scratch directory take writes.
    ReadOnly,
    /// Its visible entries that are not symlinks.
    Writable,
}

/// Only `/usr`, a few `/etc` entries, the shared directories and `inputs`,
/// with `home` as `$HOME` and the working directory, no network, and no
/// `AF_UNIX` sockets.
#[derive(Debug, Clone)]
struct QuarantineProfile {
    home: PathBuf,
    inputs: Vec<InputBind>,
}

/// A host file or directory a quarantine sees read-only at `at`, over a mount
/// point already there, bound from `fd`, its `O_PATH` descriptor.
#[derive(Debug, Clone)]
pub(crate) struct InputBind {
    pub(crate) fd: Arc<OwnedFd>,
    pub(crate) at: PathBuf,
}

/// Paths commands must not change, even inside the sandbox's writable
/// binds.
#[derive(Debug, Clone, Default)]
pub struct Protected {
    /// Directories mounted read-only: assistd's own configuration.
    pub dirs: Vec<PathBuf>,
    /// Sockets hidden from sandboxed commands, such as the daemon's IPC
    /// socket, through which a command could answer its own prompts.
    pub sockets: Vec<PathBuf>,
}

/// Host directories sandboxed commands see at their own paths, over the
/// sandbox's private mounts.
#[derive(Debug, Clone, Default)]
pub struct SharedDirs {
    /// Writable: where files cross between `bash` and the other commands.
    pub scratch: Option<PathBuf>,
    /// What `scratch` may hold; `None` leaves it to grow.
    pub scratch_cap: Option<ScratchCap>,
    /// Read-only, such as the directory truncated output is spilled to.
    pub read_only: Vec<PathBuf>,
}

impl SandboxInfo {
    /// A bwrap sandbox that sees none of the host but its programs, `home`
    /// (writable, and shared as the scratch directory), `read_only` and
    /// `inputs`, with no network and no `AF_UNIX` sockets, so no Unix socket
    /// made outside it is reachable.
    pub(crate) fn quarantined(
        &self,
        home: PathBuf,
        read_only: Vec<PathBuf>,
        inputs: Vec<InputBind>,
    ) -> Self {
        Self {
            programs: self.programs.clone(),
            extra_args: Vec::new(),
            protected: Protected::default(),
            shared: SharedDirs {
                scratch: Some(home.clone()),
                scratch_cap: None,
                read_only,
            },
            host_path: self.host_path.clone(),
            unit_dirs: self.unit_dirs.clone(),
            confined: None,
            profile: Profile::Quarantine(QuarantineProfile { home, inputs }),
        }
    }

    /// The spawner commands built by [`SandboxInfo::command`] must start
    /// from, if any.
    pub(crate) fn confined_spawner(&self) -> Option<&ConfinedSpawner> {
        self.confined.as_ref()
    }

    /// Where commands run under this configuration look up bare program
    /// names.
    pub fn search_path(&self) -> SearchPath {
        SearchPath {
            dirs: std::env::split_paths(SANDBOX_PATH).collect(),
        }
    }

    /// The path guards for commands run under this configuration: its
    /// `$HOME`, XDG directories and working directory, with the directories
    /// host shells look programs up in (the host `PATH`, `~/bin`, every
    /// visible `~/<entry>/bin`) and the user unit path sensitive. Resolves
    /// symlinks, so it blocks on the filesystem.
    pub(crate) fn path_guards(&self) -> PathGuards {
        let host_home = std::env::var("HOME").ok();
        let home_dirs = SandboxHome::read(host_home.clone())
            .map(|home| home.existing_command_dirs(&[]))
            .unwrap_or_default();
        let command_dirs = self
            .host_path
            .iter()
            .filter_map(|dir| std::fs::canonicalize(dir).ok())
            .chain(home_dirs)
            .collect();
        let exempt = self
            .shared
            .scratch
            .iter()
            .chain(&self.shared.read_only)
            .cloned()
            .collect();
        let roots = GuardRoots {
            command_dirs,
            unit_dirs: self.unit_dirs.clone(),
            exempt,
            ..GuardRoots::default()
        };
        PathGuards::read(match &self.profile {
            Profile::Quarantine(quarantine) => GuardRoots {
                home: Some(quarantine.home.clone()),
                cwd: Some(quarantine.home.clone()),
                ..roots
            },
            Profile::Host => GuardRoots {
                cwd: Layout::host(self, host_home.clone(), HomeAccess::Writable)
                    .mounts
                    .working_dir,
                home: host_home.map(PathBuf::from),
                config_home: std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from),
                data_home: std::env::var_os("XDG_DATA_HOME").map(PathBuf::from),
                ..roots
            },
        })
    }

    /// The recovery clause of a note about a private directory: where a
    /// file both `bash` and the other commands need should go instead.
    pub(crate) fn sharing_hint(&self) -> String {
        match &self.shared.scratch {
            Some(scratch) => format!(
                "a path under {}, which bash scripts and the other commands share",
                scratch.display()
            ),
            None => "a pipe, as in `cat FILE | bash \"<script>\"`".to_string(),
        }
    }

    /// The tmpfs mount that hides the host's `path` from sandboxed
    /// commands, if any.
    pub async fn private_dir_hiding(self: &Arc<Self>, path: PathBuf) -> Option<PathBuf> {
        let info = Arc::clone(self);
        on_blocking_pool(move || {
            let mounts = info.layout(HomeAccess::Writable).mounts;
            mounts.private_dir_of(&path).cloned()
        })
        .await
    }

    /// `program` with `args` set up to run in bubblewrap, with `home` as its access to `$HOME`.
    /// Fails when the seccomp filter cannot be written, or a quarantine input's descriptor
    /// cannot be duplicated for the command.
    pub async fn command(
        self: &Arc<Self>,
        program: &str,
        args: Vec<String>,
        home: HomeAccess,
    ) -> io::Result<SandboxedCommand> {
        let info = Arc::clone(self);
        let program = program.to_owned();
        on_blocking_pool(move || info.build_command(&program, args, home)).await
    }

    /// Blocks on the filesystem, which the layout is read from.
    fn build_command<I, S>(
        &self,
        program: &str,
        args: I,
        access: HomeAccess,
    ) -> io::Result<SandboxedCommand>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let layout = self.layout(access);
        let command = sandboxed_command(&self.programs, &layout, program, args)?;
        Ok(SandboxedCommand {
            command,
            mounts: layout.mounts,
        })
    }

    /// What a command with `access` to `$HOME` gets mounted. Reads `$HOME` and the command
    /// directories, so it blocks on the filesystem.
    fn layout(&self, access: HomeAccess) -> Layout<'_> {
        match &self.profile {
            Profile::Host => Layout::host(self, std::env::var("HOME").ok(), access),
            Profile::Quarantine(quarantine) => Layout::quarantine(self, quarantine),
        }
    }
}

/// A command built by [`SandboxInfo::command`], with what its sandbox mounts.
#[derive(Debug)]
pub struct SandboxedCommand {
    pub command: ProcCommand,
    mounts: Mounts,
}

impl SandboxedCommand {
    /// The tmpfs mounts `script` names a path in: empty directories of the command's own,
    /// sharing nothing with other commands or later calls.
    pub fn private_dirs_named(&self, script: &str) -> Vec<PathBuf> {
        self.mounts.private_dirs_named(script)
    }
}

//! The bubblewrap sandbox subprocess-spawning commands run in.

use std::ffi::{OsStr, OsString};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::process::{Child, Command as ProcCommand};
use tracing::{info, warn};

use super::allowlist::SearchPath;

mod session;

pub use session::LaunchError;
use session::{SharedDisplay, spawn_without_abstract_sockets};

/// The `PATH` sandboxed commands run with, and the fallback when the
/// daemon has no usable one.
const SANDBOX_PATH: &str = "/usr/local/bin:/usr/bin:/bin";

/// Variables sandboxed commands inherit, besides `LC_*`; the rest, such as
/// agent sockets and tokens, are cleared.
const KEPT_ENV: &[&str] = &["LANG", "LANGUAGE", "LOGNAME", "TERM", "TZ", "USER"];

/// Variables [`SandboxAccess::Session`] also keeps, locating the display.
const SESSION_ENV: &[&str] = &[
    "WAYLAND_DISPLAY",
    "XDG_CURRENT_DESKTOP",
    "XDG_RUNTIME_DIR",
    "XDG_SESSION_TYPE",
];

/// How sandboxing was requested for subprocess-spawning commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxRequest {
    /// Use bwrap if found on `PATH`; fall back to unsandboxed with a warn.
    Auto,
    /// Require bwrap; fail startup if missing.
    Bwrap,
    /// Never wrap.
    None,
}

/// How subprocesses are wrapped, as resolved once by [`probe_sandbox`].
#[derive(Debug, Clone)]
pub enum ResolvedSandboxMode {
    None,
    /// `bwrap` was found at this absolute path.
    Bwrap {
        path: PathBuf,
    },
}

/// Resolved sandbox configuration shared by every subprocess-spawning
/// command.
#[derive(Debug)]
pub struct SandboxInfo {
    pub mode: ResolvedSandboxMode,
    /// Extra args appended verbatim to the bwrap invocation before `--`.
    pub extra_args: Vec<String>,
    pub protected: Protected,
    /// Absolute directories unsandboxed commands look programs up in.
    host_path: Vec<PathBuf>,
    display: SharedDisplay,
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

/// Session resources a sandboxed command needs beyond the default profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxAccess<'a> {
    /// No network, and nothing of the session: `/tmp` and `/run` are fresh
    /// tmpfs mounts.
    Default,
    /// Shares the network and binds `restricted`, a socket over which the
    /// compositor withholds privileged protocols, at `display`, where
    /// clients look for the compositor.
    Session {
        restricted: &'a Path,
        display: &'a Path,
    },
}

impl SandboxInfo {
    /// A configuration that never wraps.
    pub fn none() -> Arc<Self> {
        Arc::new(Self {
            mode: ResolvedSandboxMode::None,
            extra_args: Vec::new(),
            protected: Protected::default(),
            host_path: host_path(&std::env::var_os("PATH").unwrap_or_default()),
            display: SharedDisplay::default(),
        })
    }

    /// Where commands run under this configuration look up bare program
    /// names.
    pub fn search_path(&self) -> SearchPath {
        match self.mode {
            ResolvedSandboxMode::None => SearchPath {
                dirs: self.host_path.clone(),
                read_only: false,
            },
            ResolvedSandboxMode::Bwrap { .. } => SearchPath {
                dirs: std::env::split_paths(SANDBOX_PATH).collect(),
                read_only: true,
            },
        }
    }

    /// The [`ProcCommand`] that runs `program` with `args`, wrapped in
    /// bubblewrap when the mode is [`ResolvedSandboxMode::Bwrap`].
    /// Unwrapped, `PATH` is limited to the directories
    /// [`SandboxInfo::search_path`] reports.
    pub fn command<I, S>(&self, access: SandboxAccess<'_>, program: &str, args: I) -> ProcCommand
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        match &self.mode {
            ResolvedSandboxMode::None => self.unwrapped_command(program, args),
            ResolvedSandboxMode::Bwrap { path } => self.bwrap_command(path, access, program, args),
        }
    }

    /// Spawn `program` as a graphical application once `configure` has set
    /// up its command. Under bwrap it reaches the compositor only through a
    /// restricted socket and no abstract Unix socket, and is refused when
    /// either protection is unavailable.
    pub(crate) async fn spawn_graphical(
        &self,
        program: &str,
        args: &[String],
        configure: fn(&mut ProcCommand),
    ) -> Result<Child, LaunchError> {
        let ResolvedSandboxMode::Bwrap { path } = &self.mode else {
            let mut cmd = self.unwrapped_command(program, args);
            configure(&mut cmd);
            return cmd.spawn().map_err(LaunchError::Spawn);
        };
        let display = self.display.get().await?;
        let mut cmd = self.bwrap_command(path, display.access(), program, args);
        configure(&mut cmd);
        spawn_without_abstract_sockets(cmd)
    }

    fn unwrapped_command<I, S>(&self, program: &str, args: I) -> ProcCommand
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut cmd = ProcCommand::new(program);
        cmd.args(args);
        if let Ok(path) = std::env::join_paths(&self.host_path) {
            cmd.env("PATH", path);
        }
        cmd
    }

    /// Flag order is load-bearing: the profile, then the `access` binds
    /// (after `--tmpfs /run`), then protections, then the operator's
    /// `extra_args` so they win.
    fn bwrap_command<I, S>(
        &self,
        bwrap: &Path,
        access: SandboxAccess<'_>,
        program: &str,
        args: I,
    ) -> ProcCommand
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let home = std::env::var("HOME").ok();
        let mut cmd = ProcCommand::new(bwrap);
        cmd.env_clear().envs(kept_env(std::env::vars_os(), access));
        cmd.args(
            default_bwrap_flags_for(home.clone(), access)
                .iter()
                .map(OsStr::new),
        );
        if let SandboxAccess::Session {
            restricted,
            display,
        } = access
        {
            cmd.arg("--ro-bind").arg(restricted).arg(display);
        }
        let writable: Vec<PathBuf> = home
            .filter(|h| is_bindable_home(h))
            .map(PathBuf::from)
            .into_iter()
            .collect();
        cmd.args(protection_flags(&self.protected, &writable));
        cmd.args(self.extra_args.iter().map(OsStr::new));
        cmd.arg("--");
        cmd.arg(program).args(args);
        cmd
    }
}

/// Why [`probe_sandbox`] could not satisfy the requested sandbox.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error(
        "tools.bash.sandbox = \"bwrap\" but `bwrap` was not found on PATH. \
         Install bubblewrap or change sandbox to \"auto\" / \"none\"."
    )]
    BwrapNotFound,
}

/// Read-only binds over the protected directories, and `/dev/null` over the
/// protected sockets, for those that exist under a `writable` bind; binding
/// anything else would make `bwrap` create mount points or fail.
fn protection_flags(protected: &Protected, writable: &[PathBuf]) -> Vec<PathBuf> {
    let visible = |path: &Path| writable.iter().any(|root| path.starts_with(root)) && path.exists();
    let dirs = protected
        .dirs
        .iter()
        .filter(|dir| visible(dir))
        .flat_map(|dir| [PathBuf::from("--ro-bind"), dir.clone(), dir.clone()]);
    let sockets = protected
        .sockets
        .iter()
        .filter(|socket| visible(socket))
        .flat_map(|socket| {
            [
                PathBuf::from("--ro-bind"),
                PathBuf::from("/dev/null"),
                socket.clone(),
            ]
        });
    dirs.chain(sockets).collect()
}

/// Read-only root with (when bindable) the visible entries of `$HOME`
/// writable, fresh `/tmp`, `/dev`, `/proc` and `/run`, isolated pid/ipc/uts
/// namespaces, and for [`SandboxAccess::Default`] no network; the operator
/// restores it with `--share-net` in `bwrap_extra_args`.
fn default_bwrap_flags_for(home: Option<String>, access: SandboxAccess<'_>) -> Vec<String> {
    let mut flags: Vec<String> = ["--ro-bind", "/", "/", "--tmpfs", "/tmp"]
        .map(String::from)
        .into();
    let home = home.filter(|h| is_bindable_home(h));
    match &home {
        Some(home) => flags.extend(home_bind_flags(home)),
        None => warn!(
            target: "assistd::policy",
            "HOME is unset or not a non-root absolute directory; sandboxed commands \
             will have no writable home"
        ),
    }
    flags.extend(
        [
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/run",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
        ]
        .map(String::from),
    );
    if access == SandboxAccess::Default {
        flags.push("--unshare-net".into());
    }
    flags.extend(["--new-session", "--die-with-parent"].map(String::from));
    if let Some(home) = home {
        flags.extend(["--setenv".into(), "HOME".into(), home]);
    }
    flags.extend(["--setenv".into(), "PATH".into(), SANDBOX_PATH.into()]);
    flags
}

fn home_bind_flags(home: &str) -> Vec<String> {
    let mut flags: Vec<String> = vec!["--ro-bind".into(), home.into(), home.into()];
    for entry in writable_home_entries(Path::new(home)) {
        flags.extend(["--bind".into(), entry.clone(), entry]);
    }
    flags
}

fn writable_home_entries(home: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(home) else {
        return Vec::new();
    };
    let mut writable: Vec<String> = entries
        .filter_map(Result::ok)
        .filter(|entry| !entry.file_name().as_encoded_bytes().starts_with(b"."))
        .filter(|entry| entry.file_type().is_ok_and(|kind| !kind.is_symlink()))
        .filter_map(|entry| entry.path().into_os_string().into_string().ok())
        .collect();
    writable.sort();
    writable
}

fn is_bindable_home(home: &str) -> bool {
    let path = Path::new(home);
    path.is_absolute()
        && std::fs::canonicalize(path).is_ok_and(|real| real.is_dir() && real.parent().is_some())
}

/// The `access` subset of `vars`: [`KEPT_ENV`], `LC_*`, and for
/// [`SandboxAccess::Session`] also [`SESSION_ENV`].
fn kept_env(
    vars: impl IntoIterator<Item = (OsString, OsString)>,
    access: SandboxAccess<'_>,
) -> Vec<(OsString, OsString)> {
    let session = matches!(access, SandboxAccess::Session { .. });
    let kept = |name: &str| {
        KEPT_ENV.contains(&name)
            || name.starts_with("LC_")
            || (session && SESSION_ENV.contains(&name))
    };
    vars.into_iter()
        .filter(|(name, _)| name.to_str().is_some_and(kept))
        .collect()
}

/// Resolve `request` against the environment, once per process. `Auto`
/// falls back to unsandboxed with a warning when `bwrap` is missing.
///
/// # Errors
/// [`SandboxError::BwrapNotFound`] when `request` is `Bwrap` and no `bwrap`
/// is on `PATH`.
pub fn probe_sandbox(
    request: SandboxRequest,
    extra_args: Vec<String>,
    protected: Protected,
) -> Result<Arc<SandboxInfo>, SandboxError> {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    probe_sandbox_with_path(request, extra_args, protected, &path_env)
}

fn probe_sandbox_with_path(
    request: SandboxRequest,
    extra_args: Vec<String>,
    protected: Protected,
    path_env: &OsStr,
) -> Result<Arc<SandboxInfo>, SandboxError> {
    Ok(Arc::new(SandboxInfo {
        mode: resolve_mode(request, path_env)?,
        extra_args,
        protected,
        host_path: host_path(path_env),
        display: SharedDisplay::default(),
    }))
}

fn resolve_mode(
    request: SandboxRequest,
    path_env: &OsStr,
) -> Result<ResolvedSandboxMode, SandboxError> {
    let mode = match request {
        SandboxRequest::None => {
            info!(target: "assistd::policy", "bash sandbox: disabled by config");
            ResolvedSandboxMode::None
        }
        SandboxRequest::Auto => match find_executable("bwrap", path_env) {
            Some(path) => {
                info!(
                    target: "assistd::policy",
                    path = %path.display(),
                    "bash sandbox: bubblewrap enabled (auto-detected)"
                );
                ResolvedSandboxMode::Bwrap { path }
            }
            None => {
                warn!(
                    target: "assistd::policy",
                    "bubblewrap not found on PATH; bash commands will run unsandboxed under the current user. \
                     Install bubblewrap (package: bubblewrap) for defence-in-depth."
                );
                ResolvedSandboxMode::None
            }
        },
        SandboxRequest::Bwrap => match find_executable("bwrap", path_env) {
            Some(path) => {
                info!(
                    target: "assistd::policy",
                    path = %path.display(),
                    "bash sandbox: bubblewrap enabled (required by config)"
                );
                ResolvedSandboxMode::Bwrap { path }
            }
            None => return Err(SandboxError::BwrapNotFound),
        },
    };
    if let ResolvedSandboxMode::Bwrap { path } = &mode
        && is_setuid(path)
    {
        warn!(
            target: "assistd::policy",
            path = %path.display(),
            "bwrap is setuid, but `wm open` runs it with no_new_privs (Landlock requires it), \
             so it gets no root privileges: launches fail unless unprivileged user namespaces \
             are enabled"
        );
    }
    Ok(mode)
}

/// The absolute directories of `path_env`, or [`SANDBOX_PATH`]'s when it
/// has none. Relative entries resolve against whatever directory a command
/// happens to be in, so they are dropped.
fn host_path(path_env: &OsStr) -> Vec<PathBuf> {
    let dirs: Vec<PathBuf> = std::env::split_paths(path_env)
        .filter(|dir| dir.is_absolute())
        .collect();
    if dirs.is_empty() {
        std::env::split_paths(SANDBOX_PATH).collect()
    } else {
        dirs
    }
}

fn find_executable(name: &str, path_env: &OsStr) -> Option<PathBuf> {
    std::env::split_paths(path_env)
        .filter(|dir| !dir.as_os_str().is_empty())
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable_file(candidate))
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|md| md.is_file() && md.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
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

    fn session() -> SandboxAccess<'static> {
        SandboxAccess::Session {
            restricted: Path::new("/run/user/1000/assistd-wayland.sock"),
            display: Path::new("/run/user/1000/wayland-1"),
        }
    }

    fn bwrap_info(extra_args: Vec<String>) -> SandboxInfo {
        SandboxInfo {
            mode: ResolvedSandboxMode::Bwrap {
                path: PathBuf::from("/usr/bin/bwrap"),
            },
            extra_args,
            protected: Protected::default(),
            host_path: Vec::new(),
            display: SharedDisplay::default(),
        }
    }

    fn argv_of(cmd: &ProcCommand) -> Vec<String> {
        cmd.as_std()
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    fn writable_binds(flags: &[String]) -> Vec<&str> {
        flags
            .windows(2)
            .filter(|w| w[0] == "--bind")
            .map(|w| w[1].as_str())
            .collect()
    }

    #[test]
    fn unsandboxed_command_passes_argv_through_verbatim() {
        let cmd =
            SandboxInfo::none().command(SandboxAccess::Default, "firefox", ["--new-window", "a b"]);
        assert_eq!(cmd.as_std().get_program(), "firefox");
        assert_eq!(argv_of(&cmd), ["--new-window", "a b"]);
    }

    #[test]
    fn bwrap_argv_is_profile_then_access_binds_then_extra_args_then_program() {
        let info = bwrap_info(vec!["--share-net".into()]);
        let tail = ["--share-net", "--", "firefox", "--new-window"].map(String::from);
        let session_binds = [
            "--ro-bind",
            "/run/user/1000/assistd-wayland.sock",
            "/run/user/1000/wayland-1",
        ]
        .map(String::from);
        for (access, binds) in [
            (SandboxAccess::Default, Vec::new()),
            (session(), session_binds.to_vec()),
        ] {
            let profile = default_bwrap_flags_for(std::env::var("HOME").ok(), access);
            assert!(profile.windows(2).any(|w| w == ["--tmpfs", "/run"]));
            let cmd = info.command(access, "firefox", ["--new-window"]);
            assert_eq!(cmd.as_std().get_program(), "/usr/bin/bwrap");
            let expected = [profile, binds, tail.to_vec()].concat();
            assert_eq!(argv_of(&cmd), expected, "{access:?}");
        }
    }

    #[test]
    fn only_the_default_access_loses_the_network_and_both_get_a_private_tmp() {
        for (access, unshares_net) in [(SandboxAccess::Default, true), (session(), false)] {
            let flags = default_bwrap_flags_for(None, access);
            assert_eq!(
                flags.iter().any(|f| f == "--unshare-net"),
                unshares_net,
                "{access:?}"
            );
            assert!(
                flags.windows(2).any(|w| w == ["--tmpfs", "/tmp"]),
                "{access:?}"
            );
        }
    }

    #[test]
    fn environment_keeps_only_locale_terminal_and_wayland_variables() {
        let vars = [
            ("LANG", "C.UTF-8"),
            ("LC_TIME", "en_GB.UTF-8"),
            ("SSH_AUTH_SOCK", "/tmp/ssh-x/agent.1"),
            ("GITHUB_TOKEN", "secret"),
            ("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus"),
            ("SWAYSOCK", "/run/user/1000/sway-ipc.sock"),
            ("WAYLAND_DISPLAY", "wayland-1"),
            ("DISPLAY", ":0"),
            ("XAUTHORITY", "/run/user/1000/xauth"),
        ]
        .map(|(name, value)| (OsString::from(name), OsString::from(value)));
        let names = |access| {
            kept_env(vars.clone(), access)
                .into_iter()
                .map(|(name, _)| name.into_string().expect("utf-8"))
                .collect::<Vec<_>>()
        };
        assert_eq!(names(SandboxAccess::Default), ["LANG", "LC_TIME"]);
        assert_eq!(names(session()), ["LANG", "LC_TIME", "WAYLAND_DISPLAY"]);
    }

    #[test]
    fn bwrap_process_starts_from_a_cleared_environment() {
        let info = bwrap_info(Vec::new());
        let cmd = info.command(SandboxAccess::Default, "true", Vec::<String>::new());
        let expected: Vec<_> = kept_env(std::env::vars_os(), SandboxAccess::Default)
            .into_iter()
            .map(|(name, value)| (name, Some(value)))
            .collect();
        let envs: Vec<_> = cmd
            .as_std()
            .get_envs()
            .map(|(name, value)| (name.to_os_string(), value.map(OsStr::to_os_string)))
            .collect();
        assert_eq!(envs, expected);
    }

    #[test]
    fn protections_cover_existing_paths_under_writable_binds() {
        let scratch = tempfile::tempdir().expect("tempdir");
        let config = scratch.path().join("config");
        std::fs::create_dir(&config).expect("config dir");
        let socket = scratch.path().join("assistd.sock");
        std::fs::write(&socket, b"").expect("socket stand-in");
        let protected = Protected {
            dirs: vec![config.clone(), scratch.path().join("missing")],
            sockets: vec![socket.clone(), PathBuf::from("/nonexistent/assistd.sock")],
        };
        let writable = [scratch.path().to_path_buf()];
        assert_eq!(
            protection_flags(&protected, &writable),
            [
                PathBuf::from("--ro-bind"),
                config.clone(),
                config,
                PathBuf::from("--ro-bind"),
                PathBuf::from("/dev/null"),
                socket,
            ]
        );
        assert!(protection_flags(&protected, &[PathBuf::from("/elsewhere")]).is_empty());
    }

    #[test]
    fn unsandboxed_path_keeps_only_absolute_entries() {
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
    fn home_is_read_only_with_visible_entries_writable() {
        let home = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(home.path().join(".config")).expect(".config");
        std::fs::write(home.path().join(".bashrc"), b"").expect(".bashrc");
        std::fs::create_dir(home.path().join("docs")).expect("docs");
        std::fs::write(home.path().join("notes.txt"), b"").expect("notes.txt");
        std::os::unix::fs::symlink(home.path().join(".config"), home.path().join("config"))
            .expect("symlink");
        let dir_str = home.path().to_string_lossy().into_owned();
        let entry = |name: &str| home.path().join(name).to_string_lossy().into_owned();

        let flags = default_bwrap_flags_for(Some(dir_str.clone()), SandboxAccess::Default);
        assert!(
            flags
                .windows(3)
                .any(|w| w == ["--ro-bind", dir_str.as_str(), dir_str.as_str()])
        );
        assert_eq!(
            writable_binds(&flags),
            [entry("docs").as_str(), entry("notes.txt").as_str()]
        );
        let setenv = flags
            .windows(3)
            .find(|w| w[0] == "--setenv" && w[1] == "HOME")
            .expect("HOME is exported into the sandbox");
        assert_eq!(setenv[2], dir_str);
    }

    #[test]
    fn unusable_home_never_binds_the_root_writable() {
        for home in [
            None,
            Some(""),
            Some("/"),
            Some("//"),
            Some("/usr/.."),
            Some("relative"),
        ] {
            let flags = default_bwrap_flags_for(home.map(str::to_string), SandboxAccess::Default);
            assert!(
                writable_binds(&flags).is_empty(),
                "HOME={home:?} must leave nothing of the host writable: {flags:?}"
            );
            assert!(
                !flags
                    .windows(2)
                    .any(|w| w[0] == "--setenv" && w[1] == "HOME"),
                "HOME={home:?} must not be exported: {flags:?}"
            );
        }
    }

    #[test]
    fn probe_sandbox_runs_unwrapped_when_disabled_or_bwrap_is_absent() {
        for request in [SandboxRequest::None, SandboxRequest::Auto] {
            let info =
                probe_sandbox_with_path(request, Vec::new(), Protected::default(), OsStr::new(""))
                    .unwrap_or_else(|e| panic!("{request:?}: {e}"));
            assert!(
                matches!(info.mode, ResolvedSandboxMode::None),
                "{request:?}"
            );
        }
    }

    #[test]
    fn setuid_is_read_from_the_mode_bits() {
        let file = tempfile::NamedTempFile::new().expect("tempfile");
        let set_mode = |mode| {
            std::fs::set_permissions(file.path(), std::fs::Permissions::from_mode(mode))
                .expect("chmod");
        };
        set_mode(0o755);
        assert!(!is_setuid(file.path()));
        set_mode(0o4755);
        assert!(is_setuid(file.path()));
    }

    #[test]
    fn probe_sandbox_bwrap_missing_fails_startup() {
        probe_sandbox_with_path(
            SandboxRequest::Bwrap,
            Vec::new(),
            Protected::default(),
            OsStr::new(""),
        )
        .expect_err("bwrap is required but absent from PATH");
    }
}

//! The sandbox of a long-lived server process: the host's programs, a home of its
//! own, the host paths it was granted, and none of the daemon's own files.

use std::collections::{BTreeSet, HashMap};
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::DirBuilder;
use std::io;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use assistd_utils::program::{ProgramError, find_program};
use command_fds::CommandFdExt;
use tokio::process::{Child, Command as ProcCommand};
use tracing::warn;

use super::super::on_blocking_pool;
use super::super::watched::{LockedPaths, WatchedPaths};
use super::command::{
    QUARANTINE_SYSTEM_LINKS, load_seccomp_filter, run_limited, scoped_bwrap, system_flags,
};
use super::confined::ConfinedSpawner;
use super::scope::SCOPE_ENV;
use super::{SANDBOX_PATH, SandboxInfo, SandboxPrograms};

/// The `/etc` entries a server sees besides the quarantine's: host names, the
/// name-service switch, and the accounts its own user is looked up in.
const SERVER_ETC: [&str; 4] = [
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/passwd",
    "/etc/group",
];

/// Why a server cannot be started in its sandbox.
#[derive(Debug, thiserror::Error)]
pub enum ServerSandboxError {
    #[error("cannot run `{}`: {reason}", command.display())]
    Command {
        command: PathBuf,
        reason: ProgramError,
    },
    #[error(
        "`{}` is outside the server's sandbox, which shows only /usr and what the server's \
         `read` and `write` grant; add the directory that holds it and what it loads to `read`",
        program.display()
    )]
    CommandHidden { program: PathBuf },
    #[error("cannot grant `{}`: {source}", path.display())]
    Grant {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "cannot grant `{}`: it is inside `{}`, which holds assistd's own files",
        path.display(),
        own.display()
    )]
    GrantsOwnFiles { path: PathBuf, own: PathBuf },
    #[error("cannot create the server's home {}: {source}", home.display())]
    Home {
        home: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "the kernel cannot keep a server with network access off abstract Unix sockets \
         (needs Landlock scoping, Linux 6.12+); set `network = false` for the server"
    )]
    AbstractSocketsReachable,
    #[error("cannot start the server's sandbox: {0}")]
    Spawn(#[source] io::Error),
}

impl ServerSandboxError {
    /// Whether starting the same server again would fail the same way.
    pub fn is_permanent(&self) -> bool {
        !matches!(self, Self::Spawn(_))
    }
}

/// One server: what it runs and what of the host it may reach. `Debug` gives
/// the arg count and env var names, never their values.
#[derive(Clone)]
pub struct SandboxedServer {
    /// A bare name is looked up on the daemon's `PATH`.
    pub command: PathBuf,
    pub args: Vec<String>,
    /// Set on top of the locale and terminal variables every server inherits.
    pub env: HashMap<String, String>,
    /// The server's `$HOME` and working directory, writable and kept between
    /// launches; created owner-only when missing.
    pub home: PathBuf,
    /// Host paths bound read-only at their own paths.
    pub read: Vec<PathBuf>,
    /// Host paths bound writable at their own paths.
    pub write: Vec<PathBuf>,
    /// Whether the server shares the host's network.
    pub network: bool,
}

impl fmt::Debug for SandboxedServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SandboxedServer")
            .field("command", &self.command)
            .field("arg_count", &self.args.len())
            .field("env_names", &self.env.keys().collect::<BTreeSet<_>>())
            .field("home", &self.home)
            .field("read", &self.read)
            .field("write", &self.write)
            .field("network", &self.network)
            .finish()
    }
}

/// A started server and the watched paths its sandbox keeps it from changing.
#[derive(Debug)]
pub struct ServerProcess {
    pub child: Child,
    pub locks: ServerLocks,
}

/// The watched paths bound read-only in one started server's sandbox. The default
/// is those of a server that can write nowhere, which never go out of date.
#[derive(Debug, Default)]
pub struct ServerLocks {
    watched: Arc<WatchedPaths>,
    /// The server's home and writable grants, as they were bound.
    writable: Vec<Grant>,
}

impl ServerLocks {
    /// Resolves once tasks watch other paths the server could write than the ones
    /// its sandbox binds read-only, so only a new sandbox follows the rule.
    pub async fn outdated(&self) {
        let mut changes = self.watched.changes();
        loop {
            let (writable, locked) = (self.writable.clone(), self.watched.locked_for(None));
            let is_outdated = on_blocking_pool(move || {
                writable
                    .iter()
                    .any(|grant| grant.locked != locked_in(&grant.source, &locked))
            })
            .await;
            if is_outdated {
                return;
            }
            if changes.changed().await.is_err() {
                return std::future::pending().await;
            }
        }
    }
}

/// A host path a server was granted, bound from `source` at `at`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Grant {
    /// The path as configured, where the server sees it.
    at: PathBuf,
    /// `at` with symlinks resolved.
    source: PathBuf,
    writable: bool,
    /// What tasks watch of a writable grant, bound read-only: `source` itself, or the
    /// watched paths that exist inside it. Sorted.
    locked: Vec<PathBuf>,
}

impl Grant {
    fn new(at: &Path, source: PathBuf, writable: bool, locked: &LockedPaths) -> Self {
        let locked = if writable {
            locked_in(&source, locked)
        } else {
            Vec::new()
        };
        Self {
            at: at.to_path_buf(),
            source,
            writable,
            locked,
        }
    }

    /// The bind of the grant, then a read-only bind over each locked path in it.
    fn bind_flags(&self) -> Vec<OsString> {
        let bind = if self.writable { "--bind" } else { "--ro-bind" };
        let mut flags = vec![
            bind.into(),
            self.source.clone().into(),
            self.at.clone().into(),
        ];
        for lock in &self.locked {
            let within = lock.strip_prefix(&self.source).unwrap_or(Path::new(""));
            let at: PathBuf = self.at.iter().chain(within.iter()).collect();
            flags.extend(["--ro-bind".into(), lock.clone().into(), at.into()]);
        }
        flags
    }

    /// Whether the server could create the missing host `path` inside the grant.
    fn can_create(&self, path: &Path) -> bool {
        self.writable && !self.locked.iter().any(|lock| path.starts_with(lock))
    }
}

/// A bwrap sandbox showing a server none of the host but its programs, its own
/// home and the paths it was granted, with no `AF_UNIX` sockets and the network
/// only when granted. A path a task watches is read-only in it. Servers run under
/// the tool sandbox's scope and limits.
#[derive(Debug)]
pub struct ServerSandbox {
    programs: SandboxPrograms,
    watched: Arc<WatchedPaths>,
    /// The daemon's files, symlinks resolved, which no grant may show.
    own_files: Vec<PathBuf>,
    /// The daemon's `PATH`, which servers look their own programs up on.
    path_env: OsString,
    /// Starts the servers that share the network; `None` on a kernel without
    /// Landlock scoping.
    confined: Option<ConfinedSpawner>,
}

impl ServerSandbox {
    /// A server sandbox started as `sandbox` starts its commands, hiding each of
    /// `own_files` from every server.
    pub fn over(sandbox: &SandboxInfo, own_files: &[PathBuf]) -> Self {
        let confined = ConfinedSpawner::start()
            .inspect_err(|error| {
                warn!(
                    target: "assistd::policy",
                    %error,
                    "servers with network access cannot be confined, so none will start"
                );
            })
            .ok();
        Self {
            programs: sandbox.programs.clone(),
            watched: Arc::clone(&sandbox.watched),
            own_files: own_files.iter().map(|path| resolved(path)).collect(),
            path_env: std::env::var_os("PATH").unwrap_or_else(|| SANDBOX_PATH.into()),
            confined,
        }
    }

    /// Start `server` in this sandbox with its stdin, stdout and stderr piped,
    /// leading its own process group and killed when dropped.
    pub async fn spawn(
        self: &Arc<Self>,
        server: &Arc<SandboxedServer>,
    ) -> Result<ServerProcess, ServerSandboxError> {
        let confined = match (server.network, &self.confined) {
            (true, None) => return Err(ServerSandboxError::AbstractSocketsReachable),
            (true, Some(confined)) => Some(confined),
            (false, _) => None,
        };
        let (sandbox, launched) = (Arc::clone(self), Arc::clone(server));
        let (mut cmd, locks) = on_blocking_pool(move || sandbox.command(&launched)).await?;
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .process_group(0);
        let child = match confined {
            Some(confined) => confined.spawn(cmd).await,
            None => cmd.spawn(),
        }
        .map_err(ServerSandboxError::Spawn)?;
        Ok(ServerProcess { child, locks })
    }

    /// The command that runs `server`, and the watched paths it binds read-only. Flag
    /// order is load-bearing: grants, parents first, each followed by its locked paths,
    /// then the mounts hiding the daemon's files in them, then the home. Creates the
    /// home; blocks on the filesystem.
    fn command(
        &self,
        server: &SandboxedServer,
    ) -> Result<(ProcCommand, ServerLocks), ServerSandboxError> {
        let locked = self.watched.locked_for(None);
        let grants = self.grants(server, &locked)?;
        let program = visible_program(server, &grants, &self.path_env)?;
        let home = created_home(&server.home, &locked)?;
        let mut cmd = scoped_bwrap(&self.programs);
        cmd.envs(std::env::var_os("SHELL").map(|shell| ("SHELL", shell)))
            .envs(
                server
                    .env
                    .iter()
                    .filter(|(name, _)| !SCOPE_ENV.contains(&name.as_str())),
            );
        cmd.args(system_flags());
        for entry in SERVER_ETC {
            cmd.args(["--ro-bind-try", entry, entry]);
        }
        for grant in &grants {
            cmd.args(grant.bind_flags());
        }
        cmd.args(self.hiding_flags(&grants));
        cmd.args(home.bind_flags());
        cmd.args(["--setenv", "HOME"]).arg(&server.home);
        cmd.arg("--chdir").arg(&server.home);
        let path_env = server.env.get("PATH").map_or(&*self.path_env, OsStr::new);
        cmd.args(["--setenv", "PATH"]).arg(path_env);
        if server.network {
            cmd.arg("--share-net");
        }
        let filter = load_seccomp_filter(&mut cmd).map_err(ServerSandboxError::Spawn)?;
        cmd.preserved_fds(vec![filter]);
        run_limited(&mut cmd, &self.programs, program, &server.args);
        let locks = ServerLocks {
            watched: Arc::clone(&self.watched),
            writable: grants
                .into_iter()
                .filter(|grant| grant.writable)
                .chain([home])
                .collect(),
        };
        Ok((cmd, locks))
    }

    /// `server`'s grants with symlinks resolved, parents before what they hold, the
    /// writable ones read-only where `locked`. Fails for one that does not exist or
    /// lies inside the daemon's own files.
    fn grants(
        &self,
        server: &SandboxedServer,
        locked: &LockedPaths,
    ) -> Result<Vec<Grant>, ServerSandboxError> {
        let read = server.read.iter().map(|path| (path, false));
        let write = server.write.iter().map(|path| (path, true));
        let mut grants = read
            .chain(write)
            .map(|(path, writable)| self.grant(path, writable, locked))
            .collect::<Result<Vec<_>, _>>()?;
        grants.sort();
        Ok(grants)
    }

    fn grant(
        &self,
        path: &Path,
        writable: bool,
        locked: &LockedPaths,
    ) -> Result<Grant, ServerSandboxError> {
        let source = std::fs::canonicalize(path).map_err(|source| ServerSandboxError::Grant {
            path: path.to_path_buf(),
            source,
        })?;
        match self.own_files.iter().find(|own| source.starts_with(own)) {
            Some(own) => Err(ServerSandboxError::GrantsOwnFiles {
                path: path.to_path_buf(),
                own: own.clone(),
            }),
            None => Ok(Grant::new(path, source, writable, locked)),
        }
    }

    /// The mounts that cover each of the daemon's files a grant holds: an empty
    /// directory over a directory, or over a missing path the server could create
    /// in a writable grant, and `/dev/null` over anything else.
    fn hiding_flags(&self, grants: &[Grant]) -> Vec<OsString> {
        let mut flags = Vec::new();
        for grant in grants {
            for own in &self.own_files {
                let Ok(within) = own.strip_prefix(&grant.source) else {
                    continue;
                };
                let at = grant.at.join(within).into_os_string();
                if own.is_dir() || (grant.can_create(own) && !own.exists()) {
                    flags.extend(["--tmpfs".into(), at]);
                } else if own.exists() {
                    flags.extend(["--ro-bind".into(), "/dev/null".into(), at]);
                }
            }
        }
        flags
    }
}

/// `path` with symlinks resolved when it exists, else as given.
fn resolved(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// The program `server.command` names on `path_env`. Fails when it cannot be run
/// or is under neither a system directory, a grant nor the server's home.
fn visible_program(
    server: &SandboxedServer,
    grants: &[Grant],
    path_env: &OsStr,
) -> Result<PathBuf, ServerSandboxError> {
    let lookup_path = server.env.get("PATH").map_or(path_env, OsStr::new);
    let program = find_program(&server.command, lookup_path).map_err(|reason| {
        ServerSandboxError::Command {
            command: server.command.clone(),
            reason,
        }
    })?;
    let is_visible = std::iter::once(Path::new("/usr"))
        .chain(QUARANTINE_SYSTEM_LINKS.map(Path::new))
        .chain([server.home.as_path()])
        .chain(grants.iter().map(|grant| grant.at.as_path()))
        .any(|shown| program.starts_with(shown));
    if is_visible {
        Ok(program)
    } else {
        Err(ServerSandboxError::CommandHidden { program })
    }
}

/// `home` as a writable grant, read-only where `locked`, created owner-only when missing.
fn created_home(home: &Path, locked: &LockedPaths) -> Result<Grant, ServerSandboxError> {
    DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(home)
        .and_then(|()| std::fs::canonicalize(home))
        .map(|source| Grant::new(home, source, true, locked))
        .map_err(|source| ServerSandboxError::Home {
            home: home.to_path_buf(),
            source,
        })
}

/// What `locked` keeps a server from changing in the writable host path `source`: all of
/// it when a watch covers it, else each watched path that exists inside it. Sorted.
fn locked_in(source: &Path, locked: &LockedPaths) -> Vec<PathBuf> {
    if locked.over(source).is_some() {
        return vec![source.to_path_buf()];
    }
    let mut inside: Vec<PathBuf> = locked
        .paths()
        .filter(|path| path.starts_with(source) && path.exists())
        .map(Path::to_path_buf)
        .collect();
    inside.sort();
    inside.dedup();
    inside
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use assistd_ipc::TaskId;

    use super::super::super::watched::WatchedPath;
    use super::super::scope::CommandScope;
    use super::*;

    fn sandbox_hiding(own_files: &[&Path]) -> ServerSandbox {
        ServerSandbox {
            programs: SandboxPrograms {
                bwrap: PathBuf::from("/usr/bin/bwrap"),
                prlimit: PathBuf::from("/usr/bin/prlimit"),
                choom: PathBuf::from("/usr/bin/choom"),
                scope: CommandScope {
                    systemd_run: PathBuf::from("/usr/bin/systemd-run"),
                    runtime_dir: OsString::from("/run/user/1000"),
                },
            },
            watched: Arc::default(),
            own_files: own_files.iter().map(|path| resolved(path)).collect(),
            path_env: OsString::from("/usr/bin"),
            confined: None,
        }
    }

    fn server_in(home: &Path) -> SandboxedServer {
        SandboxedServer {
            command: PathBuf::from("/usr/bin/env"),
            args: vec!["serve".into()],
            env: HashMap::new(),
            home: home.to_path_buf(),
            read: Vec::new(),
            write: Vec::new(),
            network: true,
        }
    }

    fn flags_of(cmd: &ProcCommand) -> Vec<String> {
        let argv: Vec<String> = cmd
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        let end = argv.iter().position(|arg| arg == "--").expect("flags end");
        argv[..end].to_vec()
    }

    fn position_of(flags: &[String], mount: &[&str]) -> usize {
        flags
            .windows(mount.len())
            .position(|window| window == mount)
            .unwrap_or_else(|| panic!("{mount:?} in {flags:?}"))
    }

    #[test]
    fn the_daemons_files_inside_a_grant_are_covered_before_the_home_is_bound() {
        let host = tempfile::tempdir().expect("tempdir");
        let host_dir = std::fs::canonicalize(host.path()).expect("canonical");
        let data = host_dir.join("share/assistd");
        let home = data.join("mcp/files");
        let socket = host_dir.join("assistd.sock");
        let tokens = host_dir.join("state/assistd");
        std::fs::create_dir_all(&data).expect("data dir");
        std::fs::write(&socket, b"").expect("socket stand-in");
        let path_of = |path: &Path| path.to_string_lossy().into_owned();
        let sandbox = sandbox_hiding(&[&data, &socket, &tokens]);

        let read_only = SandboxedServer {
            read: vec![host_dir.clone()],
            network: false,
            ..server_in(&home)
        };
        let flags = flags_of(&sandbox.command(&read_only).expect("command").0);
        let (host_path, data_path) = (path_of(&host_dir), path_of(&data));
        let (socket_path, home_path) = (path_of(&socket), path_of(&home));
        let granted = position_of(&flags, &["--ro-bind", &host_path, &host_path]);
        let data_hidden = position_of(&flags, &["--tmpfs", &data_path]);
        let socket_hidden = position_of(&flags, &["--ro-bind", "/dev/null", &socket_path]);
        let home_bound = position_of(&flags, &["--bind", &home_path, &home_path]);
        assert!(
            granted < data_hidden && granted < socket_hidden,
            "{flags:?}"
        );
        assert!(data_hidden < home_bound, "{flags:?}");
        assert!(!flags.contains(&path_of(&tokens)), "{flags:?}");
        assert!(!flags.contains(&"--share-net".to_string()), "{flags:?}");
        position_of(&flags, &["--setenv", "HOME", &home_path]);

        let writable = SandboxedServer {
            write: vec![host_dir.clone()],
            ..server_in(&home)
        };
        let flags = flags_of(&sandbox.command(&writable).expect("command").0);
        position_of(&flags, &["--tmpfs", &path_of(&tokens)]);
        let unshared = position_of(&flags, &["--unshare-net"]);
        assert!(
            unshared < position_of(&flags, &["--share-net"]),
            "{flags:?}"
        );
    }

    #[test]
    fn a_grant_inside_the_daemons_files_or_a_hidden_command_is_refused() {
        let host = tempfile::tempdir().expect("tempdir");
        let host_dir = std::fs::canonicalize(host.path()).expect("canonical");
        let config = host_dir.join("config/assistd");
        let tools = host_dir.join("tools");
        std::fs::create_dir_all(&config).expect("config dir");
        std::fs::create_dir(&tools).expect("tools dir");
        std::os::unix::fs::symlink(&config, host_dir.join("link")).expect("symlink");
        std::os::unix::fs::symlink("/usr/bin/env", tools.join("server")).expect("symlink");
        let sandbox = sandbox_hiding(&[&config]);
        let home = host_dir.join("home");

        for granted in [config.clone(), host_dir.join("link")] {
            let server = SandboxedServer {
                write: vec![granted.clone()],
                ..server_in(&home)
            };
            let refused = sandbox.command(&server).expect_err("grant refused");
            assert!(
                matches!(refused, ServerSandboxError::GrantsOwnFiles { .. }),
                "{granted:?}: {refused}"
            );
        }
        let missing = SandboxedServer {
            read: vec![host_dir.join("missing")],
            ..server_in(&home)
        };
        let refused = sandbox.command(&missing).expect_err("grant refused");
        assert!(
            matches!(refused, ServerSandboxError::Grant { .. }),
            "{refused}"
        );

        let hidden_command = SandboxedServer {
            command: tools.join("server"),
            ..server_in(&home)
        };
        let refused = sandbox
            .command(&hidden_command)
            .expect_err("command hidden");
        assert!(
            matches!(refused, ServerSandboxError::CommandHidden { .. }),
            "{refused}"
        );
        let granted_command = SandboxedServer {
            read: vec![tools.clone()],
            ..hidden_command
        };
        sandbox.command(&granted_command).expect("command granted");
    }

    #[tokio::test]
    async fn a_watched_path_a_server_could_write_is_read_only_until_the_watches_change() {
        let host = tempfile::tempdir().expect("tempdir");
        let host_dir = std::fs::canonicalize(host.path()).expect("canonical");
        let (projects, notes) = (host_dir.join("projects"), host_dir.join("notes"));
        let (inbox, home) = (projects.join("inbox"), host_dir.join("home"));
        std::fs::create_dir_all(&inbox).expect("inbox");
        std::fs::create_dir_all(notes.join("daily")).expect("notes");
        let path_of = |path: &Path| path.to_string_lossy().into_owned();
        let watched_by_task_1 = |path: &Path| WatchedPath {
            task: TaskId(1),
            task_name: "sort inbox".into(),
            path: path.to_path_buf(),
        };
        let sandbox = sandbox_hiding(&[]);
        let watched = vec![
            watched_by_task_1(&inbox),
            watched_by_task_1(&notes),
            watched_by_task_1(&projects.join("deleted")),
        ];
        sandbox.watched.replace(watched.clone());
        let server = SandboxedServer {
            write: vec![projects.clone(), notes.join("daily")],
            ..server_in(&home)
        };

        let (cmd, locks) = sandbox.command(&server).expect("command");
        let flags = flags_of(&cmd);
        let (projects_path, inbox_path) = (path_of(&projects), path_of(&inbox));
        let daily_path = path_of(&notes.join("daily"));
        let granted = position_of(&flags, &["--bind", &projects_path, &projects_path]);
        let inbox_locked = position_of(&flags, &["--ro-bind", &inbox_path, &inbox_path]);
        assert!(granted < inbox_locked, "{flags:?}");
        let daily_granted = position_of(&flags, &["--bind", &daily_path, &daily_path]);
        let daily_locked = position_of(&flags, &["--ro-bind", &daily_path, &daily_path]);
        assert!(daily_granted < daily_locked, "{flags:?}");
        assert!(!flags.contains(&path_of(&projects.join("deleted"))));

        let still_current = tokio::time::timeout(Duration::from_millis(50), locks.outdated());
        still_current.await.expect_err("the same watches");
        sandbox.watched.replace(watched.clone());
        let still_current = tokio::time::timeout(Duration::from_millis(50), locks.outdated());
        still_current.await.expect_err("the same watches again");
        sandbox.watched.replace(watched[1..].to_vec());
        tokio::time::timeout(Duration::from_secs(5), locks.outdated())
            .await
            .expect("the inbox is no longer watched");
    }
}

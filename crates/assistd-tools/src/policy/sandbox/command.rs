//! Building the command a sandboxed program runs as: the bwrap flags that lay out its
//! filesystem, then the scope and limits it runs under.

use std::ffi::{OsStr, OsString};
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::{Path, PathBuf};

use assistd_utils::env::inherited_env;
use command_fds::CommandFdExt;
use tokio::process::Command as ProcCommand;
use tracing::warn;

use super::home::SandboxHome;
use super::mounts::Mounts;
use super::{
    HomeAccess, InputBind, Protected, QuarantineProfile, SANDBOX_PATH, SandboxInfo,
    SandboxPrograms, SharedDirs, scope, seccomp,
};

/// The caps every sandboxed command runs under, as `prlimit` flags: tasks,
/// threads included, in the sandbox's own user namespace; bytes of a
/// process's data segment; bytes of any one file it writes.
const RESOURCE_LIMITS: [&str; 3] = ["--nproc=512", "--data=4294967296", "--fsize=4294967296"];

/// The `choom` flags that make every sandboxed process the kernel's first
/// pick when the host runs out of memory.
const OOM_SCORE_ADJUSTMENT: [&str; 3] = ["-n", "1000", "--"];

/// The namespaces every sandboxed command gets of its own, network
/// included, in its own session, killed with the daemon.
const ISOLATION_FLAGS: [&str; 7] = [
    "--unshare-user",
    "--unshare-pid",
    "--unshare-ipc",
    "--unshare-uts",
    "--unshare-net",
    "--new-session",
    "--die-with-parent",
];

/// The top-level system directories a quarantine mirrors from the host,
/// as the symlinks into `/usr` they are on a merged-`/usr` system.
pub(super) const QUARANTINE_SYSTEM_LINKS: [&str; 4] = ["/bin", "/sbin", "/lib", "/lib64"];

/// The `/etc` entries a quarantine exposes: certificates, name resolution
/// and the dynamic loader's configuration.
const QUARANTINE_ETC: [&str; 6] = [
    "/etc/ssl",
    "/etc/ca-certificates",
    "/etc/resolv.conf",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
];

/// The `bwrap_extra_args` flags that bind a host path writable, each with
/// its read-only form.
const WRITABLE_BINDS: [(&str, &str); 3] = [
    ("--bind", "--ro-bind"),
    ("--bind-try", "--ro-bind-try"),
    ("--bind-fd", "--ro-bind-fd"),
];

/// The `bwrap_extra_args` flags that bind a source on a destination with
/// its devices usable, which bwrap cannot also make read-only.
const DEVICE_BINDS: [&str; 2] = ["--dev-bind", "--dev-bind-try"];

/// One command's sandbox: its bwrap flags in the order they take effect, and what they mount.
#[derive(Debug)]
pub(super) struct Layout<'info> {
    /// The profile and everything mounted over it.
    flags: Vec<OsString>,
    /// Quarantine inputs, bound over `flags` from their descriptors.
    inputs: &'info [InputBind],
    /// What follows the seccomp filter so that it wins: `bwrap_extra_args`, then the
    /// working directory.
    late_flags: Vec<OsString>,
    pub(super) mounts: Mounts,
}

impl<'info> Layout<'info> {
    /// A host-profile command's sandbox with `home` as `$HOME`. Flag order is load-bearing: profile,
    /// ancestor self-binds, shared directories, read-only mounts over `home`'s command directories
    /// and protections, then `extra_args` past the seccomp filter. Blocks on the filesystem.
    pub(super) fn host(info: &SandboxInfo, home: Option<String>, access: HomeAccess) -> Self {
        let home = SandboxHome::read(home);
        let command_dirs = home.as_ref().map_or_default(|home| {
            home.locked_command_dirs(access, &info.host_path, &info.protected.dirs)
        });
        let writable: Vec<PathBuf> = home.iter().map(|home| PathBuf::from(&home.dir)).collect();
        let mut flags = os_strings(default_bwrap_flags_for(home.as_ref(), access));
        flags.extend(os_strings(command_dirs.rename_guard_flags()));
        flags.extend(os_strings(shared_flags(&info.shared)));
        flags.extend(os_strings(command_dirs.lock_flags()));
        flags.extend(os_strings(protection_flags(&info.protected, &writable)));
        let mut late_flags = os_strings(extra_args_for(&info.extra_args, access));
        let mut mounts = Mounts::of(&as_os_strs(flags.iter().chain(&late_flags)));
        let home_dir = home.as_ref().map(|home| Path::new(&home.dir));
        if let Some(dir) = host_working_dir(std::env::current_dir().ok(), &mounts, home_dir) {
            late_flags.extend(["--chdir".into(), dir.clone().into_os_string()]);
            mounts.working_dir = Some(dir);
        }
        Self {
            flags,
            inputs: &[],
            late_flags,
            mounts,
        }
    }

    /// The quarantine profile, then the shared directories, which its home is one of, then its
    /// inputs over their mount points in them; no `extra_args`, so the network stays unshared.
    /// Reads the host's system links.
    pub(super) fn quarantine(info: &SandboxInfo, quarantine: &'info QuarantineProfile) -> Self {
        let mut flags = quarantine_flags(&quarantine.home);
        flags.extend(os_strings(shared_flags(&info.shared)));
        let mut mounts = Mounts::of(&as_os_strs(&flags));
        for input in &quarantine.inputs {
            mounts.mount_bind(&input.at);
        }
        Self {
            flags,
            inputs: &quarantine.inputs,
            late_flags: Vec::new(),
            mounts,
        }
    }
}

/// The command that runs `program` with `args` in `layout`'s sandbox, off every Unix socket,
/// in a capped systemd scope, with `prlimit` applying [`RESOURCE_LIMITS`] inside. Fails when
/// the seccomp filter cannot be written or an input's descriptor cannot be duplicated.
pub(super) fn sandboxed_command<I, S>(
    programs: &SandboxPrograms,
    layout: &Layout<'_>,
    program: &str,
    args: I,
) -> io::Result<ProcCommand>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut cmd = scoped_bwrap(programs);
    cmd.args(&layout.flags);
    let mut fds = bind_inputs(&mut cmd, layout.inputs)?;
    fds.push(load_seccomp_filter(&mut cmd)?);
    cmd.preserved_fds(fds);
    cmd.args(&layout.late_flags);
    run_limited(&mut cmd, programs, program, args);
    Ok(cmd)
}

fn os_strings<S: Into<OsString>>(flags: impl IntoIterator<Item = S>) -> Vec<OsString> {
    flags.into_iter().map(Into::into).collect()
}

fn as_os_strs<'flags>(flags: impl IntoIterator<Item = &'flags OsString>) -> Vec<&'flags OsStr> {
    flags.into_iter().map(OsString::as_os_str).collect()
}

/// `bwrap` in a new capped scope, keeping only the locale, terminal,
/// time-zone and user-name variables of the daemon's environment.
pub(super) fn scoped_bwrap(programs: &SandboxPrograms) -> ProcCommand {
    let mut cmd = ProcCommand::from(programs.scope.command(&programs.bwrap));
    cmd.envs(inherited_env(std::env::vars_os(), &[]));
    cmd
}

/// Have bwrap load the seccomp filter that keeps the command off every Unix
/// socket, from the returned descriptor, which only `cmd`'s child may inherit.
pub(super) fn load_seccomp_filter(cmd: &mut ProcCommand) -> io::Result<OwnedFd> {
    let filter = seccomp::filter_fd()?;
    cmd.arg("--seccomp").arg(filter.as_raw_fd().to_string());
    Ok(filter)
}

/// Bind each of `inputs` from a duplicate of its descriptor, returned for only
/// `cmd`'s child to inherit, so every command sees what the quarantine opened.
fn bind_inputs(cmd: &mut ProcCommand, inputs: &[InputBind]) -> io::Result<Vec<OwnedFd>> {
    let fds = inputs
        .iter()
        .map(|input| input.fd.try_clone())
        .collect::<io::Result<Vec<_>>>()?;
    for (fd, input) in fds.iter().zip(inputs) {
        cmd.arg("--ro-bind-fd")
            .arg(fd.as_raw_fd().to_string())
            .arg(&input.at);
    }
    Ok(fds)
}

/// End `cmd`'s bwrap flags, unsetting what the scope added to the
/// environment, and run `program` with `args` under `prlimit`'s
/// [`RESOURCE_LIMITS`] and `choom`'s [`OOM_SCORE_ADJUSTMENT`].
pub(super) fn run_limited<I, S>(
    cmd: &mut ProcCommand,
    programs: &SandboxPrograms,
    program: impl AsRef<OsStr>,
    args: I,
) where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    for name in scope::SCOPE_ENV {
        cmd.arg("--unsetenv").arg(name);
    }
    cmd.arg("--");
    cmd.arg(&programs.prlimit).args(RESOURCE_LIMITS).arg("--");
    cmd.arg(&programs.choom).args(OOM_SCORE_ADJUSTMENT);
    cmd.arg(program).args(args);
}

/// Read-only root with `$HOME`'s visible entries writable when bindable and `access` allows,
/// fresh `/tmp`, `/dev`, `/proc` and `/run`, isolated user/pid/ipc/uts namespaces, and no
/// network (`--share-net` in `bwrap_extra_args` restores it).
fn default_bwrap_flags_for(home: Option<&SandboxHome>, access: HomeAccess) -> Vec<String> {
    let mut flags: Vec<String> = ["--ro-bind", "/", "/", "--tmpfs", "/tmp"]
        .map(String::from)
        .into();
    match home {
        Some(home) => flags.extend(home.bind_flags(access)),
        None => warn!(
            target: "assistd::policy",
            "HOME is unset or not a non-root absolute directory; sandboxed commands \
             will have no writable home"
        ),
    }
    flags.extend(
        ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/run"]
            .into_iter()
            .chain(ISOLATION_FLAGS)
            .map(String::from),
    );
    if let Some(home) = home {
        flags.extend(["--setenv".into(), "HOME".into(), home.dir.clone()]);
    }
    flags.extend(["--setenv".into(), "PATH".into(), SANDBOX_PATH.into()]);
    flags
}

/// [`system_flags`] with `home` as `$HOME` and working directory. Reads the
/// host's system links.
fn quarantine_flags(home: &Path) -> Vec<OsString> {
    let mut flags = system_flags();
    flags.extend(["--setenv", "HOME"].map(OsString::from));
    flags.push(home.into());
    flags.push("--chdir".into());
    flags.push(home.into());
    flags
}

/// Read-only `/usr` and [`QUARANTINE_ETC`], the [`QUARANTINE_SYSTEM_LINKS`],
/// fresh `/tmp`, `/dev` and `/proc`, [`ISOLATION_FLAGS`], and the sandbox's
/// `PATH`: none of the host but its programs. Reads the host's system links.
pub(super) fn system_flags() -> Vec<OsString> {
    let mut flags: Vec<OsString> = ["--ro-bind", "/usr", "/usr"].map(OsString::from).into();
    for link in QUARANTINE_SYSTEM_LINKS {
        let mirror = match std::fs::read_link(link) {
            Ok(target) => ["--symlink".into(), target.into_os_string(), link.into()],
            Err(_) => ["--ro-bind-try", link, link].map(OsString::from),
        };
        flags.extend(mirror);
    }
    for entry in QUARANTINE_ETC {
        flags.extend(["--ro-bind-try", entry, entry].map(OsString::from));
    }
    flags.extend(
        ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
            .into_iter()
            .chain(ISOLATION_FLAGS)
            .chain(["--setenv", "PATH", SANDBOX_PATH])
            .map(OsString::from),
    );
    flags
}

/// Binds of the shared directories at their own paths, skipped by bwrap
/// for a directory that does not exist.
fn shared_flags(shared: &SharedDirs) -> Vec<&OsStr> {
    let scratch = shared
        .scratch
        .iter()
        .flat_map(|dir| [OsStr::new("--bind-try"), dir.as_os_str(), dir.as_os_str()]);
    let read_only = shared.read_only.iter().flat_map(|dir| {
        [
            OsStr::new("--ro-bind-try"),
            dir.as_os_str(),
            dir.as_os_str(),
        ]
    });
    scratch.chain(read_only).collect()
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

/// `extra_args` as a command with `access` gets them: as they are, or
/// [`read_only_extra_args`] under a read-only home.
fn extra_args_for(extra_args: &[String], access: HomeAccess) -> Vec<&str> {
    match access {
        HomeAccess::Writable => extra_args.iter().map(String::as_str).collect(),
        HomeAccess::ReadOnly => read_only_extra_args(extra_args),
    }
}

/// `extra_args` mounting nothing writable: [`WRITABLE_BINDS`] in their read-only form, an
/// `--overlay` as a read-only one with its upper directory the top layer, and no
/// [`DEVICE_BINDS`], leaving the profile's own `/dev`.
fn read_only_extra_args(extra_args: &[String]) -> Vec<&str> {
    let mut args = Vec::with_capacity(extra_args.len());
    let mut rest = extra_args;
    while let [flag, tail @ ..] = rest {
        rest = match (flag.as_str(), tail) {
            (flag, [_, _, tail @ ..]) if DEVICE_BINDS.contains(&flag) => tail,
            ("--overlay", [upper, _, dest, tail @ ..]) => {
                args.extend(["--overlay-src", upper, "--ro-overlay", dest]);
                tail
            }
            (flag, tail) => {
                args.push(read_only_bind(flag));
                tail
            }
        };
    }
    args
}

/// `flag` in its read-only form when it is one of [`WRITABLE_BINDS`], else itself.
fn read_only_bind(flag: &str) -> &str {
    WRITABLE_BINDS
        .iter()
        .find(|(writable, _)| *writable == flag)
        .map_or(flag, |(_, read_only)| read_only)
}

/// Where a host-profile command starts: the daemon's working directory `cwd` when `mounts`
/// show it as on the host, else `home`, so a relative path names the same file inside as the
/// path guards resolve it to outside.
fn host_working_dir(cwd: Option<PathBuf>, mounts: &Mounts, home: Option<&Path>) -> Option<PathBuf> {
    cwd.filter(|cwd| cwd.is_dir() && mounts.private_dir_of(cwd).is_none())
        .or_else(|| home.map(Path::to_path_buf))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::Profile;
    use super::super::scope::CommandScope;
    use super::*;

    fn programs() -> SandboxPrograms {
        SandboxPrograms {
            bwrap: PathBuf::from("/usr/bin/bwrap"),
            prlimit: PathBuf::from("/usr/bin/prlimit"),
            choom: PathBuf::from("/usr/bin/choom"),
            scope: CommandScope {
                systemd_run: PathBuf::from("/usr/bin/systemd-run"),
                runtime_dir: OsString::from("/run/user/1000"),
            },
        }
    }

    fn bwrap_info(extra_args: Vec<String>) -> SandboxInfo {
        SandboxInfo {
            programs: programs(),
            extra_args,
            protected: Protected::default(),
            shared: SharedDirs::default(),
            host_path: Vec::new(),
            unit_dirs: Vec::new(),
            confined: None,
            profile: Profile::Host,
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

    fn host_command(
        info: &SandboxInfo,
        home: Option<String>,
        access: HomeAccess,
        program: &str,
        args: &[&str],
    ) -> ProcCommand {
        let layout = Layout::host(info, home, access);
        sandboxed_command(&info.programs, &layout, program, args).expect("bwrap command")
    }

    #[test]
    fn bwrap_argv_is_profile_then_extra_args_then_program() {
        let home_dir = tempfile::tempdir().expect("tempdir");
        let home = Some(home_dir.path().to_string_lossy().into_owned());
        let info = bwrap_info(vec!["--share-net".into()]);
        let limits = ["/usr/bin/prlimit"]
            .into_iter()
            .chain(RESOURCE_LIMITS)
            .chain(["--", "/usr/bin/choom"])
            .chain(OOM_SCORE_ADJUSTMENT);
        let cwd = std::env::current_dir().expect("cwd");
        let unset_scope_env = [
            "--unsetenv",
            "XDG_RUNTIME_DIR",
            "--unsetenv",
            "INVOCATION_ID",
        ];
        let tail: Vec<String> = ["--share-net", "--chdir", &cwd.to_string_lossy()]
            .into_iter()
            .chain(unset_scope_env)
            .chain(["--"])
            .chain(limits)
            .chain(["make", "-j4"])
            .map(String::from)
            .collect();
        let profile = default_bwrap_flags_for(
            SandboxHome::read(home.clone()).as_ref(),
            HomeAccess::Writable,
        );
        for private in [["--tmpfs", "/tmp"], ["--tmpfs", "/run"]] {
            assert!(profile.windows(2).any(|w| w == private), "{private:?}");
        }
        assert!(profile.iter().any(|flag| flag == "--unshare-net"));
        let cmd = host_command(&info, home, HomeAccess::Writable, "make", &["-j4"]);
        assert_eq!(cmd.as_std().get_program(), "/usr/bin/systemd-run");
        let argv = argv_of(&cmd);
        let scope: Vec<String> = ["--user", "--scope", "--quiet", "--collect"]
            .into_iter()
            .chain(["--expand-environment=no", "--property=MemoryMax=4G"])
            .chain(["--property=MemorySwapMax=0", "--property=TasksMax=512"])
            .chain(["/usr/bin/bwrap"])
            .map(String::from)
            .collect();
        let filter_fd = argv
            .iter()
            .position(|flag| flag == "--seccomp")
            .and_then(|at| argv.get(at + 1))
            .expect("the seccomp filter is loaded with the network shared");
        let seccomp = vec!["--seccomp".to_string(), filter_fd.clone()];
        assert_eq!(argv, [scope, profile, seccomp, tail].concat());
    }

    #[test]
    fn a_working_directory_the_sandbox_hides_is_replaced_by_home() {
        let mounts = Mounts::of(&["--ro-bind", "/", "/", "--tmpfs", "/tmp"].map(OsStr::new));
        let hidden = tempfile::tempdir_in("/tmp").expect("tempdir");
        let visible = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let home = Path::new("/home/someone");
        for (cwd, expected) in [
            (Some(visible.clone()), visible.as_path()),
            (Some(hidden.path().to_path_buf()), home),
            (Some(visible.join("no-such-dir")), home),
            (None, home),
        ] {
            assert_eq!(
                host_working_dir(cwd.clone(), &mounts, Some(home)).as_deref(),
                Some(expected),
                "{cwd:?}"
            );
        }
    }

    #[tokio::test]
    async fn shared_dirs_are_bound_and_are_not_private() {
        let info = Arc::new(SandboxInfo {
            shared: SharedDirs {
                scratch: Some(PathBuf::from("/run/user/1000/assistd/scratch")),
                scratch_cap: None,
                read_only: vec![PathBuf::from("/tmp/assistd-output")],
            },
            ..bwrap_info(Vec::new())
        });
        let binds = [
            "--bind-try",
            "/run/user/1000/assistd/scratch",
            "/run/user/1000/assistd/scratch",
            "--ro-bind-try",
            "/tmp/assistd-output",
            "/tmp/assistd-output",
        ];
        let argv = argv_of(
            &info
                .build_command("true", [""; 0], HomeAccess::Writable)
                .expect("command")
                .command,
        );
        assert!(argv.windows(binds.len()).any(|w| w == binds), "{argv:?}");
        for shared in [
            "/run/user/1000/assistd/scratch/out.txt",
            "/tmp/assistd-output/cmd-1.txt",
        ] {
            assert_eq!(
                info.private_dir_hiding(PathBuf::from(shared)).await,
                None,
                "{shared}"
            );
        }
        assert_eq!(
            info.private_dir_hiding(PathBuf::from("/run/user/1000/other"))
                .await,
            Some(PathBuf::from("/run"))
        );
    }

    #[test]
    fn bwrap_process_starts_from_a_cleared_environment() {
        let info = bwrap_info(Vec::new());
        let cmd = info
            .build_command("true", Vec::<String>::new(), HomeAccess::Writable)
            .expect("command")
            .command;
        let runtime_dir = (OsString::from("XDG_RUNTIME_DIR"), "/run/user/1000".into());
        let mut expected: Vec<_> = inherited_env(std::env::vars_os(), &[])
            .into_iter()
            .chain([runtime_dir])
            .map(|(name, value)| (name, Some(value)))
            .collect();
        expected.sort();
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

        let sandbox_home = SandboxHome::read(Some(dir_str.clone())).expect("usable home");
        let flags = default_bwrap_flags_for(Some(&sandbox_home), HomeAccess::Writable);
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
    fn a_read_only_home_leaves_only_the_private_and_scratch_mounts_writable() {
        let home_dir = tempfile::tempdir().expect("tempdir");
        let home = std::fs::canonicalize(home_dir.path()).expect("canonical home");
        let tools_bin = home.join("projects/tools/bin");
        std::fs::create_dir_all(&tools_bin).expect("projects/tools/bin");
        std::fs::create_dir(home.join("go")).expect("go");
        let scratch = "/run/user/1000/assistd/scratch";
        let info = SandboxInfo {
            shared: SharedDirs {
                scratch: Some(PathBuf::from(scratch)),
                scratch_cap: None,
                read_only: Vec::new(),
            },
            host_path: vec![tools_bin, home.join("go/bin")],
            ..bwrap_info(
                ["--bind", "/srv/data", "/srv/data"]
                    .map(String::from)
                    .into(),
            )
        };
        let writable_mounts = |access: HomeAccess| -> Vec<String> {
            let home = Some(home.to_string_lossy().into_owned());
            let cmd = host_command(&info, home, access, "true", &[]);
            argv_of(&cmd)
                .windows(2)
                .filter(|w| matches!(w[0].as_str(), "--bind" | "--bind-try" | "--tmpfs"))
                .map(|w| w[1].clone())
                .collect()
        };
        let path_of = |entry: &str| home.join(entry).to_string_lossy().into_owned();

        assert_eq!(
            writable_mounts(HomeAccess::ReadOnly),
            ["/tmp", "/run", scratch]
        );
        let unrestricted = writable_mounts(HomeAccess::Writable);
        for mounted in ["go", "projects", "projects/tools", "go/bin"] {
            assert!(unrestricted.contains(&path_of(mounted)), "{unrestricted:?}");
        }
        assert!(unrestricted.iter().any(|mounted| mounted == "/srv/data"));
    }

    #[test]
    fn a_read_only_script_gets_the_extra_args_with_nothing_writable_in_them() {
        let extra_args: Vec<String> = ["--bind", "/srv/data", "/srv/data"]
            .into_iter()
            .chain(["--dev-bind", "/dev", "/dev", "--share-net"])
            .chain(["--overlay-src", "/srv/base"])
            .chain(["--overlay", "/srv/upper", "/srv/work", "/srv/merged"])
            .map(String::from)
            .collect();
        let read_only: Vec<&str> = ["--ro-bind", "/srv/data", "/srv/data", "--share-net"]
            .into_iter()
            .chain(["--overlay-src", "/srv/base", "--overlay-src", "/srv/upper"])
            .chain(["--ro-overlay", "/srv/merged"])
            .collect();

        assert_eq!(extra_args_for(&extra_args, HomeAccess::ReadOnly), read_only);
        assert_eq!(
            extra_args_for(&extra_args, HomeAccess::Writable),
            extra_args
        );
    }

    #[test]
    fn home_command_dirs_are_bound_read_only_after_every_writable_bind() {
        let home_dir = tempfile::tempdir().expect("tempdir");
        let home = std::fs::canonicalize(home_dir.path()).expect("canonical home");
        let go_bin = home.join("go/bin");
        std::fs::create_dir_all(&go_bin).expect("go/bin");
        let tools_bin = home.join("tools/bin");
        std::fs::create_dir(home.join("tools")).expect("tools");
        let info = SandboxInfo {
            shared: SharedDirs {
                scratch: Some(go_bin.clone()),
                scratch_cap: None,
                read_only: Vec::new(),
            },
            host_path: vec![go_bin.clone(), tools_bin.clone()],
            unit_dirs: Vec::new(),
            ..bwrap_info(Vec::new())
        };
        let home_var = Some(home.to_string_lossy().into_owned());
        let argv = argv_of(&host_command(
            &info,
            home_var,
            HomeAccess::Writable,
            "true",
            &[],
        ));
        let go_bin = go_bin.to_string_lossy().into_owned();
        let go = home.join("go").to_string_lossy().into_owned();
        assert!(writable_binds(&argv).contains(&go.as_str()), "{argv:?}");
        let last_writable = argv
            .iter()
            .rposition(|flag| flag == "--bind" || flag == "--bind-try")
            .expect("writable binds");
        let read_only = argv
            .windows(3)
            .position(|w| w == ["--ro-bind", go_bin.as_str(), go_bin.as_str()])
            .expect("go/bin is bound read-only");
        assert!(read_only > last_writable, "{argv:?}");
        let tools_bin = tools_bin.to_string_lossy().into_owned();
        let empty_mount = argv
            .windows(4)
            .position(|w| {
                w == [
                    "--tmpfs",
                    tools_bin.as_str(),
                    "--remount-ro",
                    tools_bin.as_str(),
                ]
            })
            .expect("missing tools/bin gets an empty read-only mount");
        assert!(empty_mount > last_writable, "{argv:?}");
    }

    #[test]
    fn rename_guards_follow_the_entry_binds_and_precede_every_read_only_mount() {
        let home_dir = tempfile::tempdir().expect("tempdir");
        let home = std::fs::canonicalize(home_dir.path()).expect("canonical home");
        for dir in [
            "projects/tools/bin",
            "projects/tools/conf",
            "projects/tools/scratch",
        ] {
            std::fs::create_dir_all(home.join(dir)).expect(dir);
        }
        let path_of = |dir: &str| home.join(dir).to_string_lossy().into_owned();
        let info = SandboxInfo {
            protected: Protected {
                dirs: vec![home.join("projects/tools/conf")],
                sockets: Vec::new(),
            },
            shared: SharedDirs {
                scratch: Some(home.join("projects/tools/scratch")),
                scratch_cap: None,
                read_only: Vec::new(),
            },
            host_path: vec![home.join("projects/tools/bin")],
            unit_dirs: Vec::new(),
            ..bwrap_info(Vec::new())
        };
        let argv = argv_of(&host_command(
            &info,
            Some(path_of("")),
            HomeAccess::Writable,
            "true",
            &[],
        ));
        let position_of = |flag: &str, dir: &str| {
            let dir = path_of(dir);
            argv.windows(3)
                .position(|w| w == [flag, dir.as_str(), dir.as_str()])
                .unwrap_or_else(|| panic!("{flag} {dir} in {argv:?}"))
        };
        let guard = position_of("--bind", "projects/tools");
        assert!(position_of("--bind", "projects") < guard, "{argv:?}");
        assert!(guard < position_of("--bind-try", "projects/tools/scratch"));
        assert!(guard < position_of("--ro-bind", "projects/tools/bin"));
        assert!(guard < position_of("--ro-bind", "projects/tools/conf"));
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
            let sandbox_home = SandboxHome::read(home.map(str::to_string));
            assert!(sandbox_home.is_none(), "HOME={home:?} must be unusable");
            let flags = default_bwrap_flags_for(sandbox_home.as_ref(), HomeAccess::Writable);
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
}

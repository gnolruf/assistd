//! The sandbox of a program that only reads chosen directories, run at idle
//! CPU and IO priority.

use std::ffi::{OsStr, OsString};
use std::io;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use assistd_utils::program::{ProgramError, find_program};
use command_fds::CommandFdExt;
use rustix::fs::{Mode, OFlags};
use tokio::process::Command as ProcCommand;

use super::command::{load_seccomp_filter, run_limited, scoped_bwrap, system_flags};
use super::{SANDBOX_PATH, SandboxInfo, SandboxPrograms};

/// The `chrt` arguments that put a command in the idle scheduling class, which
/// has the one priority.
const IDLE_CPU: [&str; 2] = ["--idle", "0"];

/// The `ionice` arguments that put a command in the idle IO class.
const IDLE_IO: [&str; 2] = ["-c", "3"];

/// A util-linux program the reader sandbox needs and cannot run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("cannot run `{program}` from the sandbox's PATH: {reason}; install util-linux")]
pub struct ReaderSandboxError {
    program: &'static str,
    reason: ProgramError,
}

/// The program a [`ReaderSandbox`] runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReaderProgram {
    /// The executable file, opened anew for every command.
    pub image: PathBuf,
    /// Where the sandbox shows `image`; inside a bound directory, a file must
    /// already be there.
    pub path: PathBuf,
}

/// A bwrap sandbox showing none of the host but its programs and chosen
/// directories, all read-only, with no network and no `AF_UNIX` sockets. Its
/// program runs at idle CPU and IO priority under the tool sandbox's scope and limits.
#[derive(Debug, Clone)]
pub struct ReaderSandbox {
    programs: SandboxPrograms,
    chrt: PathBuf,
    ionice: PathBuf,
    read_only: Vec<PathBuf>,
}

impl ReaderSandbox {
    /// A reader sandbox started as `sandbox` starts its commands, showing each of
    /// `read_only` at its own path. Fails when `chrt` or `ionice` is not on the
    /// sandbox's `PATH`.
    pub fn over(
        sandbox: &SandboxInfo,
        read_only: Vec<PathBuf>,
    ) -> Result<Self, ReaderSandboxError> {
        Ok(Self {
            programs: sandbox.programs.clone(),
            chrt: find_util_linux("chrt")?,
            ionice: find_util_linux("ionice")?,
            read_only,
        })
    }

    /// The command that runs `program` with `args` in this sandbox; a directory
    /// that does not exist is left out. Fails when the program's image cannot be
    /// opened or the seccomp filter cannot be written. Blocks on the filesystem.
    pub fn command<I, S>(&self, program: &ReaderProgram, args: I) -> io::Result<ProcCommand>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut cmd = scoped_bwrap(&self.programs);
        cmd.args(system_flags()).args(["--chdir", "/"]);
        for dir in &self.read_only {
            cmd.arg("--ro-bind-try").arg(dir).arg(dir);
        }
        let image = rustix::fs::open(
            &program.image,
            OFlags::PATH | OFlags::CLOEXEC,
            Mode::empty(),
        )?;
        cmd.arg("--ro-bind-fd")
            .arg(image.as_raw_fd().to_string())
            .arg(&program.path);
        let filter = load_seccomp_filter(&mut cmd)?;
        cmd.preserved_fds(vec![image, filter]);
        run_limited(
            &mut cmd,
            &self.programs,
            &self.chrt,
            self.idle_program(program, args),
        );
        Ok(cmd)
    }

    /// What `chrt` is given to run: its own arguments, then `ionice` running
    /// `program` with `args`.
    fn idle_program<I, S>(&self, program: &ReaderProgram, args: I) -> Vec<OsString>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        IDLE_CPU
            .into_iter()
            .map(OsString::from)
            .chain([self.ionice.clone().into_os_string()])
            .chain(IDLE_IO.into_iter().map(OsString::from))
            .chain([program.path.clone().into_os_string()])
            .chain(args.into_iter().map(|arg| arg.as_ref().to_os_string()))
            .collect()
    }
}

fn find_util_linux(program: &'static str) -> Result<PathBuf, ReaderSandboxError> {
    find_program(Path::new(program), OsStr::new(SANDBOX_PATH))
        .map_err(|reason| ReaderSandboxError { program, reason })
}

#[cfg(test)]
mod tests {
    use super::super::scope::CommandScope;
    use super::*;

    const WRITABLE_MOUNTS: [&str; 6] = [
        "--bind",
        "--bind-try",
        "--bind-fd",
        "--dev-bind",
        "--dev-bind-try",
        "--overlay",
    ];

    fn reader_over(read_only: &[&str]) -> ReaderSandbox {
        ReaderSandbox {
            programs: SandboxPrograms {
                bwrap: PathBuf::from("/usr/bin/bwrap"),
                prlimit: PathBuf::from("/usr/bin/prlimit"),
                choom: PathBuf::from("/usr/bin/choom"),
                scope: CommandScope {
                    systemd_run: PathBuf::from("/usr/bin/systemd-run"),
                    runtime_dir: OsString::from("/run/user/1000"),
                },
            },
            chrt: PathBuf::from("/usr/bin/chrt"),
            ionice: PathBuf::from("/usr/bin/ionice"),
            read_only: read_only.iter().map(PathBuf::from).collect(),
        }
    }

    #[test]
    fn the_program_runs_at_idle_priority_over_read_only_binds_with_no_network() {
        let program = ReaderProgram {
            image: PathBuf::from("/proc/self/exe"),
            path: PathBuf::from("/opt/app/bin/worker"),
        };
        let cmd = reader_over(&["/home/someone/notes", "/opt/app/bin"])
            .command(&program, ["serve"])
            .expect("command");
        let argv: Vec<String> = cmd
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        let flags_end = argv
            .iter()
            .position(|arg| arg == "--")
            .expect("bwrap flags end");
        let flags = &argv[..flags_end];

        assert!(
            !flags
                .iter()
                .any(|flag| WRITABLE_MOUNTS.contains(&flag.as_str())),
            "{flags:?}"
        );
        assert!(
            flags.iter().any(|flag| flag == "--unshare-net"),
            "{flags:?}"
        );
        for dir in ["/home/someone/notes", "/opt/app/bin"] {
            assert!(
                flags.windows(3).any(|w| w == ["--ro-bind-try", dir, dir]),
                "{dir}: {flags:?}"
            );
        }
        let image_bind = flags
            .windows(3)
            .find(|w| w[0] == "--ro-bind-fd")
            .expect("the image is bound from its descriptor");
        assert_eq!(image_bind[2], "/opt/app/bin/worker");
        let idle = argv
            .iter()
            .position(|arg| arg == "/usr/bin/chrt")
            .expect("chrt wraps the program");
        assert_eq!(
            argv[idle..],
            ["/usr/bin/chrt", "--idle", "0", "/usr/bin/ionice", "-c", "3"]
                .into_iter()
                .chain(["/opt/app/bin/worker", "serve"])
                .collect::<Vec<_>>()
        );
    }
}

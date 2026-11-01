//! `assistd-quarantine-exec [--egress SOCKET PORT] -- PROGRAM [ARGS...]`: run
//! PROGRAM barred by Landlock from every Unix socket made outside it but
//! SOCKET, and with `--egress`, forward each connection to `127.0.0.1:PORT`
//! to SOCKET until PROGRAM exits.

use std::error::Error;
use std::ffi::OsString;
use std::io::{self, Write};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{ExitCode, ExitStatus};

use landlock::{
    AccessFs, CompatLevel, Compatible, PathBeneath, PathFd, Ruleset, RulesetAttr,
    RulesetCreatedAttr,
};
use tokio::net::{TcpListener, TcpStream, UnixStream};
use tokio::process::Command;
use tokio::task::JoinSet;

const USAGE: &str = "usage: assistd-quarantine-exec [--egress SOCKET PORT] -- PROGRAM [ARGS...]";

/// Exit status when PROGRAM cannot be started, as a shell reports a
/// command it cannot run.
const CANNOT_RUN: u8 = 126;

/// Exit status offset of a PROGRAM killed by a signal, as a shell reports it.
const SIGNALLED: i32 = 128;

/// What the command line asks for.
#[derive(Debug)]
struct Invocation {
    egress: Option<Forward>,
    program: OsString,
    args: Vec<OsString>,
}

/// The loopback `port` to forward to the proxy's `socket`.
#[derive(Debug)]
struct Forward {
    socket: OsString,
    port: u16,
}

impl Invocation {
    fn parse(mut argv: impl Iterator<Item = OsString>) -> Option<Self> {
        let egress = match argv.next()?.to_str()? {
            "--" => None,
            "--egress" => {
                let socket = argv.next()?;
                let port = argv.next()?.to_str()?.parse().ok()?;
                if argv.next()? != "--" {
                    return None;
                }
                Some(Forward { socket, port })
            }
            _ => return None,
        };
        Some(Self {
            egress,
            program: argv.next()?,
            args: argv.collect(),
        })
    }
}

fn main() -> ExitCode {
    let Some(invocation) = Invocation::parse(std::env::args_os().skip(1)) else {
        return fail(USAGE);
    };
    let allowed = invocation.egress.as_ref().map(|forward| &forward.socket);
    if let Err(error) = bar_host_sockets(allowed) {
        return fail(&format!(
            "assistd-quarantine-exec: cannot bar the command from host sockets: {error}"
        ));
    }
    let Some(forward) = invocation.egress else {
        let error = std::process::Command::new(&invocation.program)
            .args(&invocation.args)
            .exec();
        return fail(&format!("assistd-quarantine-exec: {error}"));
    };
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => return fail(&format!("assistd-quarantine-exec: {error}")),
    };
    match runtime.block_on(run(forward, invocation.program, invocation.args)) {
        Ok(status) => exit_code_of(status),
        Err(error) => fail(&format!("assistd-quarantine-exec: {error}")),
    }
}

/// Restrict this process and everything it starts from connecting to any
/// pathname Unix socket made outside it but `allowed`; fails when the kernel
/// cannot enforce that.
fn bar_host_sockets(allowed: Option<&OsString>) -> Result<(), Box<dyn Error>> {
    let mut ruleset = Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(AccessFs::ResolveUnix)?
        .create()?;
    if let Some(socket) = allowed {
        ruleset = ruleset.add_rule(PathBeneath::new(
            PathFd::new(socket)?,
            AccessFs::ResolveUnix,
        ))?;
    }
    ruleset.restrict_self()?;
    Ok(())
}

/// Listen, start the program, and forward until it exits.
async fn run(forward: Forward, program: OsString, args: Vec<OsString>) -> io::Result<ExitStatus> {
    let listener = TcpListener::bind(("127.0.0.1", forward.port)).await?;
    let mut child = Command::new(&program).args(&args).spawn()?;
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            status = child.wait() => return status,
            accepted = listener.accept() => {
                if let Ok((client, _)) = accepted {
                    connections.spawn(relay(client, forward.socket.clone()));
                }
            }
            Some(_) = connections.join_next() => {}
        }
    }
}

async fn relay(mut client: TcpStream, socket: OsString) {
    if let Ok(mut proxy) = UnixStream::connect(socket).await {
        let _ = tokio::io::copy_bidirectional(&mut client, &mut proxy).await;
    }
}

fn exit_code_of(status: ExitStatus) -> ExitCode {
    let code = status
        .code()
        .or_else(|| status.signal().map(|signal| SIGNALLED + signal))
        .unwrap_or(1);
    ExitCode::from(u8::try_from(code & 0xff).unwrap_or(1))
}

fn fail(message: &str) -> ExitCode {
    let _ = writeln!(io::stderr(), "{message}");
    ExitCode::from(CANNOT_RUN)
}

//! `assistd-egress-forward SOCKET PORT PROGRAM [ARGS...]`: run PROGRAM, and
//! until it exits forward every connection to `127.0.0.1:PORT` to SOCKET.

use std::ffi::OsString;
use std::io::{self, Write};
use std::os::unix::process::ExitStatusExt;
use std::process::{ExitCode, ExitStatus};

use tokio::net::{TcpListener, TcpStream, UnixStream};
use tokio::process::Command;
use tokio::task::JoinSet;

const USAGE: &str = "usage: assistd-egress-forward SOCKET PORT PROGRAM [ARGS...]";

/// Exit status when PROGRAM cannot be started, as a shell reports a
/// command it cannot run.
const CANNOT_RUN: u8 = 126;

/// Exit status offset of a PROGRAM killed by a signal, as a shell reports it.
const SIGNALLED: i32 = 128;

/// What the command line asks for.
#[derive(Debug)]
struct Forward {
    socket: OsString,
    port: u16,
    program: OsString,
    args: Vec<OsString>,
}

impl Forward {
    fn parse(mut argv: impl Iterator<Item = OsString>) -> Option<Self> {
        let socket = argv.next()?;
        let port = argv.next()?.to_str()?.parse().ok()?;
        let program = argv.next()?;
        Some(Self {
            socket,
            port,
            program,
            args: argv.collect(),
        })
    }
}

fn main() -> ExitCode {
    let Some(forward) = Forward::parse(std::env::args_os().skip(1)) else {
        return fail(USAGE);
    };
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => return fail(&format!("assistd-egress-forward: {error}")),
    };
    match runtime.block_on(run(forward)) {
        Ok(status) => exit_code_of(status),
        Err(error) => fail(&format!("assistd-egress-forward: {error}")),
    }
}

/// Listen, start the program, and forward until it exits.
async fn run(forward: Forward) -> io::Result<ExitStatus> {
    let listener = TcpListener::bind(("127.0.0.1", forward.port)).await?;
    let mut child = Command::new(&forward.program).args(&forward.args).spawn()?;
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

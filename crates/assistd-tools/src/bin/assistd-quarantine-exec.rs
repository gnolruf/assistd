//! `assistd-quarantine-exec -- PROGRAM [ARGS...]`: run PROGRAM barred by
//! Landlock from every Unix socket made outside it.

use std::error::Error;
use std::ffi::OsString;
use std::io::{self, Write};
use std::os::unix::process::CommandExt;
use std::process::ExitCode;

use landlock::{AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr};

const USAGE: &str = "usage: assistd-quarantine-exec -- PROGRAM [ARGS...]";

/// Exit status when PROGRAM cannot be started, as a shell reports a
/// command it cannot run.
const CANNOT_RUN: u8 = 126;

/// What the command line asks for.
#[derive(Debug)]
struct Invocation {
    program: OsString,
    args: Vec<OsString>,
}

impl Invocation {
    fn parse(mut argv: impl Iterator<Item = OsString>) -> Option<Self> {
        if argv.next()? != "--" {
            return None;
        }
        Some(Self {
            program: argv.next()?,
            args: argv.collect(),
        })
    }
}

fn main() -> ExitCode {
    let Some(invocation) = Invocation::parse(std::env::args_os().skip(1)) else {
        return fail(USAGE);
    };
    if let Err(error) = bar_host_sockets() {
        return fail(&format!(
            "assistd-quarantine-exec: cannot bar the command from host sockets: {error}"
        ));
    }
    let error = std::process::Command::new(&invocation.program)
        .args(&invocation.args)
        .exec();
    fail(&format!("assistd-quarantine-exec: {error}"))
}

/// Restrict this process and everything it starts from connecting to any
/// pathname Unix socket made outside it; fails when the kernel cannot
/// enforce that.
fn bar_host_sockets() -> Result<(), Box<dyn Error>> {
    Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(AccessFs::ResolveUnix)?
        .create()?
        .restrict_self()?;
    Ok(())
}

fn fail(message: &str) -> ExitCode {
    let _ = writeln!(io::stderr(), "{message}");
    ExitCode::from(CANNOT_RUN)
}

//! Extra command-line arguments handed to the model server, split with POSIX
//! shell quoting but never run through a shell.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::errors::CustomArgsError;

/// Flags assistd sets from `[model]`, or whose server defaults its client and
/// supervisor rely on.
const MANAGED_FLAGS: &[&str] = &[
    "--host",
    "--port",
    "-c",
    "--ctx-size",
    "-ngl",
    "--gpu-layers",
    "--n-gpu-layers",
    "--jinja",
    "--no-jinja",
    "-m",
    "--model",
    "-mu",
    "--model-url",
    "-hf",
    "-hfr",
    "--hf-repo",
    "-hff",
    "--hf-file",
    "--models-dir",
    "--models-preset",
    "--api-prefix",
    "--api-key",
    "--api-key-file",
    "--ssl-key-file",
    "--ssl-cert-file",
];

/// Flags that let anything able to reach the unauthenticated port read files,
/// run tools, change server state, or send model data off the machine.
const EXPOSING_FLAGS: &[&str] = &[
    "--tools",
    "--path",
    "--media-path",
    "--slot-save-path",
    "--props",
    "--rpc",
    "--reuse-port",
];

/// Server arguments written as one string, e.g. `--flash-attn on -ot 'exps=CPU'`.
/// No shell runs, so `$VAR`, `~` and globs stay literal; a word starting with
/// `#` begins a comment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CustomArgs {
    source: String,
    args: Vec<String>,
}

impl CustomArgs {
    /// The arguments in order, one element per shell word.
    pub fn as_slice(&self) -> &[String] {
        &self.args
    }
}

impl FromStr for CustomArgs {
    type Err = CustomArgsError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let args = shlex::split(source).ok_or(CustomArgsError::Unparseable)?;
        args.iter().try_for_each(|arg| check_arg(arg))?;
        Ok(Self {
            source: source.to_owned(),
            args,
        })
    }
}

impl TryFrom<String> for CustomArgs {
    type Error = CustomArgsError;

    fn try_from(source: String) -> Result<Self, Self::Error> {
        source.parse()
    }
}

impl From<CustomArgs> for String {
    fn from(custom_args: CustomArgs) -> Self {
        custom_args.source
    }
}

fn check_arg(arg: &str) -> Result<(), CustomArgsError> {
    if arg.chars().any(char::is_control) {
        return Err(CustomArgsError::ControlCharacter(arg.to_owned()));
    }
    let flag = arg.split_once('=').map_or(arg, |(flag, _)| flag);
    if MANAGED_FLAGS.contains(&flag) {
        Err(CustomArgsError::Managed(flag.to_owned()))
    } else if EXPOSING_FLAGS.contains(&flag) {
        Err(CustomArgsError::Exposing(flag.to_owned()))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;

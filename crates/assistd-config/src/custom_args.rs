//! Extra command-line arguments handed to a model server, split with POSIX
//! shell quoting but never run through a shell.

use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::errors::CustomArgsError;

/// Flags assistd sets on every server it launches, or whose server defaults
/// its clients and supervisor rely on.
const SHARED_MANAGED_FLAGS: &[&str] = &[
    "--host",
    "--port",
    "-ngl",
    "--gpu-layers",
    "--n-gpu-layers",
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
    "--slots",
    "--no-slots",
    "--ssl-key-file",
    "--ssl-cert-file",
];

/// Flags that let anything holding the server's key read files,
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

/// A server assistd launches, naming the flags it manages beyond the shared ones.
pub trait ServerKind: Debug + Clone + Default + PartialEq {
    const MANAGED_FLAGS: &'static [&'static str];
}

/// The chat model server.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatServer;

impl ServerKind for ChatServer {
    const MANAGED_FLAGS: &'static [&'static str] = &[
        "-c",
        "--ctx-size",
        "--jinja",
        "--no-jinja",
        "--embedding",
        "--embeddings",
    ];
}

/// The embedding model server.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbeddingServer;

impl ServerKind for EmbeddingServer {
    const MANAGED_FLAGS: &'static [&'static str] = &["--pooling"];
}

/// Server arguments written as one string, e.g. `--flash-attn on -ot 'exps=CPU'`.
/// No shell runs, so `$VAR`, `~` and globs stay literal; a word starting with
/// `#` begins a comment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String", bound = "S: ServerKind")]
pub struct CustomArgs<S> {
    source: String,
    args: Vec<String>,
    server: PhantomData<S>,
}

impl<S> CustomArgs<S> {
    /// The arguments in order, one element per shell word.
    pub fn as_slice(&self) -> &[String] {
        &self.args
    }
}

impl<S: ServerKind> FromStr for CustomArgs<S> {
    type Err = CustomArgsError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let args = shlex::split(source).ok_or(CustomArgsError::Unparseable)?;
        args.iter()
            .try_for_each(|arg| check_arg(arg, S::MANAGED_FLAGS))?;
        Ok(Self {
            source: source.to_owned(),
            args,
            server: PhantomData,
        })
    }
}

impl<S: ServerKind> TryFrom<String> for CustomArgs<S> {
    type Error = CustomArgsError;

    fn try_from(source: String) -> Result<Self, Self::Error> {
        source.parse()
    }
}

impl<S> From<CustomArgs<S>> for String {
    fn from(custom_args: CustomArgs<S>) -> Self {
        custom_args.source
    }
}

fn check_arg(arg: &str, server_managed_flags: &[&str]) -> Result<(), CustomArgsError> {
    if arg.chars().any(char::is_control) {
        return Err(CustomArgsError::ControlCharacter(arg.to_owned()));
    }
    let flag = arg.split_once('=').map_or(arg, |(flag, _)| flag);
    if SHARED_MANAGED_FLAGS.contains(&flag) || server_managed_flags.contains(&flag) {
        Err(CustomArgsError::Managed(flag.to_owned()))
    } else if EXPOSING_FLAGS.contains(&flag) {
        Err(CustomArgsError::Exposing(flag.to_owned()))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;

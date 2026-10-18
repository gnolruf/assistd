//! Configuration for the local SearXNG instance web search goes through.

use std::net::IpAddr;
use std::num::{NonZeroU16, NonZeroU64};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_SEARCH_COMMAND, DEFAULT_SEARCH_ENABLED, DEFAULT_SEARCH_HOST, DEFAULT_SEARCH_PORT,
    DEFAULT_SEARCH_READY_TIMEOUT_SECS,
};
use crate::errors::SearchCommandError;

/// A daemon-supervised SearXNG instance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SearchConfig {
    /// When `false`, no instance starts and web search is unavailable.
    pub enabled: bool,
    /// The entry point to run. It must read its settings from
    /// `SEARXNG_SETTINGS_PATH` and the `SEARXNG_*` variables assistd sets
    /// (every SearXNG build does), serve on `host:port` in the foreground
    /// without daemonising, and answer `GET /healthz` with 200. uWSGI,
    /// Granian and containers need a wrapper script that forwards those
    /// variables into their own flags.
    pub command: SearchCommand,
    /// Bind host. Must be loopback: the instance has no authentication.
    pub host: IpAddr,
    /// Bind port. Must differ from `model.port` and `embedding.port`.
    pub port: NonZeroU16,
    /// Longest, in seconds, a freshly spawned instance may take to answer
    /// its health check before the attempt counts as failed.
    pub ready_timeout_secs: NonZeroU64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_SEARCH_ENABLED,
            command: DEFAULT_SEARCH_COMMAND
                .parse()
                .expect("the default search command parses"),
            host: DEFAULT_SEARCH_HOST,
            port: DEFAULT_SEARCH_PORT,
            ready_timeout_secs: DEFAULT_SEARCH_READY_TIMEOUT_SECS,
        }
    }
}

/// A program and its arguments written as one string, e.g.
/// `/opt/searxng/bin/python -m searx.webapp`. Split with POSIX shell quoting
/// but never run through a shell, so `$VAR`, `~` and globs stay literal.
/// Always names a program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SearchCommand {
    source: String,
    program: String,
    args: Vec<String>,
}

impl SearchCommand {
    /// The program to run: a path, or a name looked up on `$PATH`.
    pub fn program(&self) -> &str {
        &self.program
    }

    /// The arguments after the program, one element per shell word.
    pub fn args(&self) -> &[String] {
        &self.args
    }
}

impl FromStr for SearchCommand {
    type Err = SearchCommandError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let mut words = shlex::split(source)
            .ok_or(SearchCommandError::Unparseable)?
            .into_iter();
        let program = words.next().ok_or(SearchCommandError::Empty)?;
        Ok(Self {
            source: source.to_owned(),
            program,
            args: words.collect(),
        })
    }
}

impl TryFrom<String> for SearchCommand {
    type Error = SearchCommandError;

    fn try_from(source: String) -> Result<Self, Self::Error> {
        source.parse()
    }
}

impl From<SearchCommand> for String {
    fn from(command: SearchCommand) -> Self {
        command.source
    }
}

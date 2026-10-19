//! Configuration for the local SearXNG instance web search goes through.

use std::net::IpAddr;
use std::num::{NonZeroU16, NonZeroU64};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_SEARCH_COMMAND, DEFAULT_SEARCH_ENABLED, DEFAULT_SEARCH_ENGINES, DEFAULT_SEARCH_HOST,
    DEFAULT_SEARCH_PORT, DEFAULT_SEARCH_READY_TIMEOUT_SECS,
};
use crate::errors::SearchCommandError;

/// A daemon-supervised SearXNG instance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SearchConfig {
    /// When `false`, no instance starts and web search is unavailable.
    pub enabled: bool,
    /// The entry point to run. It must honour the `SEARXNG_*` variables assistd
    /// sets, serve `host:port` in the foreground, and answer `GET /healthz`
    /// with 200; uWSGI, Granian and containers need a wrapper that does so.
    pub command: SearchCommand,
    /// Bind host. Must be loopback: the instance has no authentication.
    pub host: IpAddr,
    /// Bind port. Must differ from `model.port` and `embedding.port`.
    pub port: NonZeroU16,
    /// Longest, in seconds, a freshly spawned instance may take to answer
    /// its health check before the attempt counts as failed.
    pub ready_timeout_secs: NonZeroU64,
    /// SearXNG engines to search with; only these load, each enabled even if
    /// SearXNG ships it disabled or inactive. Must not be empty; a name
    /// SearXNG does not know is logged and skipped.
    pub engines: Vec<SearchEngine>,
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
            engines: DEFAULT_SEARCH_ENGINES
                .map(|name| SearchEngine::Named(name.to_owned()))
                .to_vec(),
        }
    }
}

/// One `search.engines` entry: an engine name as SearXNG's `settings.yml`
/// spells it, or a table naming an engine and the file holding its API key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SearchEngine {
    Named(String),
    Keyed {
        name: String,
        /// File whose contents, trimmed of surrounding whitespace, are the
        /// key; read at each launch so it never sits in this config.
        #[serde(deserialize_with = "crate::home_path::deserialize")]
        api_key_file: PathBuf,
    },
}

impl SearchEngine {
    pub fn name(&self) -> &str {
        match self {
            SearchEngine::Named(name) | SearchEngine::Keyed { name, .. } => name,
        }
    }

    pub fn api_key_file(&self) -> Option<&Path> {
        match self {
            SearchEngine::Named(_) => None,
            SearchEngine::Keyed { api_key_file, .. } => Some(api_key_file),
        }
    }
}

/// A program and its arguments as one string, e.g. `python -m searx.webapp`,
/// split with POSIX shell quoting but never run through a shell, so `$VAR`,
/// `~` and globs stay literal. Always names a program.
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

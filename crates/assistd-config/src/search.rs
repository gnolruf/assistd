//! Configuration for the local SearXNG instance web search goes through.

use std::net::{Ipv4Addr, SocketAddr};
use std::num::{NonZeroU16, NonZeroU64};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::defaults::{nz16, nz64};
use crate::errors::SearchCommandError;
use crate::settings::{DescribeSettings, describe_fields};
use crate::values::{NonBlankString, NonEmpty};

/// A daemon-supervised SearXNG instance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SearchConfig {
    /// When `false`, no instance starts and web search is unavailable.
    pub enabled: bool,
    /// The entry point to run. It must honour the `SEARXNG_*` variables assistd
    /// sets, serve `127.0.0.1:port` from its own process group in the foreground,
    /// and answer `GET /healthz` with 200. Containers cannot: their socket is
    /// held outside that group, so the instance never becomes ready.
    pub command: SearchCommand,
    /// Port the instance binds on `127.0.0.1`; loopback because it has no
    /// authentication. When enabled, must differ from `model.port` and from
    /// an enabled embedding server's `embedding.port`.
    pub port: NonZeroU16,
    /// Longest, in seconds, a freshly spawned instance may take to answer
    /// its health check before the attempt counts as failed.
    pub ready_timeout_secs: NonZeroU64,
    /// SearXNG engines to search with; only these load, each enabled even if
    /// SearXNG ships it disabled or inactive. A name SearXNG does not know
    /// is logged and skipped. The default two answer automated queries from
    /// a home connection; most others meet them with CAPTCHAs or refusals.
    pub engines: NonEmpty<SearchEngine>,
}

impl DescribeSettings for SearchConfig {
    describe_fields! {
        enabled => ("enabled", "Enables web search through a local SearXNG instance"),
        command => (
            "SearXNG command",
            "SearXNG entry point; must serve the port below in the foreground"
        ),
        port => (
            "server port",
            "Loopback port SearXNG binds; must differ from the model and embedding ports",
            advanced
        ),
        ready_timeout_secs => (
            "startup timeout (s)",
            "Longest wait for SearXNG to pass its health check",
            advanced
        ),
        engines => ("search engines", "SearXNG engines to search with; must not be empty"),
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            command: "searxng-run"
                .parse()
                .expect("the default search command parses"),
            port: nz16(8387),
            ready_timeout_secs: nz64(60),
            engines: ["bing", "yahoo"]
                .map(default_engine)
                .to_vec()
                .try_into()
                .expect("the default search engines are not empty"),
        }
    }
}

impl SearchConfig {
    /// Where the instance listens: `port` on `127.0.0.1`.
    pub fn listen_addr(&self) -> SocketAddr {
        SocketAddr::from((Ipv4Addr::LOCALHOST, self.port.get()))
    }
}

/// One `search.engines` entry: an engine name as SearXNG's `settings.yml`
/// spells it, or a table naming an engine and the file holding its API key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SearchEngine {
    Named(NonBlankString),
    Keyed {
        name: NonBlankString,
        /// File whose contents, trimmed of surrounding whitespace, are the
        /// key; read at each launch so it never sits in this config.
        #[serde(deserialize_with = "crate::home_path::deserialize")]
        api_key_file: PathBuf,
    },
}

impl SearchEngine {
    pub fn name(&self) -> &str {
        match self {
            SearchEngine::Named(name) | SearchEngine::Keyed { name, .. } => name.as_str(),
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

fn default_engine(name: &str) -> SearchEngine {
    SearchEngine::Named(
        name.parse()
            .expect("the default engine names are not blank"),
    )
}

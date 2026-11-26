//! Sub-agent definitions: the ones assistd ships, and one per `<name>.toml` file in the
//! `agents` directory beside the config.

use std::fs;
use std::io::{self, Read};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use assistd_utils::fs::open_regular;
use assistd_utils::net::normalized_host_name;

use crate::errors::{AgentFileError, AgentFileIssue};
use crate::mcp::McpServerName;
use crate::top::Config;

/// Most tool-calling steps one turn of any agent may take.
pub const MAX_TOOL_STEPS: u32 = 200;

/// Longest agent name, in bytes.
pub const MAX_AGENT_NAME_BYTES: usize = 48;

/// Largest agent file read, in bytes.
pub const MAX_AGENT_FILE_BYTES: u64 = 64 * 1024;

const AGENTS_DIR: &str = "agents";
const AGENT_FILE_EXTENSION: &str = "toml";

/// What a relay's `agent` names: a lowercase ASCII letter, then lowercase letters, digits
/// or `_`, at most [`MAX_AGENT_NAME_BYTES`] in all.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AgentName(String);

impl AgentName {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn parse(name: &str) -> Option<Self> {
        let mut bytes = name.bytes();
        let valid = name.len() <= MAX_AGENT_NAME_BYTES
            && bytes.next().is_some_and(|first| first.is_ascii_lowercase())
            && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_');
        valid.then(|| Self(name.to_owned()))
    }
}

/// One kind of sub-agent a relay can run a task as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSpec {
    pub name: AgentName,
    pub origin: AgentOrigin,
    /// When to pick it, as the calling model is told.
    pub description: String,
    /// Its own instructions, which follow the part of its system prompt the daemon writes.
    pub prompt: String,
    pub capabilities: AgentCapabilities,
    /// The only commands its `run` offers, out of those its capabilities give it; `None`
    /// offers them all.
    pub commands: Option<Vec<AgentCommand>>,
    /// Web hosts it is handed on every run; valid host names, empty unless it reaches the
    /// web.
    pub sites: Vec<String>,
    pub limits: AgentSpecLimits,
}

impl AgentSpec {
    /// The definitions assistd ships, whose names no file may take.
    pub fn built_ins() -> Vec<Self> {
        vec![
            Self::built_in(
                "file_review",
                "For documents, logs and code too large to read here.",
                "You review local files handed to you: documents, logs and code.",
                AgentCapabilities {
                    files: true,
                    ..AgentCapabilities::default()
                },
            ),
            Self::built_in(
                "web_research",
                "For research and reading web pages.",
                "You research on the web: search, read the pages that answer the task, and \
                 name the URLs your report relies on.",
                AgentCapabilities {
                    web: true,
                    ..AgentCapabilities::default()
                },
            ),
            Self::built_in(
                "file_web_research",
                "For a task that needs local files and the web together, such as checking a \
                 project's dependencies for advisories or a config against its upstream \
                 documentation.",
                "You work from local files handed to you and from the web: read what the task \
                 needs from each, and name the paths and URLs your report relies on.",
                AgentCapabilities {
                    files: true,
                    web: true,
                    ..AgentCapabilities::default()
                },
            ),
        ]
    }

    fn built_in(
        name: &str,
        description: &str,
        prompt: &str,
        capabilities: AgentCapabilities,
    ) -> Self {
        Self {
            name: AgentName(name.to_owned()),
            origin: AgentOrigin::BuiltIn,
            description: description.to_owned(),
            prompt: prompt.to_owned(),
            capabilities,
            commands: None,
            sites: Vec::new(),
            limits: AgentSpecLimits::default(),
        }
    }

    /// The definition the file at `path` holds, named for the file. Errors when the file
    /// cannot be read or breaks a rule of [`AgentFileIssue`] under `config`.
    fn from_file(path: &Path, config: &Config) -> Result<Self, AgentFileIssue> {
        let name = agent_name(path)?;
        let text = read_capped(path)?;
        let raw: RawAgentFile = toml::from_str(&text).map_err(|error| AgentFileIssue::Parse {
            line: error_line(&text, &error),
            message: error.message().to_owned(),
        })?;
        let description = raw.description.trim().to_owned();
        if description.is_empty() {
            return Err(AgentFileIssue::BlankDescription);
        }
        let capabilities = AgentCapabilities::parse(&raw.capabilities, raw.mcp_servers, config)?;
        let sites = parse_sites(&raw.sites)?;
        if !sites.is_empty() && !capabilities.web {
            return Err(AgentFileIssue::SitesWithoutWeb);
        }
        Ok(Self {
            name,
            origin: AgentOrigin::File(path.to_path_buf()),
            description,
            prompt: raw.prompt.trim().to_owned(),
            capabilities,
            commands: raw.commands.as_deref().map(parse_commands).transpose()?,
            sites,
            limits: AgentSpecLimits {
                tool_steps: raw.tool_steps,
                reply_tokens: raw.reply_tokens,
                thinking_tokens: raw.thinking_tokens,
            }
            .within(config)?,
        })
    }
}

/// Where a definition comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentOrigin {
    BuiltIn,
    File(PathBuf),
}

/// What a sub-agent may reach beyond a directory of its own. The default is nothing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AgentCapabilities {
    /// Read the inputs it is handed.
    pub files: bool,
    /// Search the web and fetch pages.
    pub web: bool,
    /// Query the document index, saved memories and past sessions, read-only.
    pub index: bool,
    /// The MCP servers whose tools it may call.
    pub mcp_servers: Vec<McpServerName>,
}

impl AgentCapabilities {
    fn parse(
        capabilities: &[String],
        mcp_servers: Vec<String>,
        config: &Config,
    ) -> Result<Self, AgentFileIssue> {
        let mut parsed = Self::default();
        for capability in capabilities {
            match capability.as_str() {
                "files" => parsed.files = true,
                "web" => parsed.web = true,
                "index" => parsed.index = true,
                unknown => return Err(AgentFileIssue::UnknownCapability(unknown.to_owned())),
            }
        }
        parsed.mcp_servers = mcp_servers
            .into_iter()
            .map(|name| {
                config
                    .mcp
                    .servers
                    .iter()
                    .map(|server| &server.name)
                    .find(|server| server.as_str() == name)
                    .cloned()
                    .ok_or(AgentFileIssue::UnknownMcpServer(name))
            })
            .collect::<Result<_, _>>()?;
        Ok(parsed)
    }

    /// The names a file's `capabilities` lists for these.
    pub fn names(&self) -> Vec<&'static str> {
        [
            (self.files, "files"),
            (self.web, "web"),
            (self.index, "index"),
        ]
        .into_iter()
        .filter_map(|(held, name)| held.then_some(name))
        .collect()
    }
}

/// A command a sub-agent's `run` can offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentCommand {
    Bash,
    Cat,
    Docs,
    Echo,
    Grep,
    Head,
    Ls,
    Search,
    Sort,
    Tail,
    Uniq,
    Wc,
    Web,
    Write,
}

impl AgentCommand {
    pub const ALL: [Self; 14] = [
        Self::Bash,
        Self::Cat,
        Self::Docs,
        Self::Echo,
        Self::Grep,
        Self::Head,
        Self::Ls,
        Self::Search,
        Self::Sort,
        Self::Tail,
        Self::Uniq,
        Self::Wc,
        Self::Web,
        Self::Write,
    ];

    /// The name the command is run by.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bash => "bash",
            Self::Cat => "cat",
            Self::Docs => "docs",
            Self::Echo => "echo",
            Self::Grep => "grep",
            Self::Head => "head",
            Self::Ls => "ls",
            Self::Search => "search",
            Self::Sort => "sort",
            Self::Tail => "tail",
            Self::Uniq => "uniq",
            Self::Wc => "wc",
            Self::Web => "web",
            Self::Write => "write",
        }
    }
}

/// The limits a definition sets below the daemon's own; `None` keeps the daemon's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AgentSpecLimits {
    /// Tool-calling steps one run may take, at most [`MAX_TOOL_STEPS`].
    pub tool_steps: Option<NonZeroU32>,
    /// Tokens per model response, thinking included, at most `chat.max_response_tokens`.
    pub reply_tokens: Option<NonZeroU32>,
    /// Tokens the model may think for in one response, at most `chat.max_reasoning_tokens`
    /// and below the reply limit in force.
    pub thinking_tokens: Option<NonZeroU32>,
}

impl AgentSpecLimits {
    /// These limits when none is above the daemon's under `config`.
    fn within(self, config: &Config) -> Result<Self, AgentFileIssue> {
        let chat = &config.chat;
        let checked = [
            ("tool_steps", self.tool_steps, MAX_TOOL_STEPS),
            (
                "reply_tokens",
                self.reply_tokens,
                chat.max_response_tokens.get(),
            ),
            (
                "thinking_tokens",
                self.thinking_tokens,
                chat.max_reasoning_tokens.get(),
            ),
        ];
        for (field, value, max) in checked {
            if let Some(value) = value.map(NonZeroU32::get).filter(|value| *value > max) {
                return Err(AgentFileIssue::LimitAboveDaemon { field, value, max });
            }
        }
        let reply = self.reply_tokens.unwrap_or(chat.max_response_tokens).get();
        let thinking = self
            .thinking_tokens
            .unwrap_or(chat.max_reasoning_tokens)
            .get();
        if thinking >= reply {
            return Err(AgentFileIssue::ThinkingNotBelowReply { thinking, reply });
        }
        Ok(self)
    }
}

/// The sub-agents a daemon offers: the built-in definitions, then each file that was
/// accepted, by name; and why each other file was refused.
#[derive(Debug)]
pub struct AgentCatalog {
    pub agents: Vec<AgentSpec>,
    pub refused: Vec<AgentFileError>,
}

impl AgentCatalog {
    /// The `agents` directory beside the config file at `config_path`.
    pub fn dir_beside(config_path: &Path) -> PathBuf {
        config_path
            .parent()
            .unwrap_or(Path::new(""))
            .join(AGENTS_DIR)
    }

    /// The built-ins and the `.toml` files of `dir`, each checked against `config`. A
    /// missing directory holds no files; one that cannot be listed is itself refused.
    pub fn load(dir: &Path, config: &Config) -> Self {
        let mut catalog = Self {
            agents: AgentSpec::built_ins(),
            refused: Vec::new(),
        };
        let files = match agent_files(dir) {
            Ok(files) => files,
            Err(source) => {
                catalog.refused.push(AgentFileError {
                    path: dir.to_path_buf(),
                    issue: AgentFileIssue::Read(source),
                });
                return catalog;
            }
        };
        for path in files {
            match AgentSpec::from_file(&path, config) {
                Ok(agent) => catalog.agents.push(agent),
                Err(issue) => catalog.refused.push(AgentFileError { path, issue }),
            }
        }
        catalog
    }
}

/// An agent file as written.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgentFile {
    description: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    mcp_servers: Vec<String>,
    commands: Option<Vec<String>>,
    #[serde(default)]
    sites: Vec<String>,
    tool_steps: Option<NonZeroU32>,
    reply_tokens: Option<NonZeroU32>,
    thinking_tokens: Option<NonZeroU32>,
}

/// The `.toml` files directly in `dir`, sorted by path; none when `dir` does not exist.
fn agent_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut files = entries
        .map(|entry| entry.map(|entry| entry.path()))
        .filter(|path| {
            path.as_ref().map_or(true, |path| {
                path.extension()
                    .is_some_and(|extension| extension == AGENT_FILE_EXTENSION)
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    files.sort();
    Ok(files)
}

/// The agent the file at `path` defines, named by its file stem.
fn agent_name(path: &Path) -> Result<AgentName, AgentFileIssue> {
    let stem = path
        .file_stem()
        .map_or_default(|stem| stem.to_string_lossy().into_owned());
    let name = AgentName::parse(&stem)
        .ok_or_else(|| AgentFileIssue::InvalidName { name: stem.clone() })?;
    if AgentSpec::built_ins()
        .iter()
        .any(|built_in| built_in.name == name)
    {
        return Err(AgentFileIssue::BuiltInName(stem));
    }
    Ok(name)
}

/// The text of the regular file at `path`, refused past [`MAX_AGENT_FILE_BYTES`].
fn read_capped(path: &Path) -> Result<String, AgentFileIssue> {
    let (file, bytes) = open_regular(path).map_err(AgentFileIssue::Read)?;
    if bytes > MAX_AGENT_FILE_BYTES {
        return Err(AgentFileIssue::TooLarge { bytes });
    }
    let mut text = String::new();
    file.take(MAX_AGENT_FILE_BYTES)
        .read_to_string(&mut text)
        .map_err(AgentFileIssue::Read)?;
    Ok(text)
}

/// The line of `text` that `error` points at.
fn error_line(text: &str, error: &toml::de::Error) -> Option<usize> {
    error
        .span()
        .and_then(|span| text.get(..span.start))
        .map(|before| before.matches('\n').count() + 1)
}

fn parse_commands(names: &[String]) -> Result<Vec<AgentCommand>, AgentFileIssue> {
    names
        .iter()
        .map(|name| {
            AgentCommand::ALL
                .into_iter()
                .find(|command| command.as_str() == name)
                .ok_or_else(|| AgentFileIssue::UnknownCommand(name.clone()))
        })
        .collect()
}

fn parse_sites(sites: &[String]) -> Result<Vec<String>, AgentFileIssue> {
    sites
        .iter()
        .map(|site| {
            normalized_host_name(site).ok_or_else(|| AgentFileIssue::NotAHost(site.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::nz32;

    const REVIEWER: &str = "\
        description = \"For reviewing Rust changes against my checklist.\"\n\
        capabilities = [\"files\", \"web\"]\n\
        commands = [\"cat\", \"grep\", \"web\"]\n\
        sites = [\"Docs.RS\"]\n\
        tool_steps = 40\n\
        reply_tokens = 800\n\
        thinking_tokens = 300\n\
        prompt = '''\n\
        You review Rust changes.\n\
        Match paths with \\.rs$ only.\n\
        '''\n";

    /// What loading a directory holding `text` as `<name>.toml` gives for that file.
    fn loaded(name: &str, text: &str) -> Result<AgentSpec, AgentFileIssue> {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(format!("{name}.toml")), text).unwrap();
        fs::write(dir.path().join("README.md"), "not an agent").unwrap();
        let mut catalog = AgentCatalog::load(dir.path(), &Config::default());
        assert_eq!(
            catalog.agents.len() + catalog.refused.len(),
            AgentSpec::built_ins().len() + 1
        );
        match catalog.refused.pop() {
            Some(refused) => Err(refused.issue),
            None => Ok(catalog.agents.pop().unwrap()),
        }
    }

    fn with_keys(keys: &str) -> String {
        format!("description = \"d\"\n{keys}\n")
    }

    #[test]
    fn a_valid_file_loads_after_the_built_ins_with_every_field() {
        let agent = loaded("rust_review", REVIEWER).unwrap();

        assert_eq!(agent.name.as_str(), "rust_review");
        assert!(matches!(agent.origin, AgentOrigin::File(_)));
        assert_eq!(
            agent.description,
            "For reviewing Rust changes against my checklist."
        );
        assert_eq!(
            agent.prompt,
            "You review Rust changes.\nMatch paths with \\.rs$ only."
        );
        assert_eq!(agent.capabilities.names(), ["files", "web"]);
        assert_eq!(
            agent.commands,
            Some(vec![
                AgentCommand::Cat,
                AgentCommand::Grep,
                AgentCommand::Web
            ])
        );
        assert_eq!(agent.sites, ["docs.rs"]);
        assert_eq!(
            agent.limits,
            AgentSpecLimits {
                tool_steps: Some(nz32(40)),
                reply_tokens: Some(nz32(800)),
                thinking_tokens: Some(nz32(300)),
            }
        );
    }

    #[test]
    fn a_missing_directory_holds_only_the_built_ins() {
        let dir = tempfile::tempdir().unwrap();

        let catalog = AgentCatalog::load(&dir.path().join("agents"), &Config::default());

        assert_eq!(catalog.agents, AgentSpec::built_ins());
        assert!(catalog.refused.is_empty());
    }

    #[test]
    fn a_file_is_refused_for_its_name_or_its_shape() {
        let valid = with_keys("");
        assert!(matches!(
            loaded("Rust-Review", &valid),
            Err(AgentFileIssue::InvalidName { name }) if name == "Rust-Review"
        ));
        assert!(matches!(
            loaded("web_research", &valid),
            Err(AgentFileIssue::BuiltInName(name)) if name == "web_research"
        ));
        assert!(matches!(
            loaded("blank", "description = \" \"\n"),
            Err(AgentFileIssue::BlankDescription)
        ));
        let prose = loaded("prose", "You review Rust changes.\n");
        assert!(
            matches!(&prose, Err(AgentFileIssue::Parse { line: Some(1), .. })),
            "{prose:?}"
        );
        let mistyped = loaded("mistyped", &with_keys("capabilties = [\"web\"]"));
        assert!(
            matches!(
                &mistyped,
                Err(AgentFileIssue::Parse { line: Some(2), message })
                    if message.contains("capabilties")
            ),
            "{mistyped:?}"
        );
    }

    #[test]
    fn a_file_is_refused_for_what_it_names_that_does_not_exist() {
        let refused = |keys: &str| loaded("agent", &with_keys(keys));
        assert!(matches!(
            refused("capabilities = [\"files\", \"shell\"]"),
            Err(AgentFileIssue::UnknownCapability(name)) if name == "shell"
        ));
        assert!(matches!(
            refused("commands = [\"cat\", \"curl\"]"),
            Err(AgentFileIssue::UnknownCommand(name)) if name == "curl"
        ));
        assert!(matches!(
            refused("mcp_servers = [\"github\"]"),
            Err(AgentFileIssue::UnknownMcpServer(name)) if name == "github"
        ));
        assert!(matches!(
            refused("capabilities = [\"web\"]\nsites = [\"https://docs.rs/\"]"),
            Err(AgentFileIssue::NotAHost(site)) if site == "https://docs.rs/"
        ));
        assert!(matches!(
            refused("capabilities = [\"files\"]\nsites = [\"docs.rs\"]"),
            Err(AgentFileIssue::SitesWithoutWeb)
        ));
    }

    #[test]
    fn a_file_is_refused_for_a_limit_above_the_daemons() {
        let chat = Config::default().chat;
        let (reply, thinking) = (
            chat.max_response_tokens.get(),
            chat.max_reasoning_tokens.get(),
        );
        let refused = |limits: &str| loaded("agent", &with_keys(limits));
        assert!(matches!(
            refused("tool_steps = 201"),
            Err(AgentFileIssue::LimitAboveDaemon {
                field: "tool_steps",
                value: 201,
                max: 200
            })
        ));
        assert!(matches!(
            refused(&format!("reply_tokens = {}", reply + 1)),
            Err(AgentFileIssue::LimitAboveDaemon { field: "reply_tokens", max, .. })
                if max == reply
        ));
        assert!(matches!(
            refused(&format!("thinking_tokens = {}", thinking + 1)),
            Err(AgentFileIssue::LimitAboveDaemon { field: "thinking_tokens", max, .. })
                if max == thinking
        ));
        assert!(matches!(
            refused(&format!("reply_tokens = {thinking}")),
            Err(AgentFileIssue::ThinkingNotBelowReply { .. })
        ));
        assert!(refused("tool_steps = 200").is_ok());
    }
}

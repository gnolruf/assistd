//! MCP (Model Context Protocol) client configuration.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::str::FromStr;

use http::header::AUTHORIZATION;
use http::{HeaderMap, HeaderName, HeaderValue};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use thiserror::Error;
use url::{Host, Url};

use crate::defaults::{DEFAULT_MCP_ENABLED, DEFAULT_MCP_REQUEST_TIMEOUT_SECS};
use crate::errors::McpServerNameError;
use crate::settings::{DescribeSettings, describe_fields};

/// Model Context Protocol client settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct McpConfig {
    /// When `false`, no server in `servers` is connected.
    pub enabled: bool,
    pub servers: Vec<McpServerConfig>,
}

impl DescribeSettings for McpConfig {
    describe_fields! {
        enabled => ("enabled", "Connects the MCP servers listed below"),
        servers => ("servers", "MCP servers to run or reach by URL; edit in config.toml"),
    }
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_MCP_ENABLED,
            servers: Vec::new(),
        }
    }
}

/// One `[[mcp.servers]]` entry. A `command` key makes it a stdio server and
/// a `url` key a Streamable HTTP one; an entry with both, neither, or keys
/// of the other transport fails to parse.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct McpServerConfig {
    /// Label in tool names (`mcp__<name>__<tool>`) and logs; must be unique.
    pub name: McpServerName,
    #[serde(flatten)]
    pub transport: McpTransport,
    /// Timeout in seconds for the initialize handshake and each request.
    pub request_timeout_secs: NonZeroU64,
}

/// A server's label in tool names `mcp__<name>__<tool>`: non-empty ASCII letters, digits, `_`
/// or `-`, with no `__` and no trailing `_`, so each tool name splits into server and tool
/// only one way.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct McpServerName(String);

/// How the daemon reaches an MCP server.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum McpTransport {
    Stdio(McpStdioTransport),
    Http(McpHttpTransport),
}

/// A server run as a child process and spoken to with newline-delimited
/// JSON-RPC over its stdin/stdout. `Debug` gives the arg count and env var
/// names, never their values.
#[derive(Clone, Serialize, PartialEq, Eq)]
pub struct McpStdioTransport {
    /// Command to spawn; a bare name is resolved via `$PATH`.
    pub command: PathBuf,
    /// Arguments passed to `command`. Logs show only their count; still,
    /// put tokens and keys in `env` rather than here.
    pub args: Vec<String>,
    /// Environment variables for the child, which inherits from the daemon
    /// only `HOME`, `LANG`, `LANGUAGE`, `LC_*`, `LOGNAME`, `PATH`, `SHELL`,
    /// `TERM`, `TMPDIR`, `TZ` and `USER`. Set credentials here, and keep the
    /// config file mode `0600`.
    pub env: HashMap<String, String>,
}

/// A remote server spoken to over MCP Streamable HTTP.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct McpHttpTransport {
    pub url: McpServerUrl,
    /// Static headers sent with every request, e.g. `Authorization` for a
    /// server that takes an API key. Keep the config file mode `0600`.
    pub headers: McpHeaders,
    #[serde(flatten)]
    pub auth: McpAuth,
}

/// An MCP endpoint URL: `https`, or `http` to a loopback host (`127.0.0.0/8`,
/// `::1` or `localhost`), with no user name or password.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerUrl(Url);

/// Request headers whose names and values are valid HTTP. `Debug` lists the
/// header names only.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct McpHeaders(BTreeMap<String, (HeaderName, HeaderValue)>);

/// How an HTTP server authenticates the daemon beyond `headers`: `auth =
/// "none"` sends `headers` alone, `auth = "oauth"` adds an OAuth 2.1 bearer
/// token, configured by the entry's `client_id`, `client_secret` and `scopes`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum McpAuth {
    #[default]
    None,
    OAuth(McpOAuthClient),
}

/// The OAuth client an `auth = "oauth"` server is authorized as. `Debug`
/// never shows the secret.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct McpOAuthClient {
    /// A client ID registered with the server's authorization server, as
    /// Google requires; unset registers one dynamically, which fails when
    /// the authorization server does not offer registration.
    pub client_id: Option<String>,
    /// The secret issued with `client_id`, for a confidential client.
    pub client_secret: Option<String>,
    /// Scopes to request; empty takes the ones the server advertises.
    pub scopes: Vec<String>,
}

/// The `auth` key as written.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum McpAuthKind {
    None,
    #[serde(rename = "oauth")]
    OAuth,
}

/// A `[[mcp.servers]]` entry as written, before its transport is decided.
#[derive(Deserialize)]
struct RawMcpServerConfig {
    name: McpServerName,
    command: Option<PathBuf>,
    args: Option<Vec<String>>,
    env: Option<HashMap<String, String>>,
    url: Option<McpServerUrl>,
    headers: Option<McpHeaders>,
    auth: Option<McpAuthKind>,
    client_id: Option<String>,
    client_secret: Option<String>,
    scopes: Option<Vec<String>>,
    #[serde(default = "default_request_timeout_secs")]
    request_timeout_secs: NonZeroU64,
}

/// The OAuth keys of a `[[mcp.servers]]` entry as written.
struct RawOAuthKeys {
    client_id: Option<String>,
    client_secret: Option<String>,
    scopes: Option<Vec<String>>,
}

/// Why a `[[mcp.servers]]` entry or one of its values failed to parse.
/// Never carries a header value or URL credentials.
#[derive(Debug, Error)]
enum McpEntryError {
    #[error(
        "mcp server '{0}' sets both `command` and `url`; use `command` for a \
         stdio server or `url` for a Streamable HTTP one"
    )]
    BothTransports(McpServerName),
    #[error("mcp server '{0}' needs `command` (stdio) or `url` (Streamable HTTP)")]
    NoTransport(McpServerName),
    #[error(
        "mcp server '{0}' runs `command`, so `headers`, `auth`, `client_id`, \
         `client_secret` and `scopes` do not apply"
    )]
    HttpKeysOnStdio(McpServerName),
    #[error("mcp server '{0}' connects to `url`, so `args` and `env` do not apply")]
    StdioKeysOnHttp(McpServerName),
    #[error(
        "mcp server '{0}' sets `client_id`, `client_secret` or `scopes` without `auth = \"oauth\"`"
    )]
    OAuthKeysWithoutOAuth(McpServerName),
    #[error("mcp server '{0}' sets `client_secret` without the `client_id` it was issued with")]
    SecretWithoutClientId(McpServerName),
    #[error(
        "mcp server '{0}' uses `auth = \"oauth\"`, which sends its own `Authorization` header; \
         remove the one in `headers`"
    )]
    AuthorizationHeaderWithOAuth(McpServerName),
    #[error("invalid mcp server url: {0}")]
    UrlSyntax(#[from] url::ParseError),
    #[error("mcp server url must not carry a user name or password; put credentials in `headers`")]
    UrlUserinfo,
    #[error("mcp server url must use https (or http to a loopback host), got `{0}`")]
    UrlScheme(String),
    #[error("mcp server url may use http only to a loopback host; use https for {0}")]
    PlainHttpToRemote(String),
    #[error("invalid mcp header name `{0}`")]
    HeaderName(String),
    #[error("invalid value for mcp header `{0}`")]
    HeaderValue(String),
    #[error("mcp header `{0}` is set more than once (header names ignore case)")]
    DuplicateHeader(String),
}

impl<'de> Deserialize<'de> for McpServerConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        RawMcpServerConfig::deserialize(deserializer)?
            .into_config()
            .map_err(de::Error::custom)
    }
}

impl RawMcpServerConfig {
    fn into_config(self) -> Result<McpServerConfig, McpEntryError> {
        let Self {
            name,
            command,
            args,
            env,
            url,
            headers,
            auth,
            client_id,
            client_secret,
            scopes,
            request_timeout_secs,
        } = self;
        let oauth_keys = RawOAuthKeys {
            client_id,
            client_secret,
            scopes,
        };
        let has_stdio_keys = args.is_some() || env.is_some();
        let has_http_keys = headers.is_some() || auth.is_some() || oauth_keys.any_set();
        let transport = match (command, url) {
            (Some(command), None) if !has_http_keys => McpTransport::Stdio(McpStdioTransport {
                command,
                args: args.unwrap_or_default(),
                env: env.unwrap_or_default(),
            }),
            (None, Some(url)) if !has_stdio_keys => McpTransport::Http(McpHttpTransport::new(
                &name,
                url,
                headers.unwrap_or_default(),
                auth,
                oauth_keys,
            )?),
            (Some(_), Some(_)) => return Err(McpEntryError::BothTransports(name)),
            (None, None) => return Err(McpEntryError::NoTransport(name)),
            (Some(_), None) => return Err(McpEntryError::HttpKeysOnStdio(name)),
            (None, Some(_)) => return Err(McpEntryError::StdioKeysOnHttp(name)),
        };
        Ok(McpServerConfig {
            name,
            transport,
            request_timeout_secs,
        })
    }
}

impl RawOAuthKeys {
    fn any_set(&self) -> bool {
        self.client_id.is_some() || self.client_secret.is_some() || self.scopes.is_some()
    }
}

impl McpServerName {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for McpServerName {
    type Err = McpServerNameError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::try_from(name.to_owned())
    }
}

impl TryFrom<String> for McpServerName {
    type Error = McpServerNameError;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        let unambiguous = !name.is_empty()
            && name.chars().all(is_server_name_char)
            && !name.contains("__")
            && !name.ends_with('_');
        if unambiguous {
            Ok(Self(name))
        } else {
            Err(McpServerNameError(name))
        }
    }
}

impl From<McpServerName> for String {
    fn from(name: McpServerName) -> Self {
        name.0
    }
}

impl fmt::Display for McpServerName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl McpTransport {
    /// Whether the entry holds credentials: stdio `env`, HTTP `headers` or
    /// an OAuth `client_secret`.
    pub fn sets_secrets(&self) -> bool {
        match self {
            Self::Stdio(stdio) => !stdio.env.is_empty(),
            Self::Http(http) => {
                !http.headers.0.is_empty()
                    || matches!(&http.auth, McpAuth::OAuth(oauth) if oauth.client_secret.is_some())
            }
        }
    }
}

impl McpHttpTransport {
    fn new(
        name: &McpServerName,
        url: McpServerUrl,
        headers: McpHeaders,
        auth: Option<McpAuthKind>,
        oauth_keys: RawOAuthKeys,
    ) -> Result<Self, McpEntryError> {
        let auth = match auth {
            Some(McpAuthKind::OAuth) => McpAuth::OAuth(McpOAuthClient::new(name, oauth_keys)?),
            _ if oauth_keys.any_set() => {
                return Err(McpEntryError::OAuthKeysWithoutOAuth(name.clone()));
            }
            _ => McpAuth::None,
        };
        if matches!(auth, McpAuth::OAuth(_)) && headers.sets(&AUTHORIZATION) {
            return Err(McpEntryError::AuthorizationHeaderWithOAuth(name.clone()));
        }
        Ok(Self { url, headers, auth })
    }
}

impl McpOAuthClient {
    fn new(name: &McpServerName, keys: RawOAuthKeys) -> Result<Self, McpEntryError> {
        if keys.client_secret.is_some() && keys.client_id.is_none() {
            return Err(McpEntryError::SecretWithoutClientId(name.clone()));
        }
        Ok(Self {
            client_id: keys.client_id,
            client_secret: keys.client_secret,
            scopes: keys.scopes.unwrap_or_default(),
        })
    }
}

impl fmt::Debug for McpOAuthClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpOAuthClient")
            .field("client_id", &self.client_id)
            .field(
                "client_secret",
                &self.client_secret.as_ref().map(|_| "[REDACTED]"),
            )
            .field("scopes", &self.scopes)
            .finish()
    }
}

impl Serialize for McpAuth {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        let Self::OAuth(oauth) = self else {
            map.serialize_entry("auth", "none")?;
            return map.end();
        };
        map.serialize_entry("auth", "oauth")?;
        if let Some(client_id) = &oauth.client_id {
            map.serialize_entry("client_id", client_id)?;
        }
        if let Some(client_secret) = &oauth.client_secret {
            map.serialize_entry("client_secret", client_secret)?;
        }
        map.serialize_entry("scopes", &oauth.scopes)?;
        map.end()
    }
}

impl fmt::Debug for McpStdioTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpStdioTransport")
            .field("command", &self.command)
            .field("arg_count", &self.args.len())
            .field("env_names", &self.env.keys().collect::<BTreeSet<_>>())
            .finish()
    }
}

impl McpServerUrl {
    pub fn as_url(&self) -> &Url {
        &self.0
    }

    fn parse(raw: &str) -> Result<Self, McpEntryError> {
        let url = Url::parse(raw)?;
        if !url.username().is_empty() || url.password().is_some() {
            return Err(McpEntryError::UrlUserinfo);
        }
        match url.scheme() {
            "https" => Ok(Self(url)),
            "http" if url.host().as_ref().is_some_and(is_loopback_host) => Ok(Self(url)),
            "http" => Err(McpEntryError::PlainHttpToRemote(
                url.host_str().unwrap_or_default().to_owned(),
            )),
            scheme => Err(McpEntryError::UrlScheme(scheme.to_owned())),
        }
    }
}

impl Serialize for McpServerUrl {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0.as_str())
    }
}

impl<'de> Deserialize<'de> for McpServerUrl {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::parse(&String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

impl McpHeaders {
    pub fn to_header_map(&self) -> HeaderMap {
        self.0.values().cloned().collect()
    }

    fn sets(&self, header: &HeaderName) -> bool {
        self.0.values().any(|(name, _)| name == header)
    }

    fn parse(raw: HashMap<String, String>) -> Result<Self, McpEntryError> {
        let mut seen = HashSet::new();
        let mut headers = BTreeMap::new();
        for (configured, value) in raw {
            let name = HeaderName::from_bytes(configured.as_bytes())
                .map_err(|_| McpEntryError::HeaderName(configured.clone()))?;
            let value = HeaderValue::from_str(&value)
                .map_err(|_| McpEntryError::HeaderValue(configured.clone()))?;
            if !seen.insert(name.clone()) {
                return Err(McpEntryError::DuplicateHeader(configured));
            }
            headers.insert(configured, (name, value));
        }
        Ok(Self(headers))
    }
}

impl fmt::Debug for McpHeaders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.0.keys()).finish()
    }
}

impl Serialize for McpHeaders {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(configured, (_, value))| {
            (configured, String::from_utf8_lossy(value.as_bytes()))
        }))
    }
}

impl<'de> Deserialize<'de> for McpHeaders {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::parse(HashMap::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

fn is_server_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn is_loopback_host(host: &Host<&str>) -> bool {
    match host {
        Host::Ipv4(ip) => ip.is_loopback(),
        Host::Ipv6(ip) => ip.is_loopback(),
        Host::Domain(domain) => *domain == "localhost",
    }
}

fn default_request_timeout_secs() -> NonZeroU64 {
    DEFAULT_MCP_REQUEST_TIMEOUT_SECS
}

#[cfg(test)]
mod tests;

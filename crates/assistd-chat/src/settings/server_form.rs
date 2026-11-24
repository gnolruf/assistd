//! The form one MCP server is written in: its name, its transport, then the fields that
//! transport takes.

use super::super::text_editor::{EditorShape, TextEditor};
use super::form::{FieldValue, Form, FormFields};
use super::list_editor::entry_text;

/// A `[[mcp.servers]]` entry, its keys in the order they are written.
pub(crate) type ServerEntry = Vec<(String, toml::Value)>;

/// The form over one server's fields.
pub(crate) type ServerForm = Form<ServerFields>;

/// How the daemon reaches a server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transport {
    Stdio,
    Http,
}

/// One row of the form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServerField {
    Name,
    Transport,
    Command,
    Args,
    Env,
    Read,
    Write,
    Network,
    Url,
    Headers,
    Auth,
    ClientId,
    ClientSecret,
    Scopes,
    Timeout,
}

impl ServerField {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Name => "Name",
            Self::Transport => "Transport",
            Self::Command => "Command",
            Self::Args => "Arguments",
            Self::Env => "Environment",
            Self::Read => "Readable paths",
            Self::Write => "Writable paths",
            Self::Network => "Network",
            Self::Url => "URL",
            Self::Headers => "Headers",
            Self::Auth => "Auth",
            Self::ClientId => "Client ID",
            Self::ClientSecret => "Client secret",
            Self::Scopes => "Scopes",
            Self::Timeout => "Timeout (s)",
        }
    }

    /// One line on what the field holds, shown while it is focused.
    pub(crate) fn help(self) -> &'static str {
        match self {
            Self::Name => "the <name> in tool names mcp__<name>__<tool>: letters, digits, _ or -",
            Self::Transport => "stdio runs the server as a child process; HTTP reaches it by URL",
            Self::Command => "the program to run, under /usr or a path granted below",
            Self::Args => "arguments for the command; put tokens and keys in the environment",
            Self::Env => "NAME=value; the server inherits only PATH, locale and a few basics",
            Self::Read => "host paths the sandboxed server sees; by default only /usr and its home",
            Self::Write => "host paths the sandboxed server may also change",
            Self::Network => "off leaves the server no network but its own loopback",
            Self::Url => "https, or http to a loopback host",
            Self::Headers => "NAME=value headers sent with every request",
            Self::Auth => "oauth adds a bearer token; sign in with `assistd mcp login <name>`",
            Self::ClientId => {
                "a client registered with the server; empty registers one dynamically"
            }
            Self::ClientSecret => "the secret issued with the client ID, if it has one",
            Self::Scopes => "scopes to request; none takes the ones the server advertises",
            Self::Timeout => "seconds each request may take, 1 to 180; empty for the default",
        }
    }

    /// Whether the field's entries are written `NAME=value`.
    fn holds_pairs(self) -> bool {
        matches!(self, Self::Env | Self::Headers)
    }
}

/// A server being written: both transports' fields are kept, so switching loses nothing.
pub(crate) struct ServerFields {
    transport: Transport,
    oauth: bool,
    name: TextEditor,
    command: TextEditor,
    args: Vec<String>,
    /// `NAME=value` entries with distinct names, as are `headers`.
    env: Vec<String>,
    read: Vec<String>,
    write: Vec<String>,
    network: bool,
    url: TextEditor,
    headers: Vec<String>,
    client_id: TextEditor,
    client_secret: TextEditor,
    scopes: Vec<String>,
    timeout: TextEditor,
}

impl ServerFields {
    /// The fields of the `[[mcp.servers]]` entry `server`, or a blank stdio server.
    pub(super) fn new(server: Option<&toml::Value>) -> Self {
        let value = |key: &str| server.and_then(|server| server.get(key));
        let text =
            |key: &str| TextEditor::new(value(key).map_or_default(entry_text), EditorShape::Line);
        let list = |key: &str| -> Vec<String> {
            value(key)
                .and_then(toml::Value::as_array)
                .map_or_default(|entries| entries.iter().map(entry_text).collect())
        };
        let pairs = |key: &str| -> Vec<String> {
            value(key)
                .and_then(toml::Value::as_table)
                .map_or_default(|table| {
                    table
                        .iter()
                        .map(|(name, value)| format!("{name}={}", entry_text(value)))
                        .collect()
                })
        };
        Self {
            transport: if value("url").is_some() {
                Transport::Http
            } else {
                Transport::Stdio
            },
            oauth: value("auth").and_then(toml::Value::as_str) == Some("oauth"),
            name: text("name"),
            command: text("command"),
            args: list("args"),
            env: pairs("env"),
            read: list("read"),
            write: list("write"),
            network: value("network").and_then(toml::Value::as_bool) != Some(false),
            url: text("url"),
            headers: pairs("headers"),
            client_id: text("client_id"),
            client_secret: text("client_secret"),
            scopes: list("scopes"),
            timeout: text("request_timeout_secs"),
        }
    }
}

impl FormFields for ServerFields {
    type Field = ServerField;
    type Output = ServerEntry;

    /// The name and transport, the fields the transport takes, then the timeout.
    fn fields(&self) -> Vec<ServerField> {
        let transport: &[ServerField] = match (self.transport, self.oauth) {
            (Transport::Stdio, _) => &[
                ServerField::Command,
                ServerField::Args,
                ServerField::Env,
                ServerField::Read,
                ServerField::Write,
                ServerField::Network,
            ],
            (Transport::Http, false) => {
                &[ServerField::Url, ServerField::Headers, ServerField::Auth]
            }
            (Transport::Http, true) => &[
                ServerField::Url,
                ServerField::Headers,
                ServerField::Auth,
                ServerField::ClientId,
                ServerField::ClientSecret,
                ServerField::Scopes,
            ],
        };
        [ServerField::Name, ServerField::Transport]
            .into_iter()
            .chain(transport.iter().copied())
            .chain([ServerField::Timeout])
            .collect()
    }

    fn label(field: ServerField) -> &'static str {
        field.label()
    }

    /// A `NAME=value` list is summarised by its names alone, as its values may be credentials.
    fn value(&self, field: ServerField) -> FieldValue<'_> {
        match field {
            ServerField::Name => FieldValue::Text(&self.name),
            ServerField::Transport => FieldValue::Choice(match self.transport {
                Transport::Stdio => "stdio",
                Transport::Http => "HTTP",
            }),
            ServerField::Command => FieldValue::Text(&self.command),
            ServerField::Args => list_value(&self.args, self.args.join(" ")),
            ServerField::Env => list_value(&self.env, pair_names(&self.env)),
            ServerField::Read => list_value(&self.read, self.read.join(", ")),
            ServerField::Write => list_value(&self.write, self.write.join(", ")),
            ServerField::Network => FieldValue::Choice(if self.network { "on" } else { "off" }),
            ServerField::Url => FieldValue::Text(&self.url),
            ServerField::Headers => list_value(&self.headers, pair_names(&self.headers)),
            ServerField::Auth => FieldValue::Choice(if self.oauth { "oauth" } else { "none" }),
            ServerField::ClientId => FieldValue::Text(&self.client_id),
            ServerField::ClientSecret => FieldValue::Secret(&self.client_secret),
            ServerField::Scopes => list_value(&self.scopes, self.scopes.join(", ")),
            ServerField::Timeout => FieldValue::Text(&self.timeout),
        }
    }

    fn text_mut(&mut self, field: ServerField) -> Option<&mut TextEditor> {
        match field {
            ServerField::Name => Some(&mut self.name),
            ServerField::Command => Some(&mut self.command),
            ServerField::Url => Some(&mut self.url),
            ServerField::ClientId => Some(&mut self.client_id),
            ServerField::ClientSecret => Some(&mut self.client_secret),
            ServerField::Timeout => Some(&mut self.timeout),
            ServerField::Transport
            | ServerField::Args
            | ServerField::Env
            | ServerField::Read
            | ServerField::Write
            | ServerField::Network
            | ServerField::Headers
            | ServerField::Auth
            | ServerField::Scopes => None,
        }
    }

    fn cycle(&mut self, field: ServerField, _forward: bool) {
        match field {
            ServerField::Transport => {
                self.transport = match self.transport {
                    Transport::Stdio => Transport::Http,
                    Transport::Http => Transport::Stdio,
                };
            }
            ServerField::Auth => self.oauth = !self.oauth,
            ServerField::Network => self.network = !self.network,
            _ => {}
        }
    }

    fn list_mut(&mut self, field: ServerField) -> Option<&mut Vec<String>> {
        match field {
            ServerField::Args => Some(&mut self.args),
            ServerField::Env => Some(&mut self.env),
            ServerField::Read => Some(&mut self.read),
            ServerField::Write => Some(&mut self.write),
            ServerField::Headers => Some(&mut self.headers),
            ServerField::Scopes => Some(&mut self.scopes),
            ServerField::Name
            | ServerField::Transport
            | ServerField::Command
            | ServerField::Network
            | ServerField::Url
            | ServerField::Auth
            | ServerField::ClientId
            | ServerField::ClientSecret
            | ServerField::Timeout => None,
        }
    }

    /// A `NAME=value` list refuses an entry with no name or a name it already has.
    fn list_entry(
        field: ServerField,
        text: String,
        entries: &[String],
        index: usize,
    ) -> Result<String, String> {
        if field.holds_pairs() {
            named_pair(&text, entries, index)
        } else {
            Ok(text)
        }
    }

    /// The entry holding only the keys its transport takes.
    fn output(&self) -> Result<ServerEntry, String> {
        let mut entry = vec![text_pair("name", required(&self.name, "name")?)];
        match self.transport {
            Transport::Stdio => {
                entry.push(text_pair("command", required(&self.command, "command")?));
                entry.extend(array_pair("args", &self.args));
                entry.extend(table_pair("env", &self.env));
                entry.extend(array_pair("read", &self.read));
                entry.extend(array_pair("write", &self.write));
                if !self.network {
                    entry.push(("network".to_string(), toml::Value::Boolean(false)));
                }
            }
            Transport::Http => {
                entry.push(text_pair("url", required(&self.url, "URL")?));
                entry.extend(table_pair("headers", &self.headers));
                if self.oauth {
                    entry.push(text_pair("auth", "oauth"));
                    entry.extend(optional_pair("client_id", self.client_id.text().trim()));
                    entry.extend(optional_pair("client_secret", self.client_secret.text()));
                    entry.extend(array_pair("scopes", &self.scopes));
                }
            }
        }
        let timeout = self.timeout.text().trim();
        if !timeout.is_empty() {
            let seconds = timeout
                .parse()
                .map_err(|_| "timeout must be a whole number of seconds".to_string())?;
            entry.push((
                "request_timeout_secs".to_string(),
                toml::Value::Integer(seconds),
            ));
        }
        Ok(entry)
    }
}

fn list_value(entries: &[String], summary: String) -> FieldValue<'_> {
    FieldValue::List { entries, summary }
}

/// The trimmed text of `editor`, the field named `name`; refused when it is empty.
fn required<'a>(editor: &'a TextEditor, name: &str) -> Result<&'a str, String> {
    match editor.text().trim() {
        "" => Err(format!("{name} is empty")),
        text => Ok(text),
    }
}

fn text_pair(key: &str, text: &str) -> (String, toml::Value) {
    (key.to_string(), toml::Value::from(text))
}

/// `key` set to `text`, or nothing when `text` is empty.
fn optional_pair(key: &str, text: &str) -> Option<(String, toml::Value)> {
    (!text.is_empty()).then(|| text_pair(key, text))
}

/// `key` set to the array of `entries`, or nothing when there are none.
fn array_pair(key: &str, entries: &[String]) -> Option<(String, toml::Value)> {
    let values = entries.iter().cloned().map(toml::Value::from).collect();
    (!entries.is_empty()).then(|| (key.to_string(), toml::Value::Array(values)))
}

/// `key` set to the table the `NAME=value` `pairs` make, or nothing when there are none.
fn table_pair(key: &str, pairs: &[String]) -> Option<(String, toml::Value)> {
    let table: toml::Table = pairs
        .iter()
        .filter_map(|pair| pair.split_once('='))
        .map(|(name, value)| (name.to_string(), toml::Value::from(value)))
        .collect();
    (!table.is_empty()).then(|| (key.to_string(), toml::Value::Table(table)))
}

/// `text` as `NAME=value` with its name trimmed; refused with no name, or with the name
/// of one of `entries` other than the one at `index`.
fn named_pair(text: &str, entries: &[String], index: usize) -> Result<String, String> {
    let (name, value) = text
        .split_once('=')
        .map(|(name, value)| (name.trim(), value))
        .filter(|(name, _)| !name.is_empty())
        .ok_or("write the entry as NAME=value")?;
    let taken = entries
        .iter()
        .enumerate()
        .any(|(at, entry)| at != index && pair_name(entry) == name);
    if taken {
        return Err(format!("{name} is already set"));
    }
    Ok(format!("{name}={value}"))
}

fn pair_name(pair: &str) -> &str {
    pair.split_once('=').map_or(pair, |(name, _)| name)
}

/// The names of the `NAME=value` `pairs`, which may hold credentials, on one line.
fn pair_names(pairs: &[String]) -> String {
    let names: Vec<&str> = pairs.iter().map(|pair| pair_name(pair)).collect();
    names.join(", ")
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    use super::ServerFields;
    use crate::settings::form::{Form, FormAction};

    const SERVERS: &str = r#"
        [[servers]]
        name = "bare"
        command = "server"

        [[servers]]
        name = "files"
        command = "npx"
        args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/my docs"]
        env = { API_TOKEN = "a=b", HOME = "/tmp" }
        request_timeout_secs = 60

        [[servers]]
        name = "remote"
        url = "https://mcp.example.com/mcp"
        headers = { Authorization = "Bearer key", X-Team = "one" }

        [[servers]]
        name = "drive"
        url = "https://drivemcp.googleapis.com/mcp/v1"
        auth = "oauth"
        client_id = "id.apps.googleusercontent.com"
        client_secret = " secret "
        scopes = ["drive.readonly", "drive.metadata"]

        [[servers]]
        name = "dynamic"
        url = "https://mcp.example.com/mcp"
        auth = "oauth"
    "#;

    #[test]
    fn every_server_shape_survives_the_form_unchanged() {
        let config: toml::Table = toml::from_str(SERVERS).unwrap();
        for server in config["servers"].as_array().unwrap() {
            let mut form = Form::new(ServerFields::new(Some(server)));
            match form.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)) {
                FormAction::Apply(entry) => {
                    assert_eq!(&toml::Value::Table(entry.into_iter().collect()), server);
                }
                FormAction::Refused(reason) => panic!("{server} was refused: {reason}"),
                FormAction::Pending | FormAction::Cancel => panic!("{server} did not apply"),
            }
        }
    }
}

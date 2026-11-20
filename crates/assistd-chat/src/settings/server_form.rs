//! The form one MCP server is written in: its name, its transport, then the fields that
//! transport takes.

use assistd_config::ListItem;
use crossterm::event::{KeyCode, KeyEvent};

use super::super::selection::cycled;
use super::super::text_editor::{EditorShape, TextEditor};
use super::list_editor::{ListAction, ListEditor, entry_text};

/// A `[[mcp.servers]]` entry, its keys in the order they are written.
pub(crate) type ServerEntry = Vec<(String, toml::Value)>;

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
            Self::Command => "the program to run; a bare name is looked up on PATH",
            Self::Args => "arguments for the command; put tokens and keys in the environment",
            Self::Env => "NAME=value; the server inherits only PATH, HOME and a few basics",
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

/// What a field holds, as the form draws it.
pub(crate) enum FieldValue<'a> {
    Text(&'a TextEditor),
    /// One of two options, switched with ←/→.
    Choice(&'static str),
    /// A list's entries on one line, a `NAME=value` entry as its name alone.
    List(String),
}

/// What a key in the form asks of the list holding it.
pub(super) enum FormAction {
    Pending,
    Apply(ServerEntry),
    Cancel,
    /// The form or its open list entry cannot be applied, for the reason given.
    Refused(String),
}

/// A server being written: both transports' fields are kept, so switching loses nothing.
pub(crate) struct ServerForm {
    transport: Transport,
    oauth: bool,
    focus: ServerField,
    name: TextEditor,
    command: TextEditor,
    args: Vec<String>,
    /// `NAME=value` entries with distinct names, as are `headers`.
    env: Vec<String>,
    url: TextEditor,
    headers: Vec<String>,
    client_id: TextEditor,
    client_secret: TextEditor,
    scopes: Vec<String>,
    timeout: TextEditor,
    /// The list box open over the focused list field.
    open_list: Option<ListEditor>,
}

impl ServerForm {
    /// A form holding the `[[mcp.servers]]` entry `server`, or a blank stdio server.
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
            focus: ServerField::Name,
            name: text("name"),
            command: text("command"),
            args: list("args"),
            env: pairs("env"),
            url: text("url"),
            headers: pairs("headers"),
            client_id: text("client_id"),
            client_secret: text("client_secret"),
            scopes: list("scopes"),
            timeout: text("request_timeout_secs"),
            open_list: None,
        }
    }

    pub(crate) fn focus(&self) -> ServerField {
        self.focus
    }

    /// The name and transport, the fields the transport takes, then the timeout.
    pub(crate) fn fields(&self) -> Vec<ServerField> {
        let transport: &[ServerField] = match (self.transport, self.oauth) {
            (Transport::Stdio, _) => &[ServerField::Command, ServerField::Args, ServerField::Env],
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

    pub(crate) fn value(&self, field: ServerField) -> FieldValue<'_> {
        match field {
            ServerField::Name => FieldValue::Text(&self.name),
            ServerField::Transport => FieldValue::Choice(match self.transport {
                Transport::Stdio => "stdio",
                Transport::Http => "HTTP",
            }),
            ServerField::Command => FieldValue::Text(&self.command),
            ServerField::Args => FieldValue::List(self.args.join(" ")),
            ServerField::Env => FieldValue::List(pair_names(&self.env)),
            ServerField::Url => FieldValue::Text(&self.url),
            ServerField::Headers => FieldValue::List(pair_names(&self.headers)),
            ServerField::Auth => FieldValue::Choice(if self.oauth { "oauth" } else { "none" }),
            ServerField::ClientId => FieldValue::Text(&self.client_id),
            ServerField::ClientSecret => FieldValue::Text(&self.client_secret),
            ServerField::Scopes => FieldValue::List(self.scopes.join(", ")),
            ServerField::Timeout => FieldValue::Text(&self.timeout),
        }
    }

    /// The list box open over the focused field.
    pub(crate) fn open_list(&self) -> Option<&ListEditor> {
        self.open_list.as_ref()
    }

    /// The focused field's entries as a list box's; empty for a field that is no list.
    pub(crate) fn list_entries(&self) -> Vec<toml::Value> {
        let entries = match self.focus {
            ServerField::Args => &self.args,
            ServerField::Env => &self.env,
            ServerField::Headers => &self.headers,
            ServerField::Scopes => &self.scopes,
            _ => return Vec::new(),
        };
        entries.iter().cloned().map(toml::Value::from).collect()
    }

    fn text_field_mut(&mut self, field: ServerField) -> Option<&mut TextEditor> {
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
            | ServerField::Headers
            | ServerField::Auth
            | ServerField::Scopes => None,
        }
    }

    fn list_field_mut(&mut self, field: ServerField) -> Option<&mut Vec<String>> {
        match field {
            ServerField::Args => Some(&mut self.args),
            ServerField::Env => Some(&mut self.env),
            ServerField::Headers => Some(&mut self.headers),
            ServerField::Scopes => Some(&mut self.scopes),
            ServerField::Name
            | ServerField::Transport
            | ServerField::Command
            | ServerField::Url
            | ServerField::Auth
            | ServerField::ClientId
            | ServerField::ClientSecret
            | ServerField::Timeout => None,
        }
    }

    /// ↑/↓ and Tab move between fields, ←/→ switch a choice, typing edits a text field,
    /// Enter opens a list field and elsewhere makes the entry, and Esc drops the form.
    pub(super) fn on_key(&mut self, ev: KeyEvent) -> FormAction {
        if self.open_list.is_some() {
            return self.on_list_key(ev);
        }
        match (ev.code, self.focus) {
            (KeyCode::Up | KeyCode::BackTab, _) => self.move_focus(false),
            (KeyCode::Down | KeyCode::Tab, _) => self.move_focus(true),
            (KeyCode::Left | KeyCode::Right, ServerField::Transport) => {
                self.transport = match self.transport {
                    Transport::Stdio => Transport::Http,
                    Transport::Http => Transport::Stdio,
                };
            }
            (KeyCode::Left | KeyCode::Right, ServerField::Auth) => self.oauth = !self.oauth,
            (KeyCode::Enter, focus) if matches!(self.value(focus), FieldValue::List(_)) => {
                self.open_list = Some(ListEditor::new(ListItem::Text));
            }
            (KeyCode::Enter, _) => {
                return match self.entry() {
                    Ok(entry) => FormAction::Apply(entry),
                    Err(reason) => FormAction::Refused(reason),
                };
            }
            (KeyCode::Esc, _) => return FormAction::Cancel,
            (_, focus) => {
                if let Some(editor) = self.text_field_mut(focus) {
                    editor.on_key(ev);
                }
            }
        }
        FormAction::Pending
    }

    fn move_focus(&mut self, forward: bool) {
        self.focus = cycled(&self.fields(), self.focus, forward);
    }

    fn on_list_key(&mut self, ev: KeyEvent) -> FormAction {
        let entries = self.list_entries();
        let Some(list) = &mut self.open_list else {
            return FormAction::Pending;
        };
        match list.on_key(ev, &entries) {
            ListAction::Pending => {}
            ListAction::Close => self.open_list = None,
            ListAction::Store { index, value } => {
                if let Err(reason) = self.place_entry(index, &value) {
                    return FormAction::Refused(reason);
                }
                if let Some(list) = &mut self.open_list {
                    list.close_entry();
                }
            }
            ListAction::Delete(index) => {
                if let Some(entries) = self.list_field_mut(self.focus) {
                    entries.remove(index);
                }
            }
            ListAction::Refused(reason) => return FormAction::Refused(reason),
        }
        FormAction::Pending
    }

    /// Puts the text `value` holds at `index` of the focused list, appending when `index`
    /// is its length; a `NAME=value` list refuses an entry with no name or a name it has.
    fn place_entry(&mut self, index: usize, value: &toml::Value) -> Result<(), String> {
        let field = self.focus;
        let Some(entries) = self.list_field_mut(field) else {
            return Ok(());
        };
        let text = if field.holds_pairs() {
            named_pair(&entry_text(value), entries, index)?
        } else {
            entry_text(value)
        };
        match entries.get_mut(index) {
            Some(entry) => *entry = text,
            None => entries.push(text),
        }
        Ok(())
    }

    /// The entry the fields describe, holding only the keys its transport takes, or why
    /// they describe none.
    fn entry(&self) -> Result<ServerEntry, String> {
        let mut entry = vec![text_pair("name", required(&self.name, "name")?)];
        match self.transport {
            Transport::Stdio => {
                entry.push(text_pair("command", required(&self.command, "command")?));
                entry.extend(array_pair("args", &self.args));
                entry.extend(table_pair("env", &self.env));
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

    use super::{FormAction, ServerForm};

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
            let mut form = ServerForm::new(Some(server));
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

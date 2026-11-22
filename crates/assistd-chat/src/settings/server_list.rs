//! The box that lists the MCP servers and opens one in a form to add or change it; each
//! change is handed back to the panel to apply.

use crossterm::event::{KeyCode, KeyEvent};

use super::super::selection::Selection;
use super::form::{Form, FormAction};
use super::server_form::{ServerEntry, ServerFields, ServerForm};

/// What the panel should do after a key in the server list.
pub(crate) enum ServerListAction {
    Pending,
    Close,
    /// Puts `server` at `index`, appending when `index` is the server count.
    Store {
        index: usize,
        server: ServerEntry,
    },
    Delete(usize),
    /// The open form cannot be stored, for the reason given.
    Refused(String),
}

/// What the list is doing with the selected server.
pub(crate) enum ServerListState {
    Browse,
    /// Changing it, or adding one when the add row is selected.
    Form(Box<ServerForm>),
    /// Waiting for y or n before deleting it.
    ConfirmDelete,
}

/// Navigation over the servers, with at most one open in a form.
pub(crate) struct ServerList {
    /// Index of the selected server; the server count selects the add row.
    selected: Selection,
    state: ServerListState,
}

impl ServerList {
    pub(crate) fn new() -> Self {
        Self {
            selected: Selection::default(),
            state: ServerListState::Browse,
        }
    }

    pub(crate) fn selected(&self) -> usize {
        self.selected.index()
    }

    pub(crate) fn state(&self) -> &ServerListState {
        &self.state
    }

    /// The open form.
    pub(crate) fn form(&self) -> Option<&ServerForm> {
        match &self.state {
            ServerListState::Form(form) => Some(form),
            ServerListState::Browse | ServerListState::ConfirmDelete => None,
        }
    }

    /// Closes the open form after its server was stored.
    pub(crate) fn close_form(&mut self) {
        self.state = ServerListState::Browse;
    }

    /// `servers` are the `[[mcp.servers]]` entries in effect.
    pub(crate) fn on_key(&mut self, ev: KeyEvent, servers: &[toml::Value]) -> ServerListAction {
        match &mut self.state {
            ServerListState::Form(form) => match form.on_key(ev) {
                FormAction::Pending => ServerListAction::Pending,
                FormAction::Cancel => {
                    self.state = ServerListState::Browse;
                    ServerListAction::Pending
                }
                FormAction::Apply(server) => ServerListAction::Store {
                    index: self.selected.index(),
                    server,
                },
                FormAction::Refused(reason) => ServerListAction::Refused(reason),
            },
            ServerListState::ConfirmDelete => self.on_confirm_delete_key(ev),
            ServerListState::Browse => self.on_browse_key(ev, servers),
        }
    }

    fn on_confirm_delete_key(&mut self, ev: KeyEvent) -> ServerListAction {
        match ev.code {
            KeyCode::Char('y' | 'Y') => {
                self.state = ServerListState::Browse;
                ServerListAction::Delete(self.selected.index())
            }
            KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                self.state = ServerListState::Browse;
                ServerListAction::Pending
            }
            _ => ServerListAction::Pending,
        }
    }

    fn on_browse_key(&mut self, ev: KeyEvent, servers: &[toml::Value]) -> ServerListAction {
        let add_row = servers.len();
        match ev.code {
            KeyCode::Esc | KeyCode::Char('q') => return ServerListAction::Close,
            KeyCode::Up | KeyCode::Char('k') => self.selected.previous(),
            KeyCode::Down | KeyCode::Char('j') => self.selected.next(add_row + 1),
            KeyCode::Home => self.selected = Selection::default(),
            KeyCode::End => self.selected = Selection::at(add_row),
            KeyCode::Enter => self.open_form(servers.get(self.selected.index())),
            KeyCode::Char('a') | KeyCode::Insert => {
                self.selected = Selection::at(add_row);
                self.open_form(None);
            }
            KeyCode::Char('d') | KeyCode::Delete | KeyCode::Backspace
                if self.selected.index() < add_row =>
            {
                self.state = ServerListState::ConfirmDelete;
            }
            _ => {}
        }
        ServerListAction::Pending
    }

    fn open_form(&mut self, server: Option<&toml::Value>) {
        self.state = ServerListState::Form(Box::new(Form::new(ServerFields::new(server))));
    }
}

/// The name of the `[[mcp.servers]]` entry `server`.
pub(crate) fn server_name(server: &toml::Value) -> &str {
    server
        .get("name")
        .and_then(toml::Value::as_str)
        .unwrap_or_default()
}

/// What the `[[mcp.servers]]` entry `server` reaches: its URL, or its command and arguments.
pub(crate) fn server_target(server: &toml::Value) -> String {
    let text = |key: &str| server.get(key).and_then(toml::Value::as_str);
    if let Some(url) = text("url") {
        return url.to_string();
    }
    let args = server.get("args").and_then(toml::Value::as_array);
    text("command")
        .into_iter()
        .chain(args.into_iter().flatten().filter_map(toml::Value::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

//! Submitted input: slash commands and queries.

use assistd_config::ConfigDocument;
use assistd_ipc::{Request, RequestBody};
use uuid::Uuid;

use super::{App, BranchOp, Panel, PendingAttachment, RESUME_RECENCY_SECS, WireStream};
use crate::chat::settings::SettingsPanel;

/// A typed slash command with its trimmed argument.
enum SlashCommand {
    Agents,
    Attach(String),
    Exit,
    Fork(String),
    Undo,
    New,
    Resume,
    Settings,
    Tasks,
    WakeWord,
}

impl SlashCommand {
    /// `None` when `text` is not a slash command; unknown commands are
    /// sent to the model as ordinary text.
    fn parse(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        let (verb, arg) = match trimmed.split_once(char::is_whitespace) {
            Some((verb, rest)) => (verb, rest.trim()),
            None => (trimmed, ""),
        };
        match verb {
            "/agents" => Some(Self::Agents),
            "/attach" => Some(Self::Attach(arg.to_string())),
            "/exit" => Some(Self::Exit),
            "/fork" => Some(Self::Fork(arg.to_string())),
            "/undo" => Some(Self::Undo),
            "/new" => Some(Self::New),
            "/resume" => Some(Self::Resume),
            "/settings" => Some(Self::Settings),
            "/tasks" => Some(Self::Tasks),
            "/wakeword" => Some(Self::WakeWord),
            _ => None,
        }
    }

    /// Whether the command talks to the daemon or relies on what it reported.
    fn needs_daemon(&self) -> bool {
        !matches!(self, Self::Exit | Self::Settings)
    }
}

/// Whether the slash command `name` can run while the daemon is offline.
pub(super) fn runs_offline(name: &str) -> bool {
    SlashCommand::parse(name).is_some_and(|command| !command.needs_daemon())
}

impl App {
    pub(super) fn submit_typed(&mut self, text: String) {
        match SlashCommand::parse(&text) {
            Some(command) if command.needs_daemon() && !self.daemon_reachable => {
                let verb = text.split_whitespace().next().unwrap_or_default();
                self.set_notice(&format!("{verb} needs the daemon, which is offline"));
            }
            Some(command) => self.run_slash_command(command),
            None => self.submit_query(text),
        }
    }

    fn run_slash_command(&mut self, command: SlashCommand) {
        match command {
            SlashCommand::Agents => self.open_agent_panel(),
            SlashCommand::Attach(_) if !self.vision_enabled => {
                self.output.push_error(
                    "[error] attach: vision not available: model does not support \
                     images. Use: a model with mmproj loaded",
                );
                self.set_notice("vision not available");
            }
            SlashCommand::Attach(path) if path.is_empty() => {
                self.output
                    .push_error("/attach: expected a path. Usage: /attach <path>");
                self.set_notice("/attach: missing path");
            }
            SlashCommand::Attach(path) => self.load_attachment(&path),
            SlashCommand::Exit => self.quit(),
            SlashCommand::Fork(name) => self.fork_branch(name),
            SlashCommand::Undo => {
                let id = Uuid::new_v4().to_string();
                self.spawn_branch_command(BranchOp::Undo, Request::new(id, RequestBody::Undo));
            }
            SlashCommand::New => {
                let id = Uuid::new_v4().to_string();
                self.spawn_branch_command(BranchOp::New, Request::new(id, RequestBody::NewSession));
            }
            SlashCommand::Resume => {
                let id = Uuid::new_v4().to_string();
                self.spawn_branch_command(
                    BranchOp::ResumePicker,
                    Request::new(id, RequestBody::Branches),
                );
            }
            SlashCommand::Settings => self.open_settings_panel(),
            SlashCommand::Tasks => self.open_tasks_panel(),
            SlashCommand::WakeWord => self.open_wake_word_panel(),
        }
    }

    fn open_settings_panel(&mut self) {
        match ConfigDocument::open(&self.config_path) {
            Ok(document) => {
                self.panel = Some(Panel::Settings(Box::new(SettingsPanel::new(document))));
            }
            Err(e) => {
                self.output.push_error(&format!("[error] settings: {e}"));
                self.set_notice("settings unavailable");
            }
        }
    }

    fn submit_query(&mut self, text: String) {
        if self.generating() {
            self.set_notice("still generating, please wait");
            return;
        }
        let pending = std::mem::take(&mut self.pending_attachments);
        let names: Vec<String> = pending.iter().map(|a| a.name.clone()).collect();
        let mut attachments = Vec::with_capacity(pending.len());
        let mut thumbnails = Vec::new();
        for (attachment, protocol, name) in pending.into_iter().map(PendingAttachment::into_parts) {
            attachments.push(attachment);
            thumbnails.extend(protocol.map(|protocol| (name, protocol)));
        }
        self.begin_turn(&text, &names);
        for (name, protocol) in thumbnails {
            self.output.push_thumbnail(name, protocol);
        }
        self.spawn_query(text, attachments);
    }

    pub(crate) fn spawn_resume_or_new(&mut self) {
        let req = Request::new(
            Uuid::new_v4().to_string(),
            RequestBody::ResumeOrNew {
                recency_secs: RESUME_RECENCY_SECS,
            },
        );
        self.spawn_branch_command(BranchOp::Resume, req);
    }

    fn fork_branch(&mut self, name: String) {
        if name.trim().is_empty() {
            self.output
                .push_error("/fork: expected a name. Usage: /fork <name>");
            self.set_notice("/fork: missing name");
            return;
        }
        let req = Request::new(Uuid::new_v4().to_string(), RequestBody::Fork { name });
        self.spawn_branch_command(BranchOp::Fork, req);
    }

    pub(super) fn switch_branch(&mut self, target: String) {
        let req = Request::new(Uuid::new_v4().to_string(), RequestBody::Switch { target });
        self.spawn_branch_command(BranchOp::Switch, req);
    }

    /// Refused while another branch command or a reply is in flight.
    fn spawn_branch_command(&mut self, op: BranchOp, req: Request) {
        if self.in_flight_branch_op.is_some() {
            self.set_notice("branch command in flight, please wait");
            return;
        }
        if self.generating() {
            self.set_notice("still generating, please wait");
            return;
        }
        self.in_flight_branch_op = Some(op);
        self.branches_buffer.clear();
        self.spawn_one_shot(req, WireStream::Branch, "branch");
    }
}

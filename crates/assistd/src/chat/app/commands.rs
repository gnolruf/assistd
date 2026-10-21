//! Submitted input: slash commands and queries.

use assistd_ipc::Request;
use uuid::Uuid;

use super::{App, BranchOp, PendingAttachment, WireStream};

/// A typed slash command with its trimmed argument.
enum SlashCommand {
    Attach(String),
    Exit,
    Fork(String),
    Undo,
    New,
    Resume,
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
            "/attach" => Some(Self::Attach(arg.to_string())),
            "/exit" => Some(Self::Exit),
            "/fork" => Some(Self::Fork(arg.to_string())),
            "/undo" => Some(Self::Undo),
            "/new" => Some(Self::New),
            "/resume" => Some(Self::Resume),
            "/wakeword" => Some(Self::WakeWord),
            _ => None,
        }
    }
}

impl App {
    pub(super) fn submit_typed(&mut self, text: String) {
        match SlashCommand::parse(&text) {
            Some(command) => self.run_slash_command(command),
            None => self.submit_query(text),
        }
    }

    fn run_slash_command(&mut self, command: SlashCommand) {
        match command {
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
                self.spawn_branch_command(BranchOp::Undo, Request::Undo { id });
            }
            SlashCommand::New => {
                let id = Uuid::new_v4().to_string();
                self.spawn_branch_command(BranchOp::New, Request::NewSession { id });
            }
            SlashCommand::Resume => {
                let id = Uuid::new_v4().to_string();
                self.spawn_branch_command(BranchOp::ResumePicker, Request::Branches { id });
            }
            SlashCommand::WakeWord => self.open_wake_word_panel(),
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

    pub(crate) fn spawn_resume_or_new(&mut self, recency_secs: u64) {
        let req = Request::ResumeOrNew {
            id: Uuid::new_v4().to_string(),
            recency_secs,
        };
        self.spawn_branch_command(BranchOp::Resume, req);
    }

    fn fork_branch(&mut self, name: String) {
        if name.trim().is_empty() {
            self.output
                .push_error("/fork: expected a name. Usage: /fork <name>");
            self.set_notice("/fork: missing name");
            return;
        }
        let req = Request::Fork {
            id: Uuid::new_v4().to_string(),
            name,
        };
        self.spawn_branch_command(BranchOp::Fork, req);
    }

    pub(super) fn switch_branch(&mut self, target: String) {
        let req = Request::Switch {
            id: Uuid::new_v4().to_string(),
            target,
        };
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

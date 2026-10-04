//! Fixtures shared by the command tests.

use std::os::fd::OwnedFd;
use std::path::Path;
use std::sync::Arc;

use assistd_wm::NoWindowManager;
use async_trait::async_trait;
use parking_lot::Mutex;
use rustix::fs::{Mode, OFlags};

use super::{
    BashCommand, CatCommand, EchoCommand, GrepCommand, HeadCommand, LsCommand, ScreenshotCommand,
    SeeCommand, SortCommand, TailCommand, UniqCommand, WcCommand, WebCommand, WmCommand,
    WriteCommand,
};
use crate::command::CommandRegistry;
use crate::policy::{
    AlwaysAllowGate, Approval, ApprovalGate, Approvals, ConfirmationGate, ConfirmationRequest,
    DestructivePattern,
};

pub(crate) fn test_registry() -> CommandRegistry {
    let mut registry = CommandRegistry::new();
    registry.register(CatCommand);
    registry.register(LsCommand);
    registry.register(GrepCommand);
    registry.register(WcCommand);
    registry.register(HeadCommand);
    registry.register(TailCommand);
    registry.register(SortCommand);
    registry.register(UniqCommand);
    registry.register(EchoCommand);
    registry.register(WriteCommand::permissive_for_tests());
    registry.register(SeeCommand::default());
    registry.register(ScreenshotCommand::default());
    registry.register(WebCommand::new(ApprovalGate::new(
        Arc::new(AlwaysAllowGate),
        Arc::new(Approvals::unsaved()),
    )));
    registry.register(BashCommand::default());
    registry.register(WmCommand::for_test(Arc::new(NoWindowManager)));
    registry
}

/// Destructive patterns from whitespace-separated strings.
pub(crate) fn test_patterns(patterns: &[&str]) -> Vec<DestructivePattern> {
    patterns
        .iter()
        .map(|p| {
            DestructivePattern::new(p.split_whitespace())
                .unwrap_or_else(|| panic!("invalid pattern {p:?}"))
        })
        .collect()
}

/// Path of a new FIFO named `pipe` in `dir`.
pub(crate) fn make_fifo(dir: &Path) -> String {
    let fifo = dir.join("pipe");
    rustix::fs::mkfifoat(rustix::fs::CWD, &fifo, Mode::from_raw_mode(0o600)).unwrap();
    fifo.to_string_lossy().into_owned()
}

/// Both ends of `fifo`, so a blocking open of it succeeds and a read waits.
pub(crate) fn hold_fifo_open(fifo: &str) -> (OwnedFd, OwnedFd) {
    let reader = rustix::fs::open(fifo, OFlags::RDONLY | OFlags::NONBLOCK, Mode::empty()).unwrap();
    let writer = rustix::fs::open(fifo, OFlags::WRONLY, Mode::empty()).unwrap();
    (reader, writer)
}

/// Confirmation gate that answers every prompt the same way and records
/// each prompt.
#[derive(Debug)]
pub(crate) struct RecordingGate {
    answer: Approval,
    requests: Mutex<Vec<ConfirmationRequest>>,
}

impl RecordingGate {
    pub(crate) fn new(approve: bool) -> Arc<Self> {
        Self::answering(if approve {
            Approval::Once
        } else {
            Approval::Deny
        })
    }

    pub(crate) fn answering(answer: Approval) -> Arc<Self> {
        Arc::new(Self {
            answer,
            requests: Mutex::default(),
        })
    }

    /// Each prompt's `(tool, script, matched_pattern)`.
    pub(crate) fn prompts(&self) -> Vec<(String, String, String)> {
        self.requests
            .lock()
            .iter()
            .map(|req| {
                (
                    req.tool.clone(),
                    req.script.clone(),
                    req.matched_pattern.clone(),
                )
            })
            .collect()
    }

    pub(crate) fn requests(&self) -> Vec<ConfirmationRequest> {
        self.requests.lock().clone()
    }
}

#[async_trait]
impl ConfirmationGate for RecordingGate {
    async fn confirm(&self, req: ConfirmationRequest) -> Approval {
        self.requests.lock().push(req);
        self.answer
    }
}

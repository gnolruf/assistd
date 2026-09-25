//! Fixtures shared by the command tests.

use std::sync::Arc;

use assistd_wm::NoWindowManager;
use async_trait::async_trait;
use parking_lot::Mutex;

use super::{
    BashCommand, CatCommand, EchoCommand, GrepCommand, HeadCommand, LsCommand, ScreenshotCommand,
    SeeCommand, SortCommand, TailCommand, UniqCommand, WcCommand, WebCommand, WmCommand,
    WriteCommand,
};
use crate::command::CommandRegistry;
use crate::policy::{Approval, ConfirmationGate, ConfirmationRequest, DestructivePattern};

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
    registry.register(WebCommand::new());
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

/// Confirmation gate that answers every prompt the same way and records
/// each prompt's `(tool, script, matched_pattern)`.
pub(crate) struct RecordingGate {
    approve: bool,
    prompts: Mutex<Vec<(String, String, String)>>,
}

impl RecordingGate {
    pub(crate) fn new(approve: bool) -> Arc<Self> {
        Arc::new(Self {
            approve,
            prompts: Mutex::default(),
        })
    }

    pub(crate) fn prompts(&self) -> Vec<(String, String, String)> {
        self.prompts.lock().clone()
    }
}

#[async_trait]
impl ConfirmationGate for RecordingGate {
    async fn confirm(&self, req: ConfirmationRequest) -> Approval {
        self.prompts
            .lock()
            .push((req.tool, req.script, req.matched_pattern));
        if self.approve {
            Approval::Once
        } else {
            Approval::Deny
        }
    }
}

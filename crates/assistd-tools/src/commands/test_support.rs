//! Fixtures shared by the command tests.

use std::error::Error;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use assistd_ipc::{Event, PermissionMode};
use assistd_wm::{NoWindowManager, WindowManager};
use async_trait::async_trait;
use parking_lot::Mutex;
use rustix::fs::{Mode, OFlags};
use serde_json::{Map, Value};
use tokio::sync::mpsc;

use super::{
    BashCommand, BashPolicyCfg, CatCommand, EchoCommand, GrepCommand, HeadCommand, LsCommand,
    ScreenshotCommand, ScreenshotPolicyCfg, SeeCommand, SortCommand, TailCommand, UniqCommand,
    WcCommand, WebCommand, WmCommand, WriteCommand, WritePolicyCfg,
};
use crate::command::CommandRegistry;
use crate::policy::{
    AlwaysAllowGate, Approval, ApprovalGate, Approvals, ConfirmRouter, ConfirmationGate,
    ConfirmationRequest, JudgeModel, SafetyJudge, SandboxInfo,
};
use crate::vision::VisionGate;

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
    registry.register(permissive_write());
    registry.register(test_see());
    registry.register(test_screenshot());
    registry.register(WebCommand::new(
        ApprovalGate::new(Arc::new(AlwaysAllowGate), Arc::new(Approvals::unsaved())),
        None,
    ));
    registry.register(test_bash());
    registry.register(test_wm(Arc::new(NoWindowManager)));
    registry
}

/// `write` with an allowlist of `/`, permitting any absolute path without asking.
pub(crate) fn permissive_write() -> WriteCommand {
    WriteCommand::new(
        Arc::new(WritePolicyCfg::new(vec![PathBuf::from("/")]).expect("non-empty allowlist")),
        Arc::new(AlwaysAllowGate),
        SandboxInfo::none(),
    )
}

pub(crate) fn test_see() -> SeeCommand {
    SeeCommand::new(VisionGate::new(true))
}

pub(crate) fn test_screenshot() -> ScreenshotCommand {
    ScreenshotCommand::new(
        Arc::new(ScreenshotPolicyCfg::default()),
        VisionGate::new(true),
    )
}

pub(crate) fn test_bash() -> BashCommand {
    BashCommand::new(
        Arc::new(BashPolicyCfg::default()),
        SandboxInfo::none(),
        Arc::new(AlwaysAllowGate),
    )
}

pub(crate) fn test_wm(wm: Arc<dyn WindowManager>) -> WmCommand {
    WmCommand::new(
        wm,
        Arc::new(AlwaysAllowGate),
        Arc::new(Approvals::unsaved()),
    )
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

/// A judge model that always gives the decision it holds, `"allow"` or `"ask"`.
#[derive(Debug)]
pub(crate) struct FixedJudge(pub(crate) &'static str);

#[async_trait]
impl JudgeModel for FixedJudge {
    async fn user_messages(&self, _limit: usize) -> Vec<String> {
        Vec::new()
    }

    async fn reply(
        &self,
        _prompt: String,
        _schema: Map<String, Value>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        Ok(format!(r#"{{"decision":"{}","reason":"stub"}}"#, self.0))
    }
}

/// A router in `mode` whose safety judge allows everything and whose prompts are denied
/// unanswered at once, and the wire those prompts go out on.
pub(crate) fn router_in(mode: PermissionMode) -> (Arc<ConfirmRouter>, mpsc::Receiver<Event>) {
    let (tx, rx) = mpsc::channel(4);
    let judge = SafetyJudge::new(Arc::new(FixedJudge("allow")), Duration::from_secs(5));
    let router = ConfirmRouter::new("r".into(), tx, Duration::from_millis(10), Some(judge));
    router.set_mode(mode);
    (router, rx)
}

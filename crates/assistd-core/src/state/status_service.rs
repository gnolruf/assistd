//! `StatusService`: answers what the daemon is doing and how far it has started.

use std::sync::Arc;

use tokio::sync::watch;
use tokio::task::JoinHandle;

use assistd_config::Config;
use assistd_documents::Extractor;
use assistd_embed::EmbedderHandle;
use assistd_ipc::{Component, DocumentActivity};
use assistd_tools::ToolsDisabled;
use assistd_utils::child_server::ChildServerStatus;
use assistd_utils::readiness::ReadinessCell;
use assistd_voice::VoiceManager;

use super::subsystems::McpServerStatus;
use crate::admission::TurnAdmission;
use crate::bus::EventBus;
use crate::documents::embed_gate::keep_document_embed_gate;
use crate::presence::PresenceManager;
use crate::recovery::spawn_supervised;
use crate::vision::VisionRevalidator;

/// Handles the presence, capability, readiness and `Subscribe` requests: it reads the
/// subsystems' state and owns none of its own.
#[derive(Debug)]
pub struct StatusService {
    pub(super) config: Arc<Config>,
    pub(super) presence: Arc<PresenceManager>,
    pub(super) voice: Arc<VoiceManager>,
    pub(super) vision_revalidator: Option<Arc<VisionRevalidator>>,
    pub(super) mcp_servers: Arc<[McpServerStatus]>,
    pub(super) tools_disabled: Option<ToolsDisabled>,
    pub(super) web_search: Arc<ReadinessCell<ChildServerStatus>>,
    pub(super) documents: Arc<Extractor>,
    /// What the document index was last broadcast as doing.
    pub(super) document_activity: watch::Receiver<DocumentActivity>,
    pub(super) embedder: Arc<EmbedderHandle>,
    pub(super) admission: Arc<TurnAdmission>,
    pub(super) bus: EventBus,
}

impl StatusService {
    /// Watch whether document chunks may be embedded now: not while the user's interaction,
    /// a task run or a GPU sleep lasts, unless `[documents]` lets embedding go on through
    /// it. The task keeping it current ends once every receiver is dropped.
    pub fn document_embed_gate(&self) -> (watch::Receiver<bool>, JoinHandle<()>) {
        let (gate, open) = watch::channel(false);
        let keeper = spawn_supervised(
            "document_embed_gate",
            Component::Embed,
            keep_document_embed_gate(Arc::clone(&self.config), Arc::clone(&self.admission), gate),
        );
        (open, keeper)
    }
}

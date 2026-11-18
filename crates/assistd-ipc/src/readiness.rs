//! How far each subsystem that starts in the background has got.

use std::fmt;

use serde::{Deserialize, Serialize};

use assistd_utils::readiness::NotReady;

/// A subsystem that starts in the background after the socket opens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StartupComponent {
    /// Push-to-talk and continuous listening.
    VoiceInput,
    Speech,
    /// The embedding server behind semantic recall.
    Embedding,
    /// The SearXNG instance behind web search.
    WebSearch,
    /// The worker that extracts text from the files under the index roots.
    DocumentIndexing,
    Mcp {
        server: String,
    },
}

impl fmt::Display for StartupComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StartupComponent::VoiceInput => f.write_str("voice input"),
            StartupComponent::Speech => f.write_str("speech"),
            StartupComponent::Embedding => f.write_str("semantic memory"),
            StartupComponent::WebSearch => f.write_str("web search"),
            StartupComponent::DocumentIndexing => f.write_str("document indexing"),
            StartupComponent::Mcp { server } => write!(f, "MCP {server}"),
        }
    }
}

/// How far one [`StartupComponent`]'s startup has got.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ComponentReadiness {
    Starting,
    Ready,
    /// Disabled in config or failed to start.
    Unavailable {
        reason: String,
    },
}

impl fmt::Display for ComponentReadiness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComponentReadiness::Starting => f.write_str("starting"),
            ComponentReadiness::Ready => f.write_str("ready"),
            ComponentReadiness::Unavailable { reason } => write!(f, "unavailable ({reason})"),
        }
    }
}

/// The latest [`ComponentReadiness`] a client has seen for each
/// [`StartupComponent`], in the order each was first reported.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StartupReadiness {
    components: Vec<(StartupComponent, ComponentReadiness)>,
}

impl StartupReadiness {
    /// Record `component`'s latest `state`.
    pub fn record(&mut self, component: StartupComponent, state: ComponentReadiness) {
        match self
            .components
            .iter_mut()
            .find(|(seen, _)| *seen == component)
        {
            Some((_, current)) => *current = state,
            None => self.components.push((component, state)),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &(StartupComponent, ComponentReadiness)> {
        self.components.iter()
    }

    pub fn starting(&self) -> impl Iterator<Item = &StartupComponent> {
        self.components
            .iter()
            .filter(|(_, state)| *state == ComponentReadiness::Starting)
            .map(|(component, _)| component)
    }
}

impl<T> From<Result<T, NotReady>> for ComponentReadiness {
    fn from(ready: Result<T, NotReady>) -> Self {
        match ready {
            Ok(_) => ComponentReadiness::Ready,
            Err(NotReady::Starting) => ComponentReadiness::Starting,
            Err(NotReady::Unavailable(reason)) => ComponentReadiness::Unavailable {
                reason: reason.to_string(),
            },
        }
    }
}

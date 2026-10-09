//! The embedder as the rest of the daemon sees it: available once its
//! server has come up in the background.

use std::sync::Arc;

use assistd_utils::readiness::{NotReady, Readiness, ReadinessCell};

use crate::{EmbedError, Embedder};

/// Shared access to an [`Embedder`] that may still be starting or may
/// have failed to start.
#[derive(Debug)]
pub struct EmbedderHandle {
    embedder: ReadinessCell<Arc<dyn Embedder>>,
}

impl EmbedderHandle {
    pub fn new(readiness: Readiness<Arc<dyn Embedder>>) -> Self {
        Self {
            embedder: ReadinessCell::new(readiness),
        }
    }

    /// Record how far the embedder's startup has got.
    pub fn set(&self, readiness: Readiness<Arc<dyn Embedder>>) {
        self.embedder.set(readiness);
    }

    /// The embedder, or [`EmbedError::Unavailable`] saying why there is none.
    pub fn get(&self) -> Result<Arc<dyn Embedder>, EmbedError> {
        self.embedder.get().map_err(EmbedError::Unavailable)
    }

    /// The embedder, or why there is none.
    pub fn readiness(&self) -> Result<Arc<dyn Embedder>, NotReady> {
        self.embedder.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_says_why_there_is_no_embedder() {
        let handle = EmbedderHandle::new(Readiness::Starting);
        assert_eq!(
            handle.get().unwrap_err().to_string(),
            "embedding is still starting"
        );
        handle.set(Readiness::Unavailable("disabled in config".into()));
        assert_eq!(
            handle.get().unwrap_err().to_string(),
            "embedding is unavailable: disabled in config"
        );
    }
}

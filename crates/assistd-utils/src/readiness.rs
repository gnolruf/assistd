//! Startup state of a subsystem that comes up in the background, so its
//! owner can serve requests before it is ready.

use std::sync::Arc;

use parking_lot::RwLock;
use thiserror::Error;

/// Where a background subsystem is in its startup.
#[derive(Debug, Clone)]
pub enum Readiness<T> {
    Starting,
    Ready(T),
    /// Disabled by config or failed to start; holds the reason.
    Unavailable(Arc<str>),
}

/// Why a [`ReadinessCell`] has no value to hand out.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NotReady {
    #[error("still starting")]
    Starting,
    #[error("unavailable: {0}")]
    Unavailable(Arc<str>),
}

/// A subsystem's current [`Readiness`], replaceable as startup progresses.
#[derive(Debug)]
pub struct ReadinessCell<T> {
    readiness: RwLock<Readiness<T>>,
}

impl<T: Clone> ReadinessCell<T> {
    pub fn starting() -> Self {
        Self::new(Readiness::Starting)
    }

    pub fn new(readiness: Readiness<T>) -> Self {
        Self {
            readiness: RwLock::new(readiness),
        }
    }

    pub fn set(&self, readiness: Readiness<T>) {
        *self.readiness.write() = readiness;
    }

    /// The ready value, or why there is none.
    pub fn get(&self) -> Result<T, NotReady> {
        match &*self.readiness.read() {
            Readiness::Starting => Err(NotReady::Starting),
            Readiness::Ready(value) => Ok(value.clone()),
            Readiness::Unavailable(reason) => Err(NotReady::Unavailable(Arc::clone(reason))),
        }
    }
}

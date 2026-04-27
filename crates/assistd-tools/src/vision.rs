//! Shared, runtime-mutable vision-capability flag.
//!
//! Built once at daemon startup from a `/props` probe of llama-server
//! (see `assistd_llm::probe_capabilities`) and held inside `AppState`.
//! `SeeCommand` and `ScreenshotCommand` clone the `Arc<VisionGate>` and
//! check `supported()` per invocation rather than capturing a `bool`,
//! so a model-swap on the running llama-server (followed by a daemon
//! revalidation) flips the gate cleanly without rebuilding the
//! command registry.
//!
//! The flag is a plain `AtomicBool`. Revalidation is the daemon's
//! concern — this crate intentionally does not depend on `assistd-llm`
//! or do any HTTP itself.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct VisionGate {
    supported: AtomicBool,
}

impl VisionGate {
    pub fn new(initially_supported: bool) -> Arc<Self> {
        Arc::new(Self {
            supported: AtomicBool::new(initially_supported),
        })
    }

    pub fn supported(&self) -> bool {
        self.supported.load(Ordering::Acquire)
    }

    /// Update the flag. Call from the daemon's revalidation path when
    /// a `/props` re-probe reports a model change.
    pub fn set(&self, supported: bool) {
        self.supported.store(supported, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_constructor_value() {
        let gate = VisionGate::new(true);
        assert!(gate.supported());
        let gate = VisionGate::new(false);
        assert!(!gate.supported());
    }

    #[test]
    fn set_flips_observed_value() {
        let gate = VisionGate::new(false);
        gate.set(true);
        assert!(gate.supported());
        gate.set(false);
        assert!(!gate.supported());
    }
}

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Shared, runtime-mutable vision-capability flag. Commands that need
/// vision hold the `Arc` and check [`VisionGate::supported`] on every
/// invocation, so a model swap flips them without rebuilding the
/// registry.
pub struct VisionGate {
    supported: AtomicBool,
}

impl VisionGate {
    pub fn new(initially_supported: bool) -> Arc<Self> {
        Arc::new(Self {
            supported: AtomicBool::new(initially_supported),
        })
    }

    /// Whether the current model accepts image inputs.
    pub fn supported(&self) -> bool {
        self.supported.load(Ordering::Acquire)
    }

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

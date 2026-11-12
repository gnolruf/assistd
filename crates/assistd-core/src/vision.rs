//! `VisionRevalidator`: keeps the vision gate and slot count in step with the loaded model.

use std::num::NonZeroU32;
use std::sync::Arc;

use tokio::sync::{Mutex, watch};
use tracing::info;

use assistd_ipc::PresenceState;
use assistd_llm::{LlamaServerControl, VisionState, probe_capabilities_routed};
use assistd_tools::VisionGate;

use crate::presence::PresenceManager;

/// Keeps a [`VisionGate`] and the server's slot count in step with the model llama-server
/// has loaded, re-probing only after a presence transition or a llama-server restart. A
/// failed probe leaves both unchanged and is retried on the next revalidation.
#[derive(Debug)]
pub struct VisionRevalidator {
    gate: Arc<VisionGate>,
    /// Starts at one until a probe reports the server's `total_slots`.
    slots: watch::Sender<NonZeroU32>,
    control: LlamaServerControl,
    model_name: String,
    seen: Mutex<SeenLoad>,
}

impl VisionRevalidator {
    /// Track `presence` for loads of `model_name`, probed through `control`.
    /// The gate starts closed until the first revalidation reaches the model.
    pub fn new(
        control: LlamaServerControl,
        model_name: String,
        presence: &PresenceManager,
    ) -> Arc<Self> {
        Arc::new(Self {
            gate: VisionGate::new(false),
            slots: watch::channel(NonZeroU32::MIN).0,
            control,
            model_name,
            seen: Mutex::new(SeenLoad {
                presence: presence.subscribe(),
                llama_pid: None,
                probed: false,
            }),
        })
    }

    /// The gate this revalidator keeps current.
    pub fn gate(&self) -> Arc<VisionGate> {
        Arc::clone(&self.gate)
    }

    /// Watch how many conversations the server decodes at once.
    pub fn slot_count(&self) -> watch::Receiver<NonZeroU32> {
        self.slots.subscribe()
    }

    /// Probe llama-server now, without touching the gate.
    pub async fn probe(&self) -> VisionState {
        probe_capabilities_routed(&self.control, &self.model_name).await
    }

    /// Re-probe and update the gate if a load may have happened since the
    /// last successful probe. Call while holding presence `Active`.
    /// Cancel-safe: a dropped probe reruns on the next call.
    pub async fn revalidate_if_stale(&self, presence: &PresenceManager) {
        let mut seen = self.seen.lock().await;
        if seen.take_stale(presence.llama_pid()) {
            let probe = self.probe().await;
            seen.probed = apply_probe(&self.gate, &probe);
            if let Some(slots) = probe.total_slots.filter(|_| seen.probed) {
                self.slots.send_if_modified(|current| {
                    let changed = *current != slots;
                    *current = slots;
                    changed
                });
            }
        }
    }
}

/// The load the gate was last probed against.
#[derive(Debug)]
struct SeenLoad {
    presence: watch::Receiver<PresenceState>,
    llama_pid: Option<u32>,
    probed: bool,
}

impl SeenLoad {
    /// Whether a probe is due: the last one failed, or the presence state
    /// or llama-server child changed since. Marks both as seen either way,
    /// and a change as unprobed so a probe dropped midway reruns.
    fn take_stale(&mut self, llama_pid: Option<u32>) -> bool {
        let reloaded = self.presence.has_changed().unwrap_or(false) || self.llama_pid != llama_pid;
        self.presence.mark_unchanged();
        self.llama_pid = llama_pid;
        if reloaded {
            self.probed = false;
        }
        !self.probed
    }
}

/// Set `gate` from a probe that reached the model, returning whether it
/// did.
fn apply_probe(gate: &VisionGate, probe: &VisionState) -> bool {
    if probe.model_id.is_none() {
        return false;
    }
    if gate.supported() != probe.vision_supported {
        info!(
            target: "assistd::vision",
            model = ?probe.model_id,
            vision_supported = probe.vision_supported,
            "loaded model's vision support changed; updating vision gate"
        );
    }
    gate.set(probe.vision_supported);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_probe_trusts_every_probe_that_reached_the_model() {
        let probe = |id: Option<&str>, vision| VisionState {
            model_id: id.map(str::to_string),
            vision_supported: vision,
            total_slots: None,
        };
        let cases = [
            ("failed probe", true, probe(None, false), true, false),
            (
                "reload lost vision",
                true,
                probe(Some("m"), false),
                false,
                true,
            ),
            (
                "reload gained vision",
                false,
                probe(Some("m"), true),
                true,
                true,
            ),
            ("unchanged", true, probe(Some("m"), true), true, true),
        ];
        for (label, gate_initial, probe, expected_gate, expected_reached) in cases {
            let gate = VisionGate::new(gate_initial);
            assert_eq!(apply_probe(&gate, &probe), expected_reached, "{label}");
            assert_eq!(gate.supported(), expected_gate, "{label}");
        }
    }

    #[test]
    fn a_probe_is_due_only_after_a_failure_or_a_possible_reload() {
        let (presence_tx, presence) = watch::channel(PresenceState::Active);
        let mut seen = SeenLoad {
            presence,
            llama_pid: Some(1),
            probed: true,
        };
        assert!(!seen.take_stale(Some(1)), "nothing changed");

        presence_tx.send(PresenceState::Sleeping).unwrap();
        presence_tx.send(PresenceState::Active).unwrap();
        assert!(seen.take_stale(Some(1)), "presence round trip");
        assert!(seen.take_stale(Some(1)), "probe dropped before landing");
        seen.probed = true;
        assert!(!seen.take_stale(Some(1)), "round trip already seen");

        assert!(seen.take_stale(Some(2)), "child restarted");
        seen.probed = true;
        assert!(!seen.take_stale(Some(2)), "restart already seen");

        seen.probed = false;
        assert!(seen.take_stale(Some(2)), "last probe failed");
    }
}

//! Hotkeys bound through Sway IPC.

use assistd_wm::{Hotkey, KeyEdge, SwayHotkeys};
use tokio::sync::watch;
use tracing::{info, warn};

use super::{Binding, HotkeyEvents, parse};

/// Sway's bindings, with the [`Binding`] each one's index stands for.
pub(super) struct SwayEvents {
    hotkeys: SwayHotkeys,
    bindings: Vec<Binding>,
}

impl SwayEvents {
    pub(super) fn spawn(specs: Vec<(Binding, &str)>, shutdown: watch::Receiver<bool>) -> Self {
        let (bindings, hotkeys) = specs
            .into_iter()
            .filter_map(|(binding, spec)| sway_hotkey(binding, spec).map(|h| (binding, h)))
            .unzip();
        Self {
            hotkeys: SwayHotkeys::spawn(hotkeys, shutdown),
            bindings,
        }
    }
}

impl HotkeyEvents for SwayEvents {
    async fn next_event(&mut self) -> Option<(Binding, KeyEdge)> {
        let event = self.hotkeys.next_event().await?;
        Some((self.bindings[event.index], event.edge))
    }

    async fn release(self) {
        self.hotkeys.shutdown().await;
    }
}

fn sway_hotkey(binding: Binding, spec: &str) -> Option<Hotkey> {
    let (_, keys) = parse(binding, spec)
        .map_err(|e| warn!(target: "assistd::hotkey", "{e:#}"))
        .ok()?;
    info!(
        target: "assistd::hotkey",
        "{} {spec:?} bound through sway as {keys} ({})",
        binding.config_path(),
        binding.hint()
    );
    Some(Hotkey {
        keys,
        report_release: binding == Binding::Voice,
    })
}

//! Global hotkey listener that cycles the daemon's presence state.
//!
//! Registration uses the `global-hotkey` crate, which only supports X11 on
//! Linux. On pure Wayland sessions we skip registration entirely and point
//! the user at the compositor-binding fallback (`assistd cycle`). On mixed
//! sessions (XWayland sets `DISPLAY`) we attempt registration but warn if
//! it fails — the daemon continues to serve the socket regardless.

use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_core::{PresenceConfig, PresenceManager};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState, hotkey::HotKey};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// Validate the configured hotkey string before the daemon starts anything
/// else. Empty is accepted (listener is disabled).
pub fn validate(cfg: &PresenceConfig) -> Result<()> {
    if cfg.hotkey.is_empty() {
        return Ok(());
    }
    HotKey::from_str(&cfg.hotkey)
        .map(|_| ())
        .with_context(|| format!("invalid presence.hotkey {:?}", cfg.hotkey))
}

/// Spawn a background task that listens for the configured global hotkey
/// and cycles the daemon's presence state on each press. Returns `None`
/// when the listener is disabled or can't be registered.
pub fn spawn_listener(
    cfg: &PresenceConfig,
    presence: Arc<PresenceManager>,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    if cfg.hotkey.is_empty() {
        info!(
            target: "assistd::hotkey",
            "presence.hotkey is empty; global hotkey listener disabled"
        );
        return None;
    }

    if is_wayland_only() {
        info!(
            target: "assistd::hotkey",
            "pure Wayland session detected; global hotkey disabled. Bind \
             `assistd cycle` in your compositor instead \
             (e.g. Sway: `bindsym $mod+Escape exec assistd cycle`)"
        );
        return None;
    }

    let manager = match GlobalHotKeyManager::new() {
        Ok(m) => m,
        Err(e) => {
            warn!(target: "assistd::hotkey", "failed to create GlobalHotKeyManager: {e}; hotkey disabled");
            return None;
        }
    };

    let hotkey = match HotKey::from_str(&cfg.hotkey) {
        Ok(h) => h,
        Err(e) => {
            warn!(
                target: "assistd::hotkey",
                "failed to parse presence.hotkey {:?}: {e}; hotkey disabled",
                cfg.hotkey
            );
            return None;
        }
    };

    if let Err(e) = manager.register(hotkey) {
        warn!(
            target: "assistd::hotkey",
            "failed to register global hotkey {:?}: {e}; hotkey disabled",
            cfg.hotkey
        );
        return None;
    }

    info!(
        target: "assistd::hotkey",
        "global hotkey {:?} registered — press to cycle presence",
        cfg.hotkey
    );

    let handle = tokio::spawn(run_listener(manager, hotkey, presence, shutdown));
    Some(handle)
}

async fn run_listener(
    manager: GlobalHotKeyManager,
    hotkey: HotKey,
    presence: Arc<PresenceManager>,
    mut shutdown: watch::Receiver<bool>,
) {
    let hotkey_id = hotkey.id();
    let receiver = GlobalHotKeyEvent::receiver();
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = tick.tick() => {
                while let Ok(event) = receiver.try_recv() {
                    if event.id == hotkey_id && event.state == HotKeyState::Pressed {
                        let p = presence.clone();
                        tokio::spawn(async move {
                            match p.cycle().await {
                                Ok(target) => info!(
                                    target: "assistd::hotkey",
                                    "hotkey cycled presence → {target:?}"
                                ),
                                Err(e) => warn!(
                                    target: "assistd::hotkey",
                                    "hotkey cycle failed: {e:#}"
                                ),
                            }
                        });
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
        }
    }

    if let Err(e) = manager.unregister(hotkey) {
        warn!(target: "assistd::hotkey", "failed to unregister hotkey on shutdown: {e}");
    }
}

fn is_wayland_only() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some() && std::env::var_os("DISPLAY").is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_empty_is_ok() {
        validate(&PresenceConfig {
            hotkey: String::new(),
        })
        .expect("empty hotkey valid");
    }

    #[test]
    fn validate_well_formed_is_ok() {
        validate(&PresenceConfig {
            hotkey: "Super+Escape".into(),
        })
        .expect("Super+Escape must parse");
    }

    #[test]
    fn validate_garbage_errors() {
        let err = validate(&PresenceConfig {
            hotkey: "not a real hotkey ###".into(),
        })
        .expect_err("garbage must fail");
        assert!(err.to_string().contains("presence.hotkey"));
    }
}

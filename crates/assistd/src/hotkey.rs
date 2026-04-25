//! Global hotkey listener that drives presence cycling and push-to-talk
//! voice capture.
//!
//! Registration uses the `global-hotkey` crate, which only supports X11 on
//! Linux. On pure Wayland sessions we skip registration entirely and point
//! the user at the compositor-binding fallback (`assistd cycle` or
//! `assistd ptt-start/stop`). On mixed sessions (XWayland sets `DISPLAY`)
//! we attempt registration but warn if it fails — the daemon continues to
//! serve the socket regardless.

use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use assistd_core::{
    ContinuousListener, PresenceConfig, PresenceManager, VoiceConfig, VoiceInput,
    VoiceOutputController,
};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState, hotkey::HotKey};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// Validate every configured hotkey string before the daemon starts
/// anything else. An empty string is accepted (the listener is
/// disabled).
pub fn validate(presence: &PresenceConfig, voice: &VoiceConfig) -> Result<()> {
    if !presence.hotkey.is_empty() {
        HotKey::from_str(&presence.hotkey)
            .map(|_| ())
            .with_context(|| format!("invalid presence.hotkey {:?}", presence.hotkey))?;
    }
    if voice.enabled && !voice.hotkey.is_empty() {
        HotKey::from_str(&voice.hotkey)
            .map(|_| ())
            .with_context(|| format!("invalid voice.hotkey {:?}", voice.hotkey))?;
    }
    if voice.enabled && voice.continuous.enabled && !voice.continuous.hotkey.is_empty() {
        HotKey::from_str(&voice.continuous.hotkey)
            .map(|_| ())
            .with_context(|| {
                format!(
                    "invalid voice.continuous.hotkey {:?}",
                    voice.continuous.hotkey
                )
            })?;
    }
    if voice.enabled && voice.synthesis.enabled && !voice.synthesis.toggle_hotkey.is_empty() {
        HotKey::from_str(&voice.synthesis.toggle_hotkey)
            .map(|_| ())
            .with_context(|| {
                format!(
                    "invalid voice.synthesis.toggle_hotkey {:?}",
                    voice.synthesis.toggle_hotkey
                )
            })?;
    }
    if voice.enabled && voice.synthesis.enabled && !voice.synthesis.skip_hotkey.is_empty() {
        HotKey::from_str(&voice.synthesis.skip_hotkey)
            .map(|_| ())
            .with_context(|| {
                format!(
                    "invalid voice.synthesis.skip_hotkey {:?}",
                    voice.synthesis.skip_hotkey
                )
            })?;
    }
    Ok(())
}

/// Spawn a background task that listens for the configured global
/// hotkeys and routes events to the right subsystem.
///
/// The presence hotkey fires on press (cycles presence state). The
/// voice hotkey fires on **both** press (starts PTT recording) and
/// release (stops recording + transcribes). Returns `None` when no
/// hotkeys are active or when registration failed — the daemon's
/// socket-based fallback always works regardless.
pub fn spawn_listener(
    presence_cfg: &PresenceConfig,
    voice_cfg: &VoiceConfig,
    presence: Option<Arc<PresenceManager>>,
    voice: Arc<dyn VoiceInput>,
    listener: Option<Arc<dyn ContinuousListener>>,
    voice_output: Option<Arc<VoiceOutputController>>,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    // Only bind the presence hotkey when the caller actually passed
    // a presence manager — chat TUI mode passes None, so we skip
    // even if the config is non-empty.
    let presence_hotkey = if presence_cfg.hotkey.is_empty() || presence.is_none() {
        None
    } else {
        Some(presence_cfg.hotkey.clone())
    };
    let voice_hotkey = if voice_cfg.enabled && !voice_cfg.hotkey.is_empty() {
        Some(voice_cfg.hotkey.clone())
    } else {
        None
    };
    // Continuous-listen hotkey needs the listener arc as well — chat
    // TUI mode may not supply one.
    let listen_hotkey = if voice_cfg.enabled
        && voice_cfg.continuous.enabled
        && !voice_cfg.continuous.hotkey.is_empty()
        && listener.is_some()
    {
        Some(voice_cfg.continuous.hotkey.clone())
    } else {
        None
    };
    // TTS toggle/skip hotkeys are only registered when the daemon
    // actually owns a VoiceOutputController (chat-TUI mode passes
    // None — those clients control TTS via IPC instead).
    let toggle_hotkey = if voice_cfg.enabled
        && voice_cfg.synthesis.enabled
        && !voice_cfg.synthesis.toggle_hotkey.is_empty()
        && voice_output.is_some()
    {
        Some(voice_cfg.synthesis.toggle_hotkey.clone())
    } else {
        None
    };
    let skip_hotkey = if voice_cfg.enabled
        && voice_cfg.synthesis.enabled
        && !voice_cfg.synthesis.skip_hotkey.is_empty()
        && voice_output.is_some()
    {
        Some(voice_cfg.synthesis.skip_hotkey.clone())
    } else {
        None
    };

    if presence_hotkey.is_none()
        && voice_hotkey.is_none()
        && listen_hotkey.is_none()
        && toggle_hotkey.is_none()
        && skip_hotkey.is_none()
    {
        info!(
            target: "assistd::hotkey",
            "no global hotkeys configured; hotkey listener disabled"
        );
        return None;
    }

    if is_wayland_only() {
        info!(
            target: "assistd::hotkey",
            "pure Wayland session detected; global hotkeys disabled. Bind \
             `assistd cycle` / `assistd ptt-start` / `assistd ptt-stop` in \
             your compositor instead (Sway/Hyprland: `bindsym ... exec ...`)"
        );
        return None;
    }

    let manager = match GlobalHotKeyManager::new() {
        Ok(m) => m,
        Err(e) => {
            warn!(target: "assistd::hotkey", "failed to create GlobalHotKeyManager: {e}; hotkeys disabled");
            return None;
        }
    };

    let presence_hk = presence_hotkey.and_then(|s| match HotKey::from_str(&s) {
        Ok(h) => match manager.register(h) {
            Ok(()) => {
                info!(target: "assistd::hotkey", "presence hotkey {s:?} registered (press to cycle)");
                Some(h)
            }
            Err(e) => {
                warn!(target: "assistd::hotkey", "failed to register presence.hotkey {s:?}: {e}");
                None
            }
        },
        Err(e) => {
            warn!(target: "assistd::hotkey", "failed to parse presence.hotkey {s:?}: {e}");
            None
        }
    });

    let voice_hk = voice_hotkey.and_then(|s| match HotKey::from_str(&s) {
        Ok(h) => match manager.register(h) {
            Ok(()) => {
                info!(target: "assistd::hotkey", "voice PTT hotkey {s:?} registered (hold to talk)");
                Some(h)
            }
            Err(e) => {
                warn!(target: "assistd::hotkey", "failed to register voice.hotkey {s:?}: {e}");
                None
            }
        },
        Err(e) => {
            warn!(target: "assistd::hotkey", "failed to parse voice.hotkey {s:?}: {e}");
            None
        }
    });

    let listen_hk = listen_hotkey.and_then(|s| match HotKey::from_str(&s) {
        Ok(h) => match manager.register(h) {
            Ok(()) => {
                info!(target: "assistd::hotkey", "continuous-listen hotkey {s:?} registered (press to toggle)");
                Some(h)
            }
            Err(e) => {
                warn!(target: "assistd::hotkey", "failed to register voice.continuous.hotkey {s:?}: {e}");
                None
            }
        },
        Err(e) => {
            warn!(target: "assistd::hotkey", "failed to parse voice.continuous.hotkey {s:?}: {e}");
            None
        }
    });

    let toggle_hk = toggle_hotkey.and_then(|s| match HotKey::from_str(&s) {
        Ok(h) => match manager.register(h) {
            Ok(()) => {
                info!(target: "assistd::hotkey", "voice-output toggle hotkey {s:?} registered (press to mute/unmute)");
                Some(h)
            }
            Err(e) => {
                warn!(target: "assistd::hotkey", "failed to register voice.synthesis.toggle_hotkey {s:?}: {e}");
                None
            }
        },
        Err(e) => {
            warn!(target: "assistd::hotkey", "failed to parse voice.synthesis.toggle_hotkey {s:?}: {e}");
            None
        }
    });

    let skip_hk = skip_hotkey.and_then(|s| match HotKey::from_str(&s) {
        Ok(h) => match manager.register(h) {
            Ok(()) => {
                info!(target: "assistd::hotkey", "voice-output skip hotkey {s:?} registered (press to abort current response)");
                Some(h)
            }
            Err(e) => {
                warn!(target: "assistd::hotkey", "failed to register voice.synthesis.skip_hotkey {s:?}: {e}");
                None
            }
        },
        Err(e) => {
            warn!(target: "assistd::hotkey", "failed to parse voice.synthesis.skip_hotkey {s:?}: {e}");
            None
        }
    });

    if presence_hk.is_none()
        && voice_hk.is_none()
        && listen_hk.is_none()
        && toggle_hk.is_none()
        && skip_hk.is_none()
    {
        return None;
    }

    let handle = tokio::spawn(run_listener(
        manager,
        presence_hk,
        voice_hk,
        listen_hk,
        toggle_hk,
        skip_hk,
        presence,
        voice,
        listener,
        voice_output,
        shutdown,
    ));
    Some(handle)
}

#[allow(clippy::too_many_arguments)]
async fn run_listener(
    manager: GlobalHotKeyManager,
    presence_hk: Option<HotKey>,
    voice_hk: Option<HotKey>,
    listen_hk: Option<HotKey>,
    toggle_hk: Option<HotKey>,
    skip_hk: Option<HotKey>,
    presence: Option<Arc<PresenceManager>>,
    voice: Arc<dyn VoiceInput>,
    listener: Option<Arc<dyn ContinuousListener>>,
    voice_output: Option<Arc<VoiceOutputController>>,
    mut shutdown: watch::Receiver<bool>,
) {
    let presence_id = presence_hk.map(|h| h.id());
    let voice_id = voice_hk.map(|h| h.id());
    let listen_id = listen_hk.map(|h| h.id());
    let toggle_id = toggle_hk.map(|h| h.id());
    let skip_id = skip_hk.map(|h| h.id());
    let receiver = GlobalHotKeyEvent::receiver();
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = tick.tick() => {
                while let Ok(event) = receiver.try_recv() {
                    if Some(event.id) == presence_id
                        && event.state == HotKeyState::Pressed
                        && let Some(ref p_arc) = presence
                    {
                        let p = p_arc.clone();
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
                    } else if Some(event.id) == voice_id {
                        let v = voice.clone();
                        match event.state {
                            HotKeyState::Pressed => {
                                // PTT barge-in: stop any in-flight TTS
                                // playback before opening the mic.
                                let vo = voice_output.clone();
                                tokio::spawn(async move {
                                    if let Some(ctrl) = vo {
                                        ctrl.interrupt().await;
                                    }
                                    if let Err(e) = v.start_recording().await {
                                        warn!(
                                            target: "assistd::hotkey",
                                            "voice start_recording failed: {e:#}"
                                        );
                                    }
                                });
                            }
                            HotKeyState::Released => {
                                tokio::spawn(async move {
                                    match v.stop_and_transcribe().await {
                                        Ok(text) if text.trim().is_empty() => {
                                            info!(
                                                target: "assistd::hotkey",
                                                "voice released: no speech detected (VAD)"
                                            );
                                        }
                                        Ok(text) => {
                                            info!(
                                                target: "assistd::hotkey",
                                                chars = text.chars().count(),
                                                "voice released: transcription complete"
                                            );
                                        }
                                        Err(e) => warn!(
                                            target: "assistd::hotkey",
                                            "voice stop_and_transcribe failed: {e:#}"
                                        ),
                                    }
                                });
                            }
                        }
                    } else if Some(event.id) == listen_id
                        && event.state == HotKeyState::Pressed
                        && let Some(ref listener_arc) = listener
                    {
                        let l = listener_arc.clone();
                        tokio::spawn(async move {
                            let result = if l.is_active() {
                                l.stop().await.map(|()| false)
                            } else {
                                l.start().await.map(|()| true)
                            };
                            match result {
                                Ok(active) => info!(
                                    target: "assistd::hotkey",
                                    active,
                                    "hotkey toggled continuous listening"
                                ),
                                Err(e) => warn!(
                                    target: "assistd::hotkey",
                                    "continuous-listen toggle failed: {e:#}"
                                ),
                            }
                        });
                    } else if Some(event.id) == toggle_id
                        && event.state == HotKeyState::Pressed
                        && let Some(ref ctrl_arc) = voice_output
                    {
                        let ctrl = ctrl_arc.clone();
                        tokio::spawn(async move {
                            let new_state = !ctrl.enabled();
                            ctrl.set_enabled(new_state).await;
                            info!(
                                target: "assistd::hotkey",
                                enabled = new_state,
                                "hotkey toggled voice output"
                            );
                        });
                    } else if Some(event.id) == skip_id
                        && event.state == HotKeyState::Pressed
                        && let Some(ref ctrl_arc) = voice_output
                    {
                        let ctrl = ctrl_arc.clone();
                        tokio::spawn(async move {
                            ctrl.skip().await;
                            info!(
                                target: "assistd::hotkey",
                                "hotkey skipped current voice-output response"
                            );
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

    if let Some(h) = presence_hk {
        if let Err(e) = manager.unregister(h) {
            warn!(target: "assistd::hotkey", "failed to unregister presence hotkey on shutdown: {e}");
        }
    }
    if let Some(h) = voice_hk {
        if let Err(e) = manager.unregister(h) {
            warn!(target: "assistd::hotkey", "failed to unregister voice hotkey on shutdown: {e}");
        }
    }
    if let Some(h) = listen_hk {
        if let Err(e) = manager.unregister(h) {
            warn!(target: "assistd::hotkey", "failed to unregister continuous-listen hotkey on shutdown: {e}");
        }
    }
    if let Some(h) = toggle_hk {
        if let Err(e) = manager.unregister(h) {
            warn!(target: "assistd::hotkey", "failed to unregister voice-output toggle hotkey on shutdown: {e}");
        }
    }
    if let Some(h) = skip_hk {
        if let Err(e) = manager.unregister(h) {
            warn!(target: "assistd::hotkey", "failed to unregister voice-output skip hotkey on shutdown: {e}");
        }
    }
}

fn is_wayland_only() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some() && std::env::var_os("DISPLAY").is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_voice() -> VoiceConfig {
        VoiceConfig::default()
    }

    #[test]
    fn validate_empty_is_ok() {
        validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &empty_voice(),
        )
        .expect("empty hotkeys valid");
    }

    #[test]
    fn validate_well_formed_is_ok() {
        validate(
            &PresenceConfig {
                hotkey: "Super+Escape".into(),
            },
            &empty_voice(),
        )
        .expect("Super+Escape must parse");
    }

    #[test]
    fn validate_garbage_errors() {
        let err = validate(
            &PresenceConfig {
                hotkey: "not a real hotkey ###".into(),
            },
            &empty_voice(),
        )
        .expect_err("garbage must fail");
        assert!(err.to_string().contains("presence.hotkey"));
    }

    #[test]
    fn validate_voice_hotkey_when_enabled() {
        let v = VoiceConfig {
            enabled: true,
            hotkey: "Super+Space".into(),
            ..VoiceConfig::default()
        };
        validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &v,
        )
        .expect("Super+Space must parse");
    }

    #[test]
    fn validate_continuous_hotkey_when_enabled() {
        use assistd_core::ContinuousListenConfig;
        let v = VoiceConfig {
            enabled: true,
            hotkey: String::new(),
            continuous: ContinuousListenConfig {
                enabled: true,
                hotkey: "Super+Shift+L".into(),
                ..ContinuousListenConfig::default()
            },
            ..VoiceConfig::default()
        };
        validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &v,
        )
        .expect("Super+Shift+L must parse");
    }

    #[test]
    fn validate_garbage_continuous_hotkey_errors() {
        use assistd_core::ContinuousListenConfig;
        let v = VoiceConfig {
            enabled: true,
            hotkey: String::new(),
            continuous: ContinuousListenConfig {
                enabled: true,
                hotkey: "### bogus ###".into(),
                ..ContinuousListenConfig::default()
            },
            ..VoiceConfig::default()
        };
        let err = validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &v,
        )
        .expect_err("garbage must fail");
        assert!(err.to_string().contains("voice.continuous.hotkey"));
    }

    #[test]
    fn validate_continuous_hotkey_ignored_when_disabled() {
        use assistd_core::ContinuousListenConfig;
        let v = VoiceConfig {
            enabled: true,
            hotkey: String::new(),
            continuous: ContinuousListenConfig {
                enabled: false,
                hotkey: "### bogus ###".into(),
                ..ContinuousListenConfig::default()
            },
            ..VoiceConfig::default()
        };
        validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &v,
        )
        .expect("garbage ignored when continuous is disabled");
    }

    #[test]
    fn validate_voice_hotkey_ignored_when_disabled() {
        let v = VoiceConfig {
            enabled: false,
            hotkey: "bogus###".into(),
            ..VoiceConfig::default()
        };
        validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &v,
        )
        .expect("garbage voice hotkey ignored when voice is disabled");
    }

    #[test]
    fn validate_toggle_hotkey_when_synthesis_enabled() {
        use assistd_core::SynthesisConfig;
        let v = VoiceConfig {
            enabled: true,
            hotkey: String::new(),
            synthesis: SynthesisConfig {
                enabled: true,
                toggle_hotkey: "Super+Shift+M".into(),
                ..SynthesisConfig::default()
            },
            ..VoiceConfig::default()
        };
        validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &v,
        )
        .expect("Super+Shift+M must parse");
    }

    #[test]
    fn validate_garbage_toggle_hotkey_errors() {
        use assistd_core::SynthesisConfig;
        let v = VoiceConfig {
            enabled: true,
            hotkey: String::new(),
            synthesis: SynthesisConfig {
                enabled: true,
                toggle_hotkey: "### bogus ###".into(),
                ..SynthesisConfig::default()
            },
            ..VoiceConfig::default()
        };
        let err = validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &v,
        )
        .expect_err("garbage must fail");
        assert!(err.to_string().contains("voice.synthesis.toggle_hotkey"));
    }

    #[test]
    fn validate_toggle_hotkey_ignored_when_synthesis_disabled() {
        use assistd_core::SynthesisConfig;
        let v = VoiceConfig {
            enabled: true,
            hotkey: String::new(),
            synthesis: SynthesisConfig {
                enabled: false,
                toggle_hotkey: "### bogus ###".into(),
                ..SynthesisConfig::default()
            },
            ..VoiceConfig::default()
        };
        validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &v,
        )
        .expect("garbage ignored when synthesis is disabled");
    }

    #[test]
    fn validate_skip_hotkey_when_synthesis_enabled() {
        use assistd_core::SynthesisConfig;
        let v = VoiceConfig {
            enabled: true,
            hotkey: String::new(),
            synthesis: SynthesisConfig {
                enabled: true,
                skip_hotkey: "Super+Shift+S".into(),
                ..SynthesisConfig::default()
            },
            ..VoiceConfig::default()
        };
        validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &v,
        )
        .expect("Super+Shift+S must parse");
    }

    #[test]
    fn validate_garbage_skip_hotkey_errors() {
        use assistd_core::SynthesisConfig;
        let v = VoiceConfig {
            enabled: true,
            hotkey: String::new(),
            synthesis: SynthesisConfig {
                enabled: true,
                skip_hotkey: "### bogus ###".into(),
                ..SynthesisConfig::default()
            },
            ..VoiceConfig::default()
        };
        let err = validate(
            &PresenceConfig {
                hotkey: String::new(),
            },
            &v,
        )
        .expect_err("garbage must fail");
        assert!(err.to_string().contains("voice.synthesis.skip_hotkey"));
    }
}

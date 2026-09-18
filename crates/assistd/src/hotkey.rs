//! Global hotkey listener for presence cycling and voice control.
//! `global-hotkey` only supports X11 on Linux, so pure Wayland sessions
//! get no listener and rely on compositor bindings to the CLI instead.

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

/// The five hotkeys the listener can register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Binding {
    Presence,
    Voice,
    Listen,
    Toggle,
    Skip,
}

const BINDINGS: [Binding; 5] = [
    Binding::Presence,
    Binding::Voice,
    Binding::Listen,
    Binding::Toggle,
    Binding::Skip,
];

impl Binding {
    fn config_path(self) -> &'static str {
        match self {
            Binding::Presence => "presence.hotkey",
            Binding::Voice => "voice.hotkey",
            Binding::Listen => "voice.continuous.hotkey",
            Binding::Toggle => "voice.synthesis.toggle_hotkey",
            Binding::Skip => "voice.synthesis.skip_hotkey",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Binding::Presence => "press to cycle",
            Binding::Voice => "hold to talk",
            Binding::Listen => "press to toggle",
            Binding::Toggle => "press to mute/unmute",
            Binding::Skip => "press to abort current response",
        }
    }

    /// The configured spec, or `None` when this binding's feature is
    /// off or its hotkey is empty.
    fn spec<'a>(self, presence: &'a PresenceConfig, voice: &'a VoiceConfig) -> Option<&'a str> {
        let (enabled, spec) = match self {
            Binding::Presence => (true, &presence.hotkey),
            Binding::Voice => (voice.enabled, &voice.hotkey),
            Binding::Listen => (
                voice.enabled && voice.continuous.enabled,
                &voice.continuous.hotkey,
            ),
            Binding::Toggle => (
                voice.enabled && voice.synthesis.enabled,
                &voice.synthesis.toggle_hotkey,
            ),
            Binding::Skip => (
                voice.enabled && voice.synthesis.enabled,
                &voice.synthesis.skip_hotkey,
            ),
        };
        (enabled && !spec.is_empty()).then_some(spec.as_str())
    }

    /// Whether the listener has a target to route this binding to.
    fn has_target(self, subsystems: &Subsystems) -> bool {
        match self {
            Binding::Presence => subsystems.presence.is_some(),
            Binding::Voice => true,
            Binding::Listen => subsystems.listener.is_some(),
            Binding::Toggle | Binding::Skip => subsystems.voice_output.is_some(),
        }
    }
}

/// Validate every configured hotkey string. Empty strings are accepted
/// and disable that hotkey.
pub fn validate(presence: &PresenceConfig, voice: &VoiceConfig) -> Result<()> {
    for binding in BINDINGS {
        if let Some(spec) = binding.spec(presence, voice) {
            HotKey::from_str(spec)
                .with_context(|| format!("invalid {} {spec:?}", binding.config_path()))?;
        }
    }
    Ok(())
}

/// Targets the hotkey listener routes events to. A `None` handle leaves
/// its hotkey unregistered.
pub struct Subsystems {
    pub presence: Option<Arc<PresenceManager>>,
    pub voice: Arc<dyn VoiceInput>,
    pub listener: Option<Arc<dyn ContinuousListener>>,
    pub voice_output: Option<Arc<VoiceOutputController>>,
}

/// Spawn the hotkey listener. `None` when no hotkey is configured, the
/// session is pure Wayland, or registration failed.
pub fn spawn_listener(
    presence_cfg: &PresenceConfig,
    voice_cfg: &VoiceConfig,
    subsystems: Subsystems,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    let specs: Vec<(Binding, &str)> = BINDINGS
        .into_iter()
        .filter(|b| b.has_target(&subsystems))
        .filter_map(|b| b.spec(presence_cfg, voice_cfg).map(|spec| (b, spec)))
        .collect();

    if specs.is_empty() {
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

    let registered: Vec<(Binding, HotKey)> = specs
        .into_iter()
        .filter_map(|(binding, spec)| register(&manager, binding, spec).map(|h| (binding, h)))
        .collect();
    if registered.is_empty() {
        return None;
    }

    Some(tokio::spawn(run_listener(
        manager, registered, subsystems, shutdown,
    )))
}

fn register(manager: &GlobalHotKeyManager, binding: Binding, spec: &str) -> Option<HotKey> {
    let config_path = binding.config_path();
    let hotkey = match HotKey::from_str(spec) {
        Ok(h) => h,
        Err(e) => {
            warn!(target: "assistd::hotkey", "failed to parse {config_path} {spec:?}: {e}");
            return None;
        }
    };
    match manager.register(hotkey) {
        Ok(()) => {
            info!(
                target: "assistd::hotkey",
                "{config_path} {spec:?} registered ({})",
                binding.hint()
            );
            Some(hotkey)
        }
        Err(e) => {
            warn!(target: "assistd::hotkey", "failed to register {config_path} {spec:?}: {e}");
            None
        }
    }
}

async fn run_listener(
    manager: GlobalHotKeyManager,
    registered: Vec<(Binding, HotKey)>,
    subsystems: Subsystems,
    mut shutdown: watch::Receiver<bool>,
) {
    let receiver = GlobalHotKeyEvent::receiver();
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = tick.tick() => {
                while let Ok(event) = receiver.try_recv() {
                    let binding = registered
                        .iter()
                        .find(|(_, h)| h.id() == event.id)
                        .map(|(b, _)| *b);
                    if let Some(binding) = binding {
                        on_hotkey(binding, event.state, &subsystems);
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

    for (binding, hotkey) in registered {
        if let Err(e) = manager.unregister(hotkey) {
            warn!(
                target: "assistd::hotkey",
                "failed to unregister {} on shutdown: {e}",
                binding.config_path()
            );
        }
    }
}

/// Route one hotkey event to its subsystem on a fresh task, so the
/// listener never blocks on a slow transition.
fn on_hotkey(binding: Binding, state: HotKeyState, subsystems: &Subsystems) {
    let pressed = state == HotKeyState::Pressed;
    match binding {
        Binding::Presence if pressed => {
            let Some(presence) = subsystems.presence.clone() else {
                return;
            };
            tokio::spawn(async move {
                match presence.cycle().await {
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
        Binding::Voice if pressed => {
            let voice = subsystems.voice.clone();
            let voice_output = subsystems.voice_output.clone();
            tokio::spawn(async move {
                if let Some(ctrl) = voice_output {
                    ctrl.interrupt().await;
                }
                if let Err(e) = voice.start_recording().await {
                    warn!(
                        target: "assistd::hotkey",
                        "voice start_recording failed: {e:#}"
                    );
                }
            });
        }
        Binding::Voice => {
            let voice = subsystems.voice.clone();
            tokio::spawn(async move {
                match voice.stop_and_transcribe().await {
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
        Binding::Listen if pressed => {
            let Some(listener) = subsystems.listener.clone() else {
                return;
            };
            tokio::spawn(async move {
                let result = if listener.is_active() {
                    listener.stop().await.map(|()| false)
                } else {
                    listener.start().await.map(|()| true)
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
        }
        Binding::Toggle if pressed => {
            let Some(ctrl) = subsystems.voice_output.clone() else {
                return;
            };
            tokio::spawn(async move {
                let new_state = !ctrl.enabled();
                ctrl.set_enabled(new_state).await;
                info!(
                    target: "assistd::hotkey",
                    enabled = new_state,
                    "hotkey toggled voice output"
                );
            });
        }
        Binding::Skip if pressed => {
            let Some(ctrl) = subsystems.voice_output.clone() else {
                return;
            };
            tokio::spawn(async move {
                ctrl.skip().await;
                info!(
                    target: "assistd::hotkey",
                    "hotkey skipped current voice-output response"
                );
            });
        }
        Binding::Presence | Binding::Listen | Binding::Toggle | Binding::Skip => {}
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

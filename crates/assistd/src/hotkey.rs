//! Global hotkey listener for voice control, bound through Sway IPC on Sway
//! and grabbed on the X11 root window on i3 and other X11 sessions.

use std::future::Future;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use assistd_config::Compositor;
use assistd_core::{
    Component, HotkeysConfig, VoiceConfig, VoiceInput, VoiceManager, VoiceOutputController,
    spawn_supervised,
};
use assistd_wm::KeyEdge;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState, hotkey::HotKey};
use tokio::sync::{oneshot, watch};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::Interval;
use tracing::{info, warn};

mod key_combo;
#[cfg(feature = "sway")]
mod sway;

const X11_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// The four hotkeys the listener can register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Binding {
    Voice,
    Listen,
    Toggle,
    Skip,
}

const BINDINGS: [Binding; 4] = [
    Binding::Voice,
    Binding::Listen,
    Binding::Toggle,
    Binding::Skip,
];

impl Binding {
    fn config_path(self) -> &'static str {
        match self {
            Binding::Voice => "hotkeys.push_to_talk",
            Binding::Listen => "hotkeys.toggle_listening",
            Binding::Toggle => "hotkeys.toggle_speech",
            Binding::Skip => "hotkeys.skip_speech",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Binding::Voice => "hold to talk",
            Binding::Listen => "press to toggle",
            Binding::Toggle => "press to mute/unmute",
            Binding::Skip => "press to abort current response",
        }
    }

    /// The configured spec, or `None` when this binding's feature is
    /// off or its hotkey is empty.
    fn spec<'a>(self, hotkeys: &'a HotkeysConfig, voice: &VoiceConfig) -> Option<&'a str> {
        let (enabled, spec) = match self {
            Binding::Voice => (voice.enabled, &hotkeys.push_to_talk),
            Binding::Listen => (
                voice.enabled && voice.wakeword.enabled,
                &hotkeys.toggle_listening,
            ),
            Binding::Toggle => (
                voice.enabled && voice.synthesis.enabled,
                &hotkeys.toggle_speech,
            ),
            Binding::Skip => (
                voice.enabled && voice.synthesis.enabled,
                &hotkeys.skip_speech,
            ),
        };
        (enabled && !spec.is_empty()).then_some(spec.as_str())
    }

    /// Whether the listener has a target to route this binding to.
    fn has_target(self, subsystems: &Subsystems) -> bool {
        match self {
            Binding::Voice => true,
            Binding::Listen | Binding::Toggle | Binding::Skip => subsystems.voice.is_some(),
        }
    }
}

/// Validate every configured hotkey string, including that it translates
/// into a compositor binding. Empty strings are accepted and disable that
/// hotkey.
pub(crate) fn validate(hotkeys: &HotkeysConfig, voice: &VoiceConfig) -> Result<()> {
    for binding in BINDINGS {
        if let Some(spec) = binding.spec(hotkeys, voice) {
            parse(binding, spec)?;
        }
    }
    Ok(())
}

/// Targets the hotkey listener routes events to. A `None` handle leaves
/// its hotkey unregistered.
pub(crate) struct Subsystems {
    pub ptt: Arc<dyn VoiceInput>,
    pub voice: Option<Arc<VoiceManager>>,
}

/// Spawn the hotkey listener, registering through Sway IPC on Sway and as
/// X11 grabs on i3 or an undetected X11 session. `None` when no hotkey is
/// configured, the compositor offers neither, or registration failed.
pub(crate) fn spawn_listener(
    hotkeys_cfg: &HotkeysConfig,
    voice_cfg: &VoiceConfig,
    compositor: Option<Compositor>,
    subsystems: Subsystems,
    shutdown: watch::Receiver<bool>,
) -> Option<JoinHandle<()>> {
    let specs: Vec<(Binding, &str)> = BINDINGS
        .into_iter()
        .filter(|b| b.has_target(&subsystems))
        .filter_map(|b| b.spec(hotkeys_cfg, voice_cfg).map(|spec| (b, spec)))
        .collect();

    if specs.is_empty() {
        info!(
            target: "assistd::hotkey",
            "no global hotkeys configured; hotkey listener disabled"
        );
        return None;
    }

    match registrar(compositor)? {
        Registrar::X11 => Some(spawn_events(
            X11Hotkeys::register(specs)?,
            subsystems,
            shutdown,
        )),
        #[cfg(feature = "sway")]
        Registrar::Sway => {
            let events = sway::SwayEvents::spawn(specs, shutdown.clone());
            Some(spawn_events(events, subsystems, shutdown))
        }
    }
}

fn spawn_events(
    events: impl HotkeyEvents,
    subsystems: Subsystems,
    shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    spawn_supervised(
        "hotkey_listener",
        Component::Hotkey,
        run_listener(events, subsystems, shutdown),
    )
}

/// Where hotkeys are registered.
enum Registrar {
    X11,
    #[cfg(feature = "sway")]
    Sway,
}

/// The registrar for `compositor`, or `None` (logged) when hotkeys cannot
/// be registered on it. i3 cannot bind keys over IPC, but runs on X11.
fn registrar(compositor: Option<Compositor>) -> Option<Registrar> {
    match compositor {
        #[cfg(feature = "sway")]
        Some(Compositor::Sway) => Some(Registrar::Sway),
        #[cfg(not(feature = "sway"))]
        Some(Compositor::Sway) => {
            warn!(
                target: "assistd::hotkey",
                "sway hotkeys not compiled into this build (feature `sway`); hotkeys disabled"
            );
            None
        }
        Some(Compositor::I3) => Some(Registrar::X11),
        Some(Compositor::Hyprland) => {
            info!(
                target: "assistd::hotkey",
                "no hotkey support on Hyprland; bind `assistd ptt-start` / \
                 `assistd ptt-stop` in the compositor instead"
            );
            None
        }
        None if std::env::var_os("WAYLAND_DISPLAY").is_some() => {
            info!(
                target: "assistd::hotkey",
                "Wayland session without a supported compositor; global hotkeys \
                 disabled. Bind `assistd ptt-start` / `assistd ptt-stop` in \
                 the compositor instead"
            );
            None
        }
        None => Some(Registrar::X11),
    }
}

/// The X11 hotkey for `spec`, checked to also translate into a compositor
/// binding.
fn parse(binding: Binding, spec: &str) -> Result<HotKey> {
    let config_path = binding.config_path();
    let hotkey =
        HotKey::from_str(spec).with_context(|| format!("invalid {config_path} {spec:?}"))?;
    if key_combo::key_combo(&hotkey).is_none() {
        bail!("invalid {config_path} {spec:?}: its key cannot be bound through Sway");
    }
    Ok(hotkey)
}

/// Registered hotkeys and the events they fire.
trait HotkeyEvents: Send + 'static {
    /// The next hotkey press or release. Cancel-safe.
    fn next_event(&mut self) -> impl Future<Output = Option<(Binding, KeyEdge)>> + Send;

    /// Remove every registration. Flip the shutdown watch first.
    fn release(self) -> impl Future<Output = ()> + Send;
}

/// Hotkeys grabbed on the X11 root window.
struct X11Hotkeys {
    manager: GlobalHotKeyManager,
    registered: Vec<(Binding, HotKey)>,
    tick: Interval,
}

impl X11Hotkeys {
    fn register(specs: Vec<(Binding, &str)>) -> Option<Self> {
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
        let mut tick = tokio::time::interval(X11_POLL_INTERVAL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        Some(Self {
            manager,
            registered,
            tick,
        })
    }

    fn binding_for(&self, event: GlobalHotKeyEvent) -> Option<(Binding, KeyEdge)> {
        let binding = self
            .registered
            .iter()
            .find(|(_, h)| h.id() == event.id)
            .map(|(b, _)| *b)?;
        let edge = match event.state {
            HotKeyState::Pressed => KeyEdge::Press,
            HotKeyState::Released => KeyEdge::Release,
        };
        Some((binding, edge))
    }
}

impl HotkeyEvents for X11Hotkeys {
    async fn next_event(&mut self) -> Option<(Binding, KeyEdge)> {
        let receiver = GlobalHotKeyEvent::receiver();
        loop {
            while let Ok(event) = receiver.try_recv() {
                if let Some(fired) = self.binding_for(event) {
                    return Some(fired);
                }
            }
            self.tick.tick().await;
        }
    }

    async fn release(self) {
        for (binding, hotkey) in self.registered {
            if let Err(e) = self.manager.unregister(hotkey) {
                warn!(
                    target: "assistd::hotkey",
                    "failed to unregister {} on shutdown: {e}",
                    binding.config_path()
                );
            }
        }
    }
}

fn register(manager: &GlobalHotKeyManager, binding: Binding, spec: &str) -> Option<HotKey> {
    let config_path = binding.config_path();
    let hotkey = match parse(binding, spec) {
        Ok(h) => h,
        Err(e) => {
            warn!(target: "assistd::hotkey", "{e:#}");
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
    mut source: impl HotkeyEvents,
    subsystems: Subsystems,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut handlers = JoinSet::new();
    let mut ptt_started = None;

    loop {
        tokio::select! {
            Some((binding, edge)) = source.next_event() => {
                on_hotkey(binding, edge, &subsystems, &mut handlers, &mut ptt_started);
            }
            Some(res) = handlers.join_next() => {
                if let Err(e) = res {
                    warn!(target: "assistd::hotkey", "hotkey handler failed: {e}");
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
        }
    }

    handlers.shutdown().await;
    source.release().await;
}

/// Route one hotkey event to its subsystem on a task in `handlers`, so
/// the listener never blocks on a slow transition. `ptt_started` resolves
/// once the latest push-to-talk press has finished opening the mic, so its
/// release cannot overtake it.
fn on_hotkey(
    binding: Binding,
    edge: KeyEdge,
    subsystems: &Subsystems,
    handlers: &mut JoinSet<()>,
    ptt_started: &mut Option<oneshot::Receiver<()>>,
) {
    let pressed = edge == KeyEdge::Press;
    match binding {
        Binding::Voice if pressed => {
            let (started_tx, started_rx) = oneshot::channel();
            *ptt_started = Some(started_rx);
            handlers.spawn(begin_push_to_talk(
                subsystems.ptt.clone(),
                subsystems
                    .voice
                    .as_ref()
                    .map(|voice| voice.speech().clone()),
                started_tx,
            ));
        }
        Binding::Voice => {
            handlers.spawn(end_push_to_talk(subsystems.ptt.clone(), ptt_started.take()));
        }
        Binding::Listen if pressed => {
            if let Some(voice) = subsystems.voice.clone() {
                handlers.spawn(toggle_listening(voice));
            }
        }
        Binding::Toggle if pressed => {
            if let Some(voice) = &subsystems.voice {
                handlers.spawn(toggle_voice_output(voice.speech().clone()));
            }
        }
        Binding::Skip if pressed => {
            if let Some(voice) = &subsystems.voice {
                handlers.spawn(skip_voice_output(voice.speech().clone()));
            }
        }
        Binding::Listen | Binding::Toggle | Binding::Skip => {}
    }
}

async fn begin_push_to_talk(
    voice: Arc<dyn VoiceInput>,
    voice_output: Option<Arc<VoiceOutputController>>,
    started: oneshot::Sender<()>,
) {
    if let Some(ctrl) = voice_output {
        ctrl.interrupt().await;
    }
    if let Err(e) = voice.start_recording().await {
        warn!(
            target: "assistd::hotkey",
            "voice start_recording failed: {e:#}"
        );
    }
    let _ = started.send(());
}

async fn end_push_to_talk(
    voice: Arc<dyn VoiceInput>,
    press_started: Option<oneshot::Receiver<()>>,
) {
    if let Some(press_started) = press_started {
        let _ = press_started.await;
    }
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
}

async fn toggle_listening(voice: Arc<VoiceManager>) {
    let listener = match voice.capture() {
        Ok(capture) => capture.listener,
        Err(e) => {
            warn!(target: "assistd::hotkey", "continuous-listen toggle failed: {e}");
            return;
        }
    };
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
}

async fn toggle_voice_output(ctrl: Arc<VoiceOutputController>) {
    let enabled = !ctrl.enabled();
    ctrl.set_enabled(enabled).await;
    info!(
        target: "assistd::hotkey",
        enabled,
        "hotkey toggled voice output"
    );
}

async fn skip_voice_output(ctrl: Arc<VoiceOutputController>) {
    ctrl.skip().await;
    info!(
        target: "assistd::hotkey",
        "hotkey skipped current voice-output response"
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use assistd_core::VoiceCaptureState;
    use assistd_voice::VoiceInputError;
    use async_trait::async_trait;

    use super::*;

    const GARBAGE: &str = "### bogus ###";

    /// Records call order; `start_recording` is slow, like a real mic open.
    #[derive(Debug, Default)]
    struct CallOrderVoice {
        calls: Mutex<Vec<&'static str>>,
    }

    #[async_trait]
    impl VoiceInput for CallOrderVoice {
        async fn start_recording(&self) -> Result<(), VoiceInputError> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            self.calls.lock().unwrap().push("start");
            Ok(())
        }

        async fn stop_and_transcribe(&self) -> Result<String, VoiceInputError> {
            self.calls.lock().unwrap().push("stop");
            Ok(String::new())
        }

        fn state(&self) -> VoiceCaptureState {
            VoiceCaptureState::Idle
        }

        fn subscribe(&self) -> watch::Receiver<VoiceCaptureState> {
            watch::channel(VoiceCaptureState::Idle).1
        }
    }

    type Edit = fn(&mut HotkeysConfig, &mut VoiceConfig);

    /// Every hotkey feature enabled, every hotkey empty.
    fn configs(edit: Edit) -> (HotkeysConfig, VoiceConfig) {
        let mut hotkeys = HotkeysConfig {
            push_to_talk: String::new(),
            ..HotkeysConfig::default()
        };
        let mut voice = VoiceConfig {
            enabled: true,
            ..VoiceConfig::default()
        };
        voice.wakeword.enabled = true;
        voice.synthesis.enabled = true;
        edit(&mut hotkeys, &mut voice);
        (hotkeys, voice)
    }

    #[test]
    fn validate_rejects_garbage_only_for_enabled_features() {
        let (h, v) = configs(|h, _| h.toggle_listening = GARBAGE.into());
        let err = validate(&h, &v).expect_err("garbage hotkey must be rejected");
        assert!(
            err.to_string()
                .starts_with("invalid hotkeys.toggle_listening "),
            "{err}"
        );

        let (h, v) = configs(|h, v| {
            v.wakeword.enabled = false;
            h.toggle_listening = GARBAGE.into();
        });
        validate(&h, &v).expect("disabled feature's hotkey is ignored");
    }

    #[tokio::test]
    async fn quick_tap_stops_only_after_recording_started() {
        let voice = Arc::new(CallOrderVoice::default());
        let subsystems = Subsystems {
            ptt: voice.clone(),
            voice: None,
        };
        let mut handlers = JoinSet::new();
        let mut ptt_started = None;

        for edge in [KeyEdge::Press, KeyEdge::Release] {
            on_hotkey(
                Binding::Voice,
                edge,
                &subsystems,
                &mut handlers,
                &mut ptt_started,
            );
        }
        while let Some(res) = handlers.join_next().await {
            res.expect("handler panicked");
        }

        assert_eq!(*voice.calls.lock().unwrap(), ["start", "stop"]);
    }
}

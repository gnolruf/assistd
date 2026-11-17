//! The voice stack the worker builds: sherpa-onnx speech output, then mic
//! capture with its transcriber and, when enabled, the wake-word listener.

use std::sync::Arc;
use std::time::Duration;

use assistd_config::{SynthesisConfig, VoiceConfig};
use assistd_utils::backoff::backoff_delay;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::listen::{ContinuousListener, MicContinuousListener, NoContinuousListener};
use crate::mic::capture::{DeviceValidationError, validate};
use crate::tts::SherpaVoiceOutput;
use crate::{
    MicVoiceInput, SherpaTranscriber, SileroVad, Transcriber, VoiceCapture, VoiceOutput,
    VoiceOutputController, WakeWordSpotter,
};

/// Longest wait between checks for an unplugged `voice.mic_device`.
const MIC_POLL_CAP: Duration = Duration::from_secs(10);

/// Speech output, or why there is none.
pub(super) async fn start_output(config: &SynthesisConfig) -> Result<Arc<dyn VoiceOutput>, String> {
    if !config.enabled {
        info!("voice.synthesis: disabled in config (voice.synthesis.enabled = false)");
        return Err("disabled in config (voice.synthesis.enabled = false)".into());
    }
    info!("voice.synthesis: loading the voice ({})", config.voice);
    match SherpaVoiceOutput::start(config).await {
        Ok(output) => {
            info!("voice.synthesis: speech ready");
            Ok(Arc::new(output))
        }
        Err(e) => {
            warn!("voice.synthesis failed to initialize: {e:#}; speech output disabled");
            Err(format!("speech failed to start: {e:#}"))
        }
    }
}

/// Push-to-talk input and the listener, or why there are none. `speech`
/// supplies the reply text barge-in checks reject as echo; `capture_lost`
/// is told why whenever listening cannot open the mic or loses it. While the
/// configured device is unplugged this waits for it, handing `on_missing`
/// the reason once.
pub(super) async fn start_capture(
    config: &VoiceConfig,
    speech: Arc<VoiceOutputController>,
    capture_lost: mpsc::UnboundedSender<String>,
    on_missing: impl FnOnce(String),
) -> Result<VoiceCapture, String> {
    if !config.enabled {
        info!("voice: disabled in config (voice.enabled = false)");
        return Err("disabled in config (voice.enabled = false)".into());
    }
    wait_for_mic_device(config.mic_device.as_deref(), on_missing).await?;
    info!("voice: building mic input ({})", config.transcription.model);
    let transcriber: Arc<dyn Transcriber> = Arc::new(
        SherpaTranscriber::load(&config.transcription)
            .await
            .map_err(|e| {
                warn!("voice input failed to initialize: {e:#}; PTT commands will error");
                format!("speech-to-text failed to initialize: {e:#}")
            })?,
    );
    let mic = MicVoiceInput::new(transcriber.clone(), config.mic_device.clone());
    let listener = match start_listener(config, &transcriber, speech, capture_lost).await {
        Some(listener) => listener,
        None => Arc::new(NoContinuousListener::new()),
    };
    Ok(VoiceCapture {
        input: Arc::new(mic),
        listener,
    })
}

/// Return once the configured input device is plugged in, polling with
/// backoff while it is missing; errors when devices cannot be enumerated.
pub(super) async fn wait_for_mic_device(
    mic_device: Option<&str>,
    on_missing: impl FnOnce(String),
) -> Result<(), String> {
    let mut on_missing = Some(on_missing);
    let mut attempt = 0;
    loop {
        let missing = match check_mic_device(mic_device.map(str::to_owned)).await? {
            Ok(()) => return Ok(()),
            Err(missing @ DeviceValidationError::NotFound { .. }) => missing.to_string(),
            Err(e) => {
                warn!("voice: {e}; voice capture unavailable");
                return Err(e.to_string());
            }
        };
        if let Some(report_missing) = on_missing.take() {
            warn!("voice: {missing}; voice capture waits for the device to appear");
            report_missing(missing);
        }
        tokio::time::sleep(backoff_delay(attempt).min(MIC_POLL_CAP)).await;
        attempt = attempt.saturating_add(1);
    }
}

/// Whether the configured input device is plugged in; errors when the check
/// itself could not run.
async fn check_mic_device(
    mic_device: Option<String>,
) -> Result<Result<(), DeviceValidationError>, String> {
    tokio::task::spawn_blocking(move || validate(mic_device.as_deref()))
        .await
        .map_err(|e| format!("mic device check failed: {e}"))
}

/// Wake-word listening, or `None` when it is off in config or its Silero or
/// keyword model cannot be loaded.
async fn start_listener(
    config: &VoiceConfig,
    transcriber: &Arc<dyn Transcriber>,
    speech: Arc<VoiceOutputController>,
    capture_lost: mpsc::UnboundedSender<String>,
) -> Option<Arc<dyn ContinuousListener>> {
    if !config.wakeword.enabled {
        info!("voice.wakeword: disabled in config");
        return None;
    }
    let silero = SileroVad::load(&config.transcription)
        .await
        .inspect_err(|e| {
            warn!("voice.wakeword: Silero VAD failed to load: {e:#}; listening unavailable");
        })
        .ok()?;
    let spotter = WakeWordSpotter::load(config)
        .await
        .inspect_err(|e| warn!("voice.wakeword: {e:#}; listening unavailable"))
        .ok()?;
    info!(
        "voice.wakeword: listening for {:?} (start_on_launch={})",
        spotter.phrase(),
        config.wakeword.start_on_launch
    );
    let listener = MicContinuousListener::new(
        transcriber.clone(),
        config,
        speech,
        silero,
        Arc::new(spotter),
        capture_lost,
    );
    Some(Arc::new(listener))
}

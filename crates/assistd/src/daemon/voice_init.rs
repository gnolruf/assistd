//! Voice startup for the daemon: brings up Piper and Whisper behind a
//! [`VoiceManager`] that already serves requests.

use std::sync::Arc;

use assistd_core::{
    Config, ContinuousListener, NoContinuousListener, PresenceManager, VoiceCapture, VoiceManager,
    VoiceOutput,
};
use assistd_utils::readiness::Readiness;
use assistd_voice::{
    CpuFallbackFactory, MicContinuousListener, MicVoiceInput, PiperVoiceOutput, QueueConfig,
    QueuedTranscriber, Transcriber, WhisperTranscriberBuilder, build_cpu_fallback,
};
use tokio::sync::watch;
use tracing::info;

use super::voice_probe::PresenceGpuProbe;

/// Start speech output, then capture, recording on `voice` whether each
/// came up or why it is unavailable.
pub(super) async fn init(voice: &VoiceManager, config: &Config, presence: &Arc<PresenceManager>) {
    voice.speech().set_output(init_output(config).await);
    let speaking = voice.speech().subscribe_speaking();
    voice.set_capture(init_capture(config, presence, speaking).await);
}

async fn init_capture(
    config: &Config,
    presence: &Arc<PresenceManager>,
    output_speaking: watch::Receiver<bool>,
) -> Readiness<VoiceCapture> {
    if !config.voice.enabled {
        info!("voice: disabled in config (voice.enabled = false)");
        return Readiness::Unavailable("disabled in config (voice.enabled = false)".into());
    }

    info!(
        "voice: building mic input ({})",
        config.voice.transcription.model
    );
    let primary = match WhisperTranscriberBuilder::from_config(&config.voice.transcription)
        .build()
        .await
    {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("voice input failed to initialize: {e:#}; PTT commands will error");
            return Readiness::Unavailable(format!("Whisper failed to initialize: {e:#}").into());
        }
    };
    let is_gpu = primary.is_gpu();
    let transcriber = with_cpu_fallback(config, presence, Arc::new(primary), is_gpu);

    let mic = MicVoiceInput::new(transcriber.clone(), config.voice.mic_device.clone());
    let listener: Arc<dyn ContinuousListener> = if config.voice.continuous.enabled {
        info!(
            "voice.continuous: enabled (hotkey={:?}, start_on_launch={})",
            config.voice.continuous.hotkey, config.voice.continuous.start_on_launch
        );
        warn_if_ungated_playback(config);
        Arc::new(MicContinuousListener::new(
            transcriber,
            &config.voice,
            output_speaking,
        ))
    } else {
        info!("voice.continuous: disabled in config");
        Arc::new(NoContinuousListener::new())
    };
    Readiness::Ready(VoiceCapture {
        input: Arc::new(mic),
        listener,
    })
}

/// With the gate off and no echo cancellation, spoken replies come back
/// through the mic as new queries.
fn warn_if_ungated_playback(config: &Config) {
    if config.voice.continuous.playback_gate || !config.voice.synthesis.enabled {
        return;
    }
    tracing::warn!(
        "voice.continuous.playback_gate = false with synthesis enabled: \
         the mic stays open while replies are spoken; without an \
         echo-cancelled mic source the daemon will answer its own speech \
         (see docs/voice/echo-cancellation.md)"
    );
}

/// Wrap a GPU transcriber so it falls back to CPU while the GPU is busy;
/// CPU transcribers are returned unchanged.
fn with_cpu_fallback(
    config: &Config,
    presence: &Arc<PresenceManager>,
    primary: Arc<dyn Transcriber>,
    is_gpu: bool,
) -> Arc<dyn Transcriber> {
    let queue_cfg = QueueConfig::default();
    if !is_gpu {
        info!("voice: CPU transcription active");
        return primary;
    }
    if !queue_cfg.cpu_fallback_enabled {
        info!("voice: GPU transcription active; CPU fallback disabled by config");
        return primary;
    }

    let probe = Arc::new(PresenceGpuProbe::new(
        presence.clone(),
        config.sleep.gpu_allowlist.clone(),
    ));
    let cpu_cfg = config.voice.transcription.clone();
    let cpu_factory: CpuFallbackFactory = Arc::new(move || {
        let cfg = cpu_cfg.clone();
        Box::pin(async move {
            let fallback = build_cpu_fallback(&cfg, None).await?;
            Ok(Arc::new(fallback) as Arc<dyn Transcriber>)
        })
    });
    info!(
        "voice: GPU transcription active; CPU fallback armed \
             (gpu_busy_timeout_ms={})",
        queue_cfg.gpu_busy_timeout_ms
    );
    Arc::new(QueuedTranscriber::new(
        primary,
        cpu_factory,
        probe,
        queue_cfg,
    ))
}

async fn init_output(config: &Config) -> Readiness<Arc<dyn VoiceOutput>> {
    if !config.voice.synthesis.enabled {
        info!("voice.synthesis: disabled in config (voice.synthesis.enabled = false)");
        return Readiness::Unavailable(
            "disabled in config (voice.synthesis.enabled = false)".into(),
        );
    }
    info!(
        "voice.synthesis: starting Piper ({})",
        config.voice.synthesis.voice
    );
    match PiperVoiceOutput::start(config.voice.synthesis.clone()).await {
        Ok(p) => {
            info!("voice.synthesis: Piper ready");
            Readiness::Ready(Arc::new(p))
        }
        Err(e) => {
            tracing::warn!("voice.synthesis failed to initialize: {e:#}; speech output disabled");
            Readiness::Unavailable(format!("Piper failed to start: {e:#}").into())
        }
    }
}

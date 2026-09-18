//! Voice subsystem wiring for the daemon.

use std::sync::Arc;

use assistd_core::{
    Config, ContinuousListener, NoContinuousListener, NoVoiceInput, NoVoiceOutput, PresenceManager,
    VoiceInput, VoiceOutput,
};
use assistd_voice::{
    MicContinuousListener, MicVoiceInput, QueueConfig, QueuedTranscriber, Transcriber,
    VoiceOutputController, WhisperTranscriberBuilder, build_cpu_fallback,
};
use tracing::info;

use crate::voice_probe::PresenceGpuProbe;

pub struct VoiceSubsystem {
    pub input: Arc<dyn VoiceInput>,
    pub listener: Arc<dyn ContinuousListener>,
    pub output: Arc<VoiceOutputController>,
}

/// Every handle degrades to a no-op when its feature is disabled or
/// fails to initialise.
pub async fn init(config: &Config, presence: &Arc<PresenceManager>) -> VoiceSubsystem {
    let (input, listener) = init_input(config, presence).await;
    let output_inner = init_output(config).await;
    let output = VoiceOutputController::new(output_inner, config.voice.synthesis.enabled);
    VoiceSubsystem {
        input,
        listener,
        output,
    }
}

async fn init_input(
    config: &Config,
    presence: &Arc<PresenceManager>,
) -> (Arc<dyn VoiceInput>, Arc<dyn ContinuousListener>) {
    fn disabled_voice() -> (Arc<dyn VoiceInput>, Arc<dyn ContinuousListener>) {
        (
            Arc::new(NoVoiceInput::new()),
            Arc::new(NoContinuousListener::new()),
        )
    }

    if !config.voice.enabled {
        info!("voice: disabled in config (voice.enabled = false)");
        return disabled_voice();
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
            return disabled_voice();
        }
    };

    let is_gpu = primary.is_gpu();
    let primary: Arc<dyn Transcriber> = Arc::new(primary);

    let queue_cfg = QueueConfig::default();
    let transcriber: Arc<dyn Transcriber> = if is_gpu && queue_cfg.cpu_fallback_enabled {
        let probe = Arc::new(PresenceGpuProbe::new(
            presence.clone(),
            config.sleep.gpu_allowlist.clone(),
        ));
        let cpu_cfg = config.voice.transcription.clone();
        let cpu_factory: assistd_voice::CpuFallbackFactory = Arc::new(move || {
            let cfg = cpu_cfg.clone();
            Box::pin(async move {
                let t = build_cpu_fallback(&cfg, None).await?;
                Ok(Arc::new(t) as Arc<dyn Transcriber>)
            })
        });
        info!(
            "voice: GPU transcription active; CPU fallback armed \
                 (gpu_busy_timeout_ms={})",
            queue_cfg.gpu_busy_timeout_ms
        );
        Arc::new(QueuedTranscriber::new(
            primary.clone(),
            true,
            cpu_factory,
            probe,
            queue_cfg,
        ))
    } else {
        if is_gpu {
            info!("voice: GPU transcription active; CPU fallback disabled by config");
        } else {
            info!("voice: CPU transcription active");
        }
        primary.clone()
    };

    let mic = MicVoiceInput::new(
        transcriber.clone(),
        config.voice.mic_device.clone(),
        config.voice.max_recording_secs.get(),
    );
    let listener: Arc<dyn ContinuousListener> = if config.voice.continuous.enabled {
        info!(
            "voice.continuous: enabled (hotkey={:?}, start_on_launch={})",
            config.voice.continuous.hotkey, config.voice.continuous.start_on_launch
        );
        Arc::new(MicContinuousListener::new(transcriber, &config.voice))
    } else {
        info!("voice.continuous: disabled in config");
        Arc::new(NoContinuousListener::new())
    };
    (Arc::new(mic), listener)
}

async fn init_output(config: &Config) -> Arc<dyn VoiceOutput> {
    if !config.voice.synthesis.enabled {
        info!("voice.synthesis: disabled in config (voice.synthesis.enabled = false)");
        return Arc::new(NoVoiceOutput) as Arc<dyn VoiceOutput>;
    }
    info!(
        "voice.synthesis: starting Piper ({})",
        config.voice.synthesis.voice
    );
    match assistd_voice::PiperVoiceOutput::start(config.voice.synthesis.clone()).await {
        Ok(p) => {
            info!("voice.synthesis: Piper ready");
            Arc::new(p) as Arc<dyn VoiceOutput>
        }
        Err(e) => {
            tracing::warn!("voice.synthesis failed to initialize: {e:#}; speech output disabled");
            Arc::new(NoVoiceOutput) as Arc<dyn VoiceOutput>
        }
    }
}

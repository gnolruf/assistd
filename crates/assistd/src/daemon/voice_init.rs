//! Voice startup for the daemon: brings up the sherpa-onnx voice and
//! transcriber behind a [`VoiceManager`] that already serves requests.

use std::sync::Arc;

use assistd_core::{
    Config, ContinuousListener, NoContinuousListener, PresenceManager, VoiceCapture, VoiceManager,
    VoiceOutput,
};
use assistd_utils::readiness::Readiness;
use assistd_voice::{
    CpuFallbackFactory, MicContinuousListener, MicVoiceInput, QueuedTranscriber, SherpaTranscriber,
    SherpaVoiceOutput, SileroVad, Transcriber, VoiceOutputController, WakeWordSpotter,
};
use tracing::info;

use super::voice_probe::PresenceGpuProbe;
use super::vram::VramLedger;

/// Start speech output, then capture, recording on `voice` whether each
/// came up or why it is unavailable.
pub(super) async fn init(
    voice: &VoiceManager,
    config: &Config,
    presence: &Arc<PresenceManager>,
    vram: Option<Arc<VramLedger>>,
) {
    voice.speech().set_output(init_output(config).await);
    let speech = Arc::clone(voice.speech());
    voice.set_capture(init_capture(config, presence, vram, speech).await);
}

async fn init_capture(
    config: &Config,
    presence: &Arc<PresenceManager>,
    vram: Option<Arc<VramLedger>>,
    speech: Arc<VoiceOutputController>,
) -> Readiness<VoiceCapture> {
    if !config.voice.enabled {
        info!("voice: disabled in config (voice.enabled = false)");
        return Readiness::Unavailable("disabled in config (voice.enabled = false)".into());
    }

    info!(
        "voice: building mic input ({})",
        config.voice.transcription.model
    );
    let primary = match SherpaTranscriber::load(&config.voice.transcription).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("voice input failed to initialize: {e:#}; PTT commands will error");
            return Readiness::Unavailable(
                format!("speech-to-text failed to initialize: {e:#}").into(),
            );
        }
    };
    let is_gpu = primary.is_gpu();
    let transcriber = with_cpu_fallback(config, presence, vram, Arc::new(primary), is_gpu);

    let mic = MicVoiceInput::new(transcriber.clone(), config.voice.mic_device.clone());
    let (listener, spotter) = match init_listener(config, &transcriber, speech).await {
        Some((listener, spotter)) => (listener, Some(spotter)),
        None => (
            Arc::new(NoContinuousListener::new()) as Arc<dyn ContinuousListener>,
            None,
        ),
    };
    Readiness::Ready(VoiceCapture {
        input: Arc::new(mic),
        listener,
        transcriber,
        spotter,
    })
}

/// Wake-word listening and its spotter, or `None` when it is off in config
/// or its Silero or keyword model cannot be loaded.
async fn init_listener(
    config: &Config,
    transcriber: &Arc<dyn Transcriber>,
    speech: Arc<VoiceOutputController>,
) -> Option<(Arc<dyn ContinuousListener>, Arc<WakeWordSpotter>)> {
    if !config.voice.wakeword.enabled {
        info!("voice.wakeword: disabled in config");
        return None;
    }
    let silero = match SileroVad::load(&config.voice.transcription).await {
        Ok(silero) => silero,
        Err(e) => {
            tracing::warn!(
                "voice.wakeword: Silero VAD failed to load: {e:#}; listening unavailable"
            );
            return None;
        }
    };
    let spotter = match WakeWordSpotter::load(&config.voice).await {
        Ok(spotter) => Arc::new(spotter),
        Err(e) => {
            tracing::warn!("voice.wakeword: {e:#}; listening unavailable");
            return None;
        }
    };
    info!(
        "voice.wakeword: listening for {:?} (start_on_launch={})",
        spotter.phrase(),
        config.voice.wakeword.start_on_launch
    );
    let listener = MicContinuousListener::new(
        transcriber.clone(),
        &config.voice,
        speech,
        silero,
        Arc::clone(&spotter),
    );
    Some((Arc::new(listener), spotter))
}

/// Wrap a GPU transcriber so it falls back to CPU while the GPU is
/// unavailable; CPU transcribers are returned unchanged.
fn with_cpu_fallback(
    config: &Config,
    presence: &Arc<PresenceManager>,
    vram: Option<Arc<VramLedger>>,
    primary: Arc<dyn Transcriber>,
    is_gpu: bool,
) -> Arc<dyn Transcriber> {
    if !is_gpu {
        info!("voice: CPU transcription active");
        return primary;
    }

    let probe = Arc::new(PresenceGpuProbe::new(presence.clone(), vram));
    let cpu_cfg = config.voice.transcription.clone();
    let cpu_factory: CpuFallbackFactory = Arc::new(move || {
        let cfg = cpu_cfg.clone();
        Box::pin(async move {
            let fallback = SherpaTranscriber::load_cpu(&cfg).await?;
            Ok(Arc::new(fallback) as Arc<dyn Transcriber>)
        })
    });
    info!(
        "voice: GPU transcription active, shared with the LLM; CPU fallback armed \
         for sleep and foreign VRAM use"
    );
    Arc::new(QueuedTranscriber::new(primary, cpu_factory, probe))
}

async fn init_output(config: &Config) -> Readiness<Arc<dyn VoiceOutput>> {
    if !config.voice.synthesis.enabled {
        info!("voice.synthesis: disabled in config (voice.synthesis.enabled = false)");
        return Readiness::Unavailable(
            "disabled in config (voice.synthesis.enabled = false)".into(),
        );
    }
    info!(
        "voice.synthesis: loading the voice ({})",
        config.voice.synthesis.voice
    );
    match SherpaVoiceOutput::start(&config.voice.synthesis).await {
        Ok(output) => {
            info!("voice.synthesis: speech ready");
            Readiness::Ready(Arc::new(output))
        }
        Err(e) => {
            tracing::warn!("voice.synthesis failed to initialize: {e:#}; speech output disabled");
            Readiness::Unavailable(format!("speech failed to start: {e:#}").into())
        }
    }
}

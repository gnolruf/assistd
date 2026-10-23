//! Voice startup for the daemon: brings up Piper and Whisper behind a
//! [`VoiceManager`] that already serves requests.

use std::sync::Arc;

use assistd_core::{
    Config, ContinuousListener, NoContinuousListener, PresenceManager, VoiceCapture, VoiceManager,
    VoiceOutput,
};
use assistd_utils::readiness::Readiness;
use assistd_voice::{
    CpuFallbackFactory, MicContinuousListener, MicVoiceInput, PiperVoiceOutput, ProfileStore,
    QueuedTranscriber, SileroVad, Transcriber, VoiceOutputController, WakePhrase,
    WhisperTranscriberBuilder, build_cpu_fallback,
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
    let transcriber = with_cpu_fallback(config, presence, vram, Arc::new(primary), is_gpu);

    let mic = MicVoiceInput::new(transcriber.clone(), config.voice.mic_device.clone());
    let profiles = ProfileStore::default_location();
    let listener = init_listener(config, &transcriber, speech, &profiles).await;
    Readiness::Ready(VoiceCapture {
        input: Arc::new(mic),
        listener,
        transcriber,
        profiles,
    })
}

/// Wake-word listening, or a listener that refuses to start when it is off
/// in config or its Silero model cannot be loaded.
async fn init_listener(
    config: &Config,
    transcriber: &Arc<dyn Transcriber>,
    speech: Arc<VoiceOutputController>,
    profiles: &ProfileStore,
) -> Arc<dyn ContinuousListener> {
    if !config.voice.wakeword.enabled {
        info!("voice.wakeword: disabled in config");
        return Arc::new(NoContinuousListener::new());
    }
    let silero = match SileroVad::load(&config.voice.transcription).await {
        Ok(silero) => silero,
        Err(e) => {
            tracing::warn!(
                "voice.wakeword: Silero VAD failed to load: {e:#}; listening unavailable"
            );
            return Arc::new(NoContinuousListener::new());
        }
    };
    info!(
        "voice.wakeword: enabled (start_on_launch={})",
        config.voice.wakeword.start_on_launch
    );
    let listener = MicContinuousListener::new(transcriber.clone(), &config.voice, speech, silero);
    listener.set_wake_phrase(enrolled_wake_phrase(config, profiles).await);
    Arc::new(listener)
}

/// The enrolled phrase, or the built-in one when none is enrolled or the
/// profile cannot be read.
async fn enrolled_wake_phrase(config: &Config, profiles: &ProfileStore) -> WakePhrase {
    let similarity = config.voice.wakeword.match_similarity;
    match profiles.load().await {
        Ok(Some(profile)) => {
            info!("voice.wakeword: using enrolled phrase {:?}", profile.phrase);
            profile.wake_phrase(similarity)
        }
        Ok(None) => WakePhrase::builtin(similarity),
        Err(e) => {
            tracing::warn!("voice.wakeword: {e}; using the built-in phrase");
            WakePhrase::builtin(similarity)
        }
    }
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
            let fallback = build_cpu_fallback(&cfg, None).await?;
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

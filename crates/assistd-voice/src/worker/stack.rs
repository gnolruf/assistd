//! The voice stack the worker builds: sherpa-onnx speech output, then mic
//! capture with its transcriber and, when enabled, the wake-word listener.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use assistd_config::{SynthesisConfig, VoiceConfig};
use tracing::{info, warn};

use crate::listen::{ContinuousListener, MicContinuousListener, NoContinuousListener};
use crate::mic::capture::validate;
use crate::{
    BusyProbe, CpuFallbackFactory, MicVoiceInput, QueuedTranscriber, SherpaTranscriber,
    SherpaVoiceOutput, SileroVad, Transcriber, VoiceCapture, VoiceOutput, VoiceOutputController,
    WakeWordSpotter,
};

/// GPU availability as the daemon last reported it; the GPU counts as
/// unavailable until the first report.
#[derive(Debug, Default)]
pub(super) struct ReportedGpu {
    presence_active: AtomicBool,
    foreign_busy: AtomicBool,
}

impl ReportedGpu {
    pub(super) fn update(&self, presence_active: bool, foreign_busy: bool) {
        self.presence_active
            .store(presence_active, Ordering::SeqCst);
        self.foreign_busy.store(foreign_busy, Ordering::SeqCst);
    }
}

impl BusyProbe for ReportedGpu {
    fn foreign_gpu_busy(&self) -> bool {
        self.foreign_busy.load(Ordering::SeqCst)
    }

    fn presence_active(&self) -> bool {
        self.presence_active.load(Ordering::SeqCst)
    }
}

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
/// supplies the reply text barge-in checks reject as echo.
pub(super) async fn start_capture(
    config: &VoiceConfig,
    gpu: Arc<ReportedGpu>,
    speech: Arc<VoiceOutputController>,
) -> Result<VoiceCapture, String> {
    if !config.enabled {
        info!("voice: disabled in config (voice.enabled = false)");
        return Err("disabled in config (voice.enabled = false)".into());
    }
    check_mic_device(config.mic_device.clone()).await?;
    info!("voice: building mic input ({})", config.transcription.model);
    let primary = SherpaTranscriber::load(&config.transcription)
        .await
        .map_err(|e| {
            warn!("voice input failed to initialize: {e:#}; PTT commands will error");
            format!("speech-to-text failed to initialize: {e:#}")
        })?;
    let is_gpu = primary.is_gpu();
    let transcriber = with_cpu_fallback(config, gpu, Arc::new(primary), is_gpu);
    let mic = MicVoiceInput::new(transcriber.clone(), config.mic_device.clone());
    let listener = match start_listener(config, &transcriber, speech).await {
        Some(listener) => listener,
        None => Arc::new(NoContinuousListener::new()),
    };
    Ok(VoiceCapture {
        input: Arc::new(mic),
        listener,
    })
}

/// Whether the configured input device is plugged in, or why capture cannot use it.
async fn check_mic_device(mic_device: Option<String>) -> Result<(), String> {
    let checked = tokio::task::spawn_blocking(move || validate(mic_device.as_deref()))
        .await
        .map_err(|e| format!("mic device check failed: {e}"))?;
    checked.map_err(|e| {
        warn!("voice: {e}; voice capture unavailable");
        e.to_string()
    })
}

/// Wake-word listening, or `None` when it is off in config or its Silero or
/// keyword model cannot be loaded.
async fn start_listener(
    config: &VoiceConfig,
    transcriber: &Arc<dyn Transcriber>,
    speech: Arc<VoiceOutputController>,
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
    );
    Some(Arc::new(listener))
}

/// Wrap a GPU transcriber so it falls back to CPU while the GPU is
/// unavailable; CPU transcribers are returned unchanged.
fn with_cpu_fallback(
    config: &VoiceConfig,
    gpu: Arc<ReportedGpu>,
    primary: Arc<dyn Transcriber>,
    is_gpu: bool,
) -> Arc<dyn Transcriber> {
    if !is_gpu {
        info!("voice: CPU transcription active");
        return primary;
    }
    let cpu_config = config.transcription.clone();
    let cpu_factory: CpuFallbackFactory = Arc::new(move || {
        let cpu_config = cpu_config.clone();
        Box::pin(async move {
            let fallback = SherpaTranscriber::load_cpu(&cpu_config).await?;
            Ok(Arc::new(fallback) as Arc<dyn Transcriber>)
        })
    });
    info!(
        "voice: GPU transcription active, shared with the LLM; CPU fallback armed \
         for sleep and foreign VRAM use"
    );
    Arc::new(QueuedTranscriber::new(primary, cpu_factory, gpu))
}

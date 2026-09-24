//! End-of-speech to first-audio latency benchmark. Runs a 16 kHz mono
//! WAV through Whisper, a running llama-server, and Piper N times, and
//! reports per-stage and end-to-end timings from the
//! `assistd::voice::latency` debug events. Each iteration runs in a
//! `voice_turn` span with a fresh `correlation_id`, which a custom
//! `tracing_subscriber::Layer` uses to attribute events to it.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use assistd_config::{ChatConfig, LlamaServerConfig, ModelConfig, SynthesisConfig, TimeoutsConfig};
use assistd_llm::{LlamaChatClient, LlmBackend, LlmEvent};
use assistd_voice::{
    NoVoiceOutput, PiperVoiceOutput, SentenceBuffer, Transcriber, VoiceOutput, WhisperTranscriber,
};
use clap::Parser;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::Instrument;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::{Context as LayerContext, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;

const DEFAULT_WAV_URL: &str =
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav";

#[derive(Parser, Debug)]
#[command(about = "End-to-end voice loop latency benchmark.")]
struct Args {
    /// Path to a 16 kHz mono i16 WAV. When omitted, downloads
    /// `--wav-url` to `$XDG_CACHE_HOME/assistd/bench/<filename>` (or
    /// `~/.cache/...`) and reuses the cached copy on subsequent runs.
    #[arg(long)]
    wav: Option<PathBuf>,
    /// Source URL for the on-demand WAV download. Only used when `--wav`
    /// is omitted.
    #[arg(long, default_value = DEFAULT_WAV_URL)]
    wav_url: String,
    /// Number of pipeline iterations. Iteration 1 typically pays a
    /// cold-cache penalty; the median across all runs is the headline.
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    /// llama-server host.
    #[arg(long, default_value = "127.0.0.1")]
    llama_host: std::net::IpAddr,
    /// llama-server port.
    #[arg(long, default_value = "8080")]
    llama_port: std::num::NonZeroU16,
    /// Model name sent to `/v1/chat/completions`. llama-server usually
    /// does not validate it, so a placeholder works.
    #[arg(long, default_value = "default")]
    model: String,
    /// HuggingFace identifier of the Whisper GGML model, as in
    /// `voice.transcription.model`. Tiny, base, and small all work.
    #[arg(long, default_value = "ggerganov/whisper.cpp:ggml-tiny.en.bin")]
    whisper_model: String,
    /// Force CPU inference for Whisper. Otherwise the GPU is used when
    /// CUDA is available.
    #[arg(long, default_value_t = false)]
    whisper_cpu_only: bool,
    /// Cap on LLM response tokens so each iteration completes promptly.
    /// Reasoning models spend tokens in a `<think>` block first; a
    /// budget below the thinking length yields no visible content, so
    /// `llm_first_token` and every later stage never fire.
    #[arg(long, default_value = "1024")]
    max_response_tokens: std::num::NonZeroU32,
    /// Substitute the silent `NoVoiceOutput` for Piper, for Whisper and
    /// LLM timings only. Piper stages and end-to-end timing are omitted.
    #[arg(long)]
    no_piper: bool,
    /// Piper binary name or path, e.g. `piper-tts` where the distro
    /// package ships it under that name.
    #[arg(long, default_value = "piper")]
    piper_binary: String,
    /// Pass `--cuda` to piper. Requires a piper built against
    /// `onnxruntime-gpu`; most distro packages are CPU-only and fail
    /// with "CUDA execution provider not available".
    #[arg(long)]
    piper_cuda: bool,
    /// Prepend `/no_think ` to the transcribed prompt so Qwen3-style
    /// reasoning models skip the `<think>` block, cutting
    /// `llm_first_token` to roughly prefill plus one decoded token.
    #[arg(long)]
    no_think: bool,
    /// Emit results as JSON instead of a human-readable table.
    #[arg(long)]
    json: bool,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();

    let collector = LatencyCollector::default();
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new("info,assistd::voice::latency=debug")
    });
    tracing_subscriber::registry()
        .with(collector.clone())
        .with(tracing_subscriber::fmt::layer().with_filter(env_filter))
        .init();

    let wav_path = match args.wav.clone() {
        Some(p) => p,
        None => ensure_wav_cached(&args.wav_url).await?,
    };
    eprintln!("loading WAV: {}", wav_path.display());
    let pcm = load_wav_16k_mono(&wav_path).context("loading WAV")?;
    eprintln!(
        "loaded {} samples ({:.2}s @ 16 kHz mono i16)",
        pcm.len(),
        pcm.len() as f32 / 16_000.0
    );

    eprintln!(
        "building Whisper ({}{})",
        args.whisper_model,
        if args.whisper_cpu_only { ", CPU" } else { "" }
    );
    let whisper_inner = WhisperTranscriber::builder()
        .model(args.whisper_model.clone())
        .prefer_gpu(!args.whisper_cpu_only)
        .beams(1)
        .vad_enabled(false)
        .build()
        .await
        .context("building whisper")?;
    let whisper: Arc<dyn Transcriber> = Arc::new(whisper_inner);

    eprintln!(
        "building LLM client → http://{}:{}",
        args.llama_host, args.llama_port
    );
    let chat_cfg = ChatConfig {
        max_response_tokens: args.max_response_tokens,
        ..ChatConfig::default()
    };
    let server_cfg = LlamaServerConfig {
        host: args.llama_host,
        port: args.llama_port,
        ..LlamaServerConfig::default()
    };
    let model_cfg = ModelConfig {
        name: args.model.clone(),
        ..ModelConfig::default()
    };
    let timeouts = TimeoutsConfig::default();

    let piper: Arc<dyn VoiceOutput> = if args.no_piper {
        eprintln!("Piper disabled (--no-piper); end-to-end timing will be omitted");
        Arc::new(NoVoiceOutput)
    } else {
        eprintln!(
            "building Piper TTS (binary='{}', cuda={})...",
            args.piper_binary, args.piper_cuda
        );
        let synth_cfg = SynthesisConfig {
            binary_path: args.piper_binary.clone().into(),
            use_cuda: args.piper_cuda,
            ..SynthesisConfig::default()
        };
        match PiperVoiceOutput::start(synth_cfg).await {
            Ok(p) => Arc::new(p),
            Err(e) => {
                // Whisper and LLM timings stay comparable without piper.
                eprintln!(
                    "Piper unavailable ({e:#}); falling back to NoVoiceOutput. \
                     End-to-end timing will be omitted. Install piper or pass \
                     --no-piper to silence this warning."
                );
                Arc::new(NoVoiceOutput)
            }
        }
    };

    let mut runs: Vec<RunMetrics> = Vec::with_capacity(args.iterations);
    for i in 0..args.iterations {
        // A fresh client per iteration keeps accumulated history from
        // skewing later runs; building one costs under 1 ms.
        let llm: Arc<dyn LlmBackend> = Arc::new(
            LlamaChatClient::new(&chat_cfg, &server_cfg, &model_cfg, &timeouts, None)
                .context("building LLM client")?,
        );

        let corr = format!("bench-{i}");
        let span = tracing::info_span!("voice_turn", correlation_id = %corr);
        let t0 = Instant::now();
        let outcome = run_one(whisper.clone(), llm, piper.clone(), &pcm, args.no_think)
            .instrument(span)
            .await;
        let stages = collector.take(&corr);
        match outcome {
            Ok(()) => {
                let metrics = RunMetrics::from_stages(t0, stages);
                if let Some(e2e) = metrics.end_to_end_ms {
                    eprintln!("iter {i}: {} ms end-to-end", e2e);
                } else {
                    eprintln!(
                        "iter {i}: completed but no playback_enqueued event captured ({} stage events)",
                        metrics.per_stage.len()
                    );
                }
                runs.push(metrics);
            }
            Err(e) => {
                eprintln!("iter {i} failed: {e:#}");
            }
        }
    }

    if runs.is_empty() {
        bail!("no successful iterations");
    }

    let summary = summarize(args.iterations, &runs);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_human(&summary);
    }
    Ok(())
}

/// One pass of the streaming voice path: transcribe, generate, segment,
/// and speak. No agent loop, tools, or persistence.
async fn run_one(
    whisper: Arc<dyn Transcriber>,
    llm: Arc<dyn LlmBackend>,
    piper: Arc<dyn VoiceOutput>,
    pcm: &[i16],
    no_think: bool,
) -> Result<()> {
    tracing::debug!(
        target: "assistd::voice::latency",
        stage = "audio_capture_stop",
        "voice latency stage"
    );

    let transcript = whisper
        .transcribe(pcm)
        .await
        .map_err(|e| anyhow::anyhow!("whisper failed: {e}"))?;
    if transcript.trim().is_empty() {
        bail!("whisper returned empty transcript");
    }
    // Reasoning models that honor a leading `/no_think` skip the
    // `<think>` block; other models ignore it.
    let text = if no_think {
        format!("/no_think {transcript}")
    } else {
        transcript
    };

    // There is no presence manager to wait on; the marker keeps the
    // stage timeline complete.
    tracing::debug!(
        target: "assistd::voice::latency",
        stage = "ensure_active_done",
        "voice latency stage"
    );

    let (llm_tx, mut llm_rx) = mpsc::channel::<LlmEvent>(32);
    let (speech_tx, mut speech_rx) = mpsc::channel::<String>(32);

    let llm_clone = llm.clone();
    let prompt = text.clone();
    let llm_task = tokio::spawn(
        async move {
            // The stage table only shows what fired, so an unlogged
            // failure would look like the LLM never replied.
            if let Err(e) = llm_clone.generate(prompt, llm_tx).await {
                tracing::error!(
                    target: "voice_latency_bench",
                    error = %e,
                    "LLM generate failed; iteration will report no llm_first_token"
                );
            }
        }
        .in_current_span(),
    );

    let piper_clone = piper.clone();
    let speech_task = tokio::spawn(
        async move {
            while let Some(s) = speech_rx.recv().await {
                let _ = piper_clone.speak(s).await;
            }
            // No `wait_idle`: the bench measures `playback_enqueued`,
            // not playback completion. Queued audio keeps playing in
            // the background.
        }
        .in_current_span(),
    );

    let mut sb = SentenceBuffer::new(220);
    let mut first_emitted = false;
    while let Some(ev) = llm_rx.recv().await {
        match ev {
            LlmEvent::Delta { text } => {
                for sentence in sb.push(&text) {
                    if !first_emitted {
                        tracing::debug!(
                            target: "assistd::voice::latency",
                            stage = "first_sentence_emitted",
                            "voice latency stage"
                        );
                        first_emitted = true;
                    }
                    let _ = speech_tx.send(sentence).await;
                }
            }
            LlmEvent::Done => {
                if let Some(tail) = sb.finish() {
                    if !first_emitted {
                        tracing::debug!(
                            target: "assistd::voice::latency",
                            stage = "first_sentence_emitted",
                            "voice latency stage"
                        );
                    }
                    let _ = speech_tx.send(tail).await;
                }
                break;
            }
            // `generate` runs no agent loop, so tool events never
            // arrive, and reasoning is never spoken.
            LlmEvent::ToolCallsRequested { .. }
            | LlmEvent::ToolCall { .. }
            | LlmEvent::ToolResult { .. }
            | LlmEvent::Status { .. }
            | LlmEvent::ReasoningDelta { .. } => {}
        }
    }
    drop(speech_tx);
    let _ = speech_task.await;
    let _ = llm_task.await;
    Ok(())
}

/// Samples of a 16 kHz 16-bit signed mono WAV. Any other format is a
/// clear error here rather than a confusing whisper failure later.
fn load_wav_16k_mono(path: &Path) -> Result<Vec<i16>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!("WAV must be mono, got {} channels", spec.channels);
    }
    if spec.sample_rate != 16_000 {
        bail!("WAV must be 16 kHz, got {} Hz", spec.sample_rate);
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        bail!(
            "WAV must be 16-bit signed PCM, got {} bits {:?}",
            spec.bits_per_sample,
            spec.sample_format
        );
    }
    let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
    samples.context("decoding WAV samples")
}

/// Download the WAV at `url` to a stable cache path and return that
/// path. Reuses the cached file on subsequent runs.
async fn ensure_wav_cached(url: &str) -> Result<PathBuf> {
    let cache_dir = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("assistd")
        .join("bench");
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("creating bench cache dir {}", cache_dir.display()))?;
    let filename = url.rsplit('/').next().unwrap_or("sample.wav");
    let path = cache_dir.join(filename);
    if path.exists() {
        return Ok(path);
    }
    eprintln!("downloading {url} → {}", path.display());
    let bytes = reqwest::get(url)
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()?
        .bytes()
        .await
        .context("reading WAV bytes")?;
    std::fs::write(&path, &bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

type StageLog = Vec<(String, Instant)>;

#[derive(Default, Clone)]
struct LatencyCollector {
    inner: Arc<Mutex<HashMap<String, StageLog>>>,
}

impl LatencyCollector {
    fn take(&self, corr: &str) -> StageLog {
        self.inner.lock().remove(corr).unwrap_or_default()
    }
}

#[derive(Default)]
struct CorrIdVisitor {
    corr: Option<String>,
}

impl Visit for CorrIdVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "correlation_id" {
            self.corr = Some(value.to_string());
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "correlation_id" {
            // `%`-formatted fields arrive via `record_debug`; strip any
            // surrounding quotes so lookups match exactly.
            let raw = format!("{value:?}");
            self.corr = Some(raw.trim_matches('"').to_string());
        }
    }
}

#[derive(Default)]
struct StageVisitor {
    stage: Option<String>,
}

impl Visit for StageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "stage" {
            self.stage = Some(value.to_string());
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "stage" {
            let raw = format!("{value:?}");
            self.stage = Some(raw.trim_matches('"').to_string());
        }
    }
}

struct SpanCorr(String);

impl<S> Layer<S> for LatencyCollector
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: LayerContext<'_, S>) {
        let mut v = CorrIdVisitor::default();
        attrs.record(&mut v);
        if let Some(corr) = v.corr {
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(SpanCorr(corr.clone()));
            }
            self.inner.lock().entry(corr).or_default();
        }
    }

    fn on_event(&self, event: &tracing::Event<'_>, ctx: LayerContext<'_, S>) {
        if event.metadata().target() != "assistd::voice::latency" {
            return;
        }
        let mut sv = StageVisitor::default();
        event.record(&mut sv);
        let Some(stage) = sv.stage else {
            return;
        };
        let now = Instant::now();
        let Some(span) = ctx.event_span(event) else {
            return;
        };
        for s in span.scope() {
            let exts = s.extensions();
            if let Some(corr) = exts.get::<SpanCorr>() {
                self.inner
                    .lock()
                    .entry(corr.0.clone())
                    .or_default()
                    .push((stage, now));
                return;
            }
        }
    }
}

#[derive(Debug)]
struct RunMetrics {
    per_stage: HashMap<String, u64>,
    end_to_end_ms: Option<u64>,
}

impl RunMetrics {
    fn from_stages(t0: Instant, stages: StageLog) -> Self {
        let mut per_stage = HashMap::new();
        for (name, when) in &stages {
            // Duplicate stage names keep the earliest timestamp.
            per_stage
                .entry(name.clone())
                .or_insert_with(|| when.duration_since(t0).as_millis() as u64);
        }
        let end_to_end_ms = per_stage.get("playback_enqueued").copied();
        Self {
            per_stage,
            end_to_end_ms,
        }
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[derive(Debug)]
struct StageStats {
    min_ms: u64,
    median_ms: u64,
    p95_ms: u64,
    max_ms: u64,
    count: usize,
}

fn stats(values: &[u64]) -> StageStats {
    let mut v = values.to_vec();
    v.sort_unstable();
    StageStats {
        min_ms: *v.first().unwrap_or(&0),
        median_ms: percentile(&v, 0.5),
        p95_ms: percentile(&v, 0.95),
        max_ms: *v.last().unwrap_or(&0),
        count: v.len(),
    }
}

#[derive(Debug)]
struct Summary {
    requested_iterations: usize,
    successful_iterations: usize,
    stages: Vec<(String, StageStats)>,
    end_to_end: Option<StageStats>,
}

const STAGE_ORDER: &[&str] = &[
    "audio_capture_stop",
    "whisper_start",
    "whisper_done",
    "ensure_active_done",
    "llm_request_sent",
    "llm_first_token",
    "first_sentence_emitted",
    "piper_spawn",
    "piper_first_byte",
    "piper_synth_done",
    "playback_enqueued",
];

fn summarize(requested: usize, runs: &[RunMetrics]) -> Summary {
    let mut by_stage: HashMap<String, Vec<u64>> = HashMap::new();
    for r in runs {
        for (name, ms) in &r.per_stage {
            by_stage.entry(name.clone()).or_default().push(*ms);
        }
    }
    // Known stages in timeline order, then any others by name.
    let mut stages: Vec<(String, StageStats)> = STAGE_ORDER
        .iter()
        .filter_map(|name| {
            by_stage
                .remove(*name)
                .map(|vals| ((*name).to_string(), stats(&vals)))
        })
        .collect();
    let mut leftover: Vec<(String, StageStats)> = by_stage
        .into_iter()
        .map(|(name, vals)| (name, stats(&vals)))
        .collect();
    leftover.sort_by(|a, b| a.0.cmp(&b.0));
    stages.extend(leftover);

    let end_to_end_vals: Vec<u64> = runs.iter().filter_map(|r| r.end_to_end_ms).collect();
    let end_to_end = if end_to_end_vals.is_empty() {
        None
    } else {
        Some(stats(&end_to_end_vals))
    };

    Summary {
        requested_iterations: requested,
        successful_iterations: runs.len(),
        stages,
        end_to_end,
    }
}

fn print_human(summary: &Summary) {
    println!(
        "\nVoice latency benchmark: {} of {} iterations completed\n",
        summary.successful_iterations, summary.requested_iterations,
    );
    println!(
        "{:>26}  {:>9}  {:>9}  {:>9}  {:>9}  {:>5}",
        "stage", "min", "median", "p95", "max", "n"
    );
    println!(
        "{:>26}  {:>9}  {:>9}  {:>9}  {:>9}  {:>5}",
        "----", "---", "------", "---", "---", "-"
    );
    for (name, s) in &summary.stages {
        println!(
            "{:>26}  {:>6} ms  {:>6} ms  {:>6} ms  {:>6} ms  {:>5}",
            name, s.min_ms, s.median_ms, s.p95_ms, s.max_ms, s.count
        );
    }
    println!();
    match &summary.end_to_end {
        Some(e) => {
            println!(
                "end-to-end (T0 → playback_enqueued): min {} ms, median {} ms, p95 {} ms, max {} ms (n={})",
                e.min_ms, e.median_ms, e.p95_ms, e.max_ms, e.count
            );
            let pass = e.median_ms <= 500;
            println!(
                "acceptance gate (median ≤ 500 ms): {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        None => println!("no playback_enqueued events captured; pipeline did not reach Piper"),
    }
}

// Hand-written rather than derived so the JSON schema is explicit and
// the `serde` derive feature is not needed.
impl serde::Serialize for Summary {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let stages: serde_json::Map<String, serde_json::Value> = self
            .stages
            .iter()
            .map(|(name, s)| (name.clone(), stage_json(s)))
            .collect();
        let value = json!({
            "requested_iterations": self.requested_iterations,
            "successful_iterations": self.successful_iterations,
            "stages": serde_json::Value::Object(stages),
            "end_to_end_ms": self.end_to_end.as_ref().map(stage_json),
        });
        value.serialize(ser)
    }
}

fn stage_json(s: &StageStats) -> serde_json::Value {
    json!({
        "min": s.min_ms,
        "median": s.median_ms,
        "p95": s.p95_ms,
        "max": s.max_ms,
        "count": s.count,
    })
}

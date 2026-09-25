//! End-of-speech to first-audio latency benchmark: runs a 16 kHz mono WAV
//! through Whisper, a running llama-server, and Piper N times, attributing
//! `assistd::voice::latency` events to each iteration's `voice_turn` span.

use std::collections::HashMap;
use std::fmt::Debug;
use std::net::IpAddr;
use std::num::{NonZeroU16, NonZeroU32};
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
use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::Instrument;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::{Context as LayerContext, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;

const DEFAULT_WAV_URL: &str =
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav";

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
    llama_host: IpAddr,
    /// llama-server port.
    #[arg(long, default_value = "8080")]
    llama_port: NonZeroU16,
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
    max_response_tokens: NonZeroU32,
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

/// Settings for the LLM client built fresh each iteration, so accumulated
/// history never skews later runs.
struct LlmSettings {
    chat: ChatConfig,
    server: LlamaServerConfig,
    model: ModelConfig,
    timeouts: TimeoutsConfig,
}

impl LlmSettings {
    fn from_args(args: &Args) -> Self {
        Self {
            chat: ChatConfig {
                max_response_tokens: args.max_response_tokens,
                ..ChatConfig::default()
            },
            server: LlamaServerConfig {
                host: args.llama_host,
                port: args.llama_port,
                ..LlamaServerConfig::default()
            },
            model: ModelConfig {
                name: args.model.clone(),
                ..ModelConfig::default()
            },
            timeouts: TimeoutsConfig::default(),
        }
    }

    fn client(&self) -> Result<Arc<dyn LlmBackend>> {
        let client =
            LlamaChatClient::new(&self.chat, &self.server, &self.model, &self.timeouts, None)
                .context("building LLM client")?;
        Ok(Arc::new(client))
    }
}

type StageLog = Vec<(String, Instant)>;

/// Layer that records each latency stage under its turn's correlation id.
#[derive(Default, Clone)]
struct LatencyCollector {
    stages_by_turn: Arc<Mutex<HashMap<String, StageLog>>>,
}

impl LatencyCollector {
    fn take(&self, correlation_id: &str) -> StageLog {
        self.stages_by_turn
            .lock()
            .remove(correlation_id)
            .unwrap_or_default()
    }
}

impl<S> Layer<S> for LatencyCollector
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: LayerContext<'_, S>) {
        let mut visitor = CorrelationIdVisitor::default();
        attrs.record(&mut visitor);
        if let Some(correlation_id) = visitor.correlation_id {
            if let Some(span) = ctx.span(id) {
                span.extensions_mut()
                    .insert(SpanCorrelationId(correlation_id.clone()));
            }
            self.stages_by_turn
                .lock()
                .entry(correlation_id)
                .or_default();
        }
    }

    fn on_event(&self, event: &tracing::Event<'_>, ctx: LayerContext<'_, S>) {
        if event.metadata().target() != "assistd::voice::latency" {
            return;
        }
        let mut visitor = StageVisitor::default();
        event.record(&mut visitor);
        let Some(stage) = visitor.stage else {
            return;
        };
        let now = Instant::now();
        let Some(span) = ctx.event_span(event) else {
            return;
        };
        for ancestor in span.scope() {
            let extensions = ancestor.extensions();
            if let Some(correlation_id) = extensions.get::<SpanCorrelationId>() {
                self.stages_by_turn
                    .lock()
                    .entry(correlation_id.0.clone())
                    .or_default()
                    .push((stage, now));
                return;
            }
        }
    }
}

struct SpanCorrelationId(String);

#[derive(Default)]
struct CorrelationIdVisitor {
    correlation_id: Option<String>,
}

impl Visit for CorrelationIdVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "correlation_id" {
            self.correlation_id = Some(value.to_string());
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        if field.name() == "correlation_id" {
            self.correlation_id = Some(unquoted_debug(value));
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
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        if field.name() == "stage" {
            self.stage = Some(unquoted_debug(value));
        }
    }
}

#[derive(Debug)]
struct RunMetrics {
    per_stage: HashMap<String, u64>,
    end_to_end_ms: Option<u64>,
}

impl RunMetrics {
    /// Milliseconds from `turn_start` to each stage; a repeated stage keeps its first timestamp.
    fn from_stages(turn_start: Instant, stages: StageLog) -> Self {
        let mut per_stage = HashMap::new();
        for (name, when) in &stages {
            per_stage
                .entry(name.clone())
                .or_insert_with(|| when.duration_since(turn_start).as_millis() as u64);
        }
        let end_to_end_ms = per_stage.get("playback_enqueued").copied();
        Self {
            per_stage,
            end_to_end_ms,
        }
    }
}

#[derive(Debug)]
struct StageStats {
    min_ms: u64,
    median_ms: u64,
    p95_ms: u64,
    max_ms: u64,
    count: usize,
}

#[derive(Debug)]
struct Summary {
    requested_iterations: usize,
    successful_iterations: usize,
    stages: Vec<(String, StageStats)>,
    end_to_end: Option<StageStats>,
}

// Hand-written so the JSON schema is explicit without serde's derive feature.
impl serde::Serialize for Summary {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let stages: serde_json::Map<String, serde_json::Value> = self
            .stages
            .iter()
            .map(|(name, stats)| (name.clone(), stage_json(stats)))
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

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();

    let collector = LatencyCollector::default();
    init_tracing(collector.clone());

    let pcm = load_pcm(&args).await?;
    let whisper = build_whisper(&args).await?;

    eprintln!(
        "building LLM client → http://{}:{}",
        args.llama_host, args.llama_port
    );
    let llm_settings = LlmSettings::from_args(&args);
    let piper = start_piper(&args).await;

    let runs = run_iterations(&args, &collector, &llm_settings, whisper, piper, &pcm).await?;
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

fn init_tracing(collector: LatencyCollector) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,assistd::voice::latency=debug"));
    tracing_subscriber::registry()
        .with(collector)
        .with(tracing_subscriber::fmt::layer().with_filter(env_filter))
        .init();
}

async fn load_pcm(args: &Args) -> Result<Vec<i16>> {
    let wav_path = match args.wav.clone() {
        Some(path) => path,
        None => ensure_wav_cached(&args.wav_url).await?,
    };
    eprintln!("loading WAV: {}", wav_path.display());
    let pcm = load_wav_16k_mono(&wav_path).context("loading WAV")?;
    eprintln!(
        "loaded {} samples ({:.2}s @ 16 kHz mono i16)",
        pcm.len(),
        pcm.len() as f32 / 16_000.0
    );
    Ok(pcm)
}

async fn build_whisper(args: &Args) -> Result<Arc<dyn Transcriber>> {
    eprintln!(
        "building Whisper ({}{})",
        args.whisper_model,
        if args.whisper_cpu_only { ", CPU" } else { "" }
    );
    let whisper = WhisperTranscriber::builder()
        .model(args.whisper_model.clone())
        .prefer_gpu(!args.whisper_cpu_only)
        .beams(1)
        .vad_enabled(false)
        .build()
        .await
        .context("building whisper")?;
    Ok(Arc::new(whisper))
}

/// Falls back to `NoVoiceOutput` when piper cannot start, so Whisper and LLM
/// timings stay comparable.
async fn start_piper(args: &Args) -> Arc<dyn VoiceOutput> {
    if args.no_piper {
        eprintln!("Piper disabled (--no-piper); end-to-end timing will be omitted");
        return Arc::new(NoVoiceOutput);
    }
    eprintln!(
        "building Piper TTS (binary='{}', cuda={})...",
        args.piper_binary, args.piper_cuda
    );
    let synthesis = SynthesisConfig {
        binary_path: args.piper_binary.clone().into(),
        use_cuda: args.piper_cuda,
        ..SynthesisConfig::default()
    };
    match PiperVoiceOutput::start(synthesis).await {
        Ok(piper) => Arc::new(piper),
        Err(err) => {
            eprintln!(
                "Piper unavailable ({err:#}); falling back to NoVoiceOutput. \
                 End-to-end timing will be omitted. Install piper or pass \
                 --no-piper to silence this warning."
            );
            Arc::new(NoVoiceOutput)
        }
    }
}

async fn run_iterations(
    args: &Args,
    collector: &LatencyCollector,
    llm_settings: &LlmSettings,
    whisper: Arc<dyn Transcriber>,
    piper: Arc<dyn VoiceOutput>,
    pcm: &[i16],
) -> Result<Vec<RunMetrics>> {
    let mut runs: Vec<RunMetrics> = Vec::with_capacity(args.iterations);
    for iteration in 0..args.iterations {
        let llm = llm_settings.client()?;

        let correlation_id = format!("bench-{iteration}");
        let span = tracing::info_span!("voice_turn", correlation_id = %correlation_id);
        let turn_start = Instant::now();
        let outcome = run_one(whisper.clone(), llm, piper.clone(), pcm, args.no_think)
            .instrument(span)
            .await;
        let stages = collector.take(&correlation_id);
        match outcome {
            Ok(()) => {
                let metrics = RunMetrics::from_stages(turn_start, stages);
                if let Some(end_to_end) = metrics.end_to_end_ms {
                    eprintln!("iter {iteration}: {} ms end-to-end", end_to_end);
                } else {
                    eprintln!(
                        "iter {iteration}: completed but no playback_enqueued event captured ({} stage events)",
                        metrics.per_stage.len()
                    );
                }
                runs.push(metrics);
            }
            Err(err) => {
                eprintln!("iter {iteration} failed: {err:#}");
            }
        }
    }
    Ok(runs)
}

/// One pass of the streaming voice path: transcribe, generate, segment, and
/// speak. `ensure_active_done` fires unconditionally since there is no presence
/// manager to wait on.
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

    let prompt = transcribe_prompt(whisper.as_ref(), pcm, no_think).await?;

    tracing::debug!(
        target: "assistd::voice::latency",
        stage = "ensure_active_done",
        "voice latency stage"
    );

    let (llm_tx, llm_rx) = mpsc::channel::<LlmEvent>(32);
    let (speech_tx, speech_rx) = mpsc::channel::<String>(32);
    let llm_task = spawn_generation(llm, prompt, llm_tx);
    let speech_task = spawn_speaker(piper, speech_rx);

    forward_sentences(llm_rx, speech_tx).await;
    let _ = speech_task.await;
    let _ = llm_task.await;
    Ok(())
}

/// Transcribe `pcm`, prefixing `/no_think` when asked; reasoning models that
/// honor it skip the `<think>` block and others ignore it.
async fn transcribe_prompt(
    whisper: &dyn Transcriber,
    pcm: &[i16],
    no_think: bool,
) -> Result<String> {
    let transcript = whisper
        .transcribe(pcm)
        .await
        .map_err(|err| anyhow::anyhow!("whisper failed: {err}"))?;
    if transcript.trim().is_empty() {
        bail!("whisper returned empty transcript");
    }
    Ok(if no_think {
        format!("/no_think {transcript}")
    } else {
        transcript
    })
}

/// Logs a failed `generate`, since the stage table only shows stages that
/// fired and a silent failure would look like the LLM never replied.
fn spawn_generation(
    llm: Arc<dyn LlmBackend>,
    prompt: String,
    llm_tx: mpsc::Sender<LlmEvent>,
) -> JoinHandle<()> {
    tokio::spawn(
        async move {
            if let Err(err) = llm.generate(prompt, llm_tx).await {
                tracing::error!(
                    target: "voice_latency_bench",
                    error = %err,
                    "LLM generate failed; iteration will report no llm_first_token"
                );
            }
        }
        .in_current_span(),
    )
}

/// Speaks each sentence without `wait_idle`: the bench measures
/// `playback_enqueued`, not playback completion.
fn spawn_speaker(
    piper: Arc<dyn VoiceOutput>,
    mut speech_rx: mpsc::Receiver<String>,
) -> JoinHandle<()> {
    tokio::spawn(
        async move {
            while let Some(sentence) = speech_rx.recv().await {
                let _ = piper.speak(sentence).await;
            }
        }
        .in_current_span(),
    )
}

/// Segment streamed deltas into sentences for the speaker until `Done`.
/// `generate` runs no agent loop, so tool events never arrive; reasoning is never spoken.
async fn forward_sentences(mut llm_rx: mpsc::Receiver<LlmEvent>, speech_tx: mpsc::Sender<String>) {
    let mut sentences = SentenceBuffer::new(220);
    let mut first_emitted = false;
    while let Some(event) = llm_rx.recv().await {
        match event {
            LlmEvent::Delta { text } => {
                for sentence in sentences.push(&text) {
                    mark_first_sentence(&mut first_emitted);
                    let _ = speech_tx.send(sentence).await;
                }
            }
            LlmEvent::Done => {
                if let Some(tail) = sentences.finish() {
                    mark_first_sentence(&mut first_emitted);
                    let _ = speech_tx.send(tail).await;
                }
                break;
            }
            LlmEvent::ToolCallsRequested { .. }
            | LlmEvent::ToolCall { .. }
            | LlmEvent::ToolResult { .. }
            | LlmEvent::Status { .. }
            | LlmEvent::ReasoningDelta { .. } => {}
        }
    }
}

fn mark_first_sentence(first_emitted: &mut bool) {
    if *first_emitted {
        return;
    }
    tracing::debug!(
        target: "assistd::voice::latency",
        stage = "first_sentence_emitted",
        "voice latency stage"
    );
    *first_emitted = true;
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
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
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

/// `%`-formatted fields arrive via `record_debug`; strip any surrounding
/// quotes so lookups match exactly.
fn unquoted_debug(value: &dyn Debug) -> String {
    format!("{value:?}").trim_matches('"').to_string()
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn stats(values: &[u64]) -> StageStats {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    StageStats {
        min_ms: *sorted.first().unwrap_or(&0),
        median_ms: percentile(&sorted, 0.5),
        p95_ms: percentile(&sorted, 0.95),
        max_ms: *sorted.last().unwrap_or(&0),
        count: sorted.len(),
    }
}

fn summarize(requested: usize, runs: &[RunMetrics]) -> Summary {
    let mut by_stage: HashMap<String, Vec<u64>> = HashMap::new();
    for run in runs {
        for (name, ms) in &run.per_stage {
            by_stage.entry(name.clone()).or_default().push(*ms);
        }
    }

    let end_to_end_vals: Vec<u64> = runs.iter().filter_map(|run| run.end_to_end_ms).collect();
    let end_to_end = (!end_to_end_vals.is_empty()).then(|| stats(&end_to_end_vals));

    Summary {
        requested_iterations: requested,
        successful_iterations: runs.len(),
        stages: ordered_stage_stats(by_stage),
        end_to_end,
    }
}

/// Known stages in timeline order, then any others by name.
fn ordered_stage_stats(mut by_stage: HashMap<String, Vec<u64>>) -> Vec<(String, StageStats)> {
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
    stages
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
    for (name, stats) in &summary.stages {
        println!(
            "{:>26}  {:>6} ms  {:>6} ms  {:>6} ms  {:>6} ms  {:>5}",
            name, stats.min_ms, stats.median_ms, stats.p95_ms, stats.max_ms, stats.count
        );
    }
    println!();
    match &summary.end_to_end {
        Some(end_to_end) => {
            println!(
                "end-to-end (T0 → playback_enqueued): min {} ms, median {} ms, p95 {} ms, max {} ms (n={})",
                end_to_end.min_ms,
                end_to_end.median_ms,
                end_to_end.p95_ms,
                end_to_end.max_ms,
                end_to_end.count
            );
            let pass = end_to_end.median_ms <= 500;
            println!(
                "acceptance gate (median ≤ 500 ms): {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        None => println!("no playback_enqueued events captured; pipeline did not reach Piper"),
    }
}

fn stage_json(stats: &StageStats) -> serde_json::Value {
    json!({
        "min": stats.min_ms,
        "median": stats.median_ms,
        "p95": stats.p95_ms,
        "max": stats.max_ms,
        "count": stats.count,
    })
}

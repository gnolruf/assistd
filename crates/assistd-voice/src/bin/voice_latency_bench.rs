//! `voice_latency_bench` — in-process driver for the
//! end-of-speech → first-audio-frame voice loop, built so the project
//! has a reproducible number to track against the <500 ms budget.
//!
//! Loads a 16 kHz mono i16 WAV, runs the full pipeline N times against a
//! running llama-server, and reports per-stage and end-to-end timings
//! captured from the `assistd::voice::latency` debug events emitted at
//! every stage of the daemon's hot path. Each iteration opens its own
//! `voice_turn` span with a fresh `correlation_id`; a custom
//! `tracing_subscriber::Layer` joins the events back to the iteration
//! that produced them.

#![cfg(feature = "test-support")]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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
    llama_host: String,
    /// llama-server port.
    #[arg(long, default_value_t = 8080)]
    llama_port: u16,
    /// Model identifier sent in the `model` field of `/v1/chat/completions`.
    /// Should match what llama-server has loaded; many servers don't
    /// validate this and a placeholder works fine.
    #[arg(long, default_value = "default")]
    model: String,
    /// Whisper GGML model HuggingFace identifier (matches the daemon's
    /// `voice.transcription.model` config). Tiny / base / small all work.
    #[arg(long, default_value = "ggerganov/whisper.cpp:ggml-tiny.en.bin")]
    whisper_model: String,
    /// Force CPU inference for Whisper. Default uses GPU when CUDA is
    /// available (matches the daemon's prefer_gpu default).
    #[arg(long, default_value_t = false)]
    whisper_cpu_only: bool,
    /// Cap LLM response length so each iteration completes promptly.
    /// Note: reasoning models (Qwen3, DeepSeek-R1, etc.) spend tokens
    /// in a `<think>` block before emitting visible content. With a
    /// budget below the thinking length, the iteration finishes with
    /// zero content deltas — `llm_first_token` and everything
    /// downstream of it won't fire. 1024 is a safe default for most
    /// reasoning models and short prompts.
    #[arg(long, default_value_t = 1024)]
    max_response_tokens: u32,
    /// Skip Piper TTS startup and substitute the silent `NoVoiceOutput`.
    /// Use when piper isn't on PATH or you only want Whisper + LLM
    /// timings. The bench will still report every stage that fired
    /// (audio_capture_stop through first_sentence_emitted) but
    /// piper_spawn / piper_first_byte / piper_synth_done /
    /// playback_enqueued won't appear and end-to-end won't be reported.
    #[arg(long)]
    no_piper: bool,
    /// Override the piper binary name. Defaults to `piper`; pass
    /// `piper-tts` if your distro's package ships it under that name
    /// (e.g. the Arch AUR `piper-tts-bin` package).
    #[arg(long, default_value = "piper")]
    piper_binary: String,
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
        host: args.llama_host.clone(),
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
        eprintln!("building Piper TTS (binary='{}')...", args.piper_binary);
        let synth_cfg = SynthesisConfig {
            binary_path: args.piper_binary.clone(),
            ..SynthesisConfig::default()
        };
        match PiperVoiceOutput::start(synth_cfg).await {
            Ok(p) => Arc::new(p),
            Err(e) => {
                // Mirror the daemon's try-warn-fallback pattern. The
                // bench is still useful without piper — Whisper + LLM
                // timings remain comparable across runs.
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
        // Fresh LlamaChatClient per iter so accumulated history doesn't
        // skew results across runs. New reqwest::Client costs <1 ms.
        let llm: Arc<dyn LlmBackend> = Arc::new(
            LlamaChatClient::new(&chat_cfg, &server_cfg, &model_cfg, &timeouts)
                .context("building LLM client")?,
        );

        let corr = format!("bench-{i}");
        let span = tracing::info_span!("voice_turn", correlation_id = %corr);
        let t0 = Instant::now();
        let outcome = run_one(whisper.clone(), llm, piper.clone(), &pcm)
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

/// Drive the streaming pipeline for one iteration. Mirrors the
/// daemon's `handle_query` dispatch loop but stripped to the voice path
/// — no agent loop, no tools, no persistence.
async fn run_one(
    whisper: Arc<dyn Transcriber>,
    llm: Arc<dyn LlmBackend>,
    piper: Arc<dyn VoiceOutput>,
    pcm: &[i16],
) -> Result<()> {
    tracing::debug!(
        target: "assistd::voice::latency",
        stage = "audio_capture_stop",
        "voice latency stage"
    );

    let text = whisper
        .transcribe(pcm)
        .await
        .map_err(|e| anyhow::anyhow!("whisper failed: {e}"))?;
    if text.trim().is_empty() {
        bail!("whisper returned empty transcript");
    }

    // Skip the warmup join — the bench's LLM client is fresh each iter,
    // there's no PresenceManager, and ensure_active() is therefore a
    // no-op. Emit the marker for parity with the daemon's event stream.
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
            // Surface generate errors via tracing — the bench's stage
            // table only shows what fired, so a silent failure looks
            // like "the LLM never replied" with no hint why.
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
            // We deliberately don't call `wait_idle()` (which blocks on
            // rodio's `Player::sleep_until_end`). Two reasons:
            //   1. The bench records `playback_enqueued` per sentence,
            //      not "audio finished playing" — we already have the
            //      measurement we care about.
            //   2. `drain()` spawns a `std::thread` that runs the
            //      blocking `sleep_until_end`. If we time out the
            //      future, the thread leaks and keeps holding rodio
            //      state — empirically this then blocks the *next*
            //      iteration's `player.append` and the bench wedges.
            // Audio that's already queued continues to play (or not,
            // if the device is silent) in the background until the
            // bench process exits.
        }
        .in_current_span(),
    );

    // SentenceBuffer max-len tracks the daemon's piper default
    // (max_sentence_chars = 220). Code-block mode defaults to Skip.
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
                        // first_emitted set is unused — we break next.
                    }
                    let _ = speech_tx.send(tail).await;
                }
                break;
            }
            // The bench's prompt is a single user message and the LLM
            // is invoked via generate() (not the agent loop), so tool
            // events never appear. Ignore for completeness.
            LlmEvent::ToolCall { .. } | LlmEvent::ToolResult { .. } => {}
        }
    }
    drop(speech_tx);
    let _ = speech_task.await;
    let _ = llm_task.await;
    Ok(())
}

/// Read a WAV from disk and return its samples as 16-bit signed mono PCM
/// at 16 kHz. Errors out on any other shape so the caller surfaces a
/// clear "wrong WAV format" message instead of a confusing whisper failure.
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
    std::fs::write(&path, &bytes)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

// ---------- latency collection ----------

type StageLog = Vec<(String, Instant)>;

#[derive(Default, Clone)]
struct LatencyCollector {
    inner: Arc<Mutex<HashMap<String, StageLog>>>,
}

impl LatencyCollector {
    fn take(&self, corr: &str) -> StageLog {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(corr)
            .unwrap_or_default()
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
            // `%id` formatting routes through Display, which `record_debug`
            // captures as `"abc"`-quoted Debug repr. Strip the quotes so
            // downstream lookups are exact-string matches.
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
            self.inner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(corr)
                .or_default();
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
                    .unwrap_or_else(|e| e.into_inner())
                    .entry(corr.0.clone())
                    .or_default()
                    .push((stage, now));
                return;
            }
        }
    }
}

// ---------- summary ----------

#[derive(Debug)]
struct RunMetrics {
    /// Stage name → ms from t0
    per_stage: HashMap<String, u64>,
    end_to_end_ms: Option<u64>,
}

impl RunMetrics {
    fn from_stages(t0: Instant, stages: StageLog) -> Self {
        let mut per_stage = HashMap::new();
        for (name, when) in &stages {
            // First write wins — duplicate stage names keep the earliest
            // timestamp, which is the behaviour we want for first_byte etc.
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
    // Order stages by the canonical timeline; unknown extras append.
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
        "\nVoice latency benchmark — {} of {} iterations completed\n",
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
        None => println!("no playback_enqueued events captured — pipeline did not reach Piper"),
    }
}

// `serde_json::to_string_pretty` on `Summary` requires Serialize. Build
// the JSON value by hand so we don't drag in `serde` derive (and so the
// schema is explicit and stable).
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

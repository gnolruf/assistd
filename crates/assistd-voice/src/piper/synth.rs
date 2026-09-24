//! Per-utterance Piper subprocess.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;

use parking_lot::Mutex;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};

use crate::piper::config::PiperRuntimeConfig;
use crate::piper::error::PiperError;

/// Stderr lines kept for the error message when piper exits non-zero.
const STDERR_TAIL_LINES: usize = 20;

/// Signed 16-bit PCM plus its sample rate.
#[derive(Debug, Clone)]
pub struct SynthOutput {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
}

/// Stateless synthesizer: one piper subprocess per call. Piper's raw
/// output has no in-band frame delimiter, so per-utterance EOF on stdout
/// is the only reliable end marker; the 50-250 ms model-load cost
/// overlaps with generation of the next sentence.
pub struct OneShotSynth {
    cfg: Arc<PiperRuntimeConfig>,
}

impl OneShotSynth {
    /// A synthesizer that spawns piper as `cfg` describes.
    pub fn new(cfg: Arc<PiperRuntimeConfig>) -> Self {
        Self { cfg }
    }

    /// Spawn piper, write `text`, and drain stdout to EOF. On non-zero
    /// exit the error carries piper's last stderr lines.
    pub async fn synthesize(&self, text: &str) -> Result<SynthOutput, PiperError> {
        let cfg = &*self.cfg;
        let child = self.command().spawn().map_err(|source| PiperError::Spawn {
            binary: cfg.binary_path.clone(),
            source,
        })?;
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "piper_spawn",
            "voice latency stage"
        );

        let stderr_tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL_LINES)));
        let run = run_child(child, text, stderr_tail.clone());
        let (status, pcm) = match tokio::time::timeout(cfg.deadline, run).await {
            Ok(result) => result?,
            Err(_) => {
                tracing::warn!(
                    target: "assistd::voice::piper",
                    deadline_secs = cfg.deadline.as_secs(),
                    binary = %cfg.binary_path.display(),
                    "piper synthesis exceeded deadline"
                );
                return Err(PiperError::Deadline {
                    secs: cfg.deadline.as_secs(),
                });
            }
        };

        if !status.success() {
            let stderr_tail = stderr_tail
                .lock()
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(" | ");
            return Err(PiperError::SynthFailed {
                status,
                bytes: pcm.len(),
                stderr_tail,
            });
        }

        let samples = decode_pcm(&pcm)?;
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "piper_synth_done",
            "voice latency stage"
        );
        Ok(SynthOutput {
            samples,
            sample_rate: cfg.voice_files.sample_rate,
        })
    }

    fn command(&self) -> Command {
        let cfg = &*self.cfg;
        let mut cmd = Command::new(&cfg.binary_path);
        cmd.arg("--model")
            .arg(&cfg.voice_files.onnx)
            .arg("--output-raw")
            .arg("--length-scale")
            .arg(cfg.length_scale.to_string())
            .arg("--noise-scale")
            .arg(cfg.noise_scale.to_string())
            .arg("--noise-w")
            .arg(cfg.noise_w.to_string())
            .arg("--sentence-silence")
            .arg(cfg.sentence_silence_secs.to_string());
        if cfg.use_cuda {
            cmd.arg("--cuda");
        }
        if let Some(dir) = &cfg.espeak_data_dir {
            cmd.arg("--espeak-data").arg(dir);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        cmd
    }

    /// Synthesize a short probe so a missing binary or corrupt model
    /// fails at startup.
    pub async fn health_check(&self) -> Result<(), PiperError> {
        let out = self.synthesize("ok").await?;
        if out.samples.is_empty() {
            return Err(PiperError::SynthFailed {
                status: std::process::ExitStatus::default(),
                bytes: 0,
                stderr_tail: "health check produced 0 samples".into(),
            });
        }
        Ok(())
    }
}

/// Write `text` to piper's stdin and drain stdout to EOF. The future
/// owns the child, so dropping it on timeout lets `kill_on_drop` reap
/// the process. stderr is drained concurrently because a full pipe
/// would stall piper.
async fn run_child(
    mut child: Child,
    text: &str,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
) -> Result<(ExitStatus, Vec<u8>), PiperError> {
    let mut stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let line = format!("{}\n", text.replace('\n', " "));
    let written = stdin.write_all(line.as_bytes()).await;
    drop(stdin);
    if let Err(source) = written {
        let _ = child.start_kill();
        let _ = child.wait().await;
        return Err(pipe_error("<piper stdin>", source));
    }

    let wait = async {
        child
            .wait()
            .await
            .map_err(|source| pipe_error("<piper child>", source))
    };
    let (status, pcm, ()) = tokio::join!(
        wait,
        drain_stdout(stdout),
        drain_stderr(stderr, stderr_tail)
    );
    Ok((status?, pcm?))
}

fn pipe_error(what: &str, source: std::io::Error) -> PiperError {
    PiperError::Io {
        path: PathBuf::from(what),
        source,
    }
}

fn decode_pcm(bytes: &[u8]) -> Result<Vec<i16>, PiperError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(PiperError::OddPcmLength { bytes: bytes.len() });
    }
    let (words, _) = bytes.as_chunks::<2>();
    Ok(words.iter().copied().map(i16::from_le_bytes).collect())
}

async fn drain_stdout(mut stdout: ChildStdout) -> Result<Vec<u8>, PiperError> {
    // Chunked rather than `read_to_end` so the first PCM byte can be
    // timestamped.
    let mut buf = Vec::with_capacity(64 * 1024);
    let mut chunk = [0u8; 8192];
    let mut first = true;
    loop {
        let n = stdout
            .read(&mut chunk)
            .await
            .map_err(|source| pipe_error("<piper stdout>", source))?;
        if n == 0 {
            break;
        }
        if first {
            tracing::debug!(
                target: "assistd::voice::latency",
                stage = "piper_first_byte",
                "voice latency stage"
            );
            first = false;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok(buf)
}

async fn drain_stderr(stderr: ChildStderr, tail: Arc<Mutex<VecDeque<String>>>) {
    let mut lines = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        tracing::debug!(target: "assistd::voice::piper", "{line}");
        let mut guard = tail.lock();
        if guard.len() == STDERR_TAIL_LINES {
            guard.pop_front();
        }
        guard.push_back(line);
    }
}

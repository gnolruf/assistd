//! Per-utterance Piper subprocess. One `synthesize()` call =
//! one `Command::spawn` of `piper --output-raw ...` with the text
//! written to stdin and EOF signaled by closing the pipe. EOF on
//! stdout marks the end of audio; the child exits naturally afterward.
//!
//! Why per-utterance: piper writes raw PCM continuously to stdout
//! with no in-band frame delimiter. Long-running mode would have to
//! detect utterance boundaries via stderr-log markers, which is
//! version-fragile and races against the kernel pipe between FDs.
//! Per-utterance trades 50–250 ms of CPU model-load latency for a
//! deterministic protocol — that latency overlaps with LLM generation
//! of the next sentence, so users don't see it.

use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStderr, ChildStdout, Command};

use crate::piper::config::PiperRuntimeConfig;
use crate::piper::error::PiperError;

/// Result of one `synthesize` call: 16-bit signed little-endian PCM
/// samples plus the sample rate (read from the voice's `.onnx.json`).
#[derive(Debug, Clone)]
pub struct SynthOutput {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
}

/// Stateless synthesizer. The runtime config is shared via `Arc`; each
/// `synthesize` call reads from it without mutation.
pub struct OneShotSynth {
    cfg: Arc<PiperRuntimeConfig>,
}

impl OneShotSynth {
    pub fn new(cfg: Arc<PiperRuntimeConfig>) -> Self {
        Self { cfg }
    }

    /// Spawn piper, write `text + \n`, close stdin, drain stdout to
    /// EOF, return PCM. Stderr is read concurrently to prevent the
    /// kernel pipe from filling and blocking piper mid-synthesis.
    /// On non-zero exit, the last 20 stderr lines are included in the
    /// error so callers can see piper's complaint.
    pub async fn synthesize(&self, text: &str) -> Result<SynthOutput, PiperError> {
        let cfg = &*self.cfg;
        let mut cmd = Command::new(&cfg.binary_path);
        cmd.arg("--model")
            .arg(&cfg.voice_files.onnx)
            .arg("--output-raw")
            .arg("--length-scale")
            .arg(format!("{}", cfg.length_scale))
            .arg("--noise-scale")
            .arg(format!("{}", cfg.noise_scale))
            .arg("--noise-w")
            .arg(format!("{}", cfg.noise_w))
            .arg("--sentence-silence")
            .arg(format!("{}", cfg.sentence_silence_secs));
        if let Some(ref dir) = cfg.espeak_data_dir {
            cmd.arg("--espeak-data").arg(dir);
        }

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(unix)]
        cmd.process_group(0);

        let mut child = cmd.spawn().map_err(|source| PiperError::Spawn {
            binary: cfg.binary_path.clone(),
            source,
        })?;

        let mut stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");

        let stderr_tail = Arc::new(Mutex::new(VecDeque::<String>::with_capacity(20)));
        let stderr_tail_for_drain = stderr_tail.clone();

        // Concurrent helpers: `read_to_end` on stdout, line-buffered
        // tail on stderr. Reading only stdout would let stderr fill the
        // kernel pipe and stall piper mid-synthesis.
        //
        // Everything (child + stdout + stderr) is `move`d into the
        // inner future. On timeout the future is dropped, which drops
        // `child` — and `kill_on_drop(true)` ensures the OS process
        // dies. The drain tasks live for the duration of the future
        // via `tokio::join!`, so they're cancelled together.
        let deadline = cfg.deadline;
        let binary = cfg.binary_path.clone();
        let text_owned = text.replace('\n', " ");
        let synth_fut = async move {
            let write_res = stdin.write_all(format!("{text_owned}\n").as_bytes()).await;
            drop(stdin); // signal EOF unconditionally
            if let Err(e) = write_res {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err::<(std::process::ExitStatus, Vec<u8>), PiperError>(PiperError::Io {
                    path: std::path::PathBuf::from("<piper stdin>"),
                    source: e,
                });
            }

            let stdout_drain = drain_stdout(stdout);
            let stderr_drain = drain_stderr(stderr, stderr_tail_for_drain);
            let wait_fut = async {
                child.wait().await.map_err(|source| PiperError::Io {
                    path: std::path::PathBuf::from("<piper child>"),
                    source,
                })
            };

            let (status_res, pcm_res, _) = tokio::join!(wait_fut, stdout_drain, stderr_drain);
            let status = status_res?;
            let pcm = pcm_res?;
            Ok((status, pcm))
        };

        let (status, pcm) = match tokio::time::timeout(deadline, synth_fut).await {
            Ok(Ok(pair)) => pair,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                // Future dropped → child dropped → kill_on_drop fires.
                tracing::warn!(
                    target: "assistd::voice::piper",
                    deadline_secs = deadline.as_secs(),
                    binary = %binary,
                    "piper synthesis exceeded deadline"
                );
                return Err(PiperError::Deadline {
                    secs: deadline.as_secs(),
                });
            }
        };

        if !status.success() {
            let tail_joined = stderr_tail
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(" | ");
            return Err(PiperError::SynthFailed {
                status,
                bytes: pcm.len(),
                stderr_tail: tail_joined,
            });
        }

        if pcm.len() % 2 != 0 {
            return Err(PiperError::OddPcmLength { bytes: pcm.len() });
        }

        let samples: Vec<i16> = pcm
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        Ok(SynthOutput {
            samples,
            sample_rate: cfg.voice_files.sample_rate,
        })
    }

    /// Tiny synthesis used by `PiperVoiceOutput::start()` to fail fast
    /// at daemon startup if the binary is missing, the model file is
    /// corrupt, or piper exits non-zero for any reason.
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

async fn drain_stdout(mut stdout: ChildStdout) -> Result<Vec<u8>, PiperError> {
    let mut buf = Vec::with_capacity(64 * 1024);
    stdout
        .read_to_end(&mut buf)
        .await
        .map_err(|source| PiperError::Io {
            path: std::path::PathBuf::from("<piper stdout>"),
            source,
        })?;
    Ok(buf)
}

async fn drain_stderr(stderr: ChildStderr, tail: Arc<Mutex<VecDeque<String>>>) {
    let mut lines = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        tracing::debug!(target: "assistd::voice::piper", "{line}");
        let mut guard = tail.lock().unwrap_or_else(|e| e.into_inner());
        if guard.len() == 20 {
            guard.pop_front();
        }
        guard.push_back(line);
    }
}

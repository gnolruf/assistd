//! Background probe that reports live VRAM usage in the TUI status bar.
//!
//! Shells out to `nvidia-smi` every [`POLL_INTERVAL`] and publishes the
//! parsed used/total memory on a `tokio::sync::watch` channel. If
//! `nvidia-smi` is not on the system `PATH`, the probe marks itself
//! `Disabled` and exits — there is nothing to recover from.

use std::time::Duration;

use tokio::process::Command;
use tokio::sync::watch;
use tracing::warn;

const POLL_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramInfo {
    pub used_mb: u64,
    pub total_mb: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum VramState {
    #[default]
    Unknown,
    Disabled,
    Ok(VramInfo),
    Err(String),
}

pub fn spawn_probe(mut shutdown: watch::Receiver<bool>) -> watch::Receiver<VramState> {
    let (tx, rx) = watch::channel(VramState::Unknown);

    tokio::spawn(async move {
        loop {
            match run_nvidia_smi().await {
                Ok(info) => {
                    let _ = tx.send(VramState::Ok(info));
                }
                Err(ProbeError::NotInstalled) => {
                    let _ = tx.send(VramState::Disabled);
                    return;
                }
                Err(ProbeError::Transient(msg)) => {
                    warn!(target: "assistd::chat::vram", "nvidia-smi probe failed: {msg}");
                    let _ = tx.send(VramState::Err(msg));
                }
            }

            tokio::select! {
                _ = tokio::time::sleep(POLL_INTERVAL) => {}
                _ = shutdown.changed() => return,
            }
            if *shutdown.borrow() {
                return;
            }
        }
    });

    rx
}

enum ProbeError {
    NotInstalled,
    Transient(String),
}

async fn run_nvidia_smi() -> Result<VramInfo, ProbeError> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => ProbeError::NotInstalled,
            _ => ProbeError::Transient(format!("spawn nvidia-smi: {e}")),
        })?;

    if !out.status.success() {
        return Err(ProbeError::Transient(format!(
            "nvidia-smi exited {}",
            out.status
        )));
    }

    parse_output(&out.stdout).map_err(ProbeError::Transient)
}

/// Parse `memory.used,memory.total` CSV from nvidia-smi. Multiple GPUs sum.
fn parse_output(stdout: &[u8]) -> Result<VramInfo, String> {
    let text = std::str::from_utf8(stdout).map_err(|e| format!("invalid utf8: {e}"))?;
    let mut used = 0u64;
    let mut total = 0u64;
    let mut any = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',').map(str::trim);
        let u = parts
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| format!("unparsable used field: {line}"))?;
        let t = parts
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| format!("unparsable total field: {line}"))?;
        used += u;
        total += t;
        any = true;
    }
    if !any {
        return Err("nvidia-smi returned no GPUs".into());
    }
    Ok(VramInfo {
        used_mb: used,
        total_mb: total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_gpu() {
        let info = parse_output(b"1024, 24576\n").unwrap();
        assert_eq!(info.used_mb, 1024);
        assert_eq!(info.total_mb, 24576);
    }

    #[test]
    fn parse_multi_gpu_sums() {
        let info = parse_output(b"1024, 24576\n2048, 24576\n").unwrap();
        assert_eq!(info.used_mb, 3072);
        assert_eq!(info.total_mb, 49152);
    }

    #[test]
    fn parse_rejects_empty() {
        assert!(parse_output(b"\n\n").is_err());
    }

    #[test]
    fn parse_rejects_non_numeric() {
        assert!(parse_output(b"not, numbers\n").is_err());
    }

    #[test]
    fn parse_handles_trailing_whitespace() {
        let info = parse_output(b"  512 ,  8192  \n").unwrap();
        assert_eq!(info.used_mb, 512);
        assert_eq!(info.total_mb, 8192);
    }
}

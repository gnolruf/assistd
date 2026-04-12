//! Background probe that reports live VRAM and RAM usage in the TUI status
//! bar.
//!
//! Shells out to `nvidia-smi` every [`POLL_INTERVAL`] for VRAM and reads
//! `/proc/meminfo` for RAM. If `nvidia-smi` is not on the system `PATH`,
//! VRAM is marked `Disabled` while RAM continues to be polled.

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamInfo {
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RamState {
    #[default]
    Unknown,
    Ok(RamInfo),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceState {
    pub vram: VramState,
    pub ram: RamState,
}

pub fn spawn_probe(mut shutdown: watch::Receiver<bool>) -> watch::Receiver<ResourceState> {
    let (tx, rx) = watch::channel(ResourceState::default());
    let mut vram_disabled = false;

    tokio::spawn(async move {
        loop {
            let vram = if vram_disabled {
                VramState::Disabled
            } else {
                match run_nvidia_smi().await {
                    Ok(info) => VramState::Ok(info),
                    Err(ProbeError::NotInstalled) => {
                        vram_disabled = true;
                        VramState::Disabled
                    }
                    Err(ProbeError::Transient(msg)) => {
                        warn!(target: "assistd::chat::vram", "nvidia-smi probe failed: {msg}");
                        VramState::Err(msg)
                    }
                }
            };

            let ram = match read_meminfo() {
                Ok(info) => RamState::Ok(info),
                Err(e) => {
                    warn!(target: "assistd::chat::vram", "meminfo read failed: {e}");
                    RamState::Unknown
                }
            };

            let _ = tx.send(ResourceState { vram, ram });

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

/// Read used/total RAM from `/proc/meminfo`. Used = Total - Available.
fn read_meminfo() -> Result<RamInfo, String> {
    let content =
        std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("read /proc/meminfo: {e}"))?;
    parse_meminfo(&content)
}

fn parse_meminfo(content: &str) -> Result<RamInfo, String> {
    let mut total_kb = None;
    let mut available_kb = None;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = parse_meminfo_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available_kb = parse_meminfo_kb(rest);
        }
        if total_kb.is_some() && available_kb.is_some() {
            break;
        }
    }
    let total = total_kb.ok_or("MemTotal not found in /proc/meminfo")?;
    let available = available_kb.ok_or("MemAvailable not found in /proc/meminfo")?;
    Ok(RamInfo {
        used_mb: total.saturating_sub(available) / 1024,
        total_mb: total / 1024,
    })
}

fn parse_meminfo_kb(value: &str) -> Option<u64> {
    value.trim().strip_suffix("kB")?.trim().parse().ok()
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

    #[test]
    fn meminfo_typical() {
        let content = "\
MemTotal:       65536000 kB
MemFree:         2048000 kB
MemAvailable:   32768000 kB
Buffers:          512000 kB
";
        let info = parse_meminfo(content).unwrap();
        assert_eq!(info.total_mb, 64000);
        assert_eq!(info.used_mb, 32000);
    }

    #[test]
    fn meminfo_missing_available_errors() {
        let content = "MemTotal:       65536000 kB\n";
        assert!(parse_meminfo(content).is_err());
    }
}

//! The single source of truth for every config default.

use std::net::{IpAddr, Ipv4Addr};
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};
use std::path::PathBuf;

/// Build a `NonZero` from a literal, rejecting a zero at compile time
/// rather than at config load.
pub const fn nz16(v: u16) -> NonZeroU16 {
    match NonZeroU16::new(v) {
        Some(n) => n,
        None => panic!("default must be non-zero"),
    }
}

/// See [`nz16`].
pub const fn nz32(v: u32) -> NonZeroU32 {
    match NonZeroU32::new(v) {
        Some(n) => n,
        None => panic!("default must be non-zero"),
    }
}

/// See [`nz16`].
pub const fn nz64(v: u64) -> NonZeroU64 {
    match NonZeroU64::new(v) {
        Some(n) => n,
        None => panic!("default must be non-zero"),
    }
}

pub const DEFAULT_LLAMA_BINARY: &str = "llama-server";
pub const DEFAULT_LLAMA_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
pub const DEFAULT_LLAMA_PORT: NonZeroU16 = nz16(8385);
pub const DEFAULT_GPU_LAYERS: u32 = 9999;
pub const DEFAULT_READY_TIMEOUT_SECS: NonZeroU64 = nz64(300);

pub const DEFAULT_MODEL_NAME: &str = "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M";
pub const DEFAULT_MODEL_CONTEXT_LENGTH: NonZeroU32 = nz32(8192);

pub const DEFAULT_CHAT_MAX_HISTORY_TOKENS: NonZeroU32 = nz32(6000);
pub const DEFAULT_CHAT_SUMMARY_TARGET_TOKENS: NonZeroU32 = nz32(1000);
pub const DEFAULT_CHAT_PRESERVE_RECENT_TURNS: NonZeroU32 = nz32(4);
pub const DEFAULT_CHAT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_CHAT_MAX_RESPONSE_TOKENS: NonZeroU32 = nz32(1024);
pub const DEFAULT_CHAT_REQUEST_TIMEOUT_SECS: NonZeroU64 = nz64(120);
pub const DEFAULT_CHAT_SUMMARY_TEMPERATURE: f32 = 0.3;
/// Role-and-voice prose only. Tools reach the model through each
/// request's `tools` array, so this must not name them.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are assistd, a concise local desktop assistant \
     running on a Linux workstation. When a question is about this machine or its files, \
     prefer calling a tool over guessing. Answer precisely and in a conversational tone.";

pub const DEFAULT_VOICE_HOTKEY: &str = "Super+Space";
pub const DEFAULT_VOICE_MAX_RECORDING_SECS: NonZeroU32 = nz32(60);

pub const DEFAULT_WHISPER_MODEL: &str = "ggerganov/whisper.cpp:ggml-large-v3-turbo-q5_0.bin";
pub const DEFAULT_WHISPER_VAD_MODEL: &str = "ggml-org/whisper-vad:ggml-silero-v6.2.0.bin";
pub const DEFAULT_WHISPER_PREFER_GPU: bool = true;
pub const DEFAULT_WHISPER_BEAMS: NonZeroU32 = nz32(1);
pub const DEFAULT_WHISPER_VAD_ENABLED: bool = true;

pub const DEFAULT_PIPER_ENABLED: bool = false;
pub const DEFAULT_PIPER_BINARY: &str = "piper";
pub const DEFAULT_PIPER_VOICE: &str =
    "rhasspy/piper-voices:en/en_US/lessac/medium/en_US-lessac-medium.onnx";
pub const DEFAULT_PIPER_LENGTH_SCALE: f32 = 1.0;
pub const DEFAULT_PIPER_DEADLINE_SECS: NonZeroU32 = nz32(30);
pub const DEFAULT_PIPER_MAX_SENTENCE_CHARS: NonZeroU32 = nz32(400);
pub const DEFAULT_PIPER_PARTIAL_FLUSH_MS: u32 = 750;
pub const DEFAULT_PIPER_TOGGLE_HOTKEY: &str = "";
pub const DEFAULT_PIPER_SKIP_HOTKEY: &str = "";

pub const DEFAULT_LISTEN_ENABLED: bool = false;
pub const DEFAULT_LISTEN_START_ON_LAUNCH: bool = false;
pub const DEFAULT_LISTEN_HOTKEY: &str = "";
pub const DEFAULT_LISTEN_SILENCE_MS: NonZeroU32 = nz32(800);
pub const DEFAULT_LISTEN_MAX_UTTERANCE_SECS: NonZeroU32 = nz32(30);

pub const DEFAULT_PRESENCE_HOTKEY: &str = "Super+Escape";

pub const DEFAULT_DAEMON_SHUTDOWN_GRACE_SECS: u64 = 5;

pub const DEFAULT_TIMEOUT_PRESENCE_SLEEP_SECS: u64 = 30;
pub const DEFAULT_TIMEOUT_PRESENCE_DROWSE_SECS: u64 = 10;
pub const DEFAULT_TIMEOUT_DISPATCH_ENVELOPE_SECS: u64 = 600;
pub const DEFAULT_TIMEOUT_STREAM_INACTIVITY_SECS: u64 = 30;
pub const DEFAULT_TIMEOUT_TOOL_CALL_SECS: u64 = 300;

pub const DEFAULT_IDLE_TO_DROWSY_MINS: u64 = 30;
pub const DEFAULT_IDLE_TO_SLEEP_MINS: u64 = 120;
pub const DEFAULT_GPU_MONITOR_ENABLED: bool = true;
pub const DEFAULT_GPU_POLL_SECS: NonZeroU64 = nz64(5);
pub const DEFAULT_GPU_VRAM_THRESHOLD_MB: NonZeroU64 = nz64(2048);

pub const DEFAULT_TOOLS_MAX_LINES: NonZeroU32 = nz32(200);
pub const DEFAULT_TOOLS_MAX_KB: NonZeroU32 = nz32(50);
pub const DEFAULT_TOOLS_OVERFLOW_DIR: &str = "/tmp/assistd-output";
pub const DEFAULT_BASH_TIMEOUT_SECS: NonZeroU64 = nz64(30);

pub const DEFAULT_MEMORY_ENABLED: bool = true;

pub const DEFAULT_EMBEDDING_ENABLED: bool = true;
pub const DEFAULT_EMBEDDING_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M";
pub const DEFAULT_EMBEDDING_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
pub const DEFAULT_EMBEDDING_PORT: NonZeroU16 = nz16(8386);
/// CPU-only so the embedder never contends with the chat model for VRAM.
pub const DEFAULT_EMBEDDING_GPU_LAYERS: u32 = 0;
pub const DEFAULT_EMBEDDING_TOP_K: NonZeroU32 = nz32(5);
pub const DEFAULT_EMBEDDING_AUTO_INJECT: bool = true;

pub const DEFAULT_MCP_ENABLED: bool = false;
pub const DEFAULT_MCP_REQUEST_TIMEOUT_SECS: NonZeroU64 = nz64(30);

pub const DEFAULT_TRAY_POPUP_ENABLED: bool = true;
pub const DEFAULT_TRAY_POPUP_WIDTH: u32 = 360;
pub const DEFAULT_TRAY_POPUP_HEIGHT: u32 = 120;
pub const DEFAULT_TRAY_POPUP_OFFSET_X: i32 = -10;
pub const DEFAULT_TRAY_POPUP_OFFSET_Y: i32 = 10;
pub const DEFAULT_TRAY_POPUP_AUTO_HIDE_MS: u64 = 3000;
/// `WM_CLASS` / `app_id` of the popup window, used both to create it
/// and to place it, so it is not configurable.
pub const DEFAULT_TRAY_POPUP_APP_ID: &str = "dev.assistd.popup";
pub const DEFAULT_TRAY_POPUP_WAKE_TOOL_CALL: bool = true;
pub const DEFAULT_TRAY_POPUP_WAKE_DELTA: bool = true;
pub const DEFAULT_TRAY_POPUP_WAKE_ERROR: bool = true;

/// `$XDG_DATA_HOME/assistd/memory.db`, or
/// `$HOME/.local/share/assistd/memory.db`.
pub fn default_memory_db_path() -> PathBuf {
    let data_dir = match std::env::var_os("XDG_DATA_HOME") {
        Some(d) if !d.is_empty() => PathBuf::from(d),
        _ => {
            let home = std::env::var_os("HOME").unwrap_or_default();
            PathBuf::from(home).join(".local/share")
        }
    };
    data_dir.join("assistd").join("memory.db")
}

/// Process basenames whose GPU use never triggers sleep.
pub fn default_gpu_allowlist() -> Vec<String> {
    vec![
        "Xorg".into(),
        "Xwayland".into(),
        "gnome-shell".into(),
        "kwin_x11".into(),
        "kwin_wayland".into(),
        "firefox".into(),
        "chromium".into(),
        "chrome".into(),
    ]
}

/// Literal command substrings rejected before spawn.
pub fn default_bash_denylist() -> Vec<String> {
    vec![
        "rm -rf /".into(),
        "rm -rf /*".into(),
        "rm -rf /home".into(),
        "mkfs".into(),
        "dd if=/dev/zero".into(),
        ":(){ :|:& };:".into(),
        "> /dev/sda".into(),
        "> /dev/nvme".into(),
    ]
}

/// Command prefixes that require confirmation.
pub fn default_bash_destructive_patterns() -> Vec<String> {
    vec![
        "rm -rf".into(),
        "rm -fr".into(),
        "git push --force".into(),
        "git push -f".into(),
        "git reset --hard".into(),
        "dd of=".into(),
        "shutdown".into(),
        "reboot".into(),
        "kill -9 -1".into(),
    ]
}

/// Path prefixes the `write` command may create files under.
pub fn default_writable_paths() -> Vec<String> {
    vec!["~".into(), "/tmp".into()]
}

//! Centralised default values for every config field.
//!
//! Each constant is the single source of truth for its field: referenced
//! by the `Default` impl in the owning section module and by tests that
//! assert on defaults. Changing the literal here propagates everywhere
//! with no drift.

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
/// Slim role-and-voice prose. Tool surface (native + MCP) is appended at
/// daemon startup by `assistd_tools::prompt::format_tool_listing` and
/// `assistd_mcp::prompt::format_mcp_listing`, so this constant must not
/// re-enumerate tool names — adding one would create a divergence point
/// with the registry that the dynamic listings exist to prevent.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are assistd, a concise local desktop assistant \
     running on a Linux workstation. When a question is about this machine or its files, \
     prefer calling a tool over guessing. Answer precisely and in a conversational tone.";

pub const DEFAULT_VOICE_HOTKEY: &str = "Super+Space";
/// Upper bound on push-to-talk recording length, in seconds. A held
/// hotkey past this is truncated to the first N seconds (the ring buffer
/// drops newer samples once full). Also determines the buffer size
/// pre-allocated when recording starts.
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
/// Idle gap (ms) between LLM deltas after which the sentence buffer is
/// flushed even without a terminator. `0` disables the timeout flush;
/// only the terminal `Done`-based flush is used. Inhibited while a tool
/// call is in flight (the LLM is waiting on a tool, not stalled).
pub const DEFAULT_PIPER_PARTIAL_FLUSH_MS: u32 = 750;
/// Empty by default; opt-in like `DEFAULT_LISTEN_HOTKEY`. When non-empty
/// and synthesis is enabled, pressing the hotkey flips TTS on/off at
/// runtime (also silences current playback when turning off).
pub const DEFAULT_PIPER_TOGGLE_HOTKEY: &str = "";
/// Empty by default. When non-empty, pressing the hotkey aborts the
/// current response: drops queued audio and any pending sentences for
/// the in-flight query without starting recording. TTS stays armed for
/// the next response.
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
/// HuggingFace id passed verbatim to the embed server's `--hf-repo`.
/// The `:` suffix must be a quant tag, not a `.gguf` filename: llama-server
/// resolves it against its preset manifest and a filename fails with a
/// misleading "no GGUF files found". 768-dim, ~140 MB Q4.
pub const DEFAULT_EMBEDDING_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M";
pub const DEFAULT_EMBEDDING_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
/// Distinct from `DEFAULT_LLAMA_PORT` (8385). Validated for collisions in
/// `Config::validate()`.
pub const DEFAULT_EMBEDDING_PORT: NonZeroU16 = nz16(8386);
/// CPU-only by default: small embedders are CPU-fast, and pinning them
/// off the GPU prevents VRAM contention with the chat model.
pub const DEFAULT_EMBEDDING_GPU_LAYERS: u32 = 0;
pub const DEFAULT_EMBEDDING_TOP_K: NonZeroU32 = nz32(5);
pub const DEFAULT_EMBEDDING_AUTO_INJECT: bool = true;

/// MCP (Model Context Protocol), opt-in. Existing users on upgrade
/// haven't authored any servers; default-off keeps their startup
/// noise-free. They flip `enabled = true` when they add their first
/// `[[mcp.servers]]` block.
pub const DEFAULT_MCP_ENABLED: bool = false;
pub const DEFAULT_MCP_REQUEST_TIMEOUT_SECS: NonZeroU64 = nz64(30);

/// Borderless floating popup spawned by `assistd tray` (feature
/// `tray-popup`). Geometry is in CSS pixels at the compositor's logical
/// scale. Offsets are taken from the anchored corner: a negative
/// `offset_x` on a right-anchored popup moves it inward (toward the
/// screen centre), the same on a left-anchored popup moves it
/// off-screen.
pub const DEFAULT_TRAY_POPUP_ENABLED: bool = true;
pub const DEFAULT_TRAY_POPUP_WIDTH: u32 = 360;
pub const DEFAULT_TRAY_POPUP_HEIGHT: u32 = 120;
pub const DEFAULT_TRAY_POPUP_OFFSET_X: i32 = -10;
pub const DEFAULT_TRAY_POPUP_OFFSET_Y: i32 = 10;
pub const DEFAULT_TRAY_POPUP_AUTO_HIDE_MS: u64 = 3000;
/// X11 `WM_CLASS` and Wayland `app_id` of the popup window. The popup
/// GUI builder sets it; the `[app_id="…"]` placement criteria sent
/// through `assistd-wm` matches against it. Not exposed in
/// `TrayPopupConfig` because both sides must agree — a config knob is
/// all footgun and no upside.
pub const DEFAULT_TRAY_POPUP_APP_ID: &str = "dev.assistd.popup";
/// Wake the popup on a `Event::ToolCall`. Catches every MCP tool /
/// bash / web invocation.
pub const DEFAULT_TRAY_POPUP_WAKE_TOOL_CALL: bool = true;
/// Wake the popup on the first `Event::LastDelta` of a turn — i.e. as
/// soon as the model starts replying. Default-on; flip to false if
/// you're chatting in the TUI and don't want the popup on every turn.
pub const DEFAULT_TRAY_POPUP_WAKE_DELTA: bool = true;
/// Wake the popup on `Event::Error`. Useful for noticing failures
/// you'd otherwise miss in the tracing log.
pub const DEFAULT_TRAY_POPUP_WAKE_ERROR: bool = true;

/// Returns the default SQLite memory database path, honouring `$XDG_DATA_HOME`.
///
/// Resolves to `$XDG_DATA_HOME/assistd/memory.db` when set and non-empty,
/// otherwise `$HOME/.local/share/assistd/memory.db`.
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

/// Returns the default GPU contention allowlist of process basenames that never trigger sleep.
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

/// Returns the default bash denylist of literal command strings that are rejected before spawn.
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

/// Returns the default list of shell-command prefixes that require confirmation before execution.
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

/// Returns the default list of path prefixes under which the `write` command may create files.
pub fn default_writable_paths() -> Vec<String> {
    vec!["~".into(), "/tmp".into()]
}

//! The single source of truth for every config default.

use std::net::{IpAddr, Ipv4Addr};
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};
use std::path::PathBuf;

use crate::voice::UnitInterval;

/// Builds a `NonZero` from a literal; a zero fails at compile time.
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

/// Builds a [`UnitInterval`] from a literal; a value outside `0.0..=1.0`
/// fails at compile time.
pub const fn unit(v: f32) -> UnitInterval {
    match UnitInterval::new(v) {
        Some(u) => u,
        None => panic!("default must be in 0.0..=1.0"),
    }
}

pub const DEFAULT_MODEL_NAME: &str = "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M";
pub const DEFAULT_MODEL_CONTEXT_LENGTH: NonZeroU32 = nz32(8192);
pub const DEFAULT_MODEL_SERVER_BINARY: &str = "llama-server";
pub const DEFAULT_MODEL_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
pub const DEFAULT_MODEL_PORT: NonZeroU16 = nz16(8385);
pub const DEFAULT_MODEL_GPU_LAYERS: u32 = 9999;
pub const DEFAULT_MODEL_READY_TIMEOUT_SECS: NonZeroU64 = nz64(300);

pub const DEFAULT_CHAT_MAX_HISTORY_TOKENS: NonZeroU32 = nz32(6000);
pub const DEFAULT_CHAT_SUMMARY_TARGET_TOKENS: NonZeroU32 = nz32(1000);
pub const DEFAULT_CHAT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_CHAT_MAX_RESPONSE_TOKENS: NonZeroU32 = nz32(1024);
pub const DEFAULT_CHAT_MAX_REASONING_TOKENS: NonZeroU32 = nz32(512);
pub const DEFAULT_CHAT_REQUEST_TIMEOUT_SECS: NonZeroU64 = nz64(120);
pub const DEFAULT_CHAT_SUMMARY_TEMPERATURE: f32 = 0.3;
/// Must not name tools: each request already carries their schemas.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are assistd, a concise local desktop assistant \
     running on a Linux workstation. When a question is about this machine or its files, \
     prefer calling a tool over guessing. Answer precisely and in a conversational tone.";

pub const DEFAULT_STT_MODEL: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16\
     @a6cc4e24ed264a93a17575dad416dacc4238497a:encoder.fp16.onnx";
pub const DEFAULT_SILERO_VAD_MODEL: &str =
    "Tinysoft/silero-vad-6.2@3034960e5d3beb943b860ec0ea1a857bdae4ae96:silero_vad.onnx";
pub const DEFAULT_STT_PREFER_GPU: bool = false;
pub const DEFAULT_STT_THREADS: NonZeroU32 = nz32(4);
pub const DEFAULT_STT_VAD_ENABLED: bool = true;

pub const DEFAULT_TTS_ENABLED: bool = false;
pub const DEFAULT_TTS_VOICE: &str = "csukuangfj/vits-piper-en_US-lessac-medium\
     @83f7470750b36549037551ef7007fa3b4b8697e0:en_US-lessac-medium.onnx";
pub const DEFAULT_TTS_LENGTH_SCALE: f32 = 1.0;
pub const DEFAULT_TTS_MAX_SENTENCE_CHARS: NonZeroU32 = nz32(400);
pub const DEFAULT_TTS_PARTIAL_FLUSH_MS: u32 = 750;

pub const DEFAULT_WAKEWORD_ENABLED: bool = false;
pub const DEFAULT_WAKEWORD_START_ON_LAUNCH: bool = false;
pub const DEFAULT_WAKEWORD_SILENCE_MS: NonZeroU32 = nz32(800);
pub const DEFAULT_WAKEWORD_MAX_UTTERANCE_SECS: NonZeroU32 = nz32(30);
pub const DEFAULT_WAKEWORD_BARGE_IN: bool = true;
pub const DEFAULT_WAKEWORD_COMMAND_TIMEOUT_MS: NonZeroU32 = nz32(4000);
pub const DEFAULT_WAKEWORD_FOLLOW_UP_MS: u32 = 8000;
pub const DEFAULT_WAKE_PHRASE: &str = "hey computer";
pub const DEFAULT_WAKEWORD_THRESHOLD: UnitInterval = unit(0.25);

pub const DEFAULT_HOTKEY_PUSH_TO_TALK: &str = "Super+Space";

pub const DEFAULT_TIMEOUT_PRESENCE_SLEEP_SECS: u64 = 30;
pub const DEFAULT_TIMEOUT_PRESENCE_DROWSE_SECS: u64 = 10;
pub const DEFAULT_TIMEOUT_DISPATCH_ENVELOPE_SECS: u64 = 600;
pub const DEFAULT_TIMEOUT_STREAM_INACTIVITY_SECS: u64 = 30;
pub const DEFAULT_TIMEOUT_TOOL_CALL_SECS: u64 = 300;

pub const DEFAULT_IDLE_TO_DROWSY_MINS: u64 = 30;
pub const DEFAULT_IDLE_TO_SLEEP_MINS: u64 = 120;
pub const DEFAULT_GPU_MONITOR_ENABLED: bool = true;
pub const DEFAULT_GPU_POLL_SECS: NonZeroU64 = nz64(5);
pub const DEFAULT_GPU_HEADROOM_MB: u64 = 500;

pub const DEFAULT_TOOLS_MAX_LINES: NonZeroU32 = nz32(200);
pub const DEFAULT_TOOLS_MAX_KB: NonZeroU32 = nz32(4);
pub const DEFAULT_TOOLS_SCRATCH_RETENTION_DAYS: u32 = 30;
pub const DEFAULT_BASH_TIMEOUT_SECS: NonZeroU64 = nz64(30);

pub const DEFAULT_MEMORY_ENABLED: bool = true;

pub const DEFAULT_EMBEDDING_ENABLED: bool = true;
pub const DEFAULT_EMBEDDING_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M";
pub const DEFAULT_EMBEDDING_SERVER_BINARY: &str = "llama-server";
pub const DEFAULT_EMBEDDING_READY_TIMEOUT_SECS: NonZeroU64 = nz64(300);
pub const DEFAULT_EMBEDDING_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
pub const DEFAULT_EMBEDDING_PORT: NonZeroU16 = nz16(8386);
/// CPU-only so the embedder never contends with the chat model for VRAM.
pub const DEFAULT_EMBEDDING_GPU_LAYERS: u32 = 0;
pub const DEFAULT_EMBEDDING_TOP_K: NonZeroU32 = nz32(5);
pub const DEFAULT_EMBEDDING_AUTO_INJECT: bool = true;

pub const DEFAULT_SEARCH_ENABLED: bool = false;
/// The console script upstream SearXNG installs.
pub const DEFAULT_SEARCH_COMMAND: &str = "searxng-run";
pub const DEFAULT_SEARCH_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
pub const DEFAULT_SEARCH_PORT: NonZeroU16 = nz16(8387);
pub const DEFAULT_SEARCH_READY_TIMEOUT_SECS: NonZeroU64 = nz64(60);
/// The engines that answer automated queries from a home connection; most
/// others meet them with CAPTCHAs, rate limits or outright refusals.
pub const DEFAULT_SEARCH_ENGINES: [&str; 2] = ["bing", "yahoo"];

pub const DEFAULT_MCP_ENABLED: bool = false;
pub const DEFAULT_MCP_REQUEST_TIMEOUT_SECS: NonZeroU64 = nz64(30);

pub const DEFAULT_TRAY_ICON_ACTIVE: &str = "user-available";
pub const DEFAULT_TRAY_ICON_SLEEPING: &str = "user-offline";
pub const DEFAULT_TRAY_ICON_LISTENING: &str = "audio-input-microphone";
pub const DEFAULT_TRAY_ICON_GENERATING: &str = "system-run";
pub const DEFAULT_TRAY_ICON_DISCONNECTED: &str = "network-offline";

pub const DEFAULT_NOTIFICATIONS_ENABLED: bool = true;
pub const DEFAULT_NOTIFICATIONS_AUTO_HIDE_MS: u64 = 3000;
pub const DEFAULT_NOTIFICATIONS_BRIEF_WHEN_AWAY: bool = true;

/// `$XDG_RUNTIME_DIR/assistd/output`, or `$XDG_CACHE_HOME/assistd/output`, or
/// `$HOME/.cache/assistd/output`.
pub fn default_tools_overflow_dir() -> PathBuf {
    let base = dirs::runtime_dir()
        .or_else(dirs::cache_dir)
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("assistd").join("output")
}

/// `$XDG_RUNTIME_DIR/assistd/scratch`, or `$XDG_CACHE_HOME/assistd/scratch`,
/// or `$HOME/.cache/assistd/scratch`.
pub fn default_tools_scratch_dir() -> PathBuf {
    let base = dirs::runtime_dir()
        .or_else(dirs::cache_dir)
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("assistd").join("scratch")
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

/// Programs that run without confirmation: tools that neither run another
/// program nor write files except through options the destructive patterns
/// catch, plus wrappers whose command is checked itself.
pub fn default_bash_allowed_programs() -> Vec<String> {
    [
        "[",
        "b2sum",
        "base64",
        "basename",
        "cat",
        "cksum",
        "cmp",
        "column",
        "comm",
        "cut",
        "date",
        "df",
        "diff",
        "dirname",
        "du",
        "echo",
        "egrep",
        "env",
        "expand",
        "false",
        "fgrep",
        "file",
        "find",
        "fold",
        "grep",
        "head",
        "hexdump",
        "hostname",
        "id",
        "join",
        "jq",
        "ls",
        "md5sum",
        "nice",
        "nl",
        "nohup",
        "od",
        "paste",
        "printenv",
        "printf",
        "pwd",
        "readlink",
        "realpath",
        "rev",
        "rg",
        "seq",
        "sha1sum",
        "sha256sum",
        "sha512sum",
        "sleep",
        "sort",
        "stat",
        "stdbuf",
        "strings",
        "tac",
        "tail",
        "test",
        "timeout",
        "tr",
        "tree",
        "true",
        "uname",
        "unexpand",
        "wc",
        "which",
        "whoami",
        "xargs",
    ]
    .map(String::from)
    .into()
}

/// Commands that need confirmation even when their program is allowed:
/// options that delete or write files or run another program, and risky
/// uses of commonly approved programs.
pub fn default_bash_destructive_patterns() -> Vec<String> {
    [
        "rm -r|--recursive",
        "find -delete",
        "find -fprint|-fprint0|-fprintf|-fls",
        "sort -o|--output",
        "tree -o",
        "tree -R",
        "file -C|--compile",
        "dd of=",
        "git push -f|--force|--force-with-lease",
        "git reset --hard",
        "git clean -f|--force",
        "systemctl poweroff|reboot|halt|kexec",
        "kill -1",
        "sort --compress-program",
        "rg --pre",
        "rg --hostname-bin",
    ]
    .map(String::from)
    .into()
}

/// Path prefixes the `write` command may create files under.
pub fn default_writable_paths() -> Vec<String> {
    vec!["~".into()]
}

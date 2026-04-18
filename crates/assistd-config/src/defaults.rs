//! Centralised default values for every config field.
//!
//! Each constant is the single source of truth for its field: referenced
//! both by the `#[serde(default = "...")]` helpers in the section modules
//! and by tests that need to assert on defaults. Changing the literal
//! here propagates everywhere with no drift.

pub const DEFAULT_LLAMA_BINARY: &str = "llama-server";
pub const DEFAULT_LLAMA_HOST: &str = "127.0.0.1";
pub const DEFAULT_LLAMA_PORT: u16 = 8385;
pub const DEFAULT_GPU_LAYERS: u32 = 9999;
pub const DEFAULT_READY_TIMEOUT_SECS: u64 = 300;

pub const DEFAULT_MODEL_NAME: &str = "bartowski/Qwen3-14B-GGUF:Q4_K_M";
pub const DEFAULT_MODEL_CONTEXT_LENGTH: u32 = 8192;

pub const DEFAULT_CHAT_MAX_HISTORY_TOKENS: u32 = 6000;
pub const DEFAULT_CHAT_SUMMARY_TARGET_TOKENS: u32 = 1000;
pub const DEFAULT_CHAT_PRESERVE_RECENT_TURNS: u32 = 4;
pub const DEFAULT_CHAT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_CHAT_MAX_RESPONSE_TOKENS: u32 = 1024;
pub const DEFAULT_CHAT_MAX_SUMMARY_TOKENS: u32 = 1200;
pub const DEFAULT_CHAT_REQUEST_TIMEOUT_SECS: u64 = 120;
pub const DEFAULT_CHAT_SUMMARY_TEMPERATURE: f32 = 0.3;
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are assistd, a concise local desktop assistant running on a Linux \
     workstation. You have a `run` tool that executes shell-style commands \
     (bash, cat, ls, grep, wc, echo, write, see, web) — prefer calling `run` \
     over guessing when a question is about this machine or its files. Pipes \
     (|), and/or (&&, ||), and sequencing (;) are supported inside a single \
     `run` call, so a one-liner like `cat log.txt | grep ERROR | wc -l` is \
     usually better than three separate tool calls. Answer precisely and in \
     a conversational tone.";

pub const DEFAULT_VOICE_HOTKEY: &str = "Super+V";

pub const DEFAULT_REMOTE_BIND_ADDRESS: &str = "127.0.0.1";
pub const DEFAULT_REMOTE_PORT: u16 = 8384;

pub const DEFAULT_PRESENCE_HOTKEY: &str = "Super+Escape";

pub const DEFAULT_DAEMON_SHUTDOWN_GRACE_SECS: u64 = 5;

pub const DEFAULT_IDLE_TO_DROWSY_MINS: u64 = 30;
pub const DEFAULT_IDLE_TO_SLEEP_MINS: u64 = 120;
pub const DEFAULT_GPU_MONITOR_ENABLED: bool = true;
pub const DEFAULT_GPU_POLL_SECS: u64 = 5;
pub const DEFAULT_GPU_VRAM_THRESHOLD_MB: u64 = 2048;

pub const DEFAULT_TOOLS_MAX_LINES: u32 = 200;
pub const DEFAULT_TOOLS_MAX_KB: u32 = 50;
pub const DEFAULT_TOOLS_OVERFLOW_DIR: &str = "/tmp/assistd-output";
pub const DEFAULT_BASH_TIMEOUT_SECS: u64 = 30;

pub const DEFAULT_AGENT_MAX_ITERATIONS: u32 = 20;

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

pub fn default_writable_paths() -> Vec<String> {
    vec!["~".into(), "/tmp".into()]
}

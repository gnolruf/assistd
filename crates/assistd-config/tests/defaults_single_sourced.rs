//! Verifies that every default constant defined in [`defaults`] lines up
//! with the field initializer produced by the corresponding `Default`
//! impl. Catches the common drift where a default is changed in one
//! place but not the other.

use assistd_config::*;
use defaults::*;

#[test]
fn llama_server_defaults_match_constants() {
    let c = LlamaServerConfig::default();
    assert_eq!(c.binary_path, DEFAULT_LLAMA_BINARY);
    assert_eq!(c.host, DEFAULT_LLAMA_HOST);
    assert_eq!(c.port, DEFAULT_LLAMA_PORT);
    assert_eq!(c.gpu_layers, DEFAULT_GPU_LAYERS);
    assert_eq!(c.ready_timeout_secs, DEFAULT_READY_TIMEOUT_SECS);
    assert!(c.alias.is_none());
    assert!(c.override_tensor.is_none());
    assert!(c.flash_attn.is_none());
    assert!(c.cache_type_k.is_none());
    assert!(c.cache_type_v.is_none());
    assert!(c.threads.is_none());
    assert!(c.batch_size.is_none());
    assert!(c.ubatch_size.is_none());
    assert!(c.n_cpu_moe.is_none());
    assert!(c.cache_ram_mib.is_none());
    assert!(c.mlock.is_none());
    assert!(c.mmproj_offload.is_none());
}

#[test]
fn model_defaults_match_constants() {
    let c = ModelConfig::default();
    assert_eq!(c.name, DEFAULT_MODEL_NAME);
    assert_eq!(c.context_length, DEFAULT_MODEL_CONTEXT_LENGTH);
}

#[test]
fn chat_defaults_match_constants() {
    let c = ChatConfig::default();
    assert_eq!(c.system_prompt, DEFAULT_SYSTEM_PROMPT);
    assert_eq!(c.max_history_tokens, DEFAULT_CHAT_MAX_HISTORY_TOKENS);
    assert_eq!(c.summary_target_tokens, DEFAULT_CHAT_SUMMARY_TARGET_TOKENS);
    assert_eq!(c.preserve_recent_turns, DEFAULT_CHAT_PRESERVE_RECENT_TURNS);
    assert_eq!(c.temperature, DEFAULT_CHAT_TEMPERATURE);
    assert_eq!(c.max_response_tokens, DEFAULT_CHAT_MAX_RESPONSE_TOKENS);
    assert_eq!(c.max_summary_tokens, DEFAULT_CHAT_MAX_SUMMARY_TOKENS);
    assert_eq!(c.request_timeout_secs, DEFAULT_CHAT_REQUEST_TIMEOUT_SECS);
    assert_eq!(c.summary_temperature, DEFAULT_CHAT_SUMMARY_TEMPERATURE);
    assert!(c.top_p.is_none());
    assert!(c.top_k.is_none());
    assert!(c.min_p.is_none());
    assert!(c.presence_penalty.is_none());
}

#[test]
fn voice_defaults_match_constants() {
    let c = VoiceConfig::default();
    assert!(!c.enabled);
    assert!(c.mic_device.is_none());
    assert_eq!(c.hotkey, DEFAULT_VOICE_HOTKEY);
}

#[test]
fn continuous_listen_defaults_match_constants() {
    let c = ContinuousListenConfig::default();
    assert_eq!(c.enabled, DEFAULT_LISTEN_ENABLED);
    assert_eq!(c.start_on_launch, DEFAULT_LISTEN_START_ON_LAUNCH);
    assert_eq!(c.hotkey, DEFAULT_LISTEN_HOTKEY);
    assert_eq!(c.silence_ms, DEFAULT_LISTEN_SILENCE_MS);
    assert_eq!(c.min_utterance_ms, DEFAULT_LISTEN_MIN_UTTERANCE_MS);
    assert_eq!(c.max_utterance_secs, DEFAULT_LISTEN_MAX_UTTERANCE_SECS);
    assert_eq!(c.preroll_ms, DEFAULT_LISTEN_PREROLL_MS);
    assert_eq!(c.onset_confirm_ms, DEFAULT_LISTEN_ONSET_CONFIRM_MS);
    assert_eq!(c.aggressiveness, DEFAULT_LISTEN_AGGRESSIVENESS);
}

#[test]
fn transcription_defaults_match_constants() {
    let c = TranscriptionConfig::default();
    assert_eq!(c.model, DEFAULT_WHISPER_MODEL);
    assert_eq!(c.vad_model, DEFAULT_WHISPER_VAD_MODEL);
    assert_eq!(c.prefer_gpu, DEFAULT_WHISPER_PREFER_GPU);
    assert!(c.threads.is_none());
    assert_eq!(c.beams, DEFAULT_WHISPER_BEAMS);
    assert_eq!(c.vad_enabled, DEFAULT_WHISPER_VAD_ENABLED);
    assert_eq!(c.vad_silence_secs, DEFAULT_WHISPER_VAD_SILENCE_SECS);
    assert!(c.model_cache_dir.is_none());
}

#[test]
fn remote_defaults_match_constants() {
    let c = RemoteConfig::default();
    assert!(!c.enabled);
    assert_eq!(c.bind_address, DEFAULT_REMOTE_BIND_ADDRESS);
    assert_eq!(c.port, DEFAULT_REMOTE_PORT);
}

#[test]
fn presence_defaults_match_constants() {
    let c = PresenceConfig::default();
    assert_eq!(c.hotkey, DEFAULT_PRESENCE_HOTKEY);
}

#[test]
fn daemon_defaults_match_constants() {
    let c = DaemonConfig::default();
    assert_eq!(c.shutdown_grace_secs, DEFAULT_DAEMON_SHUTDOWN_GRACE_SECS);
}

#[test]
fn sleep_defaults_match_constants() {
    let c = SleepConfig::default();
    assert_eq!(c.idle_to_drowsy_mins, DEFAULT_IDLE_TO_DROWSY_MINS);
    assert_eq!(c.idle_to_sleep_mins, DEFAULT_IDLE_TO_SLEEP_MINS);
    assert!(!c.suspend);
    assert_eq!(c.gpu_monitor_enabled, DEFAULT_GPU_MONITOR_ENABLED);
    assert_eq!(c.gpu_poll_secs, DEFAULT_GPU_POLL_SECS);
    assert_eq!(c.gpu_vram_threshold_mb, DEFAULT_GPU_VRAM_THRESHOLD_MB);
    assert!(!c.gpu_auto_wake);
}

#[test]
fn tools_output_defaults_match_constants() {
    let c = ToolsOutputConfig::default();
    assert_eq!(c.max_lines, DEFAULT_TOOLS_MAX_LINES);
    assert_eq!(c.max_kb, DEFAULT_TOOLS_MAX_KB);
    assert_eq!(c.overflow_dir, DEFAULT_TOOLS_OVERFLOW_DIR);
}

#[test]
fn tools_bash_defaults_match_constants() {
    let c = ToolsBashConfig::default();
    assert_eq!(c.timeout_secs, DEFAULT_BASH_TIMEOUT_SECS);
    assert_eq!(c.sandbox, BashSandboxMode::Auto);
    assert!(!c.denylist.is_empty());
    assert!(!c.destructive_patterns.is_empty());
}

#[test]
fn agent_defaults_match_constants() {
    let c = AgentConfig::default();
    assert_eq!(c.max_iterations, DEFAULT_AGENT_MAX_ITERATIONS);
}

#[test]
fn minimal_fixture_parses_and_validates() {
    let cfg = fixtures::minimal();
    cfg.validate().expect("minimal fixture must validate");
}

//! Per-test `fake_llama_server` fixture shared by the integration tests.

use std::path::PathBuf;

use tempfile::TempDir;

const FAKE_BIN: &str = env!("CARGO_BIN_EXE_fake_llama_server");

/// A private symlink to `fake_llama_server` plus the `mode` file it reads
/// at startup. Each test owns its own, so modes never leak between tests
/// running in parallel. Keep it alive for as long as anything may spawn it.
pub struct FakeLlama {
    dir: TempDir,
}

impl FakeLlama {
    /// Create the symlink with `mode` selected for the first spawn.
    pub fn new(mode: &str) -> Self {
        let dir = tempfile::tempdir().expect("fake llama tempdir");
        std::os::unix::fs::symlink(FAKE_BIN, dir.path().join("llama-server"))
            .expect("symlink fake_llama_server");
        let fake = Self { dir };
        fake.set_mode(mode);
        fake
    }

    /// Path to use as `LlamaServerConfig::binary_path`.
    pub fn binary_path(&self) -> PathBuf {
        self.dir.path().join("llama-server")
    }

    /// Select the mode for subsequent spawns; running children are unaffected.
    pub fn set_mode(&self, mode: &str) {
        std::fs::write(self.dir.path().join("mode"), mode).expect("write fake llama mode");
    }
}

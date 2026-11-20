//! How llama-server is launched: router mode, so no model is named on the
//! command line, weights load on demand through `/models/load`, and each
//! model's context comes from the presets file.

use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use assistd_config::ModelConfig;
use assistd_utils::child_server::{ApiKey, ChildServerSpec, remove_llama_env};
use tokio::process::Command;

use super::presets::ModelPresets;
use super::slots::SlotDir;

/// Launch parameters for the supervised router-mode llama-server.
#[derive(Debug)]
pub struct LlamaServerSpec {
    model: ModelConfig,
    api_key: ApiKey,
    slot_dir: SlotDir,
    presets: ModelPresets,
}

impl LlamaServerSpec {
    /// A spec whose server accepts only requests bearing `api_key`, saves slot
    /// state only into `slot_dir`, and reads its models' presets from `presets`.
    pub fn new(
        model: ModelConfig,
        api_key: ApiKey,
        slot_dir: SlotDir,
        presets: ModelPresets,
    ) -> Self {
        Self {
            model,
            api_key,
            slot_dir,
            presets,
        }
    }
}

impl ChildServerSpec for LlamaServerSpec {
    fn name(&self) -> &'static str {
        "llama-server"
    }

    fn prepare_spawn(&self) -> io::Result<()> {
        self.slot_dir.reset()?;
        self.presets.write(self.model.context_length)
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(self.model.server_binary.as_str());
        cmd.args(self.model.custom_args.as_slice());
        push_managed_args(&mut cmd, self);
        remove_llama_env(&mut cmd);
        cmd
    }

    fn listen_addr(&self) -> SocketAddr {
        self.model.listen_addr()
    }

    fn ready_timeout(&self) -> Duration {
        Duration::from_secs(self.model.ready_timeout_secs.get())
    }
}

/// Appended after the custom args so these win wherever the server takes
/// the last occurrence of a repeated flag. The context length is left to the
/// presets file, which a flag given here would override for every model.
fn push_managed_args(cmd: &mut Command, spec: &LlamaServerSpec) {
    let model = &spec.model;
    cmd.arg("--jinja")
        .arg("--no-slots")
        .arg("--slot-save-path")
        .arg(spec.slot_dir.path())
        .arg("--api-key-file")
        .arg(spec.api_key.file_path())
        .arg("-ngl")
        .arg(model.gpu_layers.to_string())
        .arg("--host")
        .arg(model.listen_addr().ip().to_string())
        .arg("--port")
        .arg(model.port.to_string())
        .arg("--models-preset")
        .arg(spec.presets.path());
}

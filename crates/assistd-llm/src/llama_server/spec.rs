//! How llama-server is launched: router mode, so no model is named on the
//! command line and weights load on demand through `/models/load`.

use std::net::SocketAddr;
use std::time::Duration;

use assistd_config::ModelConfig;
use assistd_utils::child_server::{ApiKey, ChildServerSpec, remove_llama_env};
use tokio::process::Command;

/// Launch parameters for the supervised router-mode llama-server.
#[derive(Debug)]
pub struct LlamaServerSpec {
    model: ModelConfig,
    api_key: ApiKey,
}

impl LlamaServerSpec {
    /// A spec whose server accepts only requests bearing `api_key`.
    pub fn new(model: ModelConfig, api_key: ApiKey) -> Self {
        Self { model, api_key }
    }
}

impl ChildServerSpec for LlamaServerSpec {
    fn name(&self) -> &'static str {
        "llama-server"
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.model.server_binary);
        cmd.args(self.model.custom_args.as_slice());
        push_managed_args(&mut cmd, &self.model, &self.api_key);
        remove_llama_env(&mut cmd);
        cmd
    }

    fn listen_addr(&self) -> SocketAddr {
        SocketAddr::new(self.model.host, self.model.port.get())
    }

    fn ready_timeout(&self) -> Duration {
        Duration::from_secs(self.model.ready_timeout_secs.get())
    }
}

/// Appended after the custom args so these win wherever the server takes
/// the last occurrence of a repeated flag.
fn push_managed_args(cmd: &mut Command, model: &ModelConfig, api_key: &ApiKey) {
    cmd.arg("--jinja")
        .arg("--no-slots")
        .arg("--api-key-file")
        .arg(api_key.file_path())
        .arg("-ngl")
        .arg(model.gpu_layers.to_string())
        .arg("--host")
        .arg(model.host.to_string())
        .arg("--port")
        .arg(model.port.to_string())
        .arg("-c")
        .arg(model.context_length.to_string());
}

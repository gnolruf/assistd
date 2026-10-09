//! How the embedding llama-server is launched.

use std::net::SocketAddr;
use std::time::Duration;

use assistd_config::EmbeddingConfig;
use assistd_utils::child_server::{ApiKey, ChildServerSpec, remove_llama_env};
use tokio::process::Command;

/// Launch parameters for the supervised embedding llama-server. With
/// `gpu_layers == 0` the child sees no CUDA devices.
#[derive(Debug)]
pub struct EmbedServerSpec {
    cfg: EmbeddingConfig,
    api_key: ApiKey,
}

impl EmbedServerSpec {
    /// A spec whose server accepts only requests bearing `api_key`.
    pub fn new(cfg: EmbeddingConfig, api_key: ApiKey) -> Self {
        Self { cfg, api_key }
    }
}

impl ChildServerSpec for EmbedServerSpec {
    fn name(&self) -> &'static str {
        "embed-server"
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.cfg.server_binary);
        cmd.args(self.cfg.custom_args.as_slice());
        push_managed_args(&mut cmd, &self.cfg, &self.api_key);
        remove_llama_env(&mut cmd);
        if self.cfg.gpu_layers == 0 {
            cmd.env("CUDA_VISIBLE_DEVICES", "");
        }
        cmd
    }

    fn listen_addr(&self) -> SocketAddr {
        SocketAddr::new(self.cfg.host, self.cfg.port.get())
    }

    fn ready_timeout(&self) -> Duration {
        Duration::from_secs(self.cfg.ready_timeout_secs.get())
    }
}

fn push_managed_args(cmd: &mut Command, cfg: &EmbeddingConfig, api_key: &ApiKey) {
    cmd.arg("--embedding")
        .arg("--no-slots")
        .arg("--api-key-file")
        .arg(api_key.file_path())
        .arg("--pooling")
        .arg("mean")
        .arg("--hf-repo")
        .arg(&cfg.model)
        .arg("-ngl")
        .arg(cfg.gpu_layers.to_string())
        .arg("--host")
        .arg(cfg.host.to_string())
        .arg("--port")
        .arg(cfg.port.to_string());
}

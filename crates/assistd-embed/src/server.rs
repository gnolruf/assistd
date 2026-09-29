//! How the embedding llama-server is launched.

use std::net::SocketAddr;
use std::time::Duration;

use assistd_config::EmbeddingConfig;
use assistd_utils::child_server::ChildServerSpec;
use tokio::process::Command;

/// Launch parameters for the supervised embedding llama-server. With
/// `gpu_layers == 0` the child sees no CUDA devices.
#[derive(Debug)]
pub struct EmbedServerSpec {
    cfg: EmbeddingConfig,
    ready_timeout: Duration,
}

impl EmbedServerSpec {
    pub fn new(cfg: EmbeddingConfig, ready_timeout: Duration) -> Self {
        Self { cfg, ready_timeout }
    }
}

impl ChildServerSpec for EmbedServerSpec {
    fn name(&self) -> &'static str {
        "embed-server"
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new("llama-server");
        cmd.arg("--embedding")
            .arg("--pooling")
            .arg("mean")
            .arg("--hf-repo")
            .arg(&self.cfg.model)
            .arg("-ngl")
            .arg(self.cfg.gpu_layers.to_string())
            .arg("--host")
            .arg(self.cfg.host.to_string())
            .arg("--port")
            .arg(self.cfg.port.to_string());
        if self.cfg.gpu_layers == 0 {
            cmd.env("CUDA_VISIBLE_DEVICES", "");
        }
        cmd
    }

    fn listen_addr(&self) -> SocketAddr {
        SocketAddr::new(self.cfg.host, self.cfg.port.get())
    }

    fn ready_timeout(&self) -> Duration {
        self.ready_timeout
    }
}

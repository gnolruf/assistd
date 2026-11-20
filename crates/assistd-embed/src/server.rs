//! How the embedding llama-server is launched.

use std::net::SocketAddr;
use std::time::Duration;

use assistd_config::{EmbeddingConfig, NonBlankString};
use assistd_utils::child_server::{ApiKey, ChildServerSpec, remove_llama_env};
use tokio::process::Command;

/// Tokens one input may hold unless `embedding.custom_args` sets another count.
/// llama-server's own 512 is fewer than some 512-character chunks come to.
const INPUT_TOKENS: &str = "2048";

/// Launch parameters for the supervised embedding llama-server, which embeds text
/// only and loads no image projector. With `gpu_layers == 0` it is given no
/// device, so it computes on the CPU and holds no VRAM.
#[derive(Debug)]
pub struct EmbedServerSpec {
    cfg: EmbeddingConfig,
    server_binary: NonBlankString,
    api_key: ApiKey,
}

impl EmbedServerSpec {
    /// A spec that runs `server_binary` as a server accepting only requests
    /// bearing `api_key`.
    pub fn new(cfg: EmbeddingConfig, server_binary: NonBlankString, api_key: ApiKey) -> Self {
        Self {
            cfg,
            server_binary,
            api_key,
        }
    }
}

impl ChildServerSpec for EmbedServerSpec {
    fn name(&self) -> &'static str {
        "embed-server"
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(self.server_binary.as_str());
        cmd.args(["--batch-size", INPUT_TOKENS, "--ubatch-size", INPUT_TOKENS]);
        cmd.args(self.cfg.custom_args.as_slice());
        push_managed_args(&mut cmd, &self.cfg, &self.api_key);
        remove_llama_env(&mut cmd);
        cmd
    }

    fn listen_addr(&self) -> SocketAddr {
        self.cfg.listen_addr()
    }

    fn ready_timeout(&self) -> Duration {
        Duration::from_secs(self.cfg.ready_timeout_secs.get())
    }
}

fn push_managed_args(cmd: &mut Command, cfg: &EmbeddingConfig, api_key: &ApiKey) {
    cmd.arg("--embedding")
        .arg("--no-slots")
        .arg("--no-mmproj")
        .arg("--api-key-file")
        .arg(api_key.file_path())
        .arg("--hf-repo")
        .arg(cfg.model.to_string())
        .arg("-ngl")
        .arg(cfg.gpu_layers.to_string())
        .arg("--host")
        .arg(cfg.listen_addr().ip().to_string())
        .arg("--port")
        .arg(cfg.port.to_string());
    if cfg.gpu_layers == 0 {
        cmd.args(["--device", "none"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch_args(gpu_layers: u32) -> Vec<String> {
        let cfg = EmbeddingConfig {
            gpu_layers,
            ..EmbeddingConfig::default()
        };
        let server_binary = "llama-server".parse().expect("a binary name");
        let api_key = ApiKey::generate().expect("a key file");
        EmbedServerSpec::new(cfg, server_binary, api_key)
            .command()
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn only_a_server_with_no_gpu_layers_is_given_no_device() {
        assert!(
            launch_args(0)
                .windows(2)
                .any(|pair| pair == ["--device", "none"])
        );
        assert!(!launch_args(1).iter().any(|arg| arg == "--device"));
    }
}

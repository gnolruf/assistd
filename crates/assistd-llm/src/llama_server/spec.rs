//! How llama-server is launched: router mode, so no model is named on the
//! command line and weights load on demand through `/models/load`.

use std::net::SocketAddr;
use std::time::Duration;

use assistd_config::{LlamaServerConfig, ModelConfig};
use assistd_utils::child_server::ChildServerSpec;
use tokio::process::Command;

/// Launch parameters for the supervised router-mode llama-server.
#[derive(Debug)]
pub struct LlamaServerSpec {
    cfg: LlamaServerConfig,
    model: ModelConfig,
}

impl LlamaServerSpec {
    pub fn new(cfg: LlamaServerConfig, model: ModelConfig) -> Self {
        Self { cfg, model }
    }
}

impl ChildServerSpec for LlamaServerSpec {
    fn name(&self) -> &'static str {
        "llama-server"
    }

    fn command(&self) -> Command {
        let mut cmd = router_command(&self.cfg, &self.model);
        push_tuning_args(&mut cmd, &self.cfg);
        cmd
    }

    fn listen_addr(&self) -> SocketAddr {
        SocketAddr::new(self.cfg.host, self.cfg.port.get())
    }

    fn ready_timeout(&self) -> Duration {
        Duration::from_secs(self.cfg.ready_timeout_secs.get())
    }
}

fn router_command(cfg: &LlamaServerConfig, model: &ModelConfig) -> Command {
    let mut cmd = Command::new(&cfg.binary_path);
    cmd.arg("--jinja")
        .arg("-ngl")
        .arg(cfg.gpu_layers.to_string())
        .arg("--host")
        .arg(cfg.host.to_string())
        .arg("--port")
        .arg(cfg.port.to_string())
        .arg("-c")
        .arg(model.context_length.to_string());
    cmd
}

fn push_tuning_args(cmd: &mut Command, cfg: &LlamaServerConfig) {
    if let Some(alias) = &cfg.alias {
        cmd.arg("--alias").arg(alias);
    }
    if let Some(override_tensor) = &cfg.override_tensor {
        cmd.arg("-ot").arg(override_tensor);
    }
    if let Some(flash) = cfg.flash_attn {
        cmd.arg("--flash-attn")
            .arg(if flash { "on" } else { "off" });
    }
    if let Some(cache_type_k) = &cfg.cache_type_k {
        cmd.arg("--cache-type-k").arg(cache_type_k);
    }
    if let Some(cache_type_v) = &cfg.cache_type_v {
        cmd.arg("--cache-type-v").arg(cache_type_v);
    }
    if let Some(threads) = cfg.threads {
        cmd.arg("--threads").arg(threads.to_string());
    }
    if let Some(batch_size) = cfg.batch_size {
        cmd.arg("--batch-size").arg(batch_size.to_string());
    }
    if let Some(ubatch_size) = cfg.ubatch_size {
        cmd.arg("--ubatch-size").arg(ubatch_size.to_string());
    }
    if let Some(n_cpu_moe) = cfg.n_cpu_moe {
        cmd.arg("--n-cpu-moe").arg(n_cpu_moe.to_string());
    }
    if let Some(cache_ram_mib) = cfg.cache_ram_mib {
        cmd.arg("--cache-ram").arg(cache_ram_mib.to_string());
    }
    if cfg.mlock == Some(true) {
        cmd.arg("--mlock");
    }
    if let Some(offload) = cfg.mmproj_offload {
        cmd.arg(if offload {
            "--mmproj-offload"
        } else {
            "--no-mmproj-offload"
        });
    }
}

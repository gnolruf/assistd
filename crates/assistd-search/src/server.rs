//! How the SearXNG instance is launched.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use assistd_config::SearchConfig;
use assistd_utils::child_server::{ChildServerSpec, random_secret};
use tokio::process::Command;

use crate::error::SearchServerError;
use crate::settings::SettingsFile;

const SEARXNG_ENV_PREFIX: &[u8] = b"SEARXNG_";

/// Launch parameters for the supervised SearXNG instance. Its settings come
/// only from the file and variables the daemon sets, never from the
/// daemon's own `SEARXNG_*` environment. `Debug` omits the secret.
pub struct SearxngServerSpec {
    cfg: SearchConfig,
    settings: SettingsFile,
    secret: String,
}

impl SearxngServerSpec {
    /// Draw a fresh secret and render the settings file the instance loads.
    pub fn prepare(cfg: SearchConfig) -> Result<Self, SearchServerError> {
        let secret = random_secret().map_err(SearchServerError::Secret)?;
        let settings = SettingsFile::render(cfg.engines.as_slice())?;
        Ok(Self {
            cfg,
            settings,
            secret,
        })
    }

    fn set_managed_env(&self, cmd: &mut Command, inherited: impl IntoIterator<Item = OsString>) {
        for name in inherited
            .into_iter()
            .filter(|name| is_searxng_setting(name))
        {
            cmd.env_remove(name);
        }
        cmd.env("SEARXNG_SETTINGS_PATH", self.settings.path())
            .env("SEARXNG_BIND_ADDRESS", self.cfg.host.to_string())
            .env("SEARXNG_PORT", self.cfg.port.to_string())
            .env("SEARXNG_SECRET", &self.secret);
    }
}

impl fmt::Debug for SearxngServerSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearxngServerSpec")
            .field("cfg", &self.cfg)
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl ChildServerSpec for SearxngServerSpec {
    fn name(&self) -> &'static str {
        "search-server"
    }

    fn prepare_spawn(&self) -> io::Result<()> {
        self.settings.write()
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(self.cfg.command.program());
        cmd.args(self.cfg.command.args());
        self.set_managed_env(&mut cmd, std::env::vars_os().map(|(name, _)| name));
        cmd
    }

    fn listen_addr(&self) -> SocketAddr {
        SocketAddr::new(self.cfg.host.get(), self.cfg.port.get())
    }

    fn health_path(&self) -> &'static str {
        "/healthz"
    }

    fn ready_timeout(&self) -> Duration {
        Duration::from_secs(self.cfg.ready_timeout_secs.get())
    }
}

fn is_searxng_setting(name: &OsStr) -> bool {
    name.as_encoded_bytes().starts_with(SEARXNG_ENV_PREFIX)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn spec() -> SearxngServerSpec {
        let cfg = SearchConfig::default();
        let settings = SettingsFile::render(cfg.engines.as_slice()).unwrap();
        SearxngServerSpec {
            cfg,
            settings,
            secret: "s3cret".into(),
        }
    }

    #[test]
    fn only_the_daemons_searxng_settings_reach_the_child() {
        let mut cmd = Command::new("searxng-run");
        let inherited = ["SEARXNG_LIMITER", "PATH", "SEARXNG_PORT", "SEARXNG_SECRET"];
        let spec = spec();
        spec.set_managed_env(&mut cmd, inherited.map(OsString::from));
        let env: BTreeMap<_, _> = cmd
            .as_std()
            .get_envs()
            .map(|(name, value)| (name.to_owned(), value.map(OsStr::to_owned)))
            .collect();
        let set = |value: &str| Some(OsString::from(value));
        assert_eq!(
            env,
            BTreeMap::from([
                ("SEARXNG_BIND_ADDRESS".into(), set("127.0.0.1")),
                ("SEARXNG_LIMITER".into(), None),
                ("SEARXNG_PORT".into(), set("8387")),
                ("SEARXNG_SECRET".into(), set("s3cret")),
                (
                    "SEARXNG_SETTINGS_PATH".into(),
                    Some(spec.settings.path().as_os_str().to_owned())
                ),
            ])
        );
    }
}

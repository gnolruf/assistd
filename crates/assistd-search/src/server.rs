//! How the SearXNG instance is launched.

use std::ffi::OsString;
use std::fmt;
use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use assistd_config::SearchConfig;
use assistd_utils::child_server::{ChildServerSpec, random_secret};
use assistd_utils::env::inherited_env;
use tokio::process::Command;

use crate::error::SearchServerError;
use crate::settings::SettingsFile;

/// What SearXNG inherits beyond the locale and terminal variables: where its
/// interpreter, caches and temporary files are found.
const HOST_VARS: &[&str] = &["HOME", "PATH", "SHELL", "TMPDIR"];

/// Launch parameters for the supervised SearXNG instance. Its settings come
/// only from the file and variables the daemon sets, and it inherits none of
/// the daemon's credentials, sockets or proxies. `Debug` omits the secret.
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

    fn set_env(
        &self,
        cmd: &mut Command,
        daemon_env: impl IntoIterator<Item = (OsString, OsString)>,
    ) {
        cmd.env_clear()
            .envs(inherited_env(daemon_env, HOST_VARS))
            .env("SEARXNG_SETTINGS_PATH", self.settings.path())
            .env(
                "SEARXNG_BIND_ADDRESS",
                self.cfg.listen_addr().ip().to_string(),
            )
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
        self.set_env(&mut cmd, std::env::vars_os());
        cmd
    }

    fn listen_addr(&self) -> SocketAddr {
        self.cfg.listen_addr()
    }

    fn health_path(&self) -> &'static str {
        "/healthz"
    }

    fn ready_timeout(&self) -> Duration {
        Duration::from_secs(self.cfg.ready_timeout_secs.get())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ffi::OsStr;

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
    fn the_child_gets_only_host_variables_and_the_managed_settings() {
        let mut cmd = Command::new("searxng-run");
        let daemon_env = [
            ("SEARXNG_LIMITER", "true"),
            ("SEARXNG_PORT", "1"),
            ("PATH", "/usr/bin"),
            ("LANG", "C.UTF-8"),
            ("GITHUB_TOKEN", "secret"),
            ("SSH_AUTH_SOCK", "/tmp/ssh-x/agent.1"),
            ("HTTPS_PROXY", "http://proxy:3128"),
        ]
        .map(|(name, value)| (OsString::from(name), OsString::from(value)));
        let spec = spec();
        spec.set_env(&mut cmd, daemon_env);
        let env: BTreeMap<_, _> = cmd
            .as_std()
            .get_envs()
            .map(|(name, value)| (name.to_owned(), value.map(OsStr::to_owned)))
            .collect();
        let set = |value: &str| Some(OsString::from(value));
        assert_eq!(
            env,
            BTreeMap::from([
                ("LANG".into(), set("C.UTF-8")),
                ("PATH".into(), set("/usr/bin")),
                ("SEARXNG_BIND_ADDRESS".into(), set("127.0.0.1")),
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

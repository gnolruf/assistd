//! Which of the daemon's environment variables a child process may inherit.

use std::ffi::OsString;

/// Variables any child may inherit, besides `LC_*`: the locale, terminal,
/// time zone and user name.
const NEUTRAL_VARS: &[&str] = &["LANG", "LANGUAGE", "LOGNAME", "TERM", "TZ", "USER"];

/// The subset of `vars` a child process may inherit: the locale, terminal,
/// time-zone and user-name variables, `LC_*`, and those named in `also`.
/// The rest, such as agent sockets and tokens, are dropped.
pub fn inherited_env(
    vars: impl IntoIterator<Item = (OsString, OsString)>,
    also: &[&str],
) -> Vec<(OsString, OsString)> {
    vars.into_iter()
        .filter(|(name, _)| {
            name.to_str().is_some_and(|name| {
                NEUTRAL_VARS.contains(&name) || also.contains(&name) || name.starts_with("LC_")
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_neutral_and_named_variables_are_inherited() {
        let vars = [
            ("LANG", "C.UTF-8"),
            ("LC_TIME", "en_GB.UTF-8"),
            ("PATH", "/usr/bin"),
            ("HOME", "/home/me"),
            ("SSH_AUTH_SOCK", "/tmp/ssh-x/agent.1"),
            ("GITHUB_TOKEN", "secret"),
            ("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus"),
            ("SWAYSOCK", "/run/user/1000/sway-ipc.sock"),
            ("WAYLAND_DISPLAY", "wayland-1"),
            ("DISPLAY", ":0"),
            ("XAUTHORITY", "/run/user/1000/xauth"),
        ]
        .map(|(name, value)| (OsString::from(name), OsString::from(value)));
        let names: Vec<String> = inherited_env(vars, &["PATH"])
            .into_iter()
            .map(|(name, _)| name.into_string().expect("utf-8"))
            .collect();
        assert_eq!(names, ["LANG", "LC_TIME", "PATH"]);
    }
}

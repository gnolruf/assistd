//! `assistd tray`: StatusNotifierItem client mirroring daemon state.

use std::path::{Path, PathBuf};

use anyhow::Result;
use assistd_config::{Config, ConfigError};
use assistd_ipc::IpcClient;
use clap::Args;
use ksni::TrayMethods;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use menu::TrayItem;

mod menu;
mod notifications;
mod state;
mod subscribe;

#[derive(Args)]
pub(crate) struct TrayArgs {
    /// Path to config file [default: `~/.config/assistd/config.toml`]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

pub(crate) async fn run(args: TrayArgs) -> Result<()> {
    init_tracing();

    let config_path = Config::path_or_default(args.config)?;
    let (config, config_error) = load_config(&config_path);
    match &config_error {
        None => tracing::info!(target: "tray", "loaded config from {}", config_path.display()),
        Some(e) => tracing::error!(
            target: "tray",
            "config has errors; tray will flag them until fixed: {e}"
        ),
    }
    let config_error = config_error.map(|e| e.to_string());

    if std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_none() {
        anyhow::bail!(
            "DBUS_SESSION_BUS_ADDRESS is not set; the tray needs a session DBus to register with. \
             Are you running from a graphical session?"
        );
    }

    let ipc = IpcClient::new();

    let notifications = config
        .as_ref()
        .and_then(|cfg| notifications::spawn_notifications(cfg, ipc.clone()));
    let notification_sink = notifications.as_ref().map(|n| n.sink.clone());

    let (actions_tx, actions_rx) = mpsc::unbounded_channel();
    let activate_cb = build_activate_callback(notification_sink.as_ref());
    let icons = config
        .as_ref()
        .map(|cfg| cfg.notifications.icons.clone())
        .unwrap_or_default();
    let item = TrayItem::new(actions_tx, activate_cb, icons, config_error);

    let handle = item
        .assume_sni_available(true)
        .spawn()
        .await
        .map_err(|e| anyhow::anyhow!("failed to register StatusNotifierItem on DBus: {e}"))?;
    tracing::info!(target: "tray", "tray icon registered on DBus");

    let subscribe_handle = handle.clone();
    let subscribe_ipc = ipc.clone();
    let subscribe_task = tokio::spawn(async move {
        subscribe::run(subscribe_handle, subscribe_ipc, notification_sink).await;
    });

    let action_task = tokio::spawn(menu::run_actions(actions_rx, ipc));

    wait_for_shutdown(action_task).await;

    handle.shutdown().await;
    subscribe_task.abort();
    let _ = subscribe_task.await;
    if let Some(n) = notifications {
        n.shutdown().await;
    }
    Ok(())
}

/// The config at `path` unless it cannot be parsed, beside what is wrong with it.
fn load_config(path: &Path) -> (Option<Config>, Option<ConfigError>) {
    match Config::load_with_issues(path) {
        Ok((config, issues)) => (Some(config), ConfigError::reject(issues).err()),
        Err(unparsed) => (None, Some(unparsed)),
    }
}

fn build_activate_callback(
    sink: Option<&notifications::NotificationSink>,
) -> Option<menu::ActivateCallback> {
    let sink = sink?.clone();
    Some(Box::new(move || sink.tray_activated()))
}

async fn wait_for_shutdown(action_task: JoinHandle<Result<()>>) {
    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(target: "tray", "could not install SIGTERM handler: {e}");
            let _ = action_task.await;
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!(target: "tray", "ctrl-c received, shutting down");
        }
        _ = sigterm.recv() => {
            tracing::info!(target: "tray", "SIGTERM received, shutting down");
        }
        res = action_task => {
            match res {
                Ok(Ok(())) => tracing::info!(target: "tray", "quit menu item activated"),
                Ok(Err(e)) => tracing::warn!(target: "tray", "menu action handler failed: {e:#}"),
                Err(e) => tracing::warn!(target: "tray", "menu action task panicked: {e}"),
            }
        }
    }
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(crate::env_filter_or("info"))
        .try_init();
}

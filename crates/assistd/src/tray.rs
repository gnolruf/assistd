//! `assistd tray`: StatusNotifierItem client mirroring daemon state.

use std::path::{Path, PathBuf};

use anyhow::Result;
use assistd_config::{Config, ConfigError};
use assistd_ipc::IpcClient;
use assistd_utils::tracing_init::env_filter_or;
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

    let config_path = match args.config.clone() {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = load_config(&config_path);
    match &config {
        Ok(_) => tracing::info!(target: "tray", "loaded config from {}", config_path.display()),
        Err(e) => tracing::error!(
            target: "tray",
            "config unusable; tray will flag it until fixed: {e}"
        ),
    }
    let config_error = config.as_ref().err().map(ToString::to_string);

    if std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_none() {
        anyhow::bail!(
            "DBUS_SESSION_BUS_ADDRESS is not set; the tray needs a session DBus to register with. \
             Are you running from a graphical session?"
        );
    }

    let ipc = IpcClient::new();

    let notifications = config
        .as_ref()
        .ok()
        .and_then(|cfg| notifications::spawn_notifications(cfg, ipc.clone()));
    let notification_sink = notifications.as_ref().map(|n| n.sink.clone());

    let (actions_tx, actions_rx) = mpsc::unbounded_channel();
    let activate_cb = build_activate_callback(notification_sink.as_ref());
    let item = TrayItem::new(actions_tx, activate_cb, config_error);

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

fn load_config(path: &Path) -> Result<Config, ConfigError> {
    let config = Config::load_from_file(path)?;
    config.validate()?;
    Ok(config)
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
        .with_env_filter(env_filter_or("info"))
        .try_init();
}

//! `assistd tray`: StatusNotifierItem client mirroring daemon state.

use std::path::PathBuf;

use anyhow::{Context, Result};
use assistd_config::Config;
use assistd_ipc::IpcClient;
use clap::Args;
use ksni::TrayMethods;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::mpsc;

mod menu;
#[cfg(feature = "tray-popup")]
pub mod popup;
mod state;
mod subscribe;

use menu::TrayItem;

#[derive(Args)]
pub struct TrayArgs {
    /// Path to config file [default: `~/.config/assistd/config.toml`]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

pub async fn run(args: TrayArgs) -> Result<()> {
    init_tracing();

    let config_path = match args.config.clone() {
        Some(p) => p,
        None => Config::default_path()?,
    };
    let config = Config::load_from_file(&config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    config.validate()?;
    tracing::info!(target: "tray", "loaded config from {}", config_path.display());

    if std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_none() {
        anyhow::bail!(
            "DBUS_SESSION_BUS_ADDRESS is not set; the tray needs a session DBus to register with. \
             Are you running from a graphical session?"
        );
    }

    let ipc = IpcClient::new();

    #[cfg(feature = "tray-popup")]
    let popup_handle = popup::spawn(&config, ipc.clone()).await?;
    #[cfg(feature = "tray-popup")]
    let popup_sink = popup_handle.as_ref().map(|h| h.sink.clone());
    #[cfg(not(feature = "tray-popup"))]
    let popup_sink: Option<()> = None;

    let (actions_tx, actions_rx) = mpsc::unbounded_channel();
    let activate_cb = build_activate_callback(&popup_sink);
    let item = TrayItem::new(config.tray.clone(), actions_tx, activate_cb);

    let handle = item
        .assume_sni_available(true)
        .spawn()
        .await
        .map_err(|e| anyhow::anyhow!("failed to register StatusNotifierItem on DBus: {e}"))?;
    tracing::info!(target: "tray", "tray icon registered on DBus");

    let subscribe_handle = handle.clone();
    let subscribe_ipc = ipc.clone();
    let subscribe_task =
        tokio::spawn(
            async move { subscribe::run(subscribe_handle, subscribe_ipc, popup_sink).await },
        );

    let action_task = tokio::spawn(menu::run_actions(actions_rx, ipc));

    wait_for_shutdown(action_task).await;

    handle.shutdown().await;
    subscribe_task.abort();
    let _ = subscribe_task.await;
    #[cfg(feature = "tray-popup")]
    if let Some(p) = popup_handle {
        p.shutdown().await;
    }
    Ok(())
}

#[cfg(feature = "tray-popup")]
fn build_activate_callback(sink: &Option<popup::PopupSink>) -> Option<menu::ActivateCallback> {
    let sink = sink.as_ref()?.clone();
    Some(Box::new(move || {
        let tx = sink.show_sender();
        let _ = tx.send(popup::DriverInput::Show);
    }))
}

#[cfg(not(feature = "tray-popup"))]
fn build_activate_callback(_sink: &Option<()>) -> Option<menu::ActivateCallback> {
    None
}

async fn wait_for_shutdown(action_task: tokio::task::JoinHandle<Result<()>>) {
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
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

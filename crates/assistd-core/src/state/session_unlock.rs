//! Firing `SessionUnlocked` triggers when logind's `LockedHint` on the user's
//! graphical session turns false.

use std::sync::Arc;

use futures_util::StreamExt;
use thiserror::Error;
use tracing::{info, warn};
use zbus::fdo::{PropertiesChanged, PropertiesChangedStream, PropertiesProxy};
use zbus::zvariant::OwnedObjectPath;
use zbus::{Connection, Proxy};

use super::event_triggers::Occurrence;
use super::task_service::TaskService;

const LOGIN1: &str = "org.freedesktop.login1";
const OWN_USER_PATH: &str = "/org/freedesktop/login1/user/self";
const USER_INTERFACE: &str = "org.freedesktop.login1.User";
const SESSION_INTERFACE: &str = "org.freedesktop.login1.Session";
const LOCKED_HINT: &str = "LockedHint";

/// Why session unlocks could not be watched.
#[derive(Debug, Error)]
enum UnlockWatchError {
    #[error(transparent)]
    Bus(#[from] zbus::Error),

    #[error("the user has no graphical session")]
    NoGraphicalSession,
}

impl TaskService {
    /// Fire every `SessionUnlocked` trigger each time the user's graphical session's
    /// `LockedHint` turns false, for as long as the future runs.
    pub(in crate::state) async fn watch_session_unlocks(self: Arc<Self>) {
        let mut changes = match session_property_changes().await {
            Ok(changes) => changes,
            Err(e) => {
                warn!(target: "assistd::tasks", error = %e, "watching session unlocks failed");
                return;
            }
        };
        while let Some(change) = changes.next().await {
            if is_unlock(&change) {
                info!(target: "assistd::tasks", "session unlocked");
                self.fire_event_triggers(Occurrence::SessionUnlocked).await;
            }
        }
    }
}

/// The property changes logind announces on the user's graphical session.
async fn session_property_changes() -> Result<PropertiesChangedStream, UnlockWatchError> {
    let connection = Connection::system().await?;
    let user = Proxy::new(&connection, LOGIN1, OWN_USER_PATH, USER_INTERFACE).await?;
    let (_, session): (String, OwnedObjectPath) = user.get_property("Display").await?;
    if session.as_str() == "/" {
        return Err(UnlockWatchError::NoGraphicalSession);
    }
    let properties = PropertiesProxy::builder(&connection)
        .destination(LOGIN1)?
        .path(session)?
        .build()
        .await?;
    Ok(properties.receive_properties_changed().await?)
}

/// Whether `change` sets the session's `LockedHint` to false; logind announces it only
/// when its value changes.
fn is_unlock(change: &PropertiesChanged) -> bool {
    change.args().is_ok_and(|args| {
        args.interface_name.as_str() == SESSION_INTERFACE
            && args
                .changed_properties
                .get(LOCKED_HINT)
                .and_then(|locked| locked.downcast_ref::<bool>().ok())
                == Some(false)
    })
}

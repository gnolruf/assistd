//! The kinds of sub-agent a relay step runs as: named roles, each fixed to
//! one reach, private files or the web.

use std::sync::Arc;

/// What a sub-agent may reach. A role has exactly one, so no sub-agent holds
/// private files and the network at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Reach {
    /// The files a step names as `inputs`, read-only; no network.
    Files,
    /// The web through the in-process `web` command alone; no files.
    Web,
}

/// A kind of sub-agent the main model can hand a step to.
#[derive(Debug)]
pub(super) struct Role {
    /// What the model passes as a step's `agent`, and clients show.
    pub(super) name: String,
    /// When to pick it, as the model is told.
    pub(super) summary: String,
    /// Its job, as its own system prompt states it.
    pub(super) brief: String,
    pub(super) reach: Reach,
}

/// The roles a relay offers, in the order the model is shown them; names
/// are unique.
#[derive(Debug)]
pub(super) struct Roles(Vec<Arc<Role>>);

impl Reach {
    /// What a sub-agent with this reach can touch, as the model is told.
    pub(super) fn capability(self) -> &'static str {
        match self {
            Reach::Files => {
                "sees the files and directories listed in `inputs`, read-only, and nothing \
                 else from this machine, and has no network"
            }
            Reach::Web => {
                "can search the web and fetch pages, and gets no files; `inputs` must be empty"
            }
        }
    }
}

impl Roles {
    /// The roles assistd ships, which configuration cannot change.
    pub(super) fn built_in() -> Self {
        Self(vec![
            Arc::new(Role {
                name: "file_review".into(),
                summary: "For documents, logs and code too large to read here.".into(),
                brief: "You review local files handed to you: documents, logs and code.".into(),
                reach: Reach::Files,
            }),
            Arc::new(Role {
                name: "web_research".into(),
                summary: "For research and reading web pages.".into(),
                brief: "You research on the web: search, read the pages that answer the task, \
                        and name the URLs your report relies on."
                    .into(),
                reach: Reach::Web,
            }),
        ])
    }

    pub(super) fn named(&self, name: &str) -> Option<&Arc<Role>> {
        self.0.iter().find(|role| role.name == name)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &Role> {
        self.0.iter().map(AsRef::as_ref)
    }

    /// The quoted names of the roles with `reach`, or of all roles, as a
    /// list for the model to read.
    pub(super) fn quoted_names(&self, reach: Option<Reach>) -> String {
        self.iter()
            .filter(|role| reach.is_none_or(|reach| role.reach == reach))
            .map(|role| format!("\"{}\"", role.name))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

//! Doubles and fixtures for this crate's tests and the tests of crates built on it.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::TurnContext;
use crate::policy::{
    Allowlist, Approval, BashPolicyCfg, ConfirmationGate, ConfirmationRequest, SharedDirs,
};

pub use crate::policy::inert_sandbox;

/// Gate that never approves.
#[derive(Debug, Default)]
pub struct DenyAllGate;

#[async_trait]
impl ConfirmationGate for DenyAllGate {
    async fn confirm(&self, _turn: &TurnContext, _req: ConfirmationRequest) -> Approval {
        Approval::Deny
    }
}

/// Gate that approves every request once.
#[derive(Debug, Default)]
pub struct AlwaysAllowGate;

#[async_trait]
impl ConfirmationGate for AlwaysAllowGate {
    async fn confirm(&self, _turn: &TurnContext, _req: ConfirmationRequest) -> Approval {
        Approval::Once
    }
}

/// A policy with a 30-second timeout and nothing approved, protected, shared or in the edit
/// scope.
pub fn bash_policy() -> BashPolicyCfg {
    let search_path = inert_sandbox(SharedDirs::default()).search_path();
    BashPolicyCfg {
        timeout: Duration::from_secs(30),
        allowlist: Arc::new(Allowlist::unsaved(search_path)),
        protected: Vec::new(),
        scratch: None,
        edit_scope: Vec::new(),
    }
}

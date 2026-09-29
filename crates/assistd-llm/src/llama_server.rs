//! Out-of-process `llama-server`: the launch spec run by the shared child-server
//! supervisor, plus the HTTP control plane and capability probes.

pub mod capabilities;
pub mod control;
pub mod error;
pub mod spec;

pub use capabilities::{VisionState, probe_capabilities_routed};
pub use control::LlamaServerControl;
pub use error::LlamaServerError;
pub use spec::LlamaServerSpec;

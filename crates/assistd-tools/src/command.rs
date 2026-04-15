//! Internal command abstraction. `Command`s are Rust handlers that the
//! chain executor dispatches to after the parser has converted a command
//! line into a [`crate::chain::Chain`] AST. They operate on raw bytes
//! (stdin → stdout) and surface a Unix-style `exit_code` so `&&`/`||`
//! composition works the way users expect.
//!
//! A `Command` is distinct from a [`crate::Tool`]: the LLM only ever sees
//! one `Tool` (`run`); the `CommandRegistry` is an internal lookup table
//! that `run` consults as it walks each chain stage.

use anyhow::Result;
use async_trait::async_trait;

/// Input to a single chain stage.
pub struct CommandInput {
    /// Positional arguments **after** argv[0]. The command's own name
    /// is not included here — the registry has already resolved it.
    pub args: Vec<String>,
    /// Bytes piped in from the previous chain stage (or empty for the
    /// first command in a pipeline).
    pub stdin: Vec<u8>,
}

/// Output of a single chain stage.
///
/// Returning `CommandOutput` inside `Result::Ok` (rather than surfacing
/// predictable failures as `Err`) is what lets `|| echo 'not found'`
/// catch a missing file: the `cat` handler reports `exit_code = 1` with
/// a friendly stderr, and the executor treats that as a triggerable
/// failure for `||`. `Err` is reserved for *catastrophic* failures
/// (spawn error, panic-in-trait, etc.) that should abort the whole
/// chain.
#[derive(Debug, Default, Clone)]
pub struct CommandOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: i32,
}

impl CommandOutput {
    pub fn ok(stdout: Vec<u8>) -> Self {
        Self {
            stdout,
            stderr: Vec::new(),
            exit_code: 0,
        }
    }

    pub fn failed(exit_code: i32, stderr: impl Into<Vec<u8>>) -> Self {
        Self {
            stdout: Vec::new(),
            stderr: stderr.into(),
            exit_code,
        }
    }
}

/// A single internal command (`cat`, `grep`, `bash`, …).
#[async_trait]
pub trait Command: Send + Sync + 'static {
    fn name(&self) -> &str;
    async fn run(&self, input: CommandInput) -> Result<CommandOutput>;
}

/// Lookup table of registered commands.
#[derive(Default)]
pub struct CommandRegistry {
    commands: Vec<Box<dyn Command>>,
}

impl CommandRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<C: Command>(&mut self, cmd: C) {
        self.commands.push(Box::new(cmd));
    }

    pub fn get(&self, name: &str) -> Option<&dyn Command> {
        self.commands
            .iter()
            .find(|c| c.name() == name)
            .map(|c| c.as_ref())
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Command names, sorted alphabetically. Used to format the
    /// "Available: …" line in the unknown-command error so there's a
    /// single source of truth.
    pub fn sorted_names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.commands.iter().map(|c| c.name()).collect();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Stub(&'static str);

    #[async_trait]
    impl Command for Stub {
        fn name(&self) -> &str {
            self.0
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput> {
            Ok(CommandOutput::ok(Vec::new()))
        }
    }

    #[test]
    fn sorted_names_is_alphabetical() {
        let mut reg = CommandRegistry::new();
        reg.register(Stub("grep"));
        reg.register(Stub("cat"));
        reg.register(Stub("ls"));
        assert_eq!(reg.sorted_names(), vec!["cat", "grep", "ls"]);
    }

    #[test]
    fn unknown_lookup_returns_none() {
        let reg = CommandRegistry::new();
        assert!(reg.get("nope").is_none());
    }

    #[tokio::test]
    async fn registered_command_is_runnable() {
        let mut reg = CommandRegistry::new();
        reg.register(Stub("ping"));
        let cmd = reg.get("ping").unwrap();
        let out = cmd
            .run(CommandInput {
                args: vec![],
                stdin: vec![],
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
    }
}

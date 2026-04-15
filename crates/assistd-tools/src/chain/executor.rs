//! Walks a [`super::Chain`] AST, dispatching each stage through a
//! [`crate::CommandRegistry`] and gluing the results together according to
//! Unix pipeline semantics.
//!
//! Pipelining is sequential for simplicity: the left stage runs to
//! completion, its stdout buffer becomes the right stage's stdin. That
//! trades throughput for determinism and lets unit tests assert byte
//! equality. [`PIPE_BUF_MAX`] caps the inter-stage buffer so a runaway
//! stage (e.g. `bash "cat /dev/urandom" | wc -l`) can't OOM the daemon.

use anyhow::Result;
use std::future::Future;
use std::pin::Pin;

use super::Chain;
use crate::command::{CommandInput, CommandOutput, CommandRegistry};

/// Maximum bytes we buffer between pipe stages. Prevents one command
/// from exhausting daemon memory. On overflow we return exit 141 (the
/// canonical "SIGPIPE" code) so `||` fallbacks still fire.
pub const PIPE_BUF_MAX: usize = 10 * 1024 * 1024;

/// Execute a parsed command chain.
///
/// The returned `CommandOutput`'s `stderr` is the concatenation of every
/// stage's stderr, with each command's output prefixed `"[name]\t"` so
/// the caller can tell which stage spoke. Short-circuited branches
/// (right side of `&&` on failure, right side of `||` on success) emit
/// nothing.
pub fn execute<'a>(
    chain: &'a Chain,
    registry: &'a CommandRegistry,
    stdin: Vec<u8>,
) -> Pin<Box<dyn Future<Output = Result<CommandOutput>> + Send + 'a>> {
    Box::pin(async move {
        match chain {
            Chain::Command(argv) => run_command(argv, registry, stdin).await,
            Chain::Pipe(l, r) => {
                let left = execute(l, registry, stdin).await?;
                if left.stdout.len() > PIPE_BUF_MAX {
                    return Ok(CommandOutput {
                        stdout: Vec::new(),
                        stderr: merge(
                            left.stderr,
                            format!("[pipe]\tstage output exceeded {PIPE_BUF_MAX} bytes\n")
                                .into_bytes(),
                        ),
                        exit_code: 141,
                    });
                }
                let right = execute(r, registry, left.stdout).await?;
                Ok(CommandOutput {
                    stdout: right.stdout,
                    stderr: merge(left.stderr, right.stderr),
                    exit_code: right.exit_code,
                })
            }
            Chain::And(l, r) => {
                let left = execute(l, registry, stdin.clone()).await?;
                if left.exit_code == 0 {
                    let right = execute(r, registry, stdin).await?;
                    Ok(CommandOutput {
                        stdout: merge(left.stdout, right.stdout),
                        stderr: merge(left.stderr, right.stderr),
                        exit_code: right.exit_code,
                    })
                } else {
                    Ok(left)
                }
            }
            Chain::Or(l, r) => {
                let left = execute(l, registry, stdin.clone()).await?;
                if left.exit_code != 0 {
                    let right = execute(r, registry, stdin).await?;
                    Ok(CommandOutput {
                        stdout: merge(left.stdout, right.stdout),
                        stderr: merge(left.stderr, right.stderr),
                        exit_code: right.exit_code,
                    })
                } else {
                    Ok(left)
                }
            }
            Chain::Seq(l, r) => {
                let left = execute(l, registry, stdin.clone()).await?;
                let right = execute(r, registry, stdin).await?;
                Ok(CommandOutput {
                    stdout: merge(left.stdout, right.stdout),
                    stderr: merge(left.stderr, right.stderr),
                    exit_code: right.exit_code,
                })
            }
        }
    })
}

async fn run_command(
    argv: &[String],
    registry: &CommandRegistry,
    stdin: Vec<u8>,
) -> Result<CommandOutput> {
    let name = argv.first().map(|s| s.as_str()).unwrap_or_default();
    if name.is_empty() {
        return Ok(CommandOutput::failed(
            2,
            b"[error] empty command\n".to_vec(),
        ));
    }

    let Some(cmd) = registry.get(name) else {
        let avail = registry.sorted_names().join(", ");
        let msg = format!("[error] unknown command: {name}. Available: {avail}\n");
        return Ok(CommandOutput::failed(127, msg.into_bytes()));
    };

    let out = cmd
        .run(CommandInput {
            args: argv[1..].to_vec(),
            stdin,
        })
        .await?;

    // Prefix stderr with `[name]\t` so the caller can localize errors
    // across a chain.
    let stderr = if out.stderr.is_empty() {
        Vec::new()
    } else {
        prefix_stderr(name, &out.stderr)
    };
    Ok(CommandOutput {
        stdout: out.stdout,
        stderr,
        exit_code: out.exit_code,
    })
}

fn prefix_stderr(name: &str, raw: &[u8]) -> Vec<u8> {
    // One `[name]\t` prefix per newline-terminated chunk; if the final
    // byte isn't a newline, still prefix the trailing remainder.
    let mut out = Vec::with_capacity(raw.len() + name.len() + 4);
    let mut line_start = true;
    for &b in raw {
        if line_start {
            out.push(b'[');
            out.extend_from_slice(name.as_bytes());
            out.push(b']');
            out.push(b'\t');
            line_start = false;
        }
        out.push(b);
        if b == b'\n' {
            line_start = true;
        }
    }
    out
}

fn merge(mut a: Vec<u8>, b: Vec<u8>) -> Vec<u8> {
    a.extend_from_slice(&b);
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::parse_chain;
    use crate::command::{Command, CommandInput, CommandOutput, CommandRegistry};
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::Mutex;

    /// Fake command: returns a fixed exit code and stdout; records that
    /// it was invoked. `args[0]` (when present) overrides exit code —
    /// lets a single stub model "this run fails" vs "this run succeeds"
    /// in the same test.
    struct Stub {
        name: &'static str,
        stdout: Vec<u8>,
        exit_code: i32,
        invoked: Arc<Mutex<bool>>,
    }

    impl Stub {
        fn new(name: &'static str, stdout: impl Into<Vec<u8>>, exit_code: i32) -> Self {
            Self {
                name,
                stdout: stdout.into(),
                exit_code,
                invoked: Arc::new(Mutex::new(false)),
            }
        }
        fn invoked_flag(&self) -> Arc<Mutex<bool>> {
            self.invoked.clone()
        }
    }

    #[async_trait]
    impl Command for Stub {
        fn name(&self) -> &str {
            self.name
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput> {
            *self.invoked.lock().unwrap() = true;
            Ok(CommandOutput {
                stdout: self.stdout.clone(),
                stderr: Vec::new(),
                exit_code: self.exit_code,
            })
        }
    }

    /// Echoes stdin to stdout.
    struct Echo;
    #[async_trait]
    impl Command for Echo {
        fn name(&self) -> &str {
            "echo_stdin"
        }
        async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
            Ok(CommandOutput::ok(input.stdin))
        }
    }

    /// Counts newlines in stdin.
    struct LineCount;
    #[async_trait]
    impl Command for LineCount {
        fn name(&self) -> &str {
            "lc"
        }
        async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
            let n = input.stdin.iter().filter(|b| **b == b'\n').count();
            Ok(CommandOutput::ok(format!("{n}\n").into_bytes()))
        }
    }

    /// Emits bytes of configurable length — exercises PIPE_BUF_MAX.
    struct Flood(usize);
    #[async_trait]
    impl Command for Flood {
        fn name(&self) -> &str {
            "flood"
        }
        async fn run(&self, _input: CommandInput) -> Result<CommandOutput> {
            Ok(CommandOutput::ok(vec![b'x'; self.0]))
        }
    }

    fn registry() -> CommandRegistry {
        let mut r = CommandRegistry::new();
        r.register(Echo);
        r.register(LineCount);
        r
    }

    #[tokio::test]
    async fn pipe_threads_stdout_into_stdin() {
        let mut r = CommandRegistry::new();
        r.register(Stub::new("emit", b"a\nb\nc\n".to_vec(), 0));
        r.register(LineCount);
        let chain = parse_chain("emit | lc").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert_eq!(out.stdout, b"3\n");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn pipe_three_stages() {
        let mut r = CommandRegistry::new();
        r.register(Stub::new("emit", b"a\nb\nc\n".to_vec(), 0));
        r.register(Echo);
        r.register(LineCount);
        let chain = parse_chain("emit | echo_stdin | lc").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert_eq!(out.stdout, b"3\n");
    }

    #[tokio::test]
    async fn and_runs_right_only_on_success() {
        let mut r = CommandRegistry::new();
        let ok_flag = {
            let s = Stub::new("ok", b"first".to_vec(), 0);
            let f = s.invoked_flag();
            r.register(s);
            f
        };
        let right_flag = {
            let s = Stub::new("right", b"second".to_vec(), 0);
            let f = s.invoked_flag();
            r.register(s);
            f
        };
        let chain = parse_chain("ok && right").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert!(*ok_flag.lock().unwrap());
        assert!(*right_flag.lock().unwrap());
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"firstsecond");
    }

    #[tokio::test]
    async fn and_short_circuits_on_failure() {
        let mut r = CommandRegistry::new();
        r.register(Stub::new("bad", b"x".to_vec(), 1));
        let right_flag = {
            let s = Stub::new("right", b"never".to_vec(), 0);
            let f = s.invoked_flag();
            r.register(s);
            f
        };
        let chain = parse_chain("bad && right").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert!(!*right_flag.lock().unwrap(), "right must not run");
        assert_eq!(out.exit_code, 1);
    }

    #[tokio::test]
    async fn or_runs_right_only_on_failure() {
        let mut r = CommandRegistry::new();
        r.register(Stub::new("bad", b"boom".to_vec(), 2));
        r.register(Stub::new("recover", b"ok".to_vec(), 0));
        let chain = parse_chain("bad || recover").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"boomok");
    }

    #[tokio::test]
    async fn or_short_circuits_on_success() {
        let mut r = CommandRegistry::new();
        r.register(Stub::new("good", b"g".to_vec(), 0));
        let right_flag = {
            let s = Stub::new("right", b"never".to_vec(), 0);
            let f = s.invoked_flag();
            r.register(s);
            f
        };
        let chain = parse_chain("good || right").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert!(!*right_flag.lock().unwrap());
        assert_eq!(out.stdout, b"g");
    }

    #[tokio::test]
    async fn seq_runs_both_regardless_of_exit() {
        let mut r = CommandRegistry::new();
        r.register(Stub::new("first", b"a\n".to_vec(), 5));
        r.register(Stub::new("second", b"b\n".to_vec(), 0));
        let chain = parse_chain("first ; second").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"a\nb\n");
    }

    #[tokio::test]
    async fn unknown_command_returns_127_with_available_list() {
        let r = registry();
        let chain = parse_chain("nope").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert_eq!(out.exit_code, 127);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] unknown command: nope"), "{stderr}");
        assert!(stderr.contains("echo_stdin"), "{stderr}");
        assert!(stderr.contains("lc"), "{stderr}");
    }

    #[tokio::test]
    async fn unknown_command_triggers_or_fallback() {
        let mut r = CommandRegistry::new();
        r.register(Stub::new("fallback", b"rescued\n".to_vec(), 0));
        let chain = parse_chain("nope || fallback").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert_eq!(out.exit_code, 0);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("rescued"), "{stdout}");
    }

    #[tokio::test]
    async fn unknown_then_and_shortcircuits() {
        let mut r = CommandRegistry::new();
        let right_flag = {
            let s = Stub::new("right", b"never".to_vec(), 0);
            let f = s.invoked_flag();
            r.register(s);
            f
        };
        let chain = parse_chain("nope && right").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert!(!*right_flag.lock().unwrap(), "right must not run");
        assert_eq!(out.exit_code, 127);
    }

    #[tokio::test]
    async fn empty_stdin_through_lc_is_zero() {
        let r = registry();
        let chain = parse_chain("echo_stdin | lc").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert_eq!(out.stdout, b"0\n");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn pipe_buf_max_guards_against_flood() {
        let mut r = CommandRegistry::new();
        r.register(Flood(PIPE_BUF_MAX + 1024));
        r.register(LineCount);
        let chain = parse_chain("flood | lc").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert_eq!(out.exit_code, 141);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[pipe]"), "{stderr}");
        assert!(stderr.contains("exceeded"), "{stderr}");
    }

    #[tokio::test]
    async fn combined_precedence_and_short_circuit() {
        // `a && b | c || d`
        //  → Or(And(a, Pipe(b, c)), d)  [given `&&` < `|` but `&&`/`||` same prec left-assoc]
        //  Actually with our grammar: andor is left-assoc over && and ||,
        //  so: ((a && (b | c)) || d)
        //  Run `a` (ok 0), then `b | c`: `b` emits "hi\n" → `lc` → "1\n", exit 0.
        //  Or short-circuits (exit 0), d doesn't run.
        let mut r = CommandRegistry::new();
        r.register(Stub::new("a", Vec::new(), 0));
        r.register(Stub::new("b", b"hi\n".to_vec(), 0));
        r.register(LineCount);
        let d_flag = {
            let s = Stub::new("d", b"never".to_vec(), 0);
            let f = s.invoked_flag();
            r.register(s);
            f
        };
        let chain = parse_chain("a && b | lc || d").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert!(!*d_flag.lock().unwrap());
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"1\n");
    }

    #[tokio::test]
    async fn stderr_is_prefixed_with_command_name() {
        struct Err1;
        #[async_trait]
        impl Command for Err1 {
            fn name(&self) -> &str {
                "boom"
            }
            async fn run(&self, _: CommandInput) -> Result<CommandOutput> {
                Ok(CommandOutput::failed(1, b"something went wrong\n".to_vec()))
            }
        }
        let mut r = CommandRegistry::new();
        r.register(Err1);
        let chain = parse_chain("boom").unwrap();
        let out = execute(&chain, &r, Vec::new()).await.unwrap();
        assert_eq!(out.exit_code, 1);
        assert_eq!(out.stderr, b"[boom]\tsomething went wrong\n");
    }
}

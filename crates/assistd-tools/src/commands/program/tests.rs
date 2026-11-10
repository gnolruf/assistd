use super::*;
use crate::chain::{execute, parse_chain};
use crate::command::CommandRegistry;
use crate::commands::{RecordingGate, register_readers};

fn registry_asking(gate: &Arc<RecordingGate>) -> CommandRegistry {
    let gate: Arc<dyn ConfirmationGate> = gate.clone();
    let mut registry = CommandRegistry::new();
    register_readers(
        &mut registry,
        &Arc::new(BashPolicyCfg::default()),
        &SandboxInfo::none(),
        &gate,
    );
    registry
}

async fn run(registry: &CommandRegistry, line: &str) -> CommandOutput {
    execute(&parse_chain(line).expect("parses"), registry, None).await
}

#[tokio::test]
async fn arguments_reach_the_program_as_words_not_code() {
    let dir = tempfile::tempdir().expect("tempdir");
    let planted = dir.path().join("planted");
    let gate = RecordingGate::new(false);
    let registry = registry_asking(&gate);
    let line = format!(
        "echo 'a; touch {0}' | grep -c \"; touch {0}\"",
        planted.display()
    );

    let out = run(&registry, &line).await;

    assert_eq!(out.exit_code, 0, "{out:?}");
    assert_eq!(out.stdout, b"1\n");
    assert!(!planted.exists());
    assert!(gate.prompts().is_empty(), "{:?}", gate.prompts());
}

#[tokio::test]
async fn a_writing_option_asks_and_a_refusal_stops_the_run() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("lines");
    std::fs::write(&source, b"b\na\n").expect("source file");
    let written = dir.path().join("sorted");
    let gate = RecordingGate::new(false);
    let registry = registry_asking(&gate);
    let line = format!("sort -o {} {}", written.display(), source.display());

    let out = run(&registry, &line).await;

    assert_eq!(out.exit_code, 126, "{out:?}");
    assert!(!written.exists());
    let prompts = gate.prompts();
    assert_eq!(prompts.len(), 1, "{prompts:?}");
    assert_eq!(
        (prompts[0].0.as_str(), prompts[0].1.as_str()),
        ("sort", line.as_str())
    );
}

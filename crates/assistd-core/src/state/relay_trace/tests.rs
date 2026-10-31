use serde_json::json;

use super::*;

fn plan() -> Vec<RelayStepPlan> {
    ["fetch the notes", "compare them", "summarise"]
        .into_iter()
        .enumerate()
        .map(|(index, task)| RelayStepPlan {
            task: task.into(),
            profile: if index == 0 {
                RelayProfile::Fetch
            } else {
                RelayProfile::Read
            },
        })
        .collect()
}

#[test]
fn trace_merges_deltas_pairs_results_and_stores_an_unfinished_step_as_stopped() {
    let mut trace = RelayTraceBuilder::planned(&plan());
    let events = [
        RelayEvent::Started { step: 1 },
        RelayEvent::ReasoningDelta {
            text: "grep ".into(),
        },
        RelayEvent::ReasoningDelta {
            text: "first".into(),
        },
        RelayEvent::ToolCall {
            name: "run".into(),
            args: json!({"command": "grep -rn x /in"}),
        },
        RelayEvent::ToolResult {
            name: "run".into(),
            result: json!({"output": "none", "exit_code": 1, "duration_ms": 12}),
        },
        RelayEvent::Delta {
            text: "Nothing ".into(),
        },
        RelayEvent::Delta {
            text: "found.".into(),
        },
        RelayEvent::ContextSize { total: 4100 },
        RelayEvent::Answered {
            answer_chars: 14,
            tool_steps: 1,
            truncated: false,
        },
        RelayEvent::Started { step: 2 },
        RelayEvent::ToolCall {
            name: "run".into(),
            args: json!({"command": "cat /in/a"}),
        },
        RelayEvent::Compacting,
    ];
    events.iter().for_each(|event| trace.record(event));
    let steps = trace.finish();

    let first = steps[0].run.as_ref().expect("step 1 ran");
    assert_eq!(first.end, PersistedRelayEnd::Answered { truncated: false });
    assert_eq!(first.tokens, Some(4100));
    let PersistedRelayEntry::Reasoning { text, .. } = &first.entries[0] else {
        panic!("reasoning first: {:?}", first.entries);
    };
    assert_eq!(text, "grep first");
    assert_eq!(
        first.entries[1..],
        [
            PersistedRelayEntry::Tool {
                name: "run".into(),
                arguments: json!({"command": "grep -rn x /in"}),
                output: "none".into(),
                outcome: Some(ToolOutcome {
                    exit_code: 1,
                    duration_ms: 12,
                }),
            },
            PersistedRelayEntry::Text("Nothing found.".into()),
        ]
    );

    let second = steps[1].run.as_ref().expect("step 2 started");
    assert_eq!(
        second.end,
        PersistedRelayEnd::Failed {
            message: STOPPED_MESSAGE.into()
        }
    );
    assert!(matches!(
        second.entries[..],
        [
            PersistedRelayEntry::Tool { outcome: None, .. },
            PersistedRelayEntry::Compaction { duration_ms: None },
        ]
    ));
    assert_eq!(steps[2].run, None, "step 3 never ran");
}

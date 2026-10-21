use super::*;

#[test]
fn wakeword_verbs_parse_and_quoting_enrolls_a_reserved_word() {
    let cases = [
        ("/wakeword", WakeWordCommand::Status),
        ("/wakeword  ", WakeWordCommand::Status),
        ("/wakeword test", WakeWordCommand::Test),
        ("/wakeword reset", WakeWordCommand::Reset),
        (
            "/wakeword hey jarvis",
            WakeWordCommand::Enroll("hey jarvis".into()),
        ),
        ("/wakeword \"test\"", WakeWordCommand::Enroll("test".into())),
        (
            "/wakeword \"reset now\"",
            WakeWordCommand::Enroll("reset now".into()),
        ),
        (
            "/wakeword test it",
            WakeWordCommand::Enroll("test it".into()),
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(
            SlashCommand::parse(input),
            Some(SlashCommand::WakeWord(expected)),
            "{input}"
        );
    }
}

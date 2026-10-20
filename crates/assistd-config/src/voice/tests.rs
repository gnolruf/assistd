use super::*;

#[test]
fn out_of_range_match_similarity_is_refused() {
    let parsed: WakeWordConfig =
        toml::from_str("match_similarity = 1.0\ndetector_threshold = 0.0\n")
            .expect("in-range similarity must parse");
    assert_eq!(parsed.match_similarity, crate::defaults::unit(1.0));

    for bad in ["1.5", "-0.1", "nan"] {
        let source = format!("match_similarity = {bad}\n");
        assert!(
            toml::from_str::<WakeWordConfig>(&source).is_err(),
            "match_similarity = {bad} must be refused"
        );
    }
}

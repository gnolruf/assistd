use super::*;

#[test]
fn out_of_range_match_similarity_is_refused() {
    for (edge, expected) in [("0.0", 0.0), ("1.0", 1.0)] {
        let parsed: WakeWordConfig = toml::from_str(&format!("match_similarity = {edge}\n"))
            .expect("in-range similarity must parse");
        assert_eq!(parsed.match_similarity.get(), expected);
    }

    for bad in ["1.5", "-0.1", "nan"] {
        let source = format!("match_similarity = {bad}\n");
        assert!(
            toml::from_str::<WakeWordConfig>(&source).is_err(),
            "match_similarity = {bad} must be refused"
        );
    }
}

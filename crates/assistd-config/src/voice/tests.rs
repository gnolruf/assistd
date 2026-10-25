use super::*;

#[test]
fn out_of_range_threshold_is_refused() {
    for (edge, expected) in [("0.0", 0.0), ("1.0", 1.0)] {
        let parsed: WakeWordConfig = toml::from_str(&format!("threshold = {edge}\n"))
            .expect("in-range threshold must parse");
        assert_eq!(parsed.threshold, crate::defaults::unit(expected));
    }

    for bad in ["1.5", "-0.1", "nan"] {
        let source = format!("threshold = {bad}\n");
        assert!(
            toml::from_str::<WakeWordConfig>(&source).is_err(),
            "threshold = {bad} must be refused"
        );
    }
}

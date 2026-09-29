//! Small text formatting helpers.

/// `bytes` as `B`, `KB`, `MB` or `GB`, with one decimal from megabytes up.
pub fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{}KB", bytes / KB)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_size_formats_expected_ranges() {
        assert_eq!(human_size(0), "0B");
        assert_eq!(human_size(500), "500B");
        assert_eq!(human_size(2048), "2KB");
        assert_eq!(human_size(1024 * 1024 * 3), "3.0MB");
        assert_eq!(human_size(1024 * 1024 * 1024 * 2), "2.0GB");
    }
}
